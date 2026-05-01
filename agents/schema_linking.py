"""
agents/schema_linking.py  — v2 (FIXED)

Root-cause fix:
  FIX-3: Old linker dropped tables silently when 70B gave a bad/incomplete JSON.
          The downstream query_plan and sql agents then wrote SQL against a partial
          schema — producing wrong joins and missing columns (car_1 failure pattern).

  Strategy:
    - Ask the model to be INCLUSIVE ("when in doubt, include the table").
    - After parsing, UNION the model's choice with any tables mentioned by name
      in the question (surface heuristic — catches obvious misses).
    - If JSON parse fails, fall back to ALL tables (safe default).
    - Cap linked schema at 6 tables max to stay token-efficient.
"""

import re
from agents.base_agent import BaseAgent

SYSTEM = (
    "You are a database expert. Identify the minimal but COMPLETE set of tables "
    "and columns needed to answer a SQL question. When in doubt about a table, "
    "INCLUDE it — it is better to include an extra table than to miss a required one."
)


class SchemaLinkingAgent(BaseAgent):

    def build_prompt(self, context: dict) -> str:
        question   = context["question"]
        schema_str = context["schema_str"]   # full schema string
        all_tables = context.get("all_tables", [])

        tables_hint = ", ".join(all_tables) if all_tables else "see schema below"

        return f"""Question: {question}

Available tables: {tables_hint}

Full Schema:
{schema_str}

Instructions:
1. List every table needed to answer the question (including JOIN tables).
2. When unsure, INCLUDE the table.
3. List only the columns actually needed.

Return ONLY valid JSON, no other text:
{{
  "relevant_tables": ["table1", "table2"],
  "relevant_columns": {{"table1": ["col_a", "col_b"], "table2": ["col_c"]}},
  "join_keys": [{{"from_table": "t1", "from_col": "id", "to_table": "t2", "to_col": "fk_id"}}],
  "reasoning": "one sentence"
}}"""

    def parse_response(self, raw: str) -> dict:
        parsed = self.extract_json(raw)
        if parsed and parsed.get("relevant_tables"):
            return parsed
        # Fallback: empty → caller will use all tables
        return {
            "relevant_tables" : [],
            "relevant_columns": {},
            "join_keys"       : [],
            "reasoning"       : "parse_failed_using_all_tables",
        }


def safe_link_schema(
    raw_linking: dict,
    question   : str,
    all_tables : list[str],
    full_schema: dict,
    db_executor,
) -> tuple[list[str], str]:
    """
    FIX-3 core logic.

    Returns (linked_table_names, linked_schema_str).
    Never returns an empty table list — falls back to all tables.

    Additional heuristic: if any table name appears verbatim (case-insensitive)
    in the question, force-include it even if the LLM missed it.
    """
    linked = raw_linking.get("relevant_tables", []) or []

    # Surface heuristic: include tables whose name appears in the question
    q_lower = question.lower()
    for t in all_tables:
        if t.lower() in q_lower and t not in linked:
            linked.append(t)

    # Validate against actual schema keys
    linked = [t for t in linked if t in full_schema]

    # Safety fallback: if still empty, use everything
    if not linked:
        linked = list(all_tables)

    # Cap at 8 tables (token budget)
    linked = linked[:8]

    schema_str = db_executor.schema_to_string(full_schema, linked)
    return linked, schema_str
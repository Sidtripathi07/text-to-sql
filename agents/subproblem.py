"""
agents/subproblem.py  — v2 (FIXED)

Fix: trimmed prompt to save tokens. The old prompt was ~400 tokens.
     With 8B and limited TPM, shorter prompts = more room for good output.
"""

from agents.base_agent import BaseAgent

SYSTEM = (
    "You are a SQL decomposition expert. Break a question into SQL clause "
    "subproblems. Output ONLY valid JSON."
)


class SubproblemAgent(BaseAgent):

    def build_prompt(self, context: dict) -> str:
        question      = context["question"]
        linked_schema = context.get("linked_schema_str", context.get("schema_str", ""))

        return f"""Question: {question}

Schema:
{linked_schema}

Decompose into SQL clauses. Only include clauses actually needed.
Return ONLY JSON (null for unused clauses):
{{
  "SELECT"  : "what to retrieve",
  "FROM"    : "main table",
  "JOIN"    : null,
  "WHERE"   : null,
  "GROUP_BY": null,
  "HAVING"  : null,
  "ORDER_BY": null,
  "LIMIT"   : null,
  "SUBQUERY": null
}}"""

    def parse_response(self, raw: str) -> dict:
        parsed = self.extract_json(raw)
        if parsed:
            return {k: v for k, v in parsed.items() if v and v != "null"}
        return {"SELECT": "*", "FROM": "unknown", "parse_error": True}
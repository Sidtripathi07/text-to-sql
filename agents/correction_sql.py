"""
agents/correction_sql.py  — v2 (FIXED)
"""

from agents.base_agent import BaseAgent

SYSTEM = (
    "You are a SQLite expert. Fix the SQL query per the correction plan. "
    "Return ONLY the corrected SQL — no explanation, no markdown."
)


class CorrectionSQLAgent(BaseAgent):

    def build_prompt(self, context: dict) -> str:
        question        = context["question"]
        schema_str      = context.get("linked_schema_str", context.get("schema_str", ""))
        failed_sql      = context.get("failed_sql", "")
        correction_plan = context.get("correction_plan", {})
        errors_found    = context.get("errors_identified", [])

        steps = correction_plan.get("steps", {})
        plan_str = "\n".join(
            f"  {k}: {v}" for k, v in steps.items()
        ) if steps else correction_plan.get("raw_plan", "Fix identified errors.")

        errors_str = ", ".join(errors_found) if errors_found else "see plan"

        return f"""Question: {question}

Schema:
{schema_str}

Incorrect SQL:
{failed_sql}

Errors to fix: {errors_str}

Correction steps:
{plan_str}

Write the corrected SQLite SQL. Return ONLY the SQL.
Corrected SQL:"""

    def parse_response(self, raw: str) -> str:
        return self.extract_sql(raw)
"""
agents/correction_plan.py  — v2 (FIXED)

Fix: The old correction agent sent the ENTIRE taxonomy (~800 tokens) every call.
     Under TPM limits this caused starvation. Now we send only the category names
     as a short reference and ask the model to select the most likely 1-3 codes.
     Full taxonomy is only fetched if the model flags a specific category.
"""

from agents.base_agent import BaseAgent
from taxonomy.error_taxonomy import TAXONOMY

SYSTEM = (
    "You are a SQL debugging expert. Identify the specific error in a failed SQL "
    "query and write a targeted correction plan. Be concise."
)

# Short reference — just category + subtype codes, no descriptions
SHORT_TAXONOMY = "\n".join(
    f"[{cat}]: " + ", ".join(info["subtypes"].keys())
    for cat, info in TAXONOMY.items()
)


class CorrectionPlanAgent(BaseAgent):

    def build_prompt(self, context: dict) -> str:
        question   = context["question"]
        schema_str = context.get("linked_schema_str", context.get("schema_str", ""))
        failed_sql = context.get("failed_sql", "")
        exec_error = context.get("exec_error", "Result does not match expected output.")
        attempt    = context.get("attempt", 1)

        return f"""Question: {question}

Schema:
{schema_str}

Failed SQL (attempt {attempt}):
{failed_sql}

Error: {exec_error}

Error code reference:
{SHORT_TAXONOMY}

Task (be brief):
1. State the 1-3 most likely error codes that apply.
2. One sentence explaining each error.
3. Numbered steps to fix.

ERRORS: [comma-separated codes]
ANALYSIS: [brief]
PLAN:
  STEP 1: ...
  STEP 2: ..."""

    def parse_response(self, raw: str) -> dict:
        import re
        result = {"raw_plan": raw, "errors_identified": [], "steps": {}}

        em = re.search(r"ERRORS\s*[:\-]\s*(.+)", raw, re.IGNORECASE)
        if em:
            result["errors_identified"] = [c.strip() for c in em.group(1).split(",")]

        steps = re.findall(
            r"STEP\s*(\d+)\s*:\s*(.+?)(?=STEP\s*\d+:|$)",
            raw, re.DOTALL | re.IGNORECASE,
        )
        for num, content in steps:
            result["steps"][f"step_{num}"] = content.strip()

        am = re.search(r"ANALYSIS\s*[:\-]\s*(.+?)(?=PLAN:|$)", raw,
                       re.DOTALL | re.IGNORECASE)
        if am:
            result["analysis"] = am.group(1).strip()

        return result
"""
agents/query_plan.py  — v2 (FIXED)

Fix: Added explicit SQLite constraint reminders so the plan never suggests
     unsupported syntax (ILIKE, FULL OUTER JOIN, etc.) that then break execution.
     Also shortened step format to save tokens.
"""

from agents.base_agent import BaseAgent

SYSTEM = (
    "You are a SQL query planner for SQLite. Create a precise step-by-step "
    "natural-language plan. Do NOT write any SQL code."
)

SQLITE_RULES = """SQLite constraints to respect:
- No ILIKE (use LIKE with LOWER())
- No FULL OUTER JOIN (use LEFT JOIN + UNION)
- No EXCEPT with different column counts
- String comparisons are case-sensitive unless LOWER() is used
- Use subqueries with IN/NOT IN for set operations when simpler"""


class QueryPlanAgent(BaseAgent):

    def build_prompt(self, context: dict) -> str:
        question    = context["question"]
        schema_str  = context.get("linked_schema_str", context.get("schema_str", ""))
        subproblems = context.get("subproblems", {})

        sub_str = "\n".join(
            f"  {k}: {v}" for k, v in subproblems.items()
        ) if subproblems else "  (no decomposition)"

        return f"""Question: {question}

Schema:
{schema_str}

Clause decomposition:
{sub_str}

{SQLITE_RULES}

Write a numbered execution plan in plain English. No SQL code.
STEP 1: Starting table and why
STEP 2: Joins needed (table, key, type)
STEP 3: WHERE filters
STEP 4: Aggregations / GROUP BY / HAVING
STEP 5: ORDER BY / LIMIT / set operations
STEP 6: What SELECT should return
REASONING: Why this plan answers the question"""

    def parse_response(self, raw: str) -> dict:
        import re
        steps = {}
        matches = re.findall(
            r"STEP\s*(\d+)\s*:\s*(.+?)(?=STEP\s*\d+:|REASONING:|$)",
            raw, re.DOTALL | re.IGNORECASE,
        )
        for num, content in matches:
            steps[f"step_{num}"] = content.strip()
        rm = re.search(r"REASONING\s*:\s*(.+?)$", raw, re.DOTALL | re.IGNORECASE)
        if rm:
            steps["reasoning"] = rm.group(1).strip()
        if not steps:
            steps = {"plan": raw.strip()}
        return steps
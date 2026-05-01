"""
agents/sql_agent.py  — v2 (FIXED)

Fixes:
  FIX-4: Added concrete few-shot examples for the 5 most common Spider patterns
          (COUNT, JOIN+GROUP, NOT IN, subquery MAX, INTERSECT-style HAVING).
          8B models are dramatically better with examples than with rules alone.
  FIX-5: Explicit "return ONLY SQL" repeated — 8B tends to add explanation text
          which breaks the extract_sql parser.
"""

from agents.base_agent import BaseAgent

FEW_SHOT = """-- Example 1: Simple count
-- Q: How many singers do we have?
SELECT COUNT(*) FROM singer

-- Example 2: JOIN + GROUP BY
-- Q: Show stadium name and number of concerts per stadium
SELECT T1.name, COUNT(T2.concert_id)
FROM stadium T1 JOIN concert T2 ON T1.stadium_id = T2.stadium_id
GROUP BY T1.stadium_id

-- Example 3: NOT IN subquery
-- Q: Show stadiums with no concerts
SELECT name FROM stadium
WHERE stadium_id NOT IN (SELECT stadium_id FROM concert)

-- Example 4: Subquery for max/min context
-- Q: Find weight of youngest dog
SELECT weight FROM pets
WHERE pettype = 'dog' AND pet_age = (SELECT MIN(pet_age) FROM pets WHERE pettype = 'dog')

-- Example 5: INTERSECT via double NOT IN avoided — use HAVING COUNT
-- Q: Students who have both cat and dog
SELECT fname FROM student
WHERE stuid IN (SELECT stuid FROM has_pet JOIN pets ON has_pet.petid=pets.petid WHERE pettype='cat')
  AND stuid IN (SELECT stuid FROM has_pet JOIN pets ON has_pet.petid=pets.petid WHERE pettype='dog')
"""

SYSTEM = (
    "You are a SQLite expert. Generate ONE valid SQLite SQL query. "
    "Return ONLY the SQL — no explanation, no markdown, no trailing semicolon."
)


class SQLAgent(BaseAgent):

    def build_prompt(self, context: dict) -> str:
        question   = context["question"]
        schema_str = context.get("linked_schema_str", context.get("schema_str", ""))
        plan       = context.get("query_plan", {})

        plan_str = "\n".join(
            f"  {k}: {v}" for k, v in plan.items()
        ) if isinstance(plan, dict) else str(plan)

        return f"""Schema:
{schema_str}

Reference SQL patterns:
{FEW_SHOT}

Execution plan:
{plan_str}

Question: {question}

Write the SQLite SQL query now. Return ONLY the SQL, nothing else.
SQL:"""

    def parse_response(self, raw: str) -> str:
        return self.extract_sql(raw)
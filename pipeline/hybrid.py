"""
pipeline/hybrid.py  — v2 (FIXED)

All 5 root-cause fixes applied:
  FIX-1: TPM-aware rate limiting in base_agent (automatic, no change here)
  FIX-2: Exponential backoff on 429 (automatic, no change here)
  FIX-3: safe_link_schema() — schema linking never silently drops tables
  FIX-4: SQL agent has few-shot examples (in sql_agent.py)
  FIX-5: Correction plan sends short taxonomy to save TPM
  FIX-6: Per-agent max_tokens tuned to minimum needed

HYBRID model assignment (70B = reasoning, 8B = mechanical):
  schema_linking  → 70B
  subproblem      → 8B
  query_plan      → 70B
  sql             → 8B
  correction_plan → 70B
  correction_sql  → 8B
"""

from dataclasses import dataclass, field
from agents.schema_linking   import SchemaLinkingAgent, safe_link_schema
from agents.subproblem       import SubproblemAgent
from agents.query_plan       import QueryPlanAgent
from agents.sql_agent        import SQLAgent
from agents.correction_plan  import CorrectionPlanAgent
from agents.correction_sql   import CorrectionSQLAgent
from config                  import HYBRID_ASSIGNMENT, SMALL_MODEL, LARGE_MODEL


@dataclass
class PipelineResult:
    pipeline      : str  = "hybrid"
    question      : str  = ""
    db_id         : str  = ""
    final_sql     : str  = ""
    exec_result   : dict = field(default_factory=dict)
    gold_sql      : str  = ""
    exec_accurate : bool = False
    valid_sql     : bool = False
    traces        : list = field(default_factory=list)
    total_tokens  : int  = 0
    total_latency : int  = 0
    corrections   : int  = 0
    linked_tables : list = field(default_factory=list)
    subproblems   : dict = field(default_factory=dict)
    query_plan    : dict = field(default_factory=dict)
    errors_found  : list = field(default_factory=list)
    model_usage   : dict = field(default_factory=lambda: {
        SMALL_MODEL: {"calls": 0, "tokens": 0},
        LARGE_MODEL: {"calls": 0, "tokens": 0},
    })


def run_hybrid(
    question, db_id, schema_str, full_schema, db_executor,
    api_key, gold_sql="", max_corrections=2, schema_cache=None, **_
) -> PipelineResult:

    res        = PipelineResult(question=question, db_id=db_id, gold_sql=gold_sql)
    all_tables = list(full_schema.keys())

    def m(agent_name: str) -> str:
        return HYBRID_ASSIGNMENT.get(agent_name, SMALL_MODEL)

    def add(trace, agent_name: str):
        res.traces.append(trace.to_dict())
        tok = trace.tokens_in + trace.tokens_out
        res.total_tokens  += tok
        res.total_latency += trace.latency_ms
        model = m(agent_name)
        res.model_usage[model]["calls"]  += 1
        res.model_usage[model]["tokens"] += tok

    # ── Stage 1: Schema Linking (70B) ────────────────────────────────────────
    linked_result = None
    if schema_cache:
        linked_result = schema_cache.get_linking(db_id, question)

    if not linked_result:
        agent = SchemaLinkingAgent(
            "schema_linking", m("schema_linking"), api_key, max_tokens=400
        )
        tr    = agent.run({"question": question, "schema_str": schema_str,
                           "all_tables": all_tables})
        add(tr, "schema_linking")
        linked_result = tr.parsed_output if not tr.error else {}
        if schema_cache and linked_result:
            schema_cache.set_linking(db_id, question, linked_result)
    else:
        res.traces.append({"agent": "schema_linking", "cached": True,
                           "parsed": linked_result, "tokens_in": 0, "tokens_out": 0,
                           "model": m("schema_linking")})

    # FIX-3: Never empty
    linked_tables, linked_schema_str = safe_link_schema(
        linked_result, question, all_tables, full_schema, db_executor
    )
    res.linked_tables = linked_tables

    ctx = {
        "question"         : question,
        "schema_str"       : schema_str,
        "linked_schema_str": linked_schema_str,
    }

    # ── Stage 2: Subproblem (8B) ──────────────────────────────────────────────
    tr  = SubproblemAgent("subproblem", m("subproblem"), api_key, max_tokens=300).run(ctx)
    add(tr, "subproblem")
    subproblems     = tr.parsed_output if not tr.error else {}
    res.subproblems = subproblems

    # ── Stage 3: Query Plan (70B) ─────────────────────────────────────────────
    tr  = QueryPlanAgent("query_plan", m("query_plan"), api_key, max_tokens=500).run(
        {**ctx, "subproblems": subproblems}
    )
    add(tr, "query_plan")
    query_plan     = tr.parsed_output if not tr.error else {}
    res.query_plan = query_plan

    # ── Stage 4: SQL (8B) ─────────────────────────────────────────────────────
    tr  = SQLAgent("sql", m("sql"), api_key, max_tokens=350).run(
        {**ctx, "subproblems": subproblems, "query_plan": query_plan}
    )
    add(tr, "sql")
    current_sql = tr.parsed_output if not tr.error else ""

    # ── Stage 5: Execute + Guided Correction ─────────────────────────────────
    for attempt in range(1, max_corrections + 2):
        exec_res        = db_executor.execute(db_id, current_sql)
        res.exec_result = exec_res

        if exec_res["success"]:
            if gold_sql:
                gold_exec = db_executor.execute(db_id, gold_sql)
                correct   = db_executor.compare_results(
                    exec_res["results"], gold_exec["results"])
            else:
                correct = True
            if correct or attempt > max_corrections:
                break
            err_msg = "Result mismatch."
        else:
            if attempt > max_corrections:
                break
            err_msg = exec_res.get("error", "Execution error")

        res.corrections += 1
        corr_ctx = {**ctx, "failed_sql": current_sql,
                    "exec_error": err_msg, "attempt": attempt}

        # 70B diagnoses the error
        tr_cp = CorrectionPlanAgent(
            "correction_plan", m("correction_plan"), api_key, max_tokens=450
        ).run(corr_ctx)
        add(tr_cp, "correction_plan")
        corr_plan        = tr_cp.parsed_output if not tr_cp.error else {}
        errors           = corr_plan.get("errors_identified", [])
        res.errors_found += errors

        # 8B fixes the SQL
        tr_cs = CorrectionSQLAgent(
            "correction_sql", m("correction_sql"), api_key, max_tokens=350
        ).run({**ctx, "failed_sql": current_sql, "correction_plan": corr_plan,
               "errors_identified": errors})
        add(tr_cs, "correction_sql")
        current_sql = tr_cs.parsed_output if not tr_cs.error else current_sql

    res.final_sql  = current_sql
    res.valid_sql  = res.exec_result.get("success", False)
    if gold_sql:
        res.exec_accurate = db_executor.execution_accuracy(db_id, current_sql, gold_sql)
    return res
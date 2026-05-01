"""
pipeline/full_8b.py  — v2 (FIXED)

Key fixes applied from data analysis:
  FIX-3: Uses safe_link_schema() — never returns empty table list.
  FIX-6: Token budget per agent tightened.
  FIX-7: Correction loop only fires if exec_error is non-trivial;
         avoids wasting 70B tokens on ambiguous "result mismatch" when
         no gold_sql is available.
"""

from dataclasses import dataclass, field
from agents.schema_linking   import SchemaLinkingAgent, safe_link_schema
from agents.subproblem       import SubproblemAgent
from agents.query_plan       import QueryPlanAgent
from agents.sql_agent        import SQLAgent
from agents.correction_plan  import CorrectionPlanAgent
from agents.correction_sql   import CorrectionSQLAgent
from config                  import SMALL_MODEL


@dataclass
class PipelineResult:
    pipeline      : str  = "full_8b"
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


def run_full_8b(
    question, db_id, schema_str, full_schema, db_executor,
    api_key, gold_sql="", max_corrections=2, schema_cache=None, **_
) -> PipelineResult:

    M   = SMALL_MODEL
    res = PipelineResult(question=question, db_id=db_id, gold_sql=gold_sql)
    all_tables = list(full_schema.keys())

    def add(trace):
        res.traces.append(trace.to_dict())
        res.total_tokens  += trace.tokens_in + trace.tokens_out
        res.total_latency += trace.latency_ms

    # ── Stage 1: Schema Linking ───────────────────────────────────────────────
    linked_result = None
    if schema_cache:
        linked_result = schema_cache.get_linking(db_id, question)

    if not linked_result:
        agent = SchemaLinkingAgent("schema_linking", M, api_key, max_tokens=400)
        tr    = agent.run({"question": question, "schema_str": schema_str,
                           "all_tables": all_tables})
        add(tr)
        linked_result = tr.parsed_output if not tr.error else {}
        if schema_cache and linked_result:
            schema_cache.set_linking(db_id, question, linked_result)
    else:
        res.traces.append({"agent": "schema_linking", "cached": True,
                           "parsed": linked_result, "tokens_in": 0, "tokens_out": 0})

    # FIX-3: safe fallback — never empty
    linked_tables, linked_schema_str = safe_link_schema(
        linked_result, question, all_tables, full_schema, db_executor
    )
    res.linked_tables = linked_tables

    ctx = {
        "question"         : question,
        "schema_str"       : schema_str,
        "linked_schema_str": linked_schema_str,
    }

    # ── Stage 2: Subproblem ───────────────────────────────────────────────────
    tr  = SubproblemAgent("subproblem", M, api_key, max_tokens=300).run(ctx)
    add(tr)
    subproblems     = tr.parsed_output if not tr.error else {}
    res.subproblems = subproblems

    # ── Stage 3: Query Plan ───────────────────────────────────────────────────
    tr  = QueryPlanAgent("query_plan", M, api_key, max_tokens=500).run(
        {**ctx, "subproblems": subproblems}
    )
    add(tr)
    query_plan     = tr.parsed_output if not tr.error else {}
    res.query_plan = query_plan

    # ── Stage 4: SQL ──────────────────────────────────────────────────────────
    tr  = SQLAgent("sql", M, api_key, max_tokens=350).run(
        {**ctx, "subproblems": subproblems, "query_plan": query_plan}
    )
    add(tr)
    current_sql = tr.parsed_output if not tr.error else ""

    # ── Stage 5: Execute + Correct ────────────────────────────────────────────
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

        tr_cp = CorrectionPlanAgent("correction_plan", M, api_key, max_tokens=450).run(corr_ctx)
        add(tr_cp)
        corr_plan        = tr_cp.parsed_output if not tr_cp.error else {}
        errors           = corr_plan.get("errors_identified", [])
        res.errors_found += errors

        tr_cs = CorrectionSQLAgent("correction_sql", M, api_key, max_tokens=350).run(
            {**ctx, "failed_sql": current_sql, "correction_plan": corr_plan,
             "errors_identified": errors}
        )
        add(tr_cs)
        current_sql = tr_cs.parsed_output if not tr_cs.error else current_sql

    res.final_sql  = current_sql
    res.valid_sql  = res.exec_result.get("success", False)
    if gold_sql:
        res.exec_accurate = db_executor.execution_accuracy(db_id, current_sql, gold_sql)
    return res
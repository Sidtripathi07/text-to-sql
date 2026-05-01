"""
pipeline/baseline.py  — v2
Single 8B call. No changes to logic, but uses fixed base_agent.
"""

from dataclasses import dataclass, field
from agents.base_agent import BaseAgent, AgentTrace
from config import SMALL_MODEL


@dataclass
class PipelineResult:
    pipeline      : str  = "baseline"
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
    errors_found  : list = field(default_factory=list)


class _DirectSQLAgent(BaseAgent):
    def build_prompt(self, context: dict) -> str:
        return (
            f"Schema:\n{context['schema_str']}\n\n"
            f"Question: {context['question']}\n\n"
            "Write a single valid SQLite SQL query. "
            "Return ONLY the SQL, no explanation.\nSQL:"
        )

    def parse_response(self, raw: str) -> str:
        return self.extract_sql(raw)


def run_baseline(question, db_id, schema_str, db_executor, api_key, gold_sql="",
                 full_schema=None, schema_cache=None, **_) -> PipelineResult:
    res   = PipelineResult(question=question, db_id=db_id, gold_sql=gold_sql)
    agent = _DirectSQLAgent("baseline_sql", SMALL_MODEL, api_key, max_tokens=350)
    trace = agent.run({"question": question, "schema_str": schema_str})
    res.traces.append(trace.to_dict())
    if trace.error:
        return res
    sql              = trace.parsed_output
    res.final_sql    = sql
    res.total_tokens = trace.tokens_in + trace.tokens_out
    res.total_latency= trace.latency_ms
    exec_res         = db_executor.execute(db_id, sql)
    res.exec_result  = exec_res
    res.valid_sql    = exec_res["success"]
    if gold_sql:
        res.exec_accurate = db_executor.execution_accuracy(db_id, sql, gold_sql)
    return res
"""
evaluation/evaluator.py  — v2 (FIXED)

Additions:
  - Results saved to disk incrementally (resume after crash/limit hit)
  - Live token budget monitor printed per sample
  - progress_cb receives running EA so UI can show live accuracy
"""

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional
from database.db_executor import DBExecutor
from evaluation.spider_loader import SpiderLoader
from cache.schema_cache import SchemaCache


@dataclass
class EvalMetrics:
    pipeline        : str  = ""
    total           : int  = 0
    exec_accurate   : int  = 0
    valid_sql       : int  = 0
    total_tokens    : int  = 0
    total_latency_ms: int  = 0
    corrections_made: int  = 0
    per_difficulty  : dict = field(default_factory=dict)
    per_db          : dict = field(default_factory=dict)
    error_codes     : dict = field(default_factory=dict)
    results         : list = field(default_factory=list)

    @property
    def ea(self):
        return (self.exec_accurate / self.total * 100) if self.total else 0.0

    @property
    def valid_pct(self):
        return (self.valid_sql / self.total * 100) if self.total else 0.0

    @property
    def avg_tokens(self):
        return (self.total_tokens / self.total) if self.total else 0.0

    @property
    def avg_latency_ms(self):
        return (self.total_latency_ms / self.total) if self.total else 0.0

    def summary(self) -> dict:
        return {
            "pipeline"      : self.pipeline,
            "total"         : self.total,
            "exec_accuracy" : round(self.ea, 2),
            "valid_sql_pct" : round(self.valid_pct, 2),
            "avg_tokens"    : round(self.avg_tokens, 1),
            "avg_latency_ms": round(self.avg_latency_ms, 1),
            "corrections"   : self.corrections_made,
            "per_difficulty": {
                k: {
                    "total"       : v["total"],
                    "exec_accurate": v["exec_accurate"],
                    "ea"          : round(v["exec_accurate"] / v["total"] * 100, 2)
                                    if v["total"] else 0.0,
                }
                for k, v in self.per_difficulty.items()
            },
        }


class Evaluator:

    def __init__(self, spider_root: str, api_key: str, use_cache: bool = True,
                 checkpoint_dir: str = ".checkpoints"):
        self.spider_root    = spider_root
        self.api_key        = api_key
        self.db_executor    = DBExecutor(spider_root)
        self.loader         = SpiderLoader(spider_root)
        self.cache          = SchemaCache() if use_cache else None
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _schema(self, db_id: str):
        if self.cache:
            cached = self.cache.get_raw_schema(db_id)
            if cached:
                return cached, self.db_executor.schema_to_string(cached)
        schema = self.db_executor.get_schema_raw(db_id)
        if self.cache:
            self.cache.set_raw_schema(db_id, schema)
        return schema, self.db_executor.schema_to_string(schema)

    def _ckpt_path(self, pipeline_name: str) -> Path:
        return self.checkpoint_dir / f"ckpt_{pipeline_name}.json"

    def _save_checkpoint(self, pipeline_name: str, results: list):
        self._ckpt_path(pipeline_name).write_text(json.dumps(results, indent=2))

    def _load_checkpoint(self, pipeline_name: str) -> list:
        p = self._ckpt_path(pipeline_name)
        if p.exists():
            try:
                return json.loads(p.read_text())
            except Exception:
                pass
        return []

    def run(
        self,
        pipeline_fn   : Callable,
        pipeline_name : str,
        dataset       : list[dict],
        max_samples   : int      = 100,
        progress_cb   : Optional[Callable] = None,
        resume        : bool     = True,
    ) -> EvalMetrics:

        metrics = EvalMetrics(pipeline=pipeline_name)
        data    = dataset[:max_samples]

        # Resume from checkpoint if available
        done_results = self._load_checkpoint(pipeline_name) if resume else []
        done_indices = {r["index"] for r in done_results}

        # Pre-populate metrics from checkpoint
        for r in done_results:
            metrics.results.append(r)
            metrics.total          += 1
            metrics.exec_accurate  += int(r["exec_accurate"])
            metrics.valid_sql      += int(r["valid_sql"])
            metrics.total_tokens   += r.get("tokens", 0)
            metrics.corrections_made += r.get("corrections", 0)
            d = r.get("difficulty", "unknown")
            if d not in metrics.per_difficulty:
                metrics.per_difficulty[d] = {"total": 0, "exec_accurate": 0}
            metrics.per_difficulty[d]["total"] += 1
            if r["exec_accurate"]:
                metrics.per_difficulty[d]["exec_accurate"] += 1

        for i, sample in enumerate(data):
            global_idx = i + 1
            if global_idx in done_indices:
                continue   # already computed — skip

            question = sample["question"]
            db_id    = sample["db_id"]
            gold_sql = sample["query"]

            try:
                full_schema, schema_str = self._schema(db_id)
                result = pipeline_fn(
                    question     = question,
                    db_id        = db_id,
                    schema_str   = schema_str,
                    full_schema  = full_schema,
                    db_executor  = self.db_executor,
                    api_key      = self.api_key,
                    gold_sql     = gold_sql,
                    schema_cache = self.cache,
                )
            except Exception as e:
                result = type("R", (), {
                    "exec_accurate": False, "valid_sql": False,
                    "final_sql": "", "total_tokens": 0, "total_latency": 0,
                    "corrections": 0, "errors_found": [], "traces": [{"error": str(e)}],
                })()

            difficulty = sample.get("difficulty", "unknown")
            if difficulty not in metrics.per_difficulty:
                metrics.per_difficulty[difficulty] = {"total": 0, "exec_accurate": 0}
            metrics.per_difficulty[difficulty]["total"] += 1
            if result.exec_accurate:
                metrics.exec_accurate += 1
                metrics.per_difficulty[difficulty]["exec_accurate"] += 1
            if result.valid_sql:
                metrics.valid_sql += 1

            tok = getattr(result, "total_tokens", 0)
            metrics.total          += 1
            metrics.total_tokens   += tok
            metrics.total_latency_ms += getattr(result, "total_latency", 0)
            metrics.corrections_made += getattr(result, "corrections", 0)

            for code in getattr(result, "errors_found", []):
                metrics.error_codes[code] = metrics.error_codes.get(code, 0) + 1

            if db_id not in metrics.per_db:
                metrics.per_db[db_id] = {"total": 0, "exec_accurate": 0}
            metrics.per_db[db_id]["total"] += 1
            if result.exec_accurate:
                metrics.per_db[db_id]["exec_accurate"] += 1

            row = {
                "index"        : global_idx,
                "question"     : question,
                "db_id"        : db_id,
                "gold_sql"     : gold_sql,
                "pred_sql"     : getattr(result, "final_sql", ""),
                "exec_accurate": result.exec_accurate,
                "valid_sql"    : result.valid_sql,
                "difficulty"   : difficulty,
                "tokens"       : tok,
                "corrections"  : getattr(result, "corrections", 0),
                "traces"       : getattr(result, "traces", []),
            }
            metrics.results.append(row)
            self._save_checkpoint(pipeline_name, metrics.results)

            if progress_cb:
                progress_cb(metrics.total, len(data), row, metrics.ea)

        return metrics
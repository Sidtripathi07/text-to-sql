"""
database/db_executor.py
Executes SQL queries against Spider's SQLite databases and compares results.
"""

import os
import sqlite3
import json
from pathlib import Path
from typing import Optional


class DBExecutor:
    """
    Manages SQLite connections and executes SQL for the Spider benchmark.
    """

    def __init__(self, spider_root: str):
        self.spider_root = Path(spider_root)
        self.db_root     = self.spider_root / "database"
        self._connections: dict[str, sqlite3.Connection] = {}

    # ── Connection management ─────────────────────────────────────────────────

    def get_connection(self, db_id: str) -> sqlite3.Connection:
        if db_id not in self._connections:
            db_path = self.db_root / db_id / f"{db_id}.sqlite"
            if not db_path.exists():
                raise FileNotFoundError(
                    f"SQLite file not found: {db_path}\n"
                    f"Download Spider dataset and set spider_data_path in config."
                )
            conn = sqlite3.connect(str(db_path), check_same_thread=False)
            conn.row_factory = sqlite3.Row
            self._connections[db_id] = conn
        return self._connections[db_id]

    def close_all(self):
        for conn in self._connections.values():
            conn.close()
        self._connections.clear()

    # ── Execution ─────────────────────────────────────────────────────────────

    def execute(self, db_id: str, sql: str, timeout: int = 10) -> dict:
        """
        Execute SQL against Spider DB. Returns:
        {
            "success"  : bool,
            "results"  : list[list],   # rows as lists
            "columns"  : list[str],
            "error"    : str | None,
            "row_count": int,
        }
        """
        result = {
            "success"  : False,
            "results"  : [],
            "columns"  : [],
            "error"    : None,
            "row_count": 0,
        }
        try:
            conn   = self.get_connection(db_id)
            cursor = conn.cursor()
            conn.execute(f"PRAGMA busy_timeout = {timeout * 1000}")
            cursor.execute(sql)
            rows          = cursor.fetchall()
            result["columns"]   = [d[0] for d in cursor.description] if cursor.description else []
            result["results"]   = [list(row) for row in rows]
            result["row_count"] = len(rows)
            result["success"]   = True
        except Exception as e:
            result["error"] = str(e)
        return result

    # ── Execution Accuracy (EA) comparison ───────────────────────────────────

    def compare_results(self, results_a: list, results_b: list) -> bool:
        """
        Compare two result sets for Execution Accuracy.
        Order-insensitive comparison of row sets (as frozensets).
        """
        def normalize(rows):
            return sorted(
                [tuple(str(v).strip().lower() if v is not None else "" for v in row)
                 for row in rows]
            )
        return normalize(results_a) == normalize(results_b)

    def execution_accuracy(self, db_id: str, pred_sql: str, gold_sql: str) -> bool:
        """Return True if predicted SQL produces same result as gold SQL."""
        pred = self.execute(db_id, pred_sql)
        gold = self.execute(db_id, gold_sql)
        if not pred["success"] or not gold["success"]:
            return False
        return self.compare_results(pred["results"], gold["results"])

    # ── Schema extraction ─────────────────────────────────────────────────────

    def get_schema_raw(self, db_id: str) -> dict:
        """
        Extract full schema from SQLite: tables → columns with types.
        Returns {table_name: {columns: [...], primary_keys: [...], foreign_keys: [...]}}
        """
        conn   = self.get_connection(db_id)
        cursor = conn.cursor()

        # Get tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()
                  if not row[0].startswith("sqlite_")]

        schema = {}
        for table in tables:
            cursor.execute(f"PRAGMA table_info(`{table}`)")
            cols_info = cursor.fetchall()
            columns   = []
            pks       = []
            for col in cols_info:
                cid, name, ctype, notnull, dflt, pk = col
                col_type = ctype or "TEXT"

                samples = []
                if "TEXT" in col_type.upper() or "VARCHAR" in col_type.upper():
                    try:
                        cursor.execute(f"SELECT DISTINCT `{name}` FROM `{table}` WHERE `{name}` IS NOT NULL LIMIT 2")
                        samples = [str(r[0]) for r in cursor.fetchall()]
                    except Exception:
                        pass # ignore if sample fetch fails
                columns.append({"name": name, "type": ctype or "TEXT"})
                if pk:
                    pks.append(name)

            cursor.execute(f"PRAGMA foreign_key_list(`{table}`)")
            fks_raw = cursor.fetchall()
            fks     = [{"from": r[3], "to_table": r[2], "to_col": r[4]} for r in fks_raw]

            schema[table] = {
                "columns"     : columns,
                "primary_keys": pks,
                "foreign_keys": fks,
            }
        return schema

    def schema_to_string(self, schema: dict, linked_tables: Optional[list] = None) -> str:
        """Convert schema dict to a compact string for LLM prompts."""
        lines = []
        tables = linked_tables if linked_tables else list(schema.keys())
        for table in tables:
            if table not in schema:
                continue
            info = schema[table]
            col_strs = []
            for c in info["columns"]:
                base = f"{c['name']} ({c['type']})"
                if c.get('samples'):
                    base += f" [e.g., '{c['samples'][0]}']"
                col_strs.append(base)   
            lines.append(f"TABLE {table}: {', '.join(col_strs)}")
            if info["primary_keys"]:
                lines.append(f"  PK: {', '.join(info['primary_keys'])}")
            if info["foreign_keys"]:
                for fk in info["foreign_keys"]:
                    lines.append(
                        f"  FK: {table}.{fk['from']} → {fk['to_table']}.{fk['to_col']}"
                    )
        return "\n".join(lines)
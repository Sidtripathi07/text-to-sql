"""
evaluation/spider_loader.py
Loads the Spider benchmark dataset (dev split) and its variants.
"""

import json
from pathlib import Path
from typing import Optional


class SpiderLoader:
    """
    Loads Spider dev set, Spider-Realistic, Spider-SYN.
    Expected Spider directory structure:
        spider/
          dev.json
          tables.json
          database/
            {db_id}/{db_id}.sqlite
    """

    def __init__(self, spider_root: str):
        self.root = Path(spider_root)

    def _load_json(self, path: Path) -> list:
        if not path.exists():
            raise FileNotFoundError(
                f"File not found: {path}\n"
                f"Download Spider from https://yale-lily.github.io/spider and "
                f"set spider_data_path in the sidebar."
            )
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def load_dev(self, max_samples: Optional[int] = None) -> list[dict]:
        """Load Spider dev set."""
        data = self._load_json(self.root / "dev.json")
        if max_samples:
            data = data[:max_samples]
        return self._normalize(data)

    def load_realistic(self, max_samples: Optional[int] = None) -> list[dict]:
        """Load Spider-Realistic (column names removed from questions)."""
        # Try common filenames
        for fname in ["spider_realistic.json", "realistic.json", "dev_realistic.json"]:
            p = self.root / fname
            if p.exists():
                data = self._load_json(p)
                if max_samples:
                    data = data[:max_samples]
                return self._normalize(data)
        # Fallback to dev
        print("Spider-Realistic not found, using dev set.")
        return self.load_dev(max_samples)

    def load_syn(self, max_samples: Optional[int] = None) -> list[dict]:
        """Load Spider-SYN (synonym substitutions)."""
        for fname in ["spider_syn.json", "syn.json", "dev_syn.json"]:
            p = self.root / fname
            if p.exists():
                data = self._load_json(p)
                if max_samples:
                    data = data[:max_samples]
                return self._normalize(data)
        print("Spider-SYN not found, using dev set.")
        return self.load_dev(max_samples)

    def _normalize(self, data: list) -> list[dict]:
        """Normalize Spider sample keys to a consistent format."""
        normalized = []
        for item in data:
            normalized.append({
                "question"   : item.get("question", item.get("SpiderQuestion", "")),
                "query"      : item.get("query", item.get("SpiderSQL", "")),
                "db_id"      : item.get("db_id", ""),
                "difficulty" : item.get("difficulty", "unknown"),
            })
        return normalized

    def load_tables(self) -> dict[str, dict]:
        """Load tables.json as db_id → schema info dict."""
        data    = self._load_json(self.root / "tables.json")
        schemas = {}
        for db in data:
            schemas[db["db_id"]] = db
        return schemas

    def get_difficulties(self, data: list) -> dict[str, int]:
        """Count questions by difficulty level."""
        counts: dict[str, int] = {}
        for item in data:
            d          = item.get("difficulty", "unknown")
            counts[d]  = counts.get(d, 0) + 1
        return counts

    def filter_by_difficulty(self, data: list, difficulty: str) -> list:
        return [d for d in data if d.get("difficulty", "") == difficulty]
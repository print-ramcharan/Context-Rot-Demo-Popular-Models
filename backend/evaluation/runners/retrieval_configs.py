"""Retrieval configuration constants and loader shared by the benchmark runners."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import logging
from pathlib import Path
from typing import Any, Dict

import yaml

logger = logging.getLogger(__name__)

# Resolve the experiments config relative to the parent of this file's directory
# (i.e. backend/evaluation), so the harness works regardless of CWD.
_CONFIG_ROOT = Path(__file__).resolve().parent.parent  # backend/evaluation
_CONFIG_PATH = _CONFIG_ROOT / "configs" / "experiments.yaml"

CONFIG_PATH = _CONFIG_PATH

CONFIGS = {
    "B1": {"label": "BM25 only", "mode": "bm25", "rerank": False},
    "B2": {"label": "Dense FAISS only", "mode": "dense", "rerank": False},
    "B3": {"label": "Hybrid RRF", "mode": "hybrid", "rerank": False},
    "B4": {"label": "Hybrid + Reranker", "mode": "hybrid", "rerank": True},
}


def load_config(path: str | Path = None) -> Dict[str, Any]:
    if path is None:
        path = CONFIG_PATH
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


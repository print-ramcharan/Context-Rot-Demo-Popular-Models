"""BEIR dataset download utilities."""

from pathlib import Path


def get_beir_paths(root: str | Path = "evaluation/data"):
    """Return per-dataset directories under the evaluation data root."""
    base = Path(root)
    base.mkdir(parents=True, exist_ok=True)
    return {name: base / name for name in ["scifact", "nfcorpus"]}

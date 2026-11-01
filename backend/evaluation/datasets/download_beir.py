"""Download BEIR corpora (scifact, nfcorpus) into a local data directory."""

import logging
from pathlib import Path

from beir import util

logger = logging.getLogger(__name__)

DATASETS = ["scifact", "nfcorpus"]

_URL_TEMPLATE = (
    "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"
)


def download_datasets(root: str | Path = "evaluation/data") -> dict[str, Path]:
    """Download each BEIR dataset and return a mapping of dataset name to path."""
    paths = {}
    for name in DATASETS:
        dest = Path(root) / name
        dest.mkdir(parents=True, exist_ok=True)
        url = _URL_TEMPLATE.format(name=name)
        logger.info("Downloading %s from %s", name, url)
        util.download_and_unzip(url, str(dest))
        paths[name] = dest
    return paths


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    downloaded = download_datasets()
    for name, path in downloaded.items():
        print(f"Downloaded: {name} -> {path}")

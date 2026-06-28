"""
User config for cyclo — stores per-Ndft optimal worker counts from a benchmark run.

Config is saved to ~/.cyclo/config.json and shared across all projects.
"""

import json
from pathlib import Path

CONFIG_PATH = Path.home() / ".cyclo" / "config.json"

_DEFAULT_WORKERS = 1  # conservative: threading overhead dominates for small FFTs


def load_config() -> dict | None:
    if not CONFIG_PATH.exists():
        return None
    with CONFIG_PATH.open() as f:
        return json.load(f)


def save_config(data: dict) -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CONFIG_PATH.open("w") as f:
        json.dump(data, f, indent=2)


def get_workers(Ndft: int) -> int:
    """
    Return the optimal scipy.fft workers value for a given Ndft.

    Looks up the benchmarked value from ~/.cyclo/config.json. If no config
    exists, or Ndft was not benchmarked, falls back to the nearest benchmarked
    size or a conservative default of 1.
    """
    config = load_config()
    if config is None:
        return _DEFAULT_WORKERS

    table: dict[str, int] = config.get("workers_by_ndft", {})
    if not table:
        return _DEFAULT_WORKERS

    # Exact match
    if str(Ndft) in table:
        return table[str(Ndft)]

    # Nearest benchmarked Ndft
    benchmarked = sorted(int(k) for k in table)
    nearest = min(benchmarked, key=lambda k: abs(k - Ndft))
    return table[str(nearest)]

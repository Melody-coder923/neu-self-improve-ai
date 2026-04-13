"""Shared utilities. No file-writing functions — all data lives in SQLite."""
from pathlib import Path


def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)

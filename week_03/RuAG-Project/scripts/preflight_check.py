import importlib
import json
import os
import sqlite3
import sys
from pathlib import Path
from typing import Dict, List

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


REQUIRED_PACKAGES_BASE = ["yaml", "sklearn", "tqdm"]
def ok(msg: str):
    print(f"[OK] {msg}")


def fail(msg: str):
    print(f"[FAIL] {msg}")


def warn(msg: str):
    print(f"[WARN] {msg}")


def check_api_key(required: bool) -> bool:
    if not required:
        ok("OPENAI_API_KEY check skipped (local provider mode).")
        return True
    key = os.getenv("OPENAI_API_KEY", "")
    if not key:
        fail("OPENAI_API_KEY is not set.")
        return False
    if len(key) < 20:
        warn("OPENAI_API_KEY seems unusually short. Please verify.")
    ok("OPENAI_API_KEY is set.")
    return True


def check_requirements(provider: str) -> bool:
    required = list(REQUIRED_PACKAGES_BASE)
    if provider == "openai":
        required.append("openai")
    elif provider == "local":
        required.extend(["transformers", "torch", "accelerate"])

    all_good = True
    for pkg in required:
        try:
            importlib.import_module(pkg)
            ok(f"Python package available: {pkg}")
        except Exception:
            fail(f"Missing Python package: {pkg}")
            all_good = False
    return all_good


def load_config(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"config file not found: {path}")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def check_config(cfg: Dict) -> bool:
    try:
        _ = cfg["data"]["sqlite_path"]
        _ = cfg["model"]["provider"]
        _ = cfg["model"]["name"]
        _ = cfg["experiment"]["relation_top_k"]
        ok("config.yaml required keys are present.")
        return True
    except Exception as e:
        fail(f"config.yaml missing required keys: {e}")
        return False


def check_sqlite_data(path: Path) -> bool:
    if not path.exists():
        fail(f"sqlite file not found: {path}")
        return False
    all_good = True
    with sqlite3.connect(path) as conn:
        table_names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "documents" not in table_names:
            fail("sqlite missing table: documents")
            return False
        if "relation_types" in table_names:
            n_rel = conn.execute("SELECT COUNT(*) FROM relation_types").fetchone()[0]
            if n_rel > 0:
                ok(f"sqlite relation_types present (rows={n_rel}).")
            else:
                warn("sqlite relation_types exists but empty; schema may fallback to auto-count mode.")
        if "filtered_docs" in table_names:
            n_f = conn.execute("SELECT COUNT(*) FROM filtered_docs").fetchone()[0]
            ok(f"sqlite filtered_docs present (rows={n_f}).")
        for split in ("train", "test"):
            n = conn.execute("SELECT COUNT(*) FROM documents WHERE split = ?", (split,)).fetchone()[0]
            if n <= 0:
                fail(f"sqlite split '{split}' has no rows")
                all_good = False
                continue
            sample_docs = conn.execute(
                "SELECT doc_id, content FROM documents WHERE split = ? ORDER BY doc_id LIMIT 3",
                (split,),
            ).fetchall()
            for i, (doc_id, content) in enumerate(sample_docs, start=1):
                try:
                    n_ent = conn.execute("SELECT COUNT(*) FROM entities WHERE doc_id = ?", (doc_id,)).fetchone()[0]
                    if not content or n_ent == 0:
                        fail(f"{split} sample doc {doc_id} has no content or entities")
                        all_good = False
                except Exception as e:
                    fail(f"{split} sample doc {doc_id} check error: {e}")
                    all_good = False
            if all_good:
                ok(f"sqlite split '{split}' basic parse check passed (rows={n}).")
    return all_good


def main():
    print("=== Preflight Check: RuAG Part 1 Baselines ===")
    root = Path(__file__).resolve().parents[1]
    cfg_path = root / "config.yaml"

    all_good = True

    # 1) Config (we need provider to decide key/package checks)
    try:
        cfg = load_config(cfg_path)
        all_good &= check_config(cfg)
    except Exception as e:
        fail(str(e))
        all_good = False
        cfg = None

    provider = (cfg or {}).get("model", {}).get("provider", "openai").lower().strip()
    if provider not in {"openai", "local"}:
        fail(f"Unsupported model.provider in config: {provider}")
        all_good = False
    else:
        ok(f"Provider mode: {provider}")

    # 2) API key
    all_good &= check_api_key(required=(provider == "openai"))

    # 3) Requirements
    all_good &= check_requirements(provider=provider)

    # 4) Sqlite data and raw parseability
    if cfg:
        sqlite_path = root / cfg["data"]["sqlite_path"]
        all_good &= check_sqlite_data(sqlite_path)

    print("==============================================")
    if all_good:
        print("[PASS] Preflight passed. You can run baselines now.")
        return
    print("[FAIL] Preflight failed. Fix issues above before running.")
    raise SystemExit(1)


if __name__ == "__main__":
    main()


import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from src.data_preprocessing import Example, examples_from_raw_records


def ensure_parent_dir(path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def init_db(sqlite_path: str):
    """Create all tables. Uses normalized documents/entities/relations (operate over DB via SQL)."""
    ensure_parent_dir(sqlite_path)
    with sqlite3.connect(sqlite_path) as conn:
        # Normalized tables: data stored in relational form, queried via SQL
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                doc_id TEXT NOT NULL,
                split TEXT NOT NULL,
                content TEXT NOT NULL,
                PRIMARY KEY (doc_id)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS entities (
                doc_id TEXT NOT NULL,
                name TEXT NOT NULL,
                ord_idx INTEGER NOT NULL,
                PRIMARY KEY (doc_id, name),
                FOREIGN KEY (doc_id) REFERENCES documents(doc_id)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS relations (
                doc_id TEXT NOT NULL,
                entity1 TEXT NOT NULL,
                relation TEXT NOT NULL,
                entity2 TEXT NOT NULL,
                FOREIGN KEY (doc_id) REFERENCES documents(doc_id)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS run_predictions (
                run_id TEXT NOT NULL,
                method TEXT NOT NULL,
                doc_id TEXT NOT NULL,
                gold_json TEXT NOT NULL,
                pred_json TEXT NOT NULL,
                raw_output TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS run_metrics (
                run_id TEXT NOT NULL,
                method TEXT NOT NULL,
                precision REAL NOT NULL,
                recall REAL NOT NULL,
                f1 REAL NOT NULL,
                tp INTEGER NOT NULL,
                fp INTEGER NOT NULL,
                fn INTEGER NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS relation_types (
                relation TEXT PRIMARY KEY,
                description TEXT NOT NULL,
                ord INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS filtered_docs (
                doc_id TEXT PRIMARY KEY,
                reason TEXT
            )
            """
        )
        conn.commit()


def upsert_documents_normalized(
    sqlite_path: str, split: str, rows: Iterable[Dict[str, Any]]
):
    """Load raw rows into normalized documents/entities/relations tables. Operate over DB via SQL."""
    init_db(sqlite_path)
    examples = examples_from_raw_records(list(rows))

    with sqlite3.connect(sqlite_path) as conn:
        doc_ids_to_remove = [
            r[0] for r in conn.execute(
                "SELECT doc_id FROM documents WHERE split = ?", (split,)
            ).fetchall()
        ]
        if doc_ids_to_remove:
            placeholders = ",".join("?" * len(doc_ids_to_remove))
            conn.execute(f"DELETE FROM relations WHERE doc_id IN ({placeholders})", doc_ids_to_remove)
            conn.execute(f"DELETE FROM entities WHERE doc_id IN ({placeholders})", doc_ids_to_remove)
        conn.execute("DELETE FROM documents WHERE split = ?", (split,))

        for ex in examples:
            conn.execute(
                "INSERT OR REPLACE INTO documents (doc_id, split, content) VALUES (?, ?, ?)",
                (ex.doc_id, split, ex.document),
            )
            for ord_idx, ent in enumerate(ex.entities):
                conn.execute(
                    "INSERT OR REPLACE INTO entities (doc_id, name, ord_idx) VALUES (?, ?, ?)",
                    (ex.doc_id, ent, ord_idx),
                )
            for h, r, t in ex.relations:
                conn.execute(
                    "INSERT INTO relations (doc_id, entity1, relation, entity2) VALUES (?, ?, ?, ?)",
                    (ex.doc_id, h, r, t),
                )
        conn.commit()


def load_examples_from_sqlite(
    sqlite_path: str,
    split: str,
    exclude_doc_ids: Optional[Set[str]] = None,
) -> List[Example]:
    """Load Examples from normalized tables via SQL (no JSON parsing)."""
    init_db(sqlite_path)
    with sqlite3.connect(sqlite_path) as conn:
        doc_rows = conn.execute(
            "SELECT doc_id, content FROM documents WHERE split = ? ORDER BY doc_id ASC",
            (split,),
        ).fetchall()

    result: List[Example] = []
    with sqlite3.connect(sqlite_path) as conn:
        for doc_id, content in doc_rows:
            if exclude_doc_ids and doc_id in exclude_doc_ids:
                continue
            entity_rows = conn.execute(
                "SELECT name FROM entities WHERE doc_id = ? ORDER BY ord_idx ASC",
                (doc_id,),
            ).fetchall()
            entities = [r[0] for r in entity_rows]
            rel_rows = conn.execute(
                "SELECT entity1, relation, entity2 FROM relations WHERE doc_id = ?",
                (doc_id,),
            ).fetchall()
            relations = [(r[0], r[1], r[2]) for r in rel_rows]
            result.append(
                Example(doc_id=doc_id, document=content, entities=entities, relations=relations)
            )
    return result


def get_relation_counts(sqlite_path: str, split: str) -> Dict[str, int]:
    examples = load_examples_from_sqlite(sqlite_path, split=split)
    counts: Dict[str, int] = {}
    for ex in examples:
        for _, rel, _ in ex.relations:
            key = rel.strip()
            counts[key] = counts.get(key, 0) + 1
    return counts


def upsert_relation_types(sqlite_path: str, relation_desc_items: Sequence[Tuple[str, str]]):
    init_db(sqlite_path)
    with sqlite3.connect(sqlite_path) as conn:
        conn.execute("DELETE FROM relation_types")
        payload = [
            (rel, desc, idx)
            for idx, (rel, desc) in enumerate(relation_desc_items, start=1)
        ]
        conn.executemany(
            "INSERT INTO relation_types (relation, description, ord) VALUES (?, ?, ?)",
            payload,
        )
        conn.commit()


def get_relation_types(sqlite_path: str) -> List[str]:
    init_db(sqlite_path)
    with sqlite3.connect(sqlite_path) as conn:
        rows = conn.execute(
            "SELECT relation FROM relation_types ORDER BY ord ASC, relation ASC"
        ).fetchall()
    return [x[0] for x in rows]


def get_relation_types_with_desc(sqlite_path: str) -> List[Tuple[str, str]]:
    """Return [(relation, description), ...] ordered by official ordering."""
    init_db(sqlite_path)
    with sqlite3.connect(sqlite_path) as conn:
        rows = conn.execute(
            "SELECT relation, description FROM relation_types ORDER BY ord ASC, relation ASC"
        ).fetchall()
    return [(r, d) for r, d in rows]


def upsert_filtered_docs(sqlite_path: str, doc_ids: Sequence[str], reason: str):
    init_db(sqlite_path)
    with sqlite3.connect(sqlite_path) as conn:
        conn.execute("DELETE FROM filtered_docs")
        payload = [(d, reason) for d in doc_ids]
        conn.executemany(
            "INSERT OR REPLACE INTO filtered_docs (doc_id, reason) VALUES (?, ?)",
            payload,
        )
        conn.commit()


def get_filtered_doc_ids(sqlite_path: str) -> Set[str]:
    init_db(sqlite_path)
    with sqlite3.connect(sqlite_path) as conn:
        rows = conn.execute("SELECT doc_id FROM filtered_docs").fetchall()
    return {x[0] for x in rows}


def save_run_to_sqlite(
    sqlite_path: str,
    run_id: str,
    method: str,
    rows: List[Dict[str, Any]],
    metrics: Dict[str, Any],
):
    init_db(sqlite_path)
    with sqlite3.connect(sqlite_path) as conn:
        conn.execute(
            "DELETE FROM run_predictions WHERE run_id = ? AND method = ?",
            (run_id, method),
        )
        conn.execute(
            "DELETE FROM run_metrics WHERE run_id = ? AND method = ?",
            (run_id, method),
        )
        pred_payload = [
            (
                run_id,
                method,
                str(r.get("id", "")),
                json.dumps(r.get("gold", []), ensure_ascii=False),
                json.dumps(r.get("pred", []), ensure_ascii=False),
                str(r.get("raw", "")),
            )
            for r in rows
        ]
        conn.executemany(
            """
            INSERT INTO run_predictions
            (run_id, method, doc_id, gold_json, pred_json, raw_output)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            pred_payload,
        )
        conn.execute(
            """
            INSERT INTO run_metrics
            (run_id, method, precision, recall, f1, tp, fp, fn)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                method,
                float(metrics["precision"]),
                float(metrics["recall"]),
                float(metrics["f1"]),
                int(metrics["tp"]),
                int(metrics["fp"]),
                int(metrics["fn"]),
            ),
        )
        conn.commit()

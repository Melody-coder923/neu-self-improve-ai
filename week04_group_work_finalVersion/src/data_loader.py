"""
data_loader.py - Load DWIE dataset into a single SQLite database.

This script reads all JSON files from the official RuAG dataset
and stores them in ruag.db with the following tables:
  - documents: article content and train/test split
  - entities: all entities per document
  - relations: ground truth relation triples
  - predictions: LLM predicted relations (filled later)
  - rules: MCTS discovered rules (Part 2)
"""

import json
import sqlite3
import os
from pathlib import Path


def create_tables(conn):
    """Create all tables in the SQLite database."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS documents (
            doc_id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            split TEXT NOT NULL CHECK(split IN ('train', 'test'))
        );

        CREATE TABLE IF NOT EXISTS entities (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id TEXT NOT NULL,
            name TEXT NOT NULL,
            FOREIGN KEY (doc_id) REFERENCES documents(doc_id),
            UNIQUE(doc_id, name)
        );

        CREATE TABLE IF NOT EXISTS relations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id TEXT NOT NULL,
            entity1 TEXT NOT NULL,
            relation TEXT NOT NULL,
            entity2 TEXT NOT NULL,
            FOREIGN KEY (doc_id) REFERENCES documents(doc_id)
        );

        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            doc_id TEXT NOT NULL,
            entity1 TEXT NOT NULL,
            relation TEXT NOT NULL,
            entity2 TEXT NOT NULL,
            method TEXT NOT NULL DEFAULT 'vanilla',
            FOREIGN KEY (doc_id) REFERENCES documents(doc_id)
        );

        CREATE TABLE IF NOT EXISTS rules (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            body_predicates TEXT NOT NULL,
            target TEXT NOT NULL,
            precision REAL NOT NULL,
            description TEXT
        );

        CREATE TABLE IF NOT EXISTS relation_types (
            relation TEXT PRIMARY KEY,
            description TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS filtered_docs (
            doc_id TEXT PRIMARY KEY,
            reason TEXT
        );
    """)
    conn.commit()


def load_relation_types(conn, relations_dict_path):
    """Load the 20 relation type definitions into the database."""
    with open(relations_dict_path, 'r', encoding='utf-8') as f:
        relations_dict = json.load(f)

    for relation, description in relations_dict.items():
        conn.execute(
            "INSERT OR IGNORE INTO relation_types (relation, description) VALUES (?, ?)",
            (relation, description.strip())
        )
    conn.commit()
    count = conn.execute("SELECT COUNT(*) FROM relation_types").fetchone()[0]
    print(f"  Loaded {count} relation types")


def load_documents(conn, data_dir, split):
    """Load all JSON files from a train/ or test/ directory."""
    json_files = sorted(Path(data_dir).glob("*.json"))
    doc_count = 0
    entity_count = 0
    relation_count = 0

    for json_file in json_files:
        doc_id = json_file.stem  # e.g., "DW_14739175_relations"

        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Insert document
        conn.execute(
            "INSERT OR IGNORE INTO documents (doc_id, content, split) VALUES (?, ?, ?)",
            (doc_id, data["content"], split)
        )
        doc_count += 1

        # Extract entities and relations
        entities = set()
        for triplet in data["relations"]:
            # Note: format in dataset is [entity1, entity2, relation]
            entity1, entity2, relation = triplet
            entities.add(entity1)
            entities.add(entity2)

            conn.execute(
                "INSERT INTO relations (doc_id, entity1, relation, entity2) VALUES (?, ?, ?, ?)",
                (doc_id, entity1, relation, entity2)
            )
            relation_count += 1

        # Insert entities
        for entity in entities:
            conn.execute(
                "INSERT OR IGNORE INTO entities (doc_id, name) VALUES (?, ?)",
                (doc_id, entity)
            )
            entity_count += 1

    conn.commit()
    print(f"  Loaded {doc_count} {split} documents, {entity_count} entities, {relation_count} relations")


def load_filtered_docs(conn):
    """Record documents that should be filtered (violate GPT protocol)."""
    # These are from the official code: RelationExtractor.extract_main()
    filtered = [
        "DW_16083654_relations",
        "DW_44141017_relations",
        "DW_17347807_relations",
        "DW_17736433_relations",
        "DW_18751636_relations",
        "DW_19210651_relations",
        "DW_39718698_relations",
    ]
    for doc_id in filtered:
        conn.execute(
            "INSERT OR IGNORE INTO filtered_docs (doc_id, reason) VALUES (?, ?)",
            (doc_id, "Violates GPT processing protocol / ground truth is empty")
        )
    conn.commit()
    print(f"  Recorded {len(filtered)} filtered documents")


def load_all(db_path, dataset_root):
    """Main function: create database and load all data."""
    conn = sqlite3.connect(db_path)

    print("Creating tables...")
    create_tables(conn)

    print("Loading relation types...")
    load_relation_types(conn, os.path.join(dataset_root, "relations_dict.json"))

    print("Loading training documents...")
    load_documents(conn, os.path.join(dataset_root, "entity_relations_pairs", "train"), "train")

    print("Loading test documents...")
    load_documents(conn, os.path.join(dataset_root, "entity_relations_pairs", "test"), "test")

    print("Loading filtered document list...")
    load_filtered_docs(conn)

    # Print summary
    print("\n=== Database Summary ===")
    for table in ["documents", "entities", "relations", "relation_types", "filtered_docs"]:
        count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"  {table}: {count} rows")

    train_count = conn.execute("SELECT COUNT(*) FROM documents WHERE split='train'").fetchone()[0]
    test_count = conn.execute("SELECT COUNT(*) FROM documents WHERE split='test'").fetchone()[0]
    test_active = conn.execute("""
        SELECT COUNT(*) FROM documents 
        WHERE split='test' AND doc_id NOT IN (SELECT doc_id FROM filtered_docs)
    """).fetchone()[0]
    print(f"\n  Train: {train_count}, Test: {test_count} (active: {test_active})")

    conn.close()
    print(f"\nDatabase saved to: {db_path}")


if __name__ == "__main__":
    # Default paths - adjust these to match your setup
    DB_PATH = os.path.join(os.path.dirname(__file__), "..", "ruag.db")
    DATASET_ROOT = os.path.join(os.path.dirname(__file__), "..", "dataset")

    load_all(DB_PATH, DATASET_ROOT)

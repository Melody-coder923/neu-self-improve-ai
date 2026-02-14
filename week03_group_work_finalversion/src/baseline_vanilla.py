"""
baseline_vanilla.py - Vanilla LLM baseline for relation extraction.

This script:
1. Reads test documents from SQLite
2. For each document, constructs a prompt with entities and relation definitions
3. Calls GPT-4 API to extract relation triples
4. Parses the response and stores predictions in SQLite
"""

import json
import sqlite3
import re
import time
import os
from pathlib import Path

try:
    from openai import OpenAI
except ImportError:
    print("Please install openai: pip install openai")
    exit(1)

# Gemini support
GEMINI_MODELS = {"gemini-2.0-flash", "gemini-2.0-flash-lite", "gemini-1.5-flash", "gemini-1.5-pro"}


class VanillaBaseline:
    def __init__(self, db_path, prompt_path, api_key=None, model="gpt-4-0613", provider="openai"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.model = model
        self.provider = provider  # "openai", "gemini", or "groq"

        # LLM parameters (from paper Table A5)
        self.temperature = 0
        self.top_p = 1
        self.max_tokens = 1000
        self.frequency_penalty = 0
        self.presence_penalty = 0

        # Load prompt template
        with open(prompt_path, 'r', encoding='utf-8') as f:
            self.prompt_template = f.read()

        # Load relation types and descriptions
        self.relation_types = {}
        rows = self.conn.execute("SELECT relation, description FROM relation_types").fetchall()
        for relation, description in rows:
            self.relation_types[relation] = description

        # Build relationships description string
        self.relationships_text = self._build_relationships_text()

        # Load filtered doc list
        self.filtered_docs = set(
            row[0] for row in self.conn.execute("SELECT doc_id FROM filtered_docs").fetchall()
        )

        # Initialize API client based on provider
        base_urls = {
            "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
            "groq": "https://api.groq.com/openai/v1",
        }

        if self.provider in base_urls:
            self.client = OpenAI(
                api_key=api_key,
                base_url=base_urls[self.provider]
            )
        else:
            if api_key:
                self.client = OpenAI(api_key=api_key)
            elif os.environ.get("OPENAI_API_KEY"):
                self.client = OpenAI()
            else:
                print("WARNING: No API key provided.")
                self.client = None

    def _build_relationships_text(self):
        """Build the relationship descriptions string for the prompt."""
        lines = []
        for relation, description in self.relation_types.items():
            lines.append(f"    -'{relation}': {description}")
        return "\n".join(lines)

    def get_test_documents(self):
        """Get all active test documents (excluding filtered ones)."""
        rows = self.conn.execute("""
            SELECT d.doc_id, d.content
            FROM documents d
            WHERE d.split = 'test'
            AND d.doc_id NOT IN (SELECT doc_id FROM filtered_docs)
            ORDER BY d.doc_id
        """).fetchall()
        return rows

    def get_entities_for_doc(self, doc_id):
        """Get all entities for a document."""
        rows = self.conn.execute(
            "SELECT DISTINCT name FROM entities WHERE doc_id = ?", (doc_id,)
        ).fetchall()
        return [row[0] for row in rows]

    def get_ground_truth(self, doc_id):
        """Get ground truth relations for a document."""
        rows = self.conn.execute(
            "SELECT entity1, relation, entity2 FROM relations WHERE doc_id = ?", (doc_id,)
        ).fetchall()
        return set((e1, rel, e2) for e1, rel, e2 in rows)

    def build_prompt(self, content, entities):
        """Build the prompt for a specific document."""
        entities_str = "; ".join(entities) + ".\n"
        prompt = self.prompt_template.format(
            relationships=self.relationships_text,
            document=content,
            entities=entities_str
        )
        return prompt

    def call_llm(self, prompt, max_retries=5, sleep_duration=10):
        """Call LLM API with retry logic. Works for both OpenAI and Gemini."""
        if self.client is None:
            raise RuntimeError("No API key configured.")

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    top_p=self.top_p,
                    max_tokens=self.max_tokens,
                )
                return response.choices[0].message.content
            except Exception as e:
                error_msg = str(e)
                print(f"  API error (attempt {attempt + 1}/{max_retries}): {error_msg}")
                if "429" in error_msg or "rate" in error_msg.lower() or "quota" in error_msg.lower():
                    # Rate limit - wait longer for Gemini free tier
                    wait_time = sleep_duration if self.provider == "openai" else 15
                    print(f"  Rate limited, waiting {wait_time}s...")
                    time.sleep(wait_time)
                elif attempt < max_retries - 1:
                    time.sleep(sleep_duration)

        raise RuntimeError("Max retries exceeded for API call.")

    def parse_response(self, response_text, valid_entities, valid_relations):
        """Parse LLM response to extract relation triples."""
        predicted = set()

        # Find all (entity1, relation, entity2) patterns
        # Handle both ('entity1', 'relation', 'entity2') and (entity1, relation, entity2)
        triplets_str = re.findall(r'\((.*?)\)', response_text)

        for triplet in triplets_str:
            # Split by comma and clean up
            parts = [p.strip().strip("'\"") for p in triplet.split(",")]

            if len(parts) < 3:
                continue

            # Take first 3 parts (ignore reason after //)
            entity1 = parts[0].strip()
            relation = parts[1].strip()
            entity2 = parts[2].strip()

            # Validate: relation must be in our 20 types, entities must be valid
            if relation in valid_relations and entity1 in valid_entities and entity2 in valid_entities:
                predicted.add((entity1, relation, entity2))

        return predicted

    def store_predictions(self, doc_id, predictions, method="vanilla"):
        """Store predictions in SQLite."""
        for entity1, relation, entity2 in predictions:
            self.conn.execute(
                "INSERT INTO predictions (doc_id, entity1, relation, entity2, method) VALUES (?, ?, ?, ?, ?)",
                (doc_id, entity1, relation, entity2, method)
            )
        self.conn.commit()

    def run(self):
        """Run vanilla baseline on all test documents."""
        # Clear previous vanilla predictions
        self.conn.execute("DELETE FROM predictions WHERE method = 'vanilla'")
        self.conn.commit()

        test_docs = self.get_test_documents()
        valid_relations = set(self.relation_types.keys())

        print(f"Running Vanilla baseline on {len(test_docs)} test documents...")
        print(f"Model: {self.model}")
        print(f"Temperature: {self.temperature}, Top-p: {self.top_p}")
        print()

        for i, (doc_id, content) in enumerate(test_docs):
            print(f"[{i+1}/{len(test_docs)}] Processing {doc_id}...")

            # Get entities for this document
            entities = self.get_entities_for_doc(doc_id)
            if not entities:
                print(f"  No entities found, skipping.")
                continue

            # Build prompt
            prompt = self.build_prompt(content, entities)

            # Call LLM
            try:
                response_text = self.call_llm(prompt)
            except RuntimeError as e:
                print(f"  Failed: {e}")
                continue

            # Parse response
            valid_entities = set(entities)
            predictions = self.parse_response(response_text, valid_entities, valid_relations)

            # Store predictions
            self.store_predictions(doc_id, predictions, method="vanilla")

            # Quick per-document stats
            ground_truth = self.get_ground_truth(doc_id)
            tp = len(predictions & ground_truth)
            print(f"  Predicted: {len(predictions)}, Ground truth: {len(ground_truth)}, TP: {tp}")

        print("\nVanilla baseline complete. Run evaluation.py to compute metrics.")

    def close(self):
        self.conn.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Vanilla LLM baseline")
    parser.add_argument("--db", default="../ruag.db", help="Path to SQLite database")
    parser.add_argument("--prompt", default="../prompts/vanilla_prompt.txt", help="Path to prompt template")
    parser.add_argument("--api-key", default=None, help="API key (OpenAI or Gemini)")
    parser.add_argument("--model", default="gpt-4-0613", help="Model name")
    parser.add_argument("--provider", default="openai", choices=["openai", "gemini", "groq"],
                        help="API provider: openai, gemini, or groq")
    args = parser.parse_args()

    baseline = VanillaBaseline(
        db_path=args.db,
        prompt_path=args.prompt,
        api_key=args.api_key,
        model=args.model,
        provider=args.provider,
    )
    baseline.run()
    baseline.close()

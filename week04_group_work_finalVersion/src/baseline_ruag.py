"""
baseline_ruag.py - RuAG (Learned-Rule-Augmented Generation) for relation extraction.

This script:
1. Reads MCTS-discovered rules from SQLite
2. Translates rules into natural language
3. For each test document, builds a RuAG prompt (with rules injected)
4. Calls LLM API to extract relation triples
5. Parses response and stores predictions in SQLite (method='ruag')

The only difference from baseline_vanilla.py is the prompt: it includes
logic rules discovered by MCTS from training data.
"""

import sqlite3
import re
import time
import os

try:
    from openai import OpenAI
except ImportError:
    print("Please install openai: pip install openai")
    exit(1)


class RuAGBaseline:
    def __init__(self, db_path, prompt_path, api_key=None, model="gpt-4-0613", provider="openai"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.model = model
        self.provider = provider

        # LLM parameters (from paper Table A5)
        self.temperature = 0
        self.top_p = 1
        self.max_tokens = 2000
        self.frequency_penalty = 0
        self.presence_penalty = 0

        # Load prompt template
        with open(prompt_path, 'r', encoding='utf-8') as f:
            self.prompt_template = f.read()

        # Load relation types
        self.relation_types = {}
        rows = self.conn.execute("SELECT relation, description FROM relation_types").fetchall()
        for relation, description in rows:
            self.relation_types[relation] = description

        # Build relationships description string
        self.relationships_text = self._build_relationships_text()

        # Load rules from SQLite (discovered by MCTS in rule_search.py)
        self.rules_text = self._load_rules()

        # Load filtered doc list
        self.filtered_docs = set(
            row[0] for row in self.conn.execute("SELECT doc_id FROM filtered_docs").fetchall()
        )

        # Initialize API client
        base_urls = {
            "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
            "groq": "https://api.groq.com/openai/v1",
            "mistral": "https://api.mistral.ai/v1",
        }

        if self.provider in base_urls:
            self.client = OpenAI(api_key=api_key, base_url=base_urls[self.provider])
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

    def _load_rules(self):
        """Load MCTS-discovered rules from SQLite and format as text."""
        rows = self.conn.execute(
            "SELECT description FROM rules ORDER BY precision DESC"
        ).fetchall()

        if not rows:
            print("WARNING: No rules found in database. Run --step search_rules first.")
            return "No rules available."

        rules_lines = []
        for i, (description,) in enumerate(rows):
            rules_lines.append(f"  Rule {i+1}: {description}")

        print(f"Loaded {len(rows)} logic rules from database.")
        return "\n".join(rules_lines)

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
        """Build the RuAG prompt with rules injected."""
        entities_str = "; ".join(entities) + ".\n"
        prompt = self.prompt_template.format(
            relationships=self.relationships_text,
            rules=self.rules_text,
            document=content,
            entities=entities_str
        )
        return prompt

    def call_llm(self, prompt, max_retries=5, sleep_duration=10):
        """Call LLM API with retry logic."""
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
                    wait_time = sleep_duration if self.provider == "openai" else 15
                    print(f"  Rate limited, waiting {wait_time}s...")
                    time.sleep(wait_time)
                elif attempt < max_retries - 1:
                    time.sleep(sleep_duration)

        raise RuntimeError("Max retries exceeded for API call.")

    def parse_response(self, response_text, valid_entities, valid_relations):
        """Parse LLM response to extract relation triples.
        
        Same parsing logic as baseline_vanilla.py:
        format-level cleanup only, strict exact-match validation.
        """
        predicted = set()

        if not response_text:
            return predicted

        lines = response_text.split('\n')
        for line in lines:
            matches = re.findall(r'\((.*?)\)', line)
            for match in matches:
                parts = [p.strip() for p in match.split(",")]

                if len(parts) < 3:
                    continue

                entity1 = parts[0].strip()
                relation = parts[1].strip()
                entity2 = parts[2].strip()

                # Format-level cleanup: remove quotes, backticks, asterisks
                for char in ["'", '"', '`', '*']:
                    entity1 = entity1.strip(char)
                    relation = relation.strip(char)
                    entity2 = entity2.strip(char)

                entity1 = entity1.strip()
                relation = relation.strip()
                entity2 = entity2.strip()

                # Strict exact match only
                if (relation in valid_relations
                        and entity1 in valid_entities
                        and entity2 in valid_entities):
                    predicted.add((entity1, relation, entity2))

        return predicted

    def store_predictions(self, doc_id, predictions, method="ruag"):
        """Store predictions in SQLite."""
        for entity1, relation, entity2 in predictions:
            self.conn.execute(
                "INSERT INTO predictions (doc_id, entity1, relation, entity2, method) VALUES (?, ?, ?, ?, ?)",
                (doc_id, entity1, relation, entity2, method)
            )
        self.conn.commit()

    def run(self):
        """Run RuAG baseline on all test documents."""
        # Clear previous RuAG predictions
        self.conn.execute("DELETE FROM predictions WHERE method = 'ruag'")
        self.conn.commit()

        test_docs = self.get_test_documents()
        valid_relations = set(self.relation_types.keys())

        print(f"\nRunning RuAG baseline on {len(test_docs)} test documents...")
        print(f"Model: {self.model}")
        print(f"Temperature: {self.temperature}, Top-p: {self.top_p}")
        print(f"Rules injected: {self.rules_text.count('Rule ')}")
        print()

        for i, (doc_id, content) in enumerate(test_docs):
            print(f"[{i+1}/{len(test_docs)}] Processing {doc_id}...")

            entities = self.get_entities_for_doc(doc_id)
            if not entities:
                print(f"  No entities found, skipping.")
                continue

            prompt = self.build_prompt(content, entities)

            try:
                response_text = self.call_llm(prompt)
            except RuntimeError as e:
                print(f"  Failed: {e}")
                continue

            valid_entities = set(entities)
            predictions = self.parse_response(response_text, valid_entities, valid_relations)

            self.store_predictions(doc_id, predictions, method="ruag")

            ground_truth = self.get_ground_truth(doc_id)
            tp = len(predictions & ground_truth)
            print(f"  Predicted: {len(predictions)}, Ground truth: {len(ground_truth)}, TP: {tp}")

        print("\nRuAG baseline complete. Run --step evaluate_ruag to compute metrics.")

    def close(self):
        self.conn.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run RuAG baseline")
    parser.add_argument("--db", default="../ruag.db", help="Path to SQLite database")
    parser.add_argument("--prompt", default="../prompts/ruag_prompt.txt", help="Path to RuAG prompt")
    parser.add_argument("--api-key", default=None, help="API key")
    parser.add_argument("--model", default="gpt-4-0613", help="Model name")
    parser.add_argument("--provider", default="openai", choices=["openai", "gemini", "groq"])
    args = parser.parse_args()

    baseline = RuAGBaseline(
        db_path=args.db,
        prompt_path=args.prompt,
        api_key=args.api_key,
        model=args.model,
        provider=args.provider,
    )
    baseline.run()
    baseline.close()

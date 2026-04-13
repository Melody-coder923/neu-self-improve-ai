"""
main.py - Main entry point for RuAG Relation Extraction.

Step 1: Load data
    python main.py --step load

Step 2: Part 1 - Vanilla Baseline
    python main.py --step vanilla --provider mistral --model mistral-small-latest --api-key YOUR_KEY
    python main.py --step evaluate

Step 3: Part 2 - RuAG (Rule-Augmented Generation)
    python main.py --step search_rules
    python main.py --step ruag --provider mistral --model mistral-small-latest --api-key YOUR_KEY
    python main.py --step evaluate_ruag
"""

import os
import sys
import argparse

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from src.data_loader import load_all
from src.baseline_vanilla import VanillaBaseline
from src.evaluation import evaluate


def main():
    parser = argparse.ArgumentParser(description="RuAG Relation Extraction")
    parser.add_argument("--step", required=True,
                        choices=["load", "vanilla", "evaluate",
                                 "search_rules", "ruag", "evaluate_ruag",
                                 "all"],
                        help="Which step to run")
    parser.add_argument("--api-key", default=None, help="API key (OpenAI, Gemini, or Groq)")
    parser.add_argument("--model", default="gpt-4-0613", help="Model name")
    parser.add_argument("--provider", default="openai", choices=["openai", "gemini", "groq", "mistral"],
                        help="API provider: openai, gemini, groq, or mistral")
    parser.add_argument("--db", default="ruag.db", help="Path to SQLite database")
    args = parser.parse_args()

    db_path = os.path.abspath(args.db)
    dataset_root = os.path.join(os.path.dirname(__file__), "dataset")
    vanilla_prompt_path = os.path.join(os.path.dirname(__file__), "prompts", "vanilla_prompt.txt")
    ruag_prompt_path = os.path.join(os.path.dirname(__file__), "prompts", "ruag_prompt.txt")

    # ── Part 1 Steps ──────────────────────────────────────────────────

    if args.step in ("load", "all"):
        print("=" * 60)
        print("Step 1: Loading data into SQLite")
        print("=" * 60)
        load_all(db_path, dataset_root)
        print()

    if args.step in ("vanilla", "all"):
        print("=" * 60)
        print("Step 2: Running Vanilla baseline (Part 1)")
        print("=" * 60)
        baseline = VanillaBaseline(
            db_path=db_path,
            prompt_path=vanilla_prompt_path,
            api_key=args.api_key,
            model=args.model,
            provider=args.provider,
        )
        baseline.run()
        baseline.close()
        print()

    if args.step in ("evaluate", "all"):
        print("=" * 60)
        print("Step 3: Evaluating Vanilla baseline results")
        print("=" * 60)
        evaluate(db_path, method="vanilla")
        print()

    # ── Part 2 Steps ──────────────────────────────────────────────────

    if args.step in ("search_rules", "all"):
        print("=" * 60)
        print("Step 4: MCTS Rule Search (Part 2 - Phase 1 & 2)")
        print("=" * 60)
        from src.rule_search import run_rule_search
        run_rule_search(db_path)
        print()

    if args.step in ("ruag", "all"):
        print("=" * 60)
        print("Step 5: Running RuAG baseline (Part 2 - Phase 3)")
        print("=" * 60)
        from src.baseline_ruag import RuAGBaseline
        ruag = RuAGBaseline(
            db_path=db_path,
            prompt_path=ruag_prompt_path,
            api_key=args.api_key,
            model=args.model,
            provider=args.provider,
        )
        ruag.run()
        ruag.close()
        print()

    if args.step in ("evaluate_ruag", "all"):
        print("=" * 60)
        print("Step 6: Evaluating RuAG results")
        print("=" * 60)
        evaluate(db_path, method="ruag")
        print()


if __name__ == "__main__":
    main()

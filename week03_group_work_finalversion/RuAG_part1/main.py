"""
main.py - Main entry point for RuAG Relation Extraction baseline.

Usage:
    # Step 1: Load data into SQLite (only need to run once)
    python main.py --step load

    # Step 2: Run Vanilla baseline
    python main.py --step vanilla --api-key YOUR_API_KEY

    # Step 3: Evaluate results
    python main.py --step evaluate

    # Or run all steps:
    python main.py --step all --api-key YOUR_API_KEY
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
    parser = argparse.ArgumentParser(description="RuAG Relation Extraction - Baseline Implementation")
    parser.add_argument("--step", required=True, choices=["load", "vanilla", "evaluate", "all"],
                        help="Which step to run")
    parser.add_argument("--api-key", default=None, help="API key (OpenAI or Gemini)")
    parser.add_argument("--model", default="gpt-4-0613", help="Model name")
    parser.add_argument("--provider", default="openai", choices=["openai", "gemini", "groq"],
                        help="API provider: openai, gemini, or groq")
    parser.add_argument("--db", default="ruag.db", help="Path to SQLite database")
    args = parser.parse_args()

    db_path = os.path.abspath(args.db)
    dataset_root = os.path.join(os.path.dirname(__file__), "dataset")
    prompt_path = os.path.join(os.path.dirname(__file__), "prompts", "vanilla_prompt.txt")

    if args.step in ("load", "all"):
        print("=" * 50)
        print("Step 1: Loading data into SQLite")
        print("=" * 50)
        load_all(db_path, dataset_root)
        print()

    if args.step in ("vanilla", "all"):
        print("=" * 50)
        print("Step 2: Running Vanilla baseline")
        print("=" * 50)
        baseline = VanillaBaseline(
            db_path=db_path,
            prompt_path=prompt_path,
            api_key=args.api_key,
            model=args.model,
            provider=args.provider,
        )
        baseline.run()
        baseline.close()
        print()

    if args.step in ("evaluate", "all"):
        print("=" * 50)
        print("Step 3: Evaluating results")
        print("=" * 50)
        evaluate(db_path, method="vanilla")


if __name__ == "__main__":
    main()

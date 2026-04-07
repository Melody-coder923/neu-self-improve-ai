import sqlite3, re, os, argparse
from datasets import load_dataset
from agentflow.agentflow.solver import construct_solver
import agentflow.agentflow.engine.factory_patch  # noqa: F401 — registers ModalEngine in the factory

dataset = load_dataset("yale-nlp/spider", split="validation")

def get_schema(db_path: str) -> str:
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [r[0] for r in cur.fetchall()]
    lines = []
    for t in tables:
        cur.execute(f"PRAGMA table_info({t})")
        cols = [f"{r[1]} ({r[2]})" for r in cur.fetchall()]
        lines.append(f"Table {t}: {', '.join(cols)}")
    conn.close()
    return "\n".join(lines)

def execute_sql(sql: str, db_path: str) -> list:
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        conn.close()
        return rows
    except Exception as e:
        return [("ERROR", str(e))]

def extract_sql(text: str) -> str:
    m = re.search(r"```sql\n(.*?)\n```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    lines = [l for l in text.split("\n") if l.strip().upper().startswith("SELECT")]
    return lines[-1].strip() if lines else text.strip()

def run_spider_eval(llm_engine_name: str = "modal",
                    db_dir: str = "data/spider/database"):
    solver = construct_solver(
        llm_engine_name=llm_engine_name,
        tools=["execute_sql"],
    )
    correct, total = 0, 0
    for sample in dataset:
        question = sample["question"]
        gold_sql = sample["query"]
        db_id = sample["db_id"]
        db_path = f"{db_dir}/{db_id}/{db_id}.sqlite"

        if not os.path.exists(db_path):
            print(f"[SKIP] DB not found: {db_path}")
            continue

        os.environ["SPIDER_DB_PATH"] = db_path
        schema = get_schema(db_path)
        prompt = (
            f"Database schema:\n{schema}\n\n"
            f"Question: {question}\n\n"
            "Use execute_sql tool to explore the database and find the answer. "
            "Output the final SQL query in a ```sql``` block."
        )
        output = solver.solve(prompt)
        pred_sql = extract_sql(output["direct_output"])
        pred_rows = execute_sql(pred_sql, db_path)
        gold_rows = execute_sql(gold_sql, db_path)

        if set(map(tuple, pred_rows)) == set(map(tuple, gold_rows)):
            correct += 1
        total += 1
        if total % 50 == 0:
            print(f"[{total}/{len(dataset)}] Exec Acc: {correct/total:.3f}")

    if total == 0:
        print("\nNo samples evaluated — check that --db_dir exists and contains .sqlite files.")
        return
    print(f"\nFinal Execution Accuracy: {correct/total:.3f} ({correct}/{total})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", default="modal")
    parser.add_argument("--db_dir", default="data/spider/database")
    args = parser.parse_args()
    run_spider_eval(llm_engine_name=args.engine, db_dir=args.db_dir)

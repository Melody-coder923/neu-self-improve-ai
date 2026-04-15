import sqlite3, re, os, json, sys
sys.path.insert(0, '/home/jason/AgentFlow')
from agentflow.agentflow.solver import construct_solver

DB_DIR = os.path.expanduser("~/AgentFlow/data/spider/database")
DEV_FILE = os.path.expanduser("~/AgentFlow/data/spider_data/dev.json")

def get_schema(db_path):
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
def execute_sql(sql, db_path):
    try:
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        conn.close()
        return rows
    except Exception as e:
        return [("ERROR", str(e))]

def extract_sql(text):
    m = re.search(r"```sql\n(.*?)\n```", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    lines = [l for l in text.split("\n") if l.strip().upper().startswith("SELECT")]
    return lines[-1].strip() if lines else text.strip()
solver = construct_solver(llm_engine_name="modal")

with open(DEV_FILE) as f:
    dataset = json.load(f)[:20]

correct, total = 0, 0
for sample in dataset:
    question = sample["question"]
    gold_sql = sample["query"]
    db_id = sample["db_id"]
    db_path = f"{DB_DIR}/{db_id}/{db_id}.sqlite"
    os.environ["SPIDER_DB_PATH"] = db_path
    schema = get_schema(db_path)
    prompt = f"Database schema:\n{schema}\n\nQuestion: {question}\n\nUse execute_sql tool to find the answer. Output final SQL in ```sql``` block."
    output = solver.solve(prompt)
    pred_sql = extract_sql(output["direct_output"])
    pred_rows = execute_sql(pred_sql, db_path)
    gold_rows = execute_sql(gold_sql, db_path)
    if set(map(tuple, pred_rows)) == set(map(tuple, gold_rows)):
        correct += 1
    total += 1
    print(f"[{total}/20] Acc: {correct/total:.3f} | {question[:50]}")

print(f"\nFinal: {correct/total:.3f} ({correct}/{total})")
print("Model: Qwen3.5-0.8B via Modal (full AgentFlow loop)")

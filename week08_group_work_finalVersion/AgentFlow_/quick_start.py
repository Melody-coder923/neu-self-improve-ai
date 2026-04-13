"""
Quick start script to verify AgentFlow is working.
"""
import os
import sys

def check_env():
    required = []
    optional = ["OPENAI_API_KEY", "DASHSCOPE_API_KEY"]
    missing_critical = []

    # At least one of these must be set for agent loop
    if not (os.environ.get("TOGETHER_API_KEY") or os.environ.get("DASHSCOPE_API_KEY")):
        missing_critical.append("TOGETHER_API_KEY or DASHSCOPE_API_KEY")

    if missing_critical:
        print(f"[ERROR] Missing required env vars: {missing_critical}")
        print("Please set them in agentflow/agentflow/.env")
        return False
    return True

def run_quick_test():
    try:
        from agentflow.agentflow.solver import construct_solver
    except ImportError as e:
        print(f"[ERROR] Cannot import AgentFlow: {e}")
        print("Please run: bash setup.sh && source .venv/bin/activate")
        sys.exit(1)

    engine = os.environ.get("LLM_ENGINE", "dashscope")
    print(f"Using LLM engine: {engine}")

    solver = construct_solver(llm_engine_name=engine)
    query = "What is the capital of France?"
    print(f"\nQuery: {query}")
    print("Running AgentFlow solver...")

    result = solver.solve(query)
    answer = result.get("direct_output", result)
    print(f"\n==> Answer: {answer}")
    print("==> Query Solved!")
    return True

if __name__ == "__main__":
    # Load .env if exists
    env_path = "agentflow/agentflow/.env"
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())

    if check_env():
        run_quick_test()

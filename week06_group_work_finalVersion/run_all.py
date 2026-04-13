"""Run all algorithms sequentially."""
import subprocess, sys

def run(cmd):
    print(f"\n{'='*60}\n  {cmd}\n{'='*60}\n")
    subprocess.run([sys.executable] + cmd.split(), check=True)

if __name__ == "__main__":
    run("pretrain.py --seed 0")
    run("reinforce.py --seed 0")
    run("ppo.py --seed 0")
    run("polyppo.py --seed 0")
    print("\nPaper targets: Pretrained(0.469,70.4%) REINFORCE(0.639,89.6%) "
          "PPO(0.618,89.2%) Poly-PPO(0.666,92.4%)")

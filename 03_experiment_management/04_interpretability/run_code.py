# run_code.py
import subprocess
import sys
from pathlib import Path

SCRIPTS = [
    "error_analysis_lightning.py",
    "select_wrong_prediction.py",
    "select_right_prediction.py",
    "gradcam_flower.py",
    "gradcam_flower_true.py",
    "saliency_flower.py",
    "saliency_flower_true.py",
]

def run_script(script_name: str):
    path = Path(script_name)
    if not path.exists():
        print(f"[SKIP] {script_name} not found")
        return

    print(f"\n=== Running {script_name} ===")
    result = subprocess.run(
        [sys.executable, script_name],
        stdout=sys.stdout,
        stderr=sys.stderr,
    )
    if result.returncode != 0:
        print(f"!!! {script_name} failed with return code {result.returncode}")
        sys.exit(result.returncode)
    print(f"=== Finished {script_name} ===\n")


def main():
    print("Starting interpretability pipeline...")
    for script in SCRIPTS:
        run_script(script)
    print("All steps finished.")


if __name__ == "__main__":
    main()
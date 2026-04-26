from pathlib import Path
import subprocess
import sys

CONFIG_DIR = Path("./yaml_optimizer")
TRAIN_SCRIPT = "hyperparameters_flower.py"

def main():
    config_files = sorted(CONFIG_DIR.glob("*.yaml"))

    if not config_files:
        print(f"No yaml files found in {CONFIG_DIR}")
        sys.exit(1)

    print(f"Found {len(config_files)} config files:")
    for i, cfg in enumerate(config_files, 1):
        print(f"{i}. {cfg}")

    for i, cfg in enumerate(config_files, 1):
        cmd = ["python", TRAIN_SCRIPT, "fit", "-c", str(cfg), "--trainer.max_epochs", "40"]
        print("\n" + "=" * 80)
        print(f"[{i}/{len(config_files)}] Running: {' '.join(cmd)}")
        print("=" * 80)

        result = subprocess.run(cmd)

        if result.returncode != 0:
            print(f"\nFailed on {cfg} with return code {result.returncode}")
            sys.exit(result.returncode)

    print("\nAll experiments finished successfully.")

if __name__ == "__main__":
    main()

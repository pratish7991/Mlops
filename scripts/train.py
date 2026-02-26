import argparse
import subprocess
import sys


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Pass Hydra override via CLI
    subprocess.run(
        [
            sys.executable,
            "-m",
            "src.main",
            f"training.epochs={args.epochs}",
        ],
        check=True,
    )

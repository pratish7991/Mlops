import argparse
from src.inference import run_inference

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--fingerprint", required=True)
    parser.add_argument("--left", required=True)
    parser.add_argument("--right", required=True)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_inference(
        args.model_path,
        args.fingerprint,
        args.left,
        args.right,
    )

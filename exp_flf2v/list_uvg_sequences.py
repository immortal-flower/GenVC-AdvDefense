"""List UVG sequences discoverable under a dataset path without loading Wan."""
import argparse
import json
import os
import sys

_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _project_root)

from uvg_data import find_uvg_sequences


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    sequences = [{"name": name, "path": path}
                 for name, path in find_uvg_sequences(args.data_dir)]
    if args.json:
        print(json.dumps(sequences, ensure_ascii=False))
    else:
        for item in sequences:
            print(f"{item['name']}\t{item['path']}")


if __name__ == "__main__":
    main()

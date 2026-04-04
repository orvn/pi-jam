#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECTS = ("picojam", "jam", "jamxe")


def run(args: list[str], cwd: Path) -> None:
    print("+", " ".join(args))
    subprocess.run(args, cwd=cwd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build all JAM release targets.")
    parser.add_argument("--train", action="store_true", help="retrain before building")
    args = parser.parse_args()

    for name in PROJECTS:
        cmd = [sys.executable, "build.py"]
        if args.train:
            cmd.append("--train")
        run(cmd, ROOT / name)


if __name__ == "__main__":
    main()

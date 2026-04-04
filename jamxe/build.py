#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def find_tool(name: str) -> str:
    candidates = [
        shutil.which(name),
        shutil.which(f"{name}.exe"),
        str(ROOT.parent.parent / "tools" / "cc65" / "bin" / f"{name}.exe"),
        str(ROOT.parent.parent / "tools" / "cc65" / "bin" / name),
    ]
    for cand in candidates:
        if cand and Path(cand).exists():
            return cand
    raise SystemExit(f"{name} not found. Install cc65 or add it to PATH.")


def run(args: list[str]) -> None:
    print("+", " ".join(str(a) for a in args))
    subprocess.run(args, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build JAM XE.")
    parser.add_argument("--train", action="store_true", help="retrain before build")
    parser.add_argument("--data", type=Path, default=ROOT / "td_xe.txt")
    parser.add_argument("--weights", type=Path, default=ROOT / "weights_xe.bin")
    args = parser.parse_args()

    ca65 = find_tool("ca65")
    ld65 = find_tool("ld65")
    dist = ROOT / "dist"
    dist.mkdir(exist_ok=True)

    if args.train:
        run([sys.executable, "train_xe.py", str(args.data)])
    if not args.weights.exists():
        raise SystemExit(f"weights not found: {args.weights}")

    run([sys.executable, "gen_theme.py", str(args.data), "theme_xe.inc"])

    blob = args.weights.read_bytes()
    if len(blob) < 65536:
        raise SystemExit(f"weights too small: {len(blob)} B")
    (ROOT / "weights_xe_main.bin").write_bytes(blob[65536:])

    run([ca65, "jamxe.asm", "-o", "jamxe.o"])
    run([ld65, "-C", "jamxe.cfg", "-o", "jamxe_link.xex", "jamxe.o"])
    run([sys.executable, "pack_xe_banks.py", str(args.weights), "jamxe_link.xex", str(dist / "jamxe.xex")])
    print(f"OK: {dist / 'jamxe.xex'}")


if __name__ == "__main__":
    main()

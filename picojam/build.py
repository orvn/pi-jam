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


def verify_blob(weights: Path, xex: Path) -> None:
    wb = weights.read_bytes()
    xb = xex.read_bytes()
    pos = xb.find(wb[:16])
    if pos < 0 or xb[pos : pos + len(wb)] != wb:
        raise SystemExit("weight blob not found intact inside picojam.xex")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build picojam.")
    parser.add_argument("--train", action="store_true", help="retrain before build")
    parser.add_argument("--data", type=Path, default=ROOT / "td_pico_mix.txt")
    parser.add_argument("--weights", type=Path, default=ROOT / "weights_pico.bin")
    args = parser.parse_args()

    ca65 = find_tool("ca65")
    ld65 = find_tool("ld65")
    dist = ROOT / "dist"
    dist.mkdir(exist_ok=True)

    if args.train:
        run([sys.executable, "train_pico.py", str(args.data)])
    if not args.weights.exists():
        raise SystemExit(f"weights not found: {args.weights}")

    run([sys.executable, "gen_theme.py", str(args.data), "theme_pico.inc"])
    run([ca65, "jampico.asm", "-o", "jampico.o"])
    run([ld65, "-C", "jampico.cfg", "-o", str(dist / "picojam.xex"), "jampico.o"])
    verify_blob(args.weights, dist / "picojam.xex")
    print(f"OK: {dist / 'picojam.xex'}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build JAM KID — same JAMXE runtime, kid personality.

Uses jamxe/ assembler, linker config, and packer with jamkid/ data and weights.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
JAMXE = ROOT.parent / "jamxe"


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


def run(args: list[str], cwd: Path = JAMXE) -> None:
    print("+", " ".join(str(a) for a in args))
    subprocess.run(args, cwd=cwd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build JAM KID.")
    parser.add_argument("--train", action="store_true", help="retrain before build")
    parser.add_argument("--data", type=Path, default=ROOT / "td_jamkid.txt")
    parser.add_argument("--weights", type=Path, default=ROOT / "weights_jamkid.bin")
    args = parser.parse_args()

    ca65 = find_tool("ca65")
    ld65 = find_tool("ld65")
    dist = ROOT / "dist"
    dist.mkdir(exist_ok=True)

    if args.train:
        run([sys.executable, "train_xe.py", str(args.data)], cwd=JAMXE)
        # train_xe.py writes to same dir as data file
        trained = args.data.parent / "weights_xe.bin"
        if trained.exists() and trained != args.weights:
            shutil.copy2(trained, args.weights)

    if not args.weights.exists():
        raise SystemExit(f"weights not found: {args.weights}")

    # Generate kid theme from kid data
    run([sys.executable, str(JAMXE / "gen_theme.py"), str(args.data), str(JAMXE / "theme_xe.inc")], cwd=JAMXE)

    # Split weights: first 64KB to banks, rest to main
    blob = args.weights.read_bytes()
    if len(blob) < 65536:
        raise SystemExit(f"weights too small: {len(blob)} B")
    (JAMXE / "weights_xe_main.bin").write_bytes(blob[65536:])

    # Assemble and link (using jamxe asm+cfg)
    run([ca65, "jamxe.asm", "-o", "jamxe.o"], cwd=JAMXE)
    run([ld65, "-C", "jamxe.cfg", "-o", "jamxe_link.xex", "jamxe.o"], cwd=JAMXE)

    # Pack XEX
    out_xex = dist / "jamkid.xex"
    run([sys.executable, "pack_xe_banks.py", str(args.weights), "jamxe_link.xex", str(out_xex)], cwd=JAMXE)
    print(f"OK: {out_xex}")

    # Pack ATR (if build_atr.py exists)
    atr_builder = JAMXE / "build_atr.py"
    if atr_builder.exists():
        out_atr = dist / "jamkid.atr"
        run([sys.executable, str(atr_builder), str(args.weights), "jamxe_link.xex", str(out_atr)], cwd=JAMXE)
        print(f"OK: {out_atr}")

    # Restore jamxe theme so jamxe build stays clean
    jamxe_data = JAMXE / "td_xe.txt"
    if jamxe_data.exists():
        run([sys.executable, str(JAMXE / "gen_theme.py"), str(jamxe_data), str(JAMXE / "theme_xe.inc")], cwd=JAMXE)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build the main JAMXE runtime."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CC65 = ROOT.parent / "tools" / "cc65" / "bin"
CA65 = CC65 / "ca65.exe"
LD65 = CC65 / "ld65.exe"


def run(args, cwd: Path) -> None:
    print("+", " ".join(str(a) for a in args))
    subprocess.run(args, cwd=cwd, check=True)


def main() -> None:
    data = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "td_xe.txt"
    weights = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "weights_xe.bin"
    live = ROOT / "live"
    live.mkdir(exist_ok=True)

    if not data.exists():
        raise SystemExit(f"data not found: {data}")
    if not weights.exists():
        raise SystemExit(f"weights not found: {weights}")

    run([sys.executable, "gen_theme.py", str(data), "theme_xe.inc"], ROOT)

    blob = weights.read_bytes()
    if len(blob) < 65536:
        raise SystemExit(f"weights too small: {len(blob)} B")
    (ROOT / "weights_xe_main.bin").write_bytes(blob[65536:])

    run([str(CA65), "jamxe.asm", "-o", "jamxe.o"], ROOT)
    run([str(LD65), "-C", "jamxe.cfg", "-o", "jamxe_link.xex", "jamxe.o"], ROOT)
    run([sys.executable, "pack_xe_banks.py", str(weights), "jamxe_link.xex", str(live / "jamxe.xex")], ROOT)

    xex = (live / "jamxe.xex").read_bytes()
    print(f"jamxe.xex: {len(xex)} B")


if __name__ == "__main__":
    main()

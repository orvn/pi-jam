#!/usr/bin/env python3
"""
Canonical JAMXE trainer.

Main XE lane:

- 512 input
- 512 hidden
- 45 output
- W1 = INT2
- W2 = INT4

The archived baseline is kept separately in `train_xe_stable.py`.
"""
from __future__ import annotations

import os
import sys

import train_xe_strong


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python train_xe.py <training_data.txt>")
        print("  Output: weights_xe.bin")
        raise SystemExit(1)

    data = sys.argv[1]
    out_dir = os.path.dirname(os.path.abspath(data))
    out = os.path.join(out_dir, "weights_xe.bin")
    train_xe_strong.train(data, out)


if __name__ == "__main__":
    main()

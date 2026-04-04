#!/usr/bin/env python3
"""Package JAMXE as a bank-loaded 130XE XEX.

Sequence:
- INIT stub selects bank 0, loader writes bank0 data to $4000-$7FFF
- repeat for bank 1..3
- final INIT stub restores main RAM
- append CODE and RODATA as proper XEX segments
- append RUNAD -> $0D00
"""
from __future__ import annotations

import struct
import sys
from pathlib import Path

BANK_SIZE = 16384
W1_SIZE = BANK_SIZE * 4
PORTB_BANK = [0xA3, 0xA7, 0xAB, 0xAF]
PORTB_MAIN = 0xB3
CODE_START = 0x0D00
CODE_SIZE = 0x1300
RODATA_START = 0x8000


def seg(start: int, data: bytes) -> bytes:
    return struct.pack("<HH", start, start + len(data) - 1) + data


def initad(addr: int) -> bytes:
    return struct.pack("<HHH", 0x02E2, 0x02E3, addr)


def runad(addr: int) -> bytes:
    return struct.pack("<HHH", 0x02E0, 0x02E1, addr)


def init_stub(portb_value: int) -> bytes:
    code = bytes([
        0xA9, portb_value,      # lda #imm
        0x8D, 0x01, 0xD3,       # sta $D301
        0x60,                   # rts
    ])
    return seg(0x0600, code) + initad(0x0600)


def pack(weights_file: str, linker_file: str, output_file: str) -> None:
    weights = Path(weights_file).read_bytes()
    linker = Path(linker_file).read_bytes()

    if len(weights) < W1_SIZE:
        raise SystemExit(f"weights too small: {len(weights)} B")
    if len(linker) < CODE_SIZE:
        raise SystemExit(f"linker too small: {len(linker)} B")

    code = bytearray(linker[:CODE_SIZE])
    rodata = bytearray(linker[CODE_SIZE:])
    while code and code[-1] == 0:
        code.pop()
    while rodata and rodata[-1] == 0:
        rodata.pop()

    xex = bytearray(b"\xFF\xFF")
    for bank, portb in enumerate(PORTB_BANK):
        start = bank * BANK_SIZE
        bank_data = weights[start : start + BANK_SIZE]
        if len(bank_data) != BANK_SIZE:
            raise SystemExit(f"bank {bank} size mismatch: {len(bank_data)} B")
        xex += init_stub(portb)
        xex += seg(0x4000, bank_data)

    xex += init_stub(PORTB_MAIN)
    xex += seg(CODE_START, code)
    xex += seg(RODATA_START, rodata)
    xex += runad(CODE_START)

    Path(output_file).write_bytes(xex)
    print(f"Packed {output_file}: {len(xex)} B")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python pack_xe_banks.py <weights_xe.bin> <jamxe_link.xex> <out.xex>")
        raise SystemExit(1)
    pack(sys.argv[1], sys.argv[2], sys.argv[3])

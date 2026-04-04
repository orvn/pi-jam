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
import re

BANK_SIZE = 16384
W1_SIZE = BANK_SIZE * 4
PORTB_BANK = [0xA3, 0xA7, 0xAB, 0xAF]
PORTB_MAIN = 0xB3
CODE_START = 0x0D00
CODE_MEM_SIZE = 0x1300       # filled region size from jamxe.cfg
RODATA_START = 0x8000
RODATA_MEM_SIZE = 0x4000     # filled region size from jamxe.cfg
MAX_SAFE_CODE_END = CODE_START + CODE_MEM_SIZE - 1


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


def parse_segments(map_file: str) -> tuple[int, int, int]:
    text = Path(map_file).read_text(encoding="utf-8", errors="replace")
    sizes: dict[str, int] = {"CODE": 0, "HICODE": 0, "RODATA": 0}
    in_segment_list = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line == "Segment list:":
            in_segment_list = True
            continue
        if not in_segment_list:
            continue
        m = re.match(r"^(CODE|HICODE|RODATA)\s+[0-9A-F]{6}\s+[0-9A-F]{6}\s+([0-9A-F]{6})\s+[0-9A-F]{5}$", line)
        if m:
            sizes[m.group(1)] = int(m.group(2), 16)
    if sizes["CODE"] <= 0:
        raise SystemExit(f"failed to parse CODE size from map: {map_file}")
    return sizes["CODE"], sizes["HICODE"], sizes["RODATA"]


def pack(weights_file: str, linker_file: str, map_file: str, output_file: str) -> None:
    weights = Path(weights_file).read_bytes()
    linker = Path(linker_file).read_bytes()
    code_len, hicode_len, rodata_len = parse_segments(map_file)

    if len(weights) < W1_SIZE:
        raise SystemExit(f"weights too small: {len(weights)} B")
    expected = CODE_MEM_SIZE + RODATA_MEM_SIZE
    if len(linker) < expected:
        raise SystemExit(
            f"linker too small: {len(linker)} B < {expected} B"
        )
    if code_len > CODE_MEM_SIZE:
        raise SystemExit(
            f"CODE segment ({code_len} B) exceeds CODE_MEM region ({CODE_MEM_SIZE} B)"
        )

    # Linker output is two filled regions: CODE_MEM then RODATA (which holds HICODE + RODATA)
    code = linker[:code_len]
    rodata = linker[CODE_MEM_SIZE : CODE_MEM_SIZE + hicode_len + rodata_len]

    code_end = CODE_START + code_len - 1

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
    print(f"  CODE   ${CODE_START:04X}-${code_end:04X} ({code_len} B)")
    if rodata:
        ro_end = RODATA_START + len(rodata) - 1
        print(f"  HIGH   ${RODATA_START:04X}-${ro_end:04X} ({len(rodata)} B)")


if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python pack_xe_banks.py <weights_xe.bin> <jamxe_link.xex> <mapfile> <out.xex>")
        raise SystemExit(1)
    pack(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])

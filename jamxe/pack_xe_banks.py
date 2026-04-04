#!/usr/bin/env python3
"""Package JAMXE as a bank-loaded 130XE XEX (SIO2SD-safe).

Sequence (safe: PORTB stays in main-RAM mode during all SIO transfers):
- load bank0 data to $4000-$7FFF in MAIN RAM
- INIT stub copies $4000-$7FFF from main RAM to extended bank 0
- repeat for bank 1..3
- append CODE and RODATA as proper XEX segments
- append RUNAD -> $0D00

The copy stub uses $0700-$07FF as a 256-byte page buffer and toggles
PORTB only inside the stub (no bank switching during disk I/O).
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
STUB_ADDR = 0x0600
BUF_ADDR = 0x0700


def seg(start: int, data: bytes) -> bytes:
    return struct.pack("<HH", start, start + len(data) - 1) + data


def initad(addr: int) -> bytes:
    return struct.pack("<HHH", 0x02E2, 0x02E3, addr)


def runad(addr: int) -> bytes:
    return struct.pack("<HHH", 0x02E0, 0x02E1, addr)


def sio2sd_init_stub() -> bytes:
    """INIT stub that prepares the machine for safe SIO2SD loading.

    SIO2SD loads XEX segments in small chunks. ANTIC DMA reads from
    memory being overwritten during loading, causing crashes. This stub:
    1. Ensures PORTB is in safe main-RAM mode
    2. Disables ANTIC DMA (STA $D400) to stop display reads
    3. Sets RAMTOP to $D0 to move screen buffers out of the way
    4. Closes and reopens E: so the OS reallocates screen memory high

    Must run as the very first INIT before any large segments load.
    """
    code = bytes([
        0xAD, 0x01, 0xD3,       # LDA $D301
        0x09, 0xB3,             # ORA #$B3         ; safe RAM config
        0x8D, 0x01, 0xD3,       # STA $D301
        0xA9, 0x00,             # LDA #0
        0x8D, 0x00, 0xD4,       # STA $D400        ; disable ANTIC DMA
        0xA9, 0xD0,             # LDA #$D0
        0x85, 0x6A,             # STA $6A          ; RAMTOP = $D000
        0xA9, 0x0C,             # LDA #12          ; CLOSE command
        0x8D, 0x42, 0x03,       # STA ICCOM
        0x20, 0x56, 0xE4,       # JSR CIOV         ; close E:
        0xA9, 0x03,             # LDA #3           ; OPEN command
        0x8D, 0x42, 0x03,       # STA ICCOM
        0xA9, 0x36,             # LDA #<edev       ; -> "E:",EOL at $0636
        0x8D, 0x44, 0x03,       # STA ICBAL
        0xA9, 0x06,             # LDA #>edev
        0x8D, 0x45, 0x03,       # STA ICBAH
        0xA9, 0x0C,             # LDA #$0C         ; read+write
        0x8D, 0x4A, 0x03,       # STA ICAX1
        0xA9, 0x00,             # LDA #0
        0x8D, 0x4B, 0x03,       # STA ICAX2
        0x20, 0x56, 0xE4,       # JSR CIOV         ; open E:
        0x60,                   # RTS
        0x45, 0x3A, 0x9B,       # "E:", EOL        ; at $0636
    ])
    return code


def copy_stub(portb_bank: int) -> bytes:
    """Generate INIT stub that copies 16KB from main $4000-$7FFF to a bank.

    Uses $0700-$07FF as a 256-byte page buffer. PORTB is only switched
    inside the copy loop, never during SIO disk transfers.

    Layout at $0600 (51 bytes):
      $0600  LDX #$40           start page
      $0602  STX $0611          patch src high byte
      $0605  STX $0624          patch dst high byte
      $0608  LDA #$B3           PORTB_MAIN
      $060A  STA $D301
      $060D  LDY #$00
      $060F  LDA $4000,Y        (src, high byte patched)
      $0612  STA $0700,Y        buffer
      $0615  INY
      $0616  BNE $060F
      $0618  LDA #portb_bank
      $061A  STA $D301
      $061D  LDY #$00
      $061F  LDA $0700,Y        buffer
      $0622  STA $4000,Y        (dst, high byte patched)
      $0625  INY
      $0626  BNE $061F
      $0628  INX
      $0629  CPX #$80
      $062B  BNE $0602
      $062D  LDA #$B3
      $062F  STA $D301
      $0632  RTS
    """
    src_hi_addr = STUB_ADDR + 0x11   # high byte of LDA $4000,Y
    dst_hi_addr = STUB_ADDR + 0x24   # high byte of STA $4000,Y

    code = bytes([
        # -- page loop top ($0600) --
        0xA2, 0x40,                         # LDX #$40         ; start at page $40
        # -- @nextpage ($0602) --
        0x8E, src_hi_addr & 0xFF, src_hi_addr >> 8,  # STX src+2
        0x8E, dst_hi_addr & 0xFF, dst_hi_addr >> 8,  # STX dst+2
        # Phase 1: read page from main RAM into buffer
        0xA9, PORTB_MAIN,                   # LDA #$B3
        0x8D, 0x01, 0xD3,                   # STA $D301
        0xA0, 0x00,                         # LDY #$00
        # @src ($060F):
        0xB9, 0x00, 0x40,                   # LDA $4000,Y      (high byte patched)
        0x99, 0x00, BUF_ADDR >> 8,          # STA $0700,Y
        0xC8,                               # INY
        0xD0, 0xF7,                         # BNE @src          (-9)
        # Phase 2: write buffer to bank
        0xA9, portb_bank,                   # LDA #bank_portb
        0x8D, 0x01, 0xD3,                   # STA $D301
        0xA0, 0x00,                         # LDY #$00
        # @dst ($061F):
        0xB9, 0x00, BUF_ADDR >> 8,          # LDA $0700,Y
        0x99, 0x00, 0x40,                   # STA $4000,Y      (high byte patched)
        0xC8,                               # INY
        0xD0, 0xF7,                         # BNE @dst          (-9)
        # Next page
        0xE8,                               # INX
        0xE0, 0x80,                         # CPX #$80          ; until page $80
        0xD0, 0xD5,                         # BNE @nextpage     (-43 -> $0602)
        # Restore main RAM
        0xA9, PORTB_MAIN,                   # LDA #$B3
        0x8D, 0x01, 0xD3,                   # STA $D301
        0x60,                               # RTS
    ])
    assert len(code) == 51, f"copy stub is {len(code)} bytes, expected 51"
    return code


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

    # SIO2SD fix: disable ANTIC DMA and move screen memory before loading
    sio_stub = sio2sd_init_stub()
    xex += seg(STUB_ADDR, sio_stub)
    xex += initad(STUB_ADDR)

    for bank, portb in enumerate(PORTB_BANK):
        start = bank * BANK_SIZE
        bank_data = weights[start : start + BANK_SIZE]
        if len(bank_data) != BANK_SIZE:
            raise SystemExit(f"bank {bank} size mismatch: {len(bank_data)} B")
        # SIO2SD-safe: load bank data to MAIN RAM, then INIT copies to bank
        xex += seg(0x4000, bank_data)
        xex += seg(STUB_ADDR, copy_stub(portb))
        xex += initad(STUB_ADDR)

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

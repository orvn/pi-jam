#!/usr/bin/env python3
"""Build a bootable ATR for JAMXE on SIO2SD.

SIO2SD can't handle PORTB in bank mode during SIO transfers.
Strategy: read each 16KB bank to main RAM $4000-$7FFF, then copy
to extended bank using indirect-indexed (ZP),Y with a $0600 page
buffer. No self-modifying code.

Layout (128-byte SD sectors):
  Sectors 1-3:    Boot loader
  Sectors 4-131:  Bank 0 (16384 B)
  Sectors 132-259: Bank 1
  Sectors 260-387: Bank 2
  Sectors 388-515: Bank 3
  Sectors 516+:   CODE ($0D00) then RODATA ($8000)
"""
from __future__ import annotations
import sys
from pathlib import Path

ATR_SS = 128
BOOT_SEC = 3
BOOT_ADDR = 0x0700
BOOT_INIT = 0x0706
BANK_SIZE = 16384
N_BANKS = 4
PORTB_BANK = [0xA3, 0xA7, 0xAB, 0xAF]
PORTB_MAIN = 0xB3
CODE_START = 0x0D00
CODE_MEM = 0x1300
RODATA_START = 0x8000
RODATA_MEM = 0x4000


def build_boot(code_end_hi, rodata_end_hi):
    c = bytearray()
    def emit(*b): c.extend(b)

    # ZP locations used by copy routine
    ZP_PTR = 0x82     # 2 bytes: pointer into $4000-$7FFF
    ZP_BANKPB = 0x84  # 1 byte: PORTB value for current bank

    # Boot header
    emit(0x00, BOOT_SEC)
    emit(BOOT_ADDR & 0xFF, BOOT_ADDR >> 8)
    emit(BOOT_INIT & 0xFF, BOOT_INIT >> 8)

    # Init
    emit(0xD8)                    # CLD
    emit(0xA2, 0xFF, 0x9A)       # LDX #$FF / TXS
    emit(0xAD, 0x01, 0xD3)       # LDA $D301
    emit(0x09, PORTB_MAIN)       # ORA #$B3
    emit(0x8D, 0x01, 0xD3)       # STA $D301

    # Disable ANTIC DMA + move screen above RODATA area
    emit(0xA9, 0x00)
    emit(0x8D, 0x00, 0xD4)       # STA DMACTL
    emit(0x8D, 0x2F, 0x02)       # STA SDMCTL
    emit(0xA9, 0xC0)
    emit(0x85, 0x6A)             # RAMTOP = $C0 (screen at ~$BC00, below OS ROM)

    # SIO DCB setup
    emit(0xA9, 0x31, 0x8D, 0x00, 0x03)  # DDEVIC
    emit(0xA9, 0x01, 0x8D, 0x01, 0x03)  # DUNIT
    emit(0xA9, 0x52, 0x8D, 0x02, 0x03)  # DCOMND=read
    emit(0xA9, 0x40, 0x8D, 0x03, 0x03)  # DSTATS=read
    emit(0xA9, 0x07, 0x8D, 0x06, 0x03)  # DTIMLO
    emit(0xA9, ATR_SS, 0x8D, 0x08, 0x03)  # DBYTLO
    emit(0xA9, 0x00, 0x8D, 0x09, 0x03)  # DBYTHI
    emit(0xA9, 0x04, 0x8D, 0x0A, 0x03)  # DAUX1=4
    emit(0xA9, 0x00, 0x8D, 0x0B, 0x03)  # DAUX2=0

    # --- Bank loop: read to main $4000, copy to extended ---
    emit(0xA9, 0x00, 0x85, 0x80)  # bank counter in $80

    bank_loop = len(c)
    # Read 128 sectors to $4000-$7FFF (PORTB stays main)
    emit(0xA9, 0x00, 0x8D, 0x04, 0x03)  # DBUFLO=0
    emit(0xA9, 0x40, 0x8D, 0x05, 0x03)  # DBUFHI=$40

    read_loop = len(c)
    emit(0x20, 0x59, 0xE4)       # JSR SIOV
    emit(0x18)                    # CLC
    emit(0xAD, 0x04, 0x03)       # LDA DBUFLO
    emit(0x69, 0x80)             # ADC #128
    emit(0x8D, 0x04, 0x03)       # STA DBUFLO
    emit(0x90, 0x03)             # BCC +3
    emit(0xEE, 0x05, 0x03)       # INC DBUFHI
    emit(0xEE, 0x0A, 0x03)       # INC DAUX1
    emit(0xD0, 0x03)             # BNE +3
    emit(0xEE, 0x0B, 0x03)       # INC DAUX2
    emit(0xAD, 0x05, 0x03)       # LDA DBUFHI
    emit(0xC9, 0x80)             # CMP #$80
    d = read_loop - (len(c) + 2)
    emit(0x90, d & 0xFF)          # BCC read_loop

    # --- Copy $4000-$7FFF from main to bank ---
    # Get bank PORTB value
    emit(0xA6, 0x80)              # LDX $80 (bank index)
    portb_ref = len(c) + 1
    emit(0xBD, 0x00, 0x00)       # LDA portb_tbl,X (patched)
    emit(0x85, ZP_BANKPB)        # STA $84

    # Init pointer to $4000
    emit(0xA9, 0x00, 0x85, ZP_PTR)      # PTR lo = 0
    emit(0xA9, 0x40, 0x85, ZP_PTR + 1)  # PTR hi = $40

    page_loop = len(c)
    # Phase 1: main RAM -> $0600 buffer (PORTB is already main)
    emit(0xA0, 0x00)              # LDY #0
    src_loop = len(c)
    emit(0xB1, ZP_PTR)           # LDA (ZP_PTR),Y
    emit(0x99, 0x00, 0x06)       # STA $0600,Y
    emit(0xC8)                    # INY
    d = src_loop - (len(c) + 2)
    emit(0xD0, d & 0xFF)          # BNE src_loop

    # Phase 2: $0600 buffer -> bank
    emit(0xA5, ZP_BANKPB)        # LDA $84
    emit(0x8D, 0x01, 0xD3)       # STA PORTB (select bank)
    emit(0xA0, 0x00)              # LDY #0
    dst_loop = len(c)
    emit(0xB9, 0x00, 0x06)       # LDA $0600,Y
    emit(0x91, ZP_PTR)           # STA (ZP_PTR),Y
    emit(0xC8)                    # INY
    d = dst_loop - (len(c) + 2)
    emit(0xD0, d & 0xFF)          # BNE dst_loop

    # Restore main RAM for next page
    emit(0xA9, PORTB_MAIN)
    emit(0x8D, 0x01, 0xD3)       # STA PORTB

    # Next page: inc PTR high byte
    emit(0xE6, ZP_PTR + 1)       # INC $83
    emit(0xA5, ZP_PTR + 1)       # LDA $83
    emit(0xC9, 0x80)             # CMP #$80
    d = page_loop - (len(c) + 2)
    emit(0x90, d & 0xFF)          # BCC page_loop

    # Next bank
    emit(0xE6, 0x80)              # INC $80
    emit(0xA5, 0x80)              # LDA $80
    emit(0xC9, N_BANKS)           # CMP #4
    d = bank_loop - (len(c) + 2)
    emit(0x90, d & 0xFF)          # BCC bank_loop

    # --- Load CODE ---
    emit(0xA9, CODE_START & 0xFF, 0x8D, 0x04, 0x03)
    emit(0xA9, CODE_START >> 8, 0x8D, 0x05, 0x03)
    code_read = len(c)
    emit(0x20, 0x59, 0xE4)
    emit(0x18, 0xAD, 0x04, 0x03, 0x69, 0x80, 0x8D, 0x04, 0x03)
    emit(0x90, 0x03, 0xEE, 0x05, 0x03)
    emit(0xEE, 0x0A, 0x03, 0xD0, 0x03, 0xEE, 0x0B, 0x03)
    emit(0xAD, 0x05, 0x03, 0xC9, code_end_hi)
    d = code_read - (len(c) + 2)
    emit(0x90, d & 0xFF)

    # --- Load RODATA ---
    emit(0xA9, RODATA_START & 0xFF, 0x8D, 0x04, 0x03)
    emit(0xA9, RODATA_START >> 8, 0x8D, 0x05, 0x03)
    ro_read = len(c)
    emit(0x20, 0x59, 0xE4)
    emit(0x18, 0xAD, 0x04, 0x03, 0x69, 0x80, 0x8D, 0x04, 0x03)
    emit(0x90, 0x03, 0xEE, 0x05, 0x03)
    emit(0xEE, 0x0A, 0x03, 0xD0, 0x03, 0xEE, 0x0B, 0x03)
    emit(0xAD, 0x05, 0x03, 0xC9, rodata_end_hi)
    d = ro_read - (len(c) + 2)
    emit(0x90, d & 0xFF)

    # --- Reopen E: with new RAMTOP, enable screen, run ---
    emit(0xA2, 0x00)              # LDX #0 (IOCB #0)
    emit(0xA9, 0x0C, 0x8D, 0x42, 0x03)  # ICCOM = CLOSE
    emit(0x20, 0x56, 0xE4)       # JSR CIOV
    emit(0xA2, 0x00)              # LDX #0
    emit(0xA9, 0x03, 0x8D, 0x42, 0x03)  # ICCOM = OPEN
    edev_lo = len(c) + 1
    emit(0xA9, 0x00, 0x8D, 0x44, 0x03)  # ICBAL (patched)
    edev_hi = len(c) + 1
    emit(0xA9, 0x00, 0x8D, 0x45, 0x03)  # ICBAH (patched)
    emit(0xA9, 0x0C, 0x8D, 0x4A, 0x03)  # ICAX1
    emit(0xA9, 0x00, 0x8D, 0x4B, 0x03)  # ICAX2
    emit(0x20, 0x56, 0xE4)       # JSR CIOV → screen at RAMTOP=$D0
    emit(0x4C, CODE_START & 0xFF, CODE_START >> 8)  # JMP start

    # --- Data tables ---
    edev_addr = BOOT_ADDR + len(c)
    emit(0x45, 0x3A, 0x9B)       # "E:", EOL
    c[edev_lo] = edev_addr & 0xFF
    c[edev_hi] = (edev_addr >> 8) & 0xFF

    portb_tbl = BOOT_ADDR + len(c)
    for pv in PORTB_BANK:
        emit(pv)
    c[portb_ref] = portb_tbl & 0xFF
    c[portb_ref + 1] = (portb_tbl >> 8) & 0xFF

    print(f"  Boot: {len(c)} bytes (limit {BOOT_SEC * ATR_SS})")
    while len(c) < BOOT_SEC * ATR_SS:
        c.append(0)
    assert len(c) <= BOOT_SEC * ATR_SS, f"Boot too large: {len(c)}"
    return bytes(c)


def build_atr(weights_file, linker_file, output_file):
    weights = Path(weights_file).read_bytes()
    linker = Path(linker_file).read_bytes()
    assert len(weights) >= BANK_SIZE * N_BANKS
    assert len(linker) >= CODE_MEM

    code_data = linker[:CODE_MEM]
    # Strip RODATA fill zeros — actual content ends at ~$BA88,
    # screen memory starts at ~$BC00 (RAMTOP=$C0)
    rodata_data = linker[CODE_MEM:CODE_MEM + RODATA_MEM].rstrip(b'\x00')
    r = len(rodata_data) % ATR_SS
    if r:
        rodata_data += b'\x00' * (ATR_SS - r)

    code_end_hi = (((CODE_START + len(code_data)) + 0xFF) >> 8) & 0xFF
    rodata_end_hi = (((RODATA_START + len(rodata_data)) + 0xFF) >> 8) & 0xFF

    bs = BANK_SIZE // ATR_SS
    total = max(BOOT_SEC + N_BANKS * bs + len(code_data) // ATR_SS + len(rodata_data) // ATR_SS, 720)
    print(f"  Banks: {N_BANKS}x{bs}={N_BANKS*bs} sectors")
    print(f"  CODE:  {len(code_data)} B  RODATA: {len(rodata_data)} B")

    boot = build_boot(code_end_hi, rodata_end_hi)
    disk = bytearray(total * ATR_SS)
    disk[:len(boot)] = boot
    pos = BOOT_SEC * ATR_SS
    for b in range(N_BANKS):
        disk[pos:pos+BANK_SIZE] = weights[b*BANK_SIZE:(b+1)*BANK_SIZE]
        pos += BANK_SIZE
    disk[pos:pos+len(code_data)] = code_data; pos += len(code_data)
    disk[pos:pos+len(rodata_data)] = rodata_data

    img_sz = total * ATR_SS
    par = img_sz // 16
    hdr = bytes([0x96, 0x02, par & 0xFF, (par >> 8) & 0xFF, ATR_SS, 0, 0,0,0,0,0,0,0,0,0,0])
    Path(output_file).write_bytes(hdr + disk)
    print(f"  ATR: {output_file} ({len(hdr)+len(disk)} B)")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python build_atr.py <weights.bin> <link.xex> <out.atr>")
        raise SystemExit(1)
    build_atr(sys.argv[1], sys.argv[2], sys.argv[3])

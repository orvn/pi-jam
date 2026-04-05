# Pi Jam

A generative language model for inference on Rasberry Pi Zero.

Lineage and naming of this repo comes from [Atari JAM](https://github.com/marspa73/atarijam), originally designed to run on Atari 800.

_Work in progress, more to come_

---

## Build

Requires `gcc` and `make`. No other dependencies for inference.

```sh
make
```

This builds all four variants into their respective `dist/` directories.

To cross-compile for Pi Zero (ARM1176):

```sh
CC=arm-linux-gnueabihf-gcc make
```

To build a single variant:

```sh
make -C jam       # compact model  (~18 KB weights)
make -C picojam   # ultra-tiny     (~1.6 KB weights)
make -C jamxe     # extended       (~76 KB weights)
make -C jamkid    # kid register   (~76 KB weights, same engine as jamxe)
```

## Run

Each variant is a REPL. Type a query, get a response.

```sh
./jam/dist/jam jam/weights_b2s.bin
./picojam/dist/picojam picojam/weights_pico.bin
./jamxe/dist/jamxe jamxe/weights_xe.bin
./jamkid/dist/jamkid jamkid/weights_jamkid.bin
```

When running on the Pi, the binary and weights file can sit in the same directory and the weights path can be omitted:

```sh
cd jam && ./dist/jam
```

## Try

Queries that tend to work well per variant:

| Variant | Try |
|---------|-----|
| jam | `2+3`, `COUNT 5`, `BIN 7`, `REV DOG` |
| picojam | `3+4`, `CNT5`, `BIN7`, `GPT` |
| jamxe | `HOW ARE YOU`, `WHAT IS LOVE`, `3+4`, `COUNT 5` |
| jamkid | `HELLO`, `NAME`, `ORCA`, `STORY` |

## Retrain

Training requires Python 3 and PyTorch (offline, not needed on the Pi).

```sh
cd jam
python3 train_compact_2bit.py td_compact.txt weights_b2s.bin
make
```

Same pattern for the other variants (`train_xe_strong.py` for jamxe/jamkid, `train_pico.py` for picojam).
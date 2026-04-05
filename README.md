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
make -C jamxe        # extended MLP    (~76 KB weights)
make -C jamkid       # kid register    (~76 KB weights, same engine as jamxe)
make -C transformer  # full transformer
```

## Run

Each variant is a REPL. Type a query, get a response.

```sh
./jamxe/dist/jamxe jamxe/weights_xe.bin
./jamkid/dist/jamkid jamkid/weights_jamkid.bin
./transformer/dist/transformer transformer/weights_transformer.bin
```

When running on the Pi, the binary and weights file can sit in the same directory and the weights path can be omitted:

```sh
cd jamxe && ./dist/jamxe
```

## Try

| Variant | Try |
|---------|-----|
| jamxe | `HOW ARE YOU`, `WHAT IS LOVE`, `3+4`, `COUNT 5` |
| jamkid | `HELLO`, `NAME`, `ORCA`, `STORY` |
| transformer | open-ended queries, longer responses |

## Retrain

Training requires Python 3 and PyTorch (offline, not needed on the Pi).

```sh
# MLP (jamxe / jamkid)
cd jamxe
python3 train_xe_strong.py td_xe.txt weights_xe.bin
make

# Transformer
python3 train_transformer.py td_xe.txt weights_transformer.bin
make -C transformer
```
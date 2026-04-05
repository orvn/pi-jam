# Pi JAM

Generative language models for inference on Raspberry Pi Zero.

Lineage and naming of this repo comes from [Atari JAM](https://github.com/marspa73/atarijam), originally designed to run on Atari 800.

_Just Another Model(s)_ for the smallest Rasberry Pi!

---

## Models

| Variant | Type | Weights | Description |
|---------|------|---------|-------------|
| `jamxe` | 2-layer MLP | 76 KB | 512→512→45, dual-source feature hash with frozen query anchor |
| `jamkid` | 2-layer MLP | 76 KB | Same engine as jamxe, trained on a kid-register dataset |
| `neuralpi` (WIP) | Decoder transformer | ~25 MB | 8-layer attention model, KV cache inference |

`jamxe` and `jamkid` respond in milliseconds. 
`neuralpi` _(work in progress)_ will be slower but architecturally more capable

---

## Build

Requires `gcc` and `make`. No other dependencies for inference.

```sh
make
```

To cross-compile for Pi Zero (ARM1176)
```sh
CC=arm-linux-gnueabihf-gcc make
```

To build a single variant:

```sh
make -C jamxe    # extended MLP     (~76 KB weights)
make -C jamkid   # kid register     (~76 KB weights, same engine as jamxe)
make -C neuralpi # decoder transformer
```

## Run

Each variant is a REPL: type a query, get a response

```sh
./jamxe/dist/jamxe jamxe/weights_xe.bin
./jamkid/dist/jamkid jamkid/weights_jamkid.bin
./neuralpi/dist/neuralpi neuralpi/weights_neuralpi.bin
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
| neuralpi | `HOW ARE YOU`, `WHAT IS ATARI`, `TELL A STORY` |

## Retrain

Training requires Python 3 and PyTorch (offline and independent from the Pi).

```sh
# MLP (jamxe / jamkid)
cd jamxe
python3 train_xe_strong.py td_xe.txt weights_xe.bin
make

# NeuralPi (transformer)
python3 train_transformer.py td_xe.txt neuralpi/weights_neuralpi.bin
make -C neuralpi
```

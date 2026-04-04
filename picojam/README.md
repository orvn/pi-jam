# picojam

Tiny Atari language model release.

Current shipped files:

- source: `jampico.asm`
- trainer: `train_pico.py`
- data: `td_pico_mix.txt`
- weights: `weights_pico.bin`
- binary: `dist/picojam.xex`

## Build

```powershell
python build.py
```

## Retrain

```powershell
python build.py --train
```

The default build uses the shipped weights and regenerates `theme_pico.inc` from the data file.

# jam

Compact standalone JAM release.

Current shipped files:

- source: `jambo.asm`
- trainer: `train_compact_2bit.py`
- data: `td_compact.txt`
- weights: `weights_b2s.bin`
- binary: `dist/jam.xex`

## Build

```powershell
python build.py
```

## Retrain

```powershell
python build.py --train
```

This is the compact non-banked JAM line with the current incremental `L1` engine.

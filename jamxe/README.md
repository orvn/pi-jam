# jamxe

Bank-switched Atari 130XE JAM release.

Current shipped files:

- source: `jamxe.asm`
- trainers: `train_xe.py`, `train_xe_strong.py`
- data: `td_xe.txt`
- weights: `weights_xe.bin`
- binary: `dist/jamxe.xex`

## Build

```powershell
python build.py
```

## Retrain

```powershell
python build.py --train
```

The build script splits the packed XE blob and repacks the final `.xex` with bank data.

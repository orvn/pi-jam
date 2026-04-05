
# JAM KID

Same JAMXE 512→512→45 runtime, kid personality.

A small, curious, playful machine child living in 80 KB of Atari RAM.
Short concrete language, feelings, pretend play, and gentle nonsense.

## Build

```powershell
python jamkid\build.py
```

Retrain and rebuild:

```powershell
python jamkid\build.py --train
```

## Architecture

Shares runtime with `jamxe/` (same ASM, cfg, packer).
Only the training data and weights differ.

## Output

- `dist/jamkid.xex` — XEX for emulators and disk drives
- `dist/jamkid.atr` — ATR for SIO2SD (boot from D1:)

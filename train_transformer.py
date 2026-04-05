#!/usr/bin/env python3
"""
Decoder-only character-level transformer trainer for Pi Jam.

Target (Pi Zero W v1.1 defaults):
    8 layers, d_model=512, n_heads=8, d_ff=2048, ctx=64
    ~25M parameters → ~25MB INT8 export

Same training data format as the MLP variants:
    query|ANSWER        one pair per line
    Lines starting with # or @ are skipped

The model is trained in float32 then exported to INT8 (per-tensor symmetric
quantization). Weights + float32 scales are written to a binary blob that
inference.c reads directly.

Usage:
    python3 train_transformer.py td_xe.txt weights_transformer.bin
    python3 train_transformer.py td_xe.txt weights_transformer.bin --layers 4 --dim 256
"""

from __future__ import annotations

import argparse
import math
import random
import struct
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Vocabulary  (compatible with jam/jamxe charset at indices 0-43)
# ---------------------------------------------------------------------------

CHARS = " ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-,!?'/"  # 44 chars, indices 0-43
EOS   = 44   # end of sequence
BOS   = 45   # beginning of sequence
SEP   = 46   # query / response boundary (was '>' in MLP context strings)
PAD   = 47   # sequence padding
VOCAB = 48

_c2i = {c: i for i, c in enumerate(CHARS)}

def encode_char(c: str) -> int:
    return _c2i.get(c.upper(), PAD)

def encode(text: str) -> list[int]:
    return [encode_char(c) for c in text]

def decode(ids: list[int]) -> str:
    out = []
    for i in ids:
        if i == EOS:
            break
        if i < len(CHARS):
            out.append(CHARS[i])
    return "".join(out)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class Config:
    n_layers:   int   = 8
    d_model:    int   = 512
    n_heads:    int   = 8
    d_ff:       int   = 2048
    ctx_len:    int   = 64
    dropout:    float = 0.1
    lr:         float = 3e-4
    batch_size: int   = 64
    epochs:     int   = 150
    warmup:     int   = 10      # warmup epochs
    aug_factor: int   = 20      # augmentation multiplier
    device:     str   = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def load_pairs(path: str) -> list[tuple[str, str]]:
    pairs = []
    seen = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("@"):
                continue
            if "|" not in line:
                continue
            q, a = line.split("|", 1)
            q, a = q.strip().upper(), a.strip().upper()
            if q and a and (q, a) not in seen:
                pairs.append((q, a))
                seen.add((q, a))
    return pairs


def augment_pairs(pairs: list, factor: int) -> list:
    out = list(pairs)
    for _ in range(factor - 1):
        for q, a in pairs:
            if len(q) > 3 and random.random() < 0.5:
                i = random.randint(0, len(q) - 1)
                r = random.random()
                if r < 0.33:
                    q = q[:i] + q[i + 1:]          # delete char
                elif r < 0.66:
                    q = q[:i] + q[i] + q[i:]       # duplicate char
                else:
                    j = random.randint(0, len(q) - 1)
                    q = q[:i] + q[j] + q[i + 1:]   # swap char
            out.append((q, a))
    return out


class QADataset(Dataset):
    """
    Each sample is a sequence:
        [BOS, q0, q1, ..., qn, SEP, a0, a1, ..., am, EOS, PAD, PAD, ...]

    Input  x: sequence[:-1]
    Target y: sequence[1:]   (shifted by one — standard language model target)
    Mask   m: True only at response positions (SEP onward in the target)
              so loss is computed only where the model is generating an answer
    """
    def __init__(self, pairs: list, ctx_len: int):
        self.items = []
        for q, a in pairs:
            seq = [BOS] + encode(q) + [SEP] + encode(a) + [EOS]
            sep_idx = len(q) + 1          # position of SEP in seq

            # pad / truncate to ctx_len + 1
            if len(seq) < ctx_len + 1:
                seq += [PAD] * (ctx_len + 1 - len(seq))
            else:
                seq = seq[:ctx_len + 1]

            x = seq[:-1]
            y = seq[1:]
            # loss mask: True from the position that predicts SEP onwards
            m = [i >= sep_idx for i in range(len(y))]
            self.items.append((
                torch.tensor(x, dtype=torch.long),
                torch.tensor(y, dtype=torch.long),
                torch.tensor(m, dtype=torch.bool),
            ))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads  = n_heads
        self.head_dim = d_model // n_heads
        self.wq = nn.Linear(d_model, d_model)
        self.wk = nn.Linear(d_model, d_model)
        self.wv = nn.Linear(d_model, d_model)
        self.wo = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        H, Dh = self.n_heads, self.head_dim

        q = self.wq(x).view(B, L, H, Dh).transpose(1, 2)
        k = self.wk(x).view(B, L, H, Dh).transpose(1, 2)
        v = self.wv(x).view(B, L, H, Dh).transpose(1, 2)

        scale  = math.sqrt(Dh)
        scores = (q @ k.transpose(-2, -1)) / scale
        scores = scores.masked_fill(causal_mask[:, :, :L, :L], float("-inf"))
        attn   = self.drop(F.softmax(scores, dim=-1))

        out = (attn @ v).transpose(1, 2).contiguous().view(B, L, D)
        return self.wo(out)


class FFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.w1   = nn.Linear(d_model, d_ff)
        self.w2   = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.drop(F.gelu(self.w1(x))))


class Block(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.ln1  = nn.LayerNorm(cfg.d_model)
        self.attn = MultiHeadAttention(cfg.d_model, cfg.n_heads, cfg.dropout)
        self.ln2  = nn.LayerNorm(cfg.d_model)
        self.ffn  = FFN(cfg.d_model, cfg.d_ff, cfg.dropout)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # Pre-norm with residual connections
        x = x + self.drop(self.attn(self.ln1(x), mask))
        x = x + self.drop(self.ffn(self.ln2(x)))
        return x


class Transformer(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(VOCAB, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.ctx_len, cfg.d_model)
        self.drop    = nn.Dropout(cfg.dropout)
        self.blocks  = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.ln_f    = nn.LayerNorm(cfg.d_model)
        self.head    = nn.Linear(cfg.d_model, VOCAB, bias=False)

        # Weight tying: output head shares token embedding weights
        self.head.weight = self.tok_emb.weight

        # Causal mask buffer — upper-triangular positions are masked out
        mask = torch.triu(torch.ones(cfg.ctx_len, cfg.ctx_len, dtype=torch.bool), diagonal=1)
        self.register_buffer("causal_mask", mask.unsqueeze(0).unsqueeze(0))

        self.apply(self._init)

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, 0.0, 0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, L = idx.shape
        pos  = torch.arange(L, device=idx.device)
        x    = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for block in self.blocks:
            x = block(x, self.causal_mask)
        return self.head(self.ln_f(x))

    def n_params(self) -> int:
        # Subtract tied output head (already counted in tok_emb)
        return sum(p.numel() for p in self.parameters()) - self.head.weight.numel()

# ---------------------------------------------------------------------------
# Generation (greedy, for eval)
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate(model: Transformer, query: str, max_new: int = 20) -> str:
    model.eval()
    cfg  = model.cfg
    dev  = next(model.parameters()).device
    ctx  = [BOS] + encode(query) + [SEP]
    for _ in range(max_new):
        feed   = torch.tensor([ctx[-cfg.ctx_len:]], dtype=torch.long, device=dev)
        logits = model(feed)[0, -1]                 # logits for last position
        tok    = logits.argmax().item()
        if tok == EOS:
            break
        ctx.append(tok)
    # Everything after SEP
    sep_pos = ctx.index(SEP) + 1 if SEP in ctx else len(ctx)
    return decode(ctx[sep_pos:])

# ---------------------------------------------------------------------------
# Export — INT8 weights + float32 scales + float32 biases/layernorms
# ---------------------------------------------------------------------------

def _quantize(w: torch.Tensor) -> tuple[bytes, float]:
    """Symmetric per-tensor INT8: scale = max(|w|) / 127."""
    scale = float(w.abs().max()) / 127.0
    if scale == 0.0:
        scale = 1.0
    q = (w / scale).round().clamp(-127, 127).to(torch.int8)
    return q.cpu().contiguous().numpy().tobytes(), scale


def _wf32(f, t: torch.Tensor) -> None:
    f.write(t.detach().cpu().float().contiguous().numpy().tobytes())


def _wq(f, t: torch.Tensor) -> None:
    data, scale = _quantize(t.detach().cpu().float())
    f.write(data)
    f.write(struct.pack("<f", scale))


def export(model: Transformer, path: str) -> None:
    """
    Binary blob layout:
      Header  28 bytes: magic(4) + version(4) + n_layers n_heads d_model d_ff ctx_len vocab (6×int32)
      tok_emb  VOCAB × d_model  int8 + float32 scale
      pos_emb  ctx_len × d_model  float32
      Per layer:
        ln1 weight + bias  d_model × float32 each
        wq wk wv wo        d_model×d_model int8 + float32 scale, then d_model float32 bias
        ln2 weight + bias  d_model × float32 each
        w1 (FFN up)        d_ff×d_model int8 + float32 scale, then d_ff float32 bias
        w2 (FFN down)      d_model×d_ff int8 + float32 scale, then d_model float32 bias
      ln_f weight + bias   d_model × float32 each
      (output head weight tied to tok_emb — not re-written)
    """
    cfg = model.cfg
    with open(path, "wb") as f:
        f.write(b"PJAM")
        f.write(struct.pack("<7I", 1, cfg.n_layers, cfg.n_heads,
                            cfg.d_model, cfg.d_ff, cfg.ctx_len, VOCAB))

        _wq(f, model.tok_emb.weight.data)
        _wf32(f, model.pos_emb.weight.data)

        for block in model.blocks:
            _wf32(f, block.ln1.weight.data)
            _wf32(f, block.ln1.bias.data)
            for proj in (block.attn.wq, block.attn.wk,
                         block.attn.wv, block.attn.wo):
                _wq(f,   proj.weight.data)
                _wf32(f, proj.bias.data)
            _wf32(f, block.ln2.weight.data)
            _wf32(f, block.ln2.bias.data)
            _wq(f,   block.ffn.w1.weight.data)
            _wf32(f, block.ffn.w1.bias.data)
            _wq(f,   block.ffn.w2.weight.data)
            _wf32(f, block.ffn.w2.bias.data)

        _wf32(f, model.ln_f.weight.data)
        _wf32(f, model.ln_f.bias.data)

    size_mb = Path(path).stat().st_size / 1e6
    print(f"Exported: {path} ({size_mb:.1f} MB)")

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(data_file: str, out_file: str, cfg: Config) -> None:
    pairs = load_pairs(data_file)
    if not pairs:
        raise SystemExit(f"No training pairs found in {data_file}")
    print(f"Pairs: {len(pairs)}", flush=True)

    aug_pairs = augment_pairs(pairs, cfg.aug_factor)
    dataset   = QADataset(aug_pairs, cfg.ctx_len)
    loader    = DataLoader(dataset, batch_size=cfg.batch_size,
                           shuffle=True, drop_last=False)
    print(f"Examples (augmented): {len(dataset)}", flush=True)

    model = Transformer(cfg).to(cfg.device)
    print(f"Parameters: {model.n_params() / 1e6:.1f}M", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=0.1)

    def lr_lambda(epoch: int) -> float:
        if epoch < cfg.warmup:
            return (epoch + 1) / cfg.warmup
        t = (epoch - cfg.warmup) / max(1, cfg.epochs - cfg.warmup)
        return 0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * t))

    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    for epoch in range(cfg.epochs):
        model.train()
        total_loss = 0.0
        total_tok  = 0

        for x, y, m in loader:
            x, y, m = x.to(cfg.device), y.to(cfg.device), m.to(cfg.device)

            logits = model(x)                              # B, L, VOCAB
            loss   = F.cross_entropy(
                logits.view(-1, VOCAB),
                y.view(-1),
                reduction="none",
                ignore_index=PAD,
            )
            # Only backprop on response tokens
            masked = (loss * m.view(-1).float()).sum()
            n_tok  = m.float().sum().clamp(min=1)

            (masked / n_tok).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad()

            total_loss += masked.item()
            total_tok  += n_tok.item()

        scheduler.step()

        if (epoch + 1) % 10 == 0 or epoch == 0:
            avg_loss = total_loss / max(total_tok, 1)
            lr_now   = opt.param_groups[0]["lr"]
            sample_q, sample_a = random.choice(pairs)
            got = generate(model, sample_q)
            match = "OK" if got == sample_a else "  "
            print(
                f"epoch {epoch+1:4d}  loss {avg_loss:.4f}  "
                f"lr {lr_now:.2e}  "
                f"'{sample_q}' -> {got!r:16s} [{sample_a}] {match}",
                flush=True,
            )

    # Final sample pass
    print("\nSamples:", flush=True)
    seen: set = set()
    for q, a in pairs:
        if a not in seen and len(seen) < 20:
            seen.add(a)
            got = generate(model, q)
            ok  = "OK" if got == a else "  "
            print(f"  {ok}  '{q}' -> {got!r:20s}  [{a}]", flush=True)

    export(model, out_file)

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train Pi Jam transformer.")
    parser.add_argument("data",    help="Training data file (td_*.txt)")
    parser.add_argument("output",  help="Output weights file (.bin)")
    parser.add_argument("--layers",  type=int,   default=8)
    parser.add_argument("--dim",     type=int,   default=512)
    parser.add_argument("--heads",   type=int,   default=8)
    parser.add_argument("--ff",      type=int,   default=2048)
    parser.add_argument("--ctx",     type=int,   default=64)
    parser.add_argument("--epochs",  type=int,   default=150)
    parser.add_argument("--batch",   type=int,   default=64)
    parser.add_argument("--lr",      type=float, default=3e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--aug",     type=int,   default=20,
                        help="Augmentation multiplier (default 20x)")
    args = parser.parse_args()

    cfg = Config(
        n_layers   = args.layers,
        d_model    = args.dim,
        n_heads    = args.heads,
        d_ff       = args.ff,
        ctx_len    = args.ctx,
        epochs     = args.epochs,
        batch_size = args.batch,
        lr         = args.lr,
        dropout    = args.dropout,
        aug_factor = args.aug,
    )

    train(args.data, args.output, cfg)


if __name__ == "__main__":
    main()

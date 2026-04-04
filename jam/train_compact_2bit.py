#!/usr/bin/env python3
"""
JAMBO compact trainer: 192 -> 256 -> 45
W1 = INT2 (-2,-1,0,+1), W2 = INT4, 3-phase output bias.

Input layout matches jambo.asm exactly:
  0..127   trigram hash
  128..159 bag-of-chars
  160..167 suffix last_char & 7
  168..175 suffix prev_char & 7
  176..183 suffix (last - prev) & 7
  184..191 gencnt & 7
"""
import os
import random
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

CS = " ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-,!?'/"
EOL = len(CS)
NO = EOL + 1
SEP = ">"
NT = 128
NB = 64
NI = NT + NB
NH = 256
DV = "cuda" if torch.cuda.is_available() else "cpu"
CI = {c: i for i, c in enumerate(CS)}


def compact_hash(ctx, gencnt):
    h = [0] * NI
    t = ctx.lower()

    for i in range(len(t) - 2):
        c0 = ord(t[i]) & 0x7F
        c1 = ord(t[i + 1]) & 0x7F
        c2 = ord(t[i + 2]) & 0x7F
        idx = (c0 * 31 + c1 * 7 + c2) & 0x7F
        h[idx] = min(h[idx] + 1, 255)

    for c in t:
        idx = NT + (ord(c) & 0x1F)
        h[idx] = min(h[idx] + 1, 255)

    if gencnt >= 1:
        ans = t.split(SEP, 1)[1] if SEP in t else t
        h[NT + 32 + (ord(ans[-1]) & 7)] += 1
        h[NT + 56 + (gencnt & 7)] += 1
        if gencnt >= 2:
            h[NT + 40 + (ord(ans[-2]) & 7)] += 1
            h[NT + 48 + ((ord(ans[-1]) - ord(ans[-2])) & 7)] += 1

    return h


def load_pairs(path):
    ps = []
    with open(path, encoding="utf-8") as f:
        for l in f:
            l = l.strip()
            if not l or l.startswith("#") or l.startswith("@") or "|" not in l:
                continue
            q, a = l.split("|", 1)
            ps.append((q.strip().lower(), a.strip().upper()))
    return ps


def make_examples(ps, aug=0):
    ex = []
    for rep in range(1 + aug):
        for q0, a in ps:
            q = q0
            if rep > 0 and len(q) > 3 and random.random() < 0.4:
                i = random.randint(0, len(q) - 1)
                r = random.random()
                if r < 0.33 and len(q) > 3:
                    q = q[:i] + q[i + 1 :]
                elif r < 0.66:
                    q = q[:i] + q[i] + q[i:]

            ctx = q + SEP
            pos = 0
            for c in a:
                if c not in CI:
                    continue
                ex.append((compact_hash(ctx, pos), CI[c], 0 if pos < 3 else (1 if pos < 8 else 2)))
                ctx += c.lower()
                pos += 1
            ex.append((compact_hash(ctx, pos), EOL, 0 if pos < 3 else (1 if pos < 8 else 2)))
    return ex


class STE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.round()

    @staticmethod
    def backward(ctx, grad):
        return grad


def sq4(w):
    return STE.apply(w).clamp(-8, 7)


def sq2(w):
    return w + (w.round().clamp(-2, 1) - w).detach()


def sq8(b):
    return STE.apply(b).clamp(-128, 127)


def pack_w1_int2(wm):
    code = {0: 0, -1: 1, 1: 2, -2: 3}
    blob = bytearray()
    rows, cols = wm.shape
    for r in range(rows):
        for c in range(0, cols, 4):
            b0 = code[int(wm[r, c])]
            b1 = code[int(wm[r, c + 1])] << 2
            b2 = code[int(wm[r, c + 2])] << 4
            b3 = code[int(wm[r, c + 3])] << 6
            blob.append(b0 | b1 | b2 | b3)
    return blob


def pack_w_int4(wm):
    blob = bytearray()
    rows, cols = wm.shape
    for r in range(rows):
        for c in range(0, cols, 2):
            lo = int(wm[r, c]) & 0xF
            hi = (int(wm[r, c + 1]) & 0xF) if c + 1 < cols else 0
            blob.append(lo | (hi << 4))
    return blob


def train(data_file, out_file="weights_b2s.bin"):
    ps = load_pairs(data_file)
    from collections import Counter

    rc = Counter(a for _, a in ps)
    print(f"Data: {len(ps)} pairs, {len(rc)} answers", flush=True)

    best = 0.0
    bw = None
    bs = -1

    for seed in range(8):
        random.seed(seed)
        torch.manual_seed(seed)

        ex = make_examples(ps, aug=10)
        X = torch.tensor([e[0] for e in ex], dtype=torch.float32, device=DV)
        Y = torch.tensor([e[1] for e in ex], dtype=torch.long, device=DV)
        S = torch.tensor([e[2] for e in ex], dtype=torch.long, device=DV)

        if seed == 0:
            print(f"Examples: {len(ex)}", flush=True)

        w1 = nn.Parameter(torch.randn(NH, NI, device=DV) * 0.3)
        b1 = nn.Parameter(torch.zeros(NH, device=DV))
        w2 = nn.Parameter(torch.randn(NO, NH, device=DV) * 0.3)
        b2 = nn.Parameter(torch.zeros(NO, device=DV))
        b2s = nn.Parameter(torch.zeros(NO, device=DV))
        b2m = nn.Parameter(torch.zeros(NO, device=DV))

        def fwd(x, phases, ste):
            W1 = sq2(w1) if ste else w1
            B1 = sq8(b1) if ste else b1
            W2 = sq4(w2) if ste else w2
            B2 = sq8(b2) if ste else b2
            B2S = sq8(b2s) if ste else b2s
            B2M = sq8(b2m) if ste else b2m
            h = F.relu(F.linear(x, W1, B1)).clamp(0, 255)
            lo = F.linear(h, W2)
            ph0 = phases == 0
            ph1 = phases == 1
            bi = torch.where(
                ph0.unsqueeze(1),
                B2S.unsqueeze(0),
                torch.where(ph1.unsqueeze(1), B2M.unsqueeze(0), B2.unsqueeze(0)),
            )
            return lo + bi

        opt = torch.optim.AdamW([w1, b1, w2, b2, b2s, b2m], lr=0.005, weight_decay=0.01)
        for _ in range(1200):
            loss = F.cross_entropy(fwd(X, S, False), Y)
            opt.zero_grad()
            loss.backward()
            opt.step()

        opt = torch.optim.AdamW([w1, b1, w2, b2, b2s, b2m], lr=0.005, weight_decay=0.001)
        sc = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=2500, T_mult=2)
        lb = 0.0
        ls = None
        for ep in range(12000):
            loss = F.cross_entropy(fwd(X, S, True), Y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sc.step()
            if (ep + 1) % 2500 == 0:
                acc = (fwd(X, S, True).argmax(1) == Y).float().mean().item()
                if acc > lb:
                    lb = acc
                    ls = (
                        w1.data.clone(),
                        b1.data.clone(),
                        w2.data.clone(),
                        b2.data.clone(),
                        b2s.data.clone(),
                        b2m.data.clone(),
                    )

        if ls:
            w1.data, b1.data, w2.data, b2.data, b2s.data, b2m.data = ls

        W1 = w1.data.round().clamp(-2, 1).cpu().numpy().astype(int)
        B1 = b1.data.round().clamp(-128, 127).cpu().numpy().astype(int)
        W2 = w2.data.round().clamp(-8, 7).cpu().numpy().astype(int)
        B2 = b2.data.round().clamp(-128, 127).cpu().numpy().astype(int)
        B2S = b2s.data.round().clamp(-128, 127).cpu().numpy().astype(int)
        B2M = b2m.data.round().clamp(-128, 127).cpu().numpy().astype(int)

        def sim1(h, s):
            hid = [max(0, min(255, sum(int(W1[r, c]) * h[c] for c in range(NI)) + int(B1[r]))) for r in range(NH)]
            bi = B2S if s == 0 else (B2M if s == 1 else B2)
            return [sum(int(W2[r, c]) * hid[c] for c in range(NH)) + int(bi[r]) for r in range(NO)]

        tex = make_examples(ps, aug=0)
        ok = sum(1 for h, t, s in tex if sim1(h, s).index(max(sim1(h, s))) == t)
        sa = ok / len(tex)
        print(f"  seed={seed}: STE={lb:.3f} sim={sa:.3f}", flush=True)

        if sa > best:
            best = sa
            bs = seed
            bw = (W1, B1, W2, B2, B2S, B2M, sim1)
        if best >= 0.998:
            print("  Early", flush=True)
            break

    print(f"\nBest: seed={bs} sim={best:.3f}", flush=True)
    W1, B1, W2, B2, B2S, B2M, sim1 = bw

    blob = bytearray()
    blob += pack_w1_int2(W1)
    blob += pack_w_int4(W2)
    for bv in [B1, B2, B2S, B2M]:
        blob += bytes([int(v) & 0xFF for v in bv])

    with open(out_file, "wb") as f:
        f.write(blob)
    print(f"Weights: {len(blob)} bytes -> {out_file}", flush=True)

    def gen(q):
        ctx = q + SEP
        r = ""
        pos = 0
        for _ in range(15):
            lo = sim1(compact_hash(ctx, pos), 0 if pos < 3 else (1 if pos < 8 else 2))
            i = lo.index(max(lo))
            if i == EOL:
                break
            r += CS[i]
            ctx += CS[i].lower()
            pos += 1
        return r

    print("\nSample:", flush=True)
    seen = set()
    for q, a in ps:
        if a not in seen and len(seen) < 20:
            seen.add(a)
            print(f"  '{q}' -> {gen(q)}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 train_compact_2bit.py <training_data.txt>")
        print("  Output: weights_b2s.bin")
        sys.exit(1)
    data = sys.argv[1]
    out_dir = os.path.dirname(os.path.abspath(data))
    out = os.path.join(out_dir, "weights_b2s.bin")
    train(data, out)

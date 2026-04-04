#!/usr/bin/env python3
"""
PICO JAM trainer: 96 -> 128 -> 45
W1: 1-BIT (8 per byte!), W2: INT2 (4 per byte), single bias + algorithmic EOL.

Weight layout:
  W1: 64 * 12 = 768 B  (1-bit, 8/byte)
  W2: 45 * 16 = 720 B  (INT2, 4/byte)
  B1: 64 B
  B2: 45 B
  Total: 1597 B (~1.6 KB brain)
"""
import random, sys
import torch, torch.nn as nn, torch.nn.functional as F

CS = " abcdefghijklmnopqrstuvwxyz0123456789.-,!?'/"
EOL = len(CS); NO = EOL + 1; SEP = '>'
NT = 64; NB = 32; NI = NT + NB; NH = 64
DV = 'cuda' if torch.cuda.is_available() else 'cpu'
CI = {c: i for i, c in enumerate(CS)}


def pico_hash(ctx, gencnt):
    h = [0] * NI
    t = ctx.lower()
    for i in range(len(t) - 2):
        c0, c1, c2 = ord(t[i]) & 0x7F, ord(t[i+1]) & 0x7F, ord(t[i+2]) & 0x7F
        idx = (c0 * 31 + c1 * 7 + c2) & 0x3F
        h[idx] = min(h[idx] + 1, 255)
    for c in t:
        h[NT + (ord(c) & 0x0F)] = min(h[NT + (ord(c) & 0x0F)] + 1, 255)
    if gencnt >= 1:
        ans = t.split(SEP, 1)[1] if SEP in t else t
        h[80 + (ord(ans[-1]) & 7)] += 1
        h[88 + (gencnt & 7)] += 1
    return h


def load_pairs(path):
    ps = []
    with open(path, encoding='utf-8') as f:
        for l in f:
            l = l.strip()
            if not l or l.startswith('#') or l.startswith('@') or '|' not in l:
                continue
            q, a = l.split('|', 1)
            ps.append((q.strip().lower(), a.strip().lower()))
    return ps


def make_examples(ps, aug=0):
    ex = []
    for rep in range(1 + aug):
        for q0, a in ps:
            q = q0
            if rep > 0 and len(q) > 3 and random.random() < 0.4:
                i = random.randint(0, len(q) - 1)
                r = random.random()
                if r < 0.33 and len(q) > 3: q = q[:i] + q[i+1:]
                elif r < 0.66: q = q[:i] + q[i] + q[i:]
            ctx = q + SEP; pos = 0
            for c in a:
                if c not in CI: continue
                ex.append((pico_hash(ctx, pos), CI[c], pos))
                ctx += c.lower(); pos += 1
            ex.append((pico_hash(ctx, pos), EOL, pos))
    return ex


class STE_sign(torch.autograd.Function):
    """STE: forward snaps to {-1, +1}, backward passes gradient through."""
    @staticmethod
    def forward(ctx, x):
        return torch.where(x >= 0, torch.ones_like(x), -torch.ones_like(x))
    @staticmethod
    def backward(ctx, grad):
        return grad

def sq1(w):
    """Quantize to 1-bit {-1, +1} with STE."""
    return STE_sign.apply(w)

def sq2(w):
    """Quantize to INT2 {-2,-1,0,+1} with STE."""
    return w + (w.round().clamp(-2, 1) - w).detach()

def sq8(b):
    return (b + (b.round().clamp(-128, 127) - b).detach())


def pack_1bit(wm):
    """Pack {-1,+1} matrix: +1 -> bit 1, -1 -> bit 0. LSB first."""
    blob = bytearray()
    rows, cols = wm.shape
    for r in range(rows):
        for c in range(0, cols, 8):
            byte = 0
            for k in range(8):
                if c + k < cols and int(wm[r, c + k]) > 0:
                    byte |= (1 << k)
            blob.append(byte)
    return blob


def train(data_file, out_file='weights_pico.bin'):
    ps = load_pairs(data_file)
    from collections import Counter
    rc = Counter(a for _, a in ps)
    print(f"Data: {len(ps)} pairs, {len(rc)} unique answers", flush=True)

    best = 0.0; bw = None; bs = -1

    for seed in range(12):
        random.seed(seed); torch.manual_seed(seed)
        ex = make_examples(ps, aug=15)
        X = torch.tensor([e[0] for e in ex], dtype=torch.float32, device=DV)
        Y = torch.tensor([e[1] for e in ex], dtype=torch.long, device=DV)
        P = torch.tensor([e[2] for e in ex], dtype=torch.long, device=DV)
        if seed == 0:
            print(f"Examples: {len(ex)}", flush=True)

        w1 = nn.Parameter(torch.randn(NH, NI, device=DV) * 0.5)
        b1 = nn.Parameter(torch.zeros(NH, device=DV))
        w2 = nn.Parameter(torch.randn(NO, NH, device=DV) * 0.5)
        b2 = nn.Parameter(torch.zeros(NO, device=DV))

        def fwd(x, pos, ste):
            W1 = sq1(w1) if ste else w1
            B1 = sq8(b1) if ste else b1
            W2 = sq2(w2) if ste else w2
            B2 = sq8(b2) if ste else b2
            h = F.relu(F.linear(x, W1, B1)).clamp(0, 255)
            lo = F.linear(h, W2, B2)
            # Algorithmic EOL suppression: subtract penalty for early positions
            # pos < 3: strong suppress, pos 3-7: mild suppress
            eol_penalty = torch.zeros_like(lo)
            eol_penalty[:, EOL] = torch.where(pos < 3, torch.tensor(-40.0, device=DV),
                                   torch.where(pos < 8, torch.tensor(-15.0, device=DV),
                                   torch.tensor(0.0, device=DV)))
            return lo + eol_penalty

        # Phase 1: float warmup
        opt = torch.optim.AdamW([w1, b1, w2, b2], lr=0.008, weight_decay=0.01)
        for _ in range(2000):
            loss = F.cross_entropy(fwd(X, P, False), Y)
            opt.zero_grad(); loss.backward(); opt.step()

        # Phase 2: QAT with 1-bit
        opt = torch.optim.AdamW([w1, b1, w2, b2], lr=0.005, weight_decay=0.001)
        sc = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=3000, T_mult=2)
        lb = 0.0; ls = None
        for ep in range(15000):
            loss = F.cross_entropy(fwd(X, P, True), Y)
            opt.zero_grad(); loss.backward(); opt.step(); sc.step()
            if (ep+1) % 3000 == 0:
                acc = (fwd(X, P, True).argmax(1) == Y).float().mean().item()
                if acc > lb:
                    lb = acc
                    ls = (w1.data.clone(), b1.data.clone(),
                          w2.data.clone(), b2.data.clone())

        if ls: w1.data, b1.data, w2.data, b2.data = ls

        W1 = torch.where(w1.data >= 0, 1, -1).cpu().numpy().astype(int)
        B1 = b1.data.round().clamp(-128, 127).cpu().numpy().astype(int)
        W2 = w2.data.round().clamp(-2, 1).cpu().numpy().astype(int)
        B2 = b2.data.round().clamp(-128, 127).cpu().numpy().astype(int)

        # Integer simulation with algorithmic EOL suppression
        def sim1(h, pos):
            hid = [max(0, min(255, sum(int(W1[r,c])*h[c] for c in range(NI)) + int(B1[r])))
                   for r in range(NH)]
            lo = [sum(int(W2[r,c])*hid[c] for c in range(NH)) + int(B2[r]) for r in range(NO)]
            # EOL suppression
            if pos < 3: lo[EOL] -= 40
            elif pos < 8: lo[EOL] -= 15
            return lo

        tex = make_examples(ps, aug=0)
        ok = sum(1 for h, t, p in tex if sim1(h, p).index(max(sim1(h, p))) == t)
        sa = ok / len(tex)
        print(f"  seed={seed}: STE={lb:.3f} sim={sa:.3f}", flush=True)

        if sa > best:
            best = sa; bs = seed
            bw = (W1, B1, W2, B2, sim1)
        if best >= 0.998:
            print("  Early!", flush=True); break

    if bw is None:
        print("FATAL"); sys.exit(1)

    W1, B1, W2, B2, sim1 = bw
    print(f"\nBest: seed={bs} sim={best:.3f}", flush=True)

    def pack_int2(wm):
        code = {0: 0, -1: 1, 1: 2, -2: 3}
        blob = bytearray()
        rows, cols = wm.shape
        for r in range(rows):
            for c in range(0, cols, 4):
                b = 0
                for k in range(4):
                    if c+k < cols: b |= code[int(wm[r, c+k])] << (k*2)
                blob.append(b)
        return blob

    # Pack: W1 (1-bit) + W2 (INT2) + B1 + B2
    blob = bytearray()
    blob += pack_1bit(W1)
    blob += pack_int2(W2)
    for bv in [B1, B2]:
        blob += bytes([int(v) & 0xFF for v in bv])

    with open(out_file, 'wb') as f:
        f.write(blob)
    print(f"Weights: {len(blob)} bytes -> {out_file}", flush=True)

    # Quick test
    def gen(q):
        ctx = q + SEP; r = ''; pos = 0
        for _ in range(15):
            lo = sim1(pico_hash(ctx, pos), pos)
            i = lo.index(max(lo))
            if i == EOL: break
            r += CS[i]; ctx += CS[i]; pos += 1
        return r

    print("\nSample:", flush=True)
    seen = set()
    for q, a in ps:
        if a not in seen and len(seen) < 25:
            seen.add(a)
            g = gen(q)
            ok = 'OK' if g == a else 'MISS'
            print(f"  '{q}' -> {g:15s} [{a:15s}] {ok}", flush=True)


if __name__ == '__main__':
    data = sys.argv[1] if len(sys.argv) > 1 else 'td_pico.txt'
    train(data)

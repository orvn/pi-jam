#!/usr/bin/env python3
"""
Internal trainer implementation for the canonical JAMXE lane.

Architecture:

- 512 -> 512 -> 45
- W1 = INT2
- W2 = INT4

Same 77,703-byte external blob size as the archived stable baseline, but with
the richer 512-input XE feature layout.
"""
import torch, torch.nn as nn, torch.nn.functional as F, random, sys, os

CS = " ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-,!?'/"
EOL = len(CS); NO = EOL + 1; SEP = '>'
NT = 192; NB = 64; NI = 512; NH = 512
DV = 'cuda' if torch.cuda.is_available() else 'cpu'
CI = {c: i for i, c in enumerate(CS)}

def dual_hash(t):
    h = [0] * NI; t2 = t.lower()
    if ">" in t2:
        query = t2[:t2.index(">")]
        answer = t2[t2.index(">")+1:]
    else:
        query = t2; answer = ""
    # Live trigrams [0-191]
    for i in range(len(t2)-2):
        c0,c1,c2 = ord(t2[i])&0x7F, ord(t2[i+1])&0x7F, ord(t2[i+2])&0x7F
        idx = (c0*31+c1*7+c2) % 192
        h[idx] = min(h[idx]+1, 255)
    # Live bag [192-223]
    for c in t2: h[192+(ord(c)&0x1F)] = min(h[192+(ord(c)&0x1F)]+1, 255)
    # Answer suffix one-hot [224-255]
    gc = len(answer)
    if gc >= 1:
        h[224+(ord(answer[-1])&7)] += 1
        h[248+(gc&7)] += 1
    if gc >= 2:
        h[232+(ord(answer[-2])&7)] += 1
        h[240+((ord(answer[-1])-ord(answer[-2]))&7)] += 1
    # Query trigrams FROZEN [256-383] (mod 128)
    for i in range(len(query)-2):
        c0,c1,c2 = ord(query[i])&0x7F, ord(query[i+1])&0x7F, ord(query[i+2])&0x7F
        idx = 256 + (c0*31+c1*7+c2) % 128
        h[idx] = min(h[idx]+1, 255)
    # Query bag FROZEN [384-415]
    for c in query: h[384+(ord(c)&0x1F)] = min(h[384+(ord(c)&0x1F)]+1, 255)
    # Answer bag LIVE [416-447]
    for c in answer: h[416+(ord(c)&0x1F)] = min(h[416+(ord(c)&0x1F)]+1, 255)
    # Query suffix [448-479]
    if len(query)>=1: h[448+(ord(query[-1])&7)] += 1
    if len(query)>=2:
        h[456+(ord(query[-2])&7)] += 1
        h[464+((ord(query[-1])-ord(query[-2]))&7)] += 1
    words = query.split()
    if words: h[472+(len(words[-1])&7)] += 1
    # Query act [480-511]
    if words:
        hm = {'who':0,'what':1,'why':2,'how':3,'where':4,'when':5}
        hw = words[0]
        if hw in hm: h[480+hm[hw]] += 1
        elif hw in ('is','can','do','does','are','will','would','could'): h[486] += 1
        else: h[487] += 1
    h[488+(len(query)&7)] += 1
    h[496+(min(len(words),7) if words else 0)] += 1
    if "?" in query: h[504] += 1
    if any(c.isdigit() for c in query): h[505] += 1
    return h

def load_pairs(path):
    ps = []
    with open(path) as f:
        for l in f:
            l = l.strip()
            if not l or l.startswith('#') or l.startswith('@') or '|' not in l:
                continue
            q, a = l.split('|', 1)
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
                if r < 0.33 and len(q) > 3: q = q[:i] + q[i+1:]
                elif r < 0.66: q = q[:i] + q[i] + q[i:]
            ctx = q + SEP; pos = 0
            for c in a:
                if c not in CI: continue
                ex.append((dual_hash(ctx), CI[c], 0 if pos < 3 else (1 if pos < 8 else 2)))
                ctx += c.lower(); pos += 1
            ex.append((dual_hash(ctx), EOL, 0 if pos < 3 else (1 if pos < 8 else 2)))
    return ex

class STE(torch.autograd.Function):
    @staticmethod
    def forward(c, x): return x.round()
    @staticmethod
    def backward(c, g): return g

def sq4(w): return STE.apply(w).clamp(-8, 7)
def sq2(w): return (w + (w.round().clamp(-2, 1) - w).detach())
def sq8(b): return STE.apply(b).clamp(-128, 127)

def train(data_file, out_file='weights_xe_strong.bin'):
    ps = load_pairs(data_file)
    from collections import Counter
    rc = Counter(a for _, a in ps)
    print(f"Data: {len(ps)} pairs, {len(rc)} answers", flush=True)

    best = 0; bw = None; bs = -1
    for seed in range(3):
        random.seed(seed); torch.manual_seed(seed)
        ex = make_examples(ps, aug=10)
        X = torch.tensor([e[0] for e in ex], dtype=torch.float32, device=DV)
        Y = torch.tensor([e[1] for e in ex], dtype=torch.long, device=DV)
        S = torch.tensor([e[2] for e in ex], dtype=torch.bool, device=DV)
        if seed == 0:
            print(f"Examples: {len(ex)}, starts: {S.sum().item()}", flush=True)

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
            ph0 = (phases == 0); ph1 = (phases == 1)
            bi = torch.where(ph0.unsqueeze(1), B2S.unsqueeze(0),
                 torch.where(ph1.unsqueeze(1), B2M.unsqueeze(0), B2.unsqueeze(0)))
            return lo + bi

        # Phase 1: float warmup
        opt = torch.optim.AdamW([w1, b1, w2, b2, b2s, b2m], lr=0.005, weight_decay=0.01)
        for ep in range(1500):
            loss = F.cross_entropy(fwd(X, S, False), Y)
            opt.zero_grad(); loss.backward(); opt.step()

        # Phase 2: STE quantization-aware
        opt = torch.optim.AdamW([w1, b1, w2, b2, b2s, b2m], lr=0.005, weight_decay=0.001)
        sc = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=3000, T_mult=2)
        lb = 0; ls = None
        for ep in range(15000):
            loss = F.cross_entropy(fwd(X, S, True), Y)
            opt.zero_grad(); loss.backward(); opt.step(); sc.step()
            if (ep + 1) % 3000 == 0:
                acc = (fwd(X, S, True).argmax(1) == Y).float().mean().item()
                if acc > lb:
                    lb = acc
                    ls = (w1.data.clone(), b1.data.clone(), w2.data.clone(),
                          b2.data.clone(), b2s.data.clone(), b2m.data.clone())

        if ls: w1.data, b1.data, w2.data, b2.data, b2s.data, b2m.data = ls

        # Quantize
        W1 = w1.data.round().clamp(-8, 7).cpu().numpy().astype(int)
        B1 = b1.data.round().clamp(-128, 127).cpu().numpy().astype(int)
        W2 = w2.data.round().clamp(-8, 7).cpu().numpy().astype(int)
        B2 = b2.data.round().clamp(-128, 127).cpu().numpy().astype(int)
        B2S = b2s.data.round().clamp(-128, 127).cpu().numpy().astype(int)
        B2M = b2m.data.round().clamp(-128, 127).cpu().numpy().astype(int)

        # Evaluate: full generation
        def sim1(h, s):
            hid = [max(0, min(255, sum(int(W1[r,c]) * h[c] for c in range(NI)) + int(B1[r])))
                   for r in range(NH)]
            bi = B2S if s==0 else (B2M if s==1 else B2)
            return [sum(int(W2[r,c]) * hid[c] for c in range(NH)) + int(bi[r]) for r in range(NO)]

        tex = make_examples(ps, aug=0)
        ok = sum(1 for h, t, s in tex if sim1(h, s).index(max(sim1(h, s))) == t)
        sa = ok / len(tex)
        print(f"  seed={seed}: STE={lb:.3f} sim={sa:.3f}", flush=True)

        if sa > best:
            best = sa; bs = seed
            bw = (W1, B1, W2, B2, B2S, B2M, sim1)
        if best >= 0.998:
            print("  Early", flush=True); break

    print(f"\nBest: seed={bs} sim={best:.3f}", flush=True)
    W1, B1, W2, B2, B2S, B2M, sim1 = bw

    # Pack W1 as 2-bit (4 weights/byte), W2 as INT4 (2 weights/byte)
    blob = bytearray()
    # W1: 2-bit packing {0:00, -1:01, +1:10, -2:11}
    code_map = {0:0, -1:1, 1:2, -2:3}
    r1, c1 = W1.shape
    clamped = 0
    for ri in range(r1):
        for ci in range(0, c1, 4):
            byte = 0
            for k in range(4):
                if ci+k < c1:
                    v = max(-2, min(1, int(W1[ri, ci+k])))
                    if int(W1[ri, ci+k]) != v: clamped += 1
                    byte |= code_map[v] << (k*2)
            blob.append(byte)
    w1_size = len(blob)
    # W2: INT4 nibble packing (unchanged)
    r2, c2 = W2.shape
    for ri in range(r2):
        for ci in range(0, c2, 2):
            lo = int(W2[ri, ci]) & 0xF
            hi = (int(W2[ri, ci+1]) & 0xF) if ci+1 < c2 else 0
            blob.append(lo | (hi << 4))
    print(f"  W1: {w1_size} B (2-bit, {clamped} clamped)")
    for bv in [B1, B2, B2S, B2M]:
        blob += bytes([int(v) & 0xFF for v in bv])

    with open(out_file, 'wb') as f:
        f.write(blob)
    print(f"Weights: {len(blob)} bytes -> {out_file}", flush=True)

    # Generate sample answers from the actual training data
    def gen(q):
        ctx = q + SEP; r = ''; pos = 0
        for _ in range(15):
            lo = sim1(dual_hash(ctx), pos < 3)
            i = lo.index(max(lo))
            if i == EOL: break
            r += CS[i]; ctx += CS[i].lower(); pos += 1
        return r

    print("\nSample:", flush=True)
    seen = set()
    for q, a in ps:
        if a not in seen and len(seen) < 20:
            seen.add(a)
            print(f"  '{q}' -> {gen(q)}", flush=True)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python train_xe_strong.py <training_data.txt>")
        print("  Output: weights_xe_strong.bin")
        sys.exit(1)
    data = sys.argv[1]
    out_dir = os.path.dirname(os.path.abspath(data))
    out = os.path.join(out_dir, 'weights_xe_strong.bin')
    train(data, out)

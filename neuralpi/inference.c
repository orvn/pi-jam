/*
 * Pi Jam — decoder-only transformer inference.
 *
 * Reads a PJAM binary blob produced by train_transformer.py:
 *   header   32 bytes   magic "PJAM" + 7 × uint32 config
 *   tok_emb  int8 weights + float32 scale   (vocab × d_model)
 *   pos_emb  float32                         (ctx_len × d_model)
 *   per layer:
 *     ln1 weight + bias  (d_model × f32 each)
 *     wq wk wv wo        (d_model × d_model int8 + f32 scale + d_model f32 bias)
 *     ln2 weight + bias  (d_model × f32 each)
 *     w1 (FFN up)        (d_ff × d_model int8 + f32 scale + d_ff f32 bias)
 *     w2 (FFN down)      (d_model × d_ff int8 + f32 scale + d_model f32 bias)
 *   ln_f weight + bias   (d_model × f32 each)
 *   (output head weight is tied to tok_emb — not stored separately)
 *
 * A KV cache holds K and V for every processed position so each generation
 * step is O(pos × d_model) rather than O(pos² × d_model).  On ARM1176
 * (no NEON) this keeps response time in the low-second range for typical
 * short answers.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <ctype.h>
#include "theme.h"

/* -------------------------------------------------------------------------
 * Vocabulary — must match train_transformer.py
 * ------------------------------------------------------------------------- */

#define EOS_TOK  44
#define BOS_TOK  45
#define SEP_TOK  46
#define PAD_TOK  47

/* 44 printable characters, indices 0–43 */
static const char CHARS[] = " ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-,!?'/";
#define N_CHARS  44

static int encode_char(char c)
{
    c = (char)toupper((unsigned char)c);
    for (int i = 0; i < N_CHARS; i++)
        if (CHARS[i] == c) return i;
    return PAD_TOK;
}

/* -------------------------------------------------------------------------
 * Config and model
 * ------------------------------------------------------------------------- */

typedef struct {
    int n_layers, n_heads, d_model, d_ff, ctx_len, vocab;
    int head_dim;   /* d_model / n_heads, computed on load */
} Config;

typedef struct {
    float   *ln1_w, *ln1_b;                  /* d_model each */
    int8_t  *wq_w;  float wq_s;  float *wq_b;  /* d_model × d_model */
    int8_t  *wk_w;  float wk_s;  float *wk_b;
    int8_t  *wv_w;  float wv_s;  float *wv_b;
    int8_t  *wo_w;  float wo_s;  float *wo_b;
    float   *ln2_w, *ln2_b;
    int8_t  *w1_w;  float w1_s;  float *w1_b;  /* d_ff × d_model */
    int8_t  *w2_w;  float w2_s;  float *w2_b;  /* d_model × d_ff */
} Layer;

typedef struct {
    Config  cfg;
    int8_t *tok_emb_w;  float tok_emb_s;   /* vocab × d_model */
    float  *pos_emb;                        /* ctx_len × d_model */
    Layer  *layers;
    float  *ln_f_w, *ln_f_b;
    float  *k_cache;   /* [n_layers][ctx_len][d_model] */
    float  *v_cache;
} Model;

/* -------------------------------------------------------------------------
 * Load helpers
 * ------------------------------------------------------------------------- */

static void die(const char *msg)
{
    fprintf(stderr, "error: %s\n", msg);
    exit(1);
}

static void xread_f32(FILE *f, float *dst, size_t n)
{
    if (fread(dst, sizeof(float), n, f) != n) die("unexpected end of weights file");
}

static void xread_i8(FILE *f, int8_t *dst, size_t n)
{
    if (fread(dst, 1, n, f) != n) die("unexpected end of weights file");
}

static float *xalloc_f32(size_t n)
{
    float *p = (float *)malloc(n * sizeof(float));
    if (!p) die("out of memory");
    return p;
}

static int8_t *xalloc_i8(size_t n)
{
    int8_t *p = (int8_t *)malloc(n);
    if (!p) die("out of memory");
    return p;
}

/* Read int8 blob then its float32 scale (PJAM _wq format) */
static float read_wq(FILE *f, int8_t *dst, size_t n)
{
    xread_i8(f, dst, n);
    float s;
    xread_f32(f, &s, 1);
    return s;
}

static Model *load_model(const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "error: cannot open weights: %s\n", path);
        exit(1);
    }

    char magic[4];
    if (fread(magic, 1, 4, f) != 4 || memcmp(magic, "PJAM", 4) != 0)
        die("not a PJAM weights file");

    uint32_t hdr[7];
    if (fread(hdr, sizeof(uint32_t), 7, f) != 7) die("truncated PJAM header");
    if (hdr[0] != 1) { fprintf(stderr, "error: unsupported PJAM version %u\n", hdr[0]); exit(1); }

    Model *m = (Model *)calloc(1, sizeof(Model));
    if (!m) die("out of memory");

    Config *c  = &m->cfg;
    c->n_layers = (int)hdr[1];
    c->n_heads  = (int)hdr[2];
    c->d_model  = (int)hdr[3];
    c->d_ff     = (int)hdr[4];
    c->ctx_len  = (int)hdr[5];
    c->vocab    = (int)hdr[6];
    c->head_dim = c->d_model / c->n_heads;

    int D = c->d_model, FF = c->d_ff, L = c->ctx_len, V = c->vocab, N = c->n_layers;

    m->tok_emb_w = xalloc_i8((size_t)V * D);
    m->tok_emb_s = read_wq(f, m->tok_emb_w, (size_t)V * D);

    m->pos_emb = xalloc_f32((size_t)L * D);
    xread_f32(f, m->pos_emb, (size_t)L * D);

    m->layers = (Layer *)calloc(N, sizeof(Layer));
    if (!m->layers) die("out of memory");

    for (int i = 0; i < N; i++) {
        Layer *ly = &m->layers[i];

        ly->ln1_w = xalloc_f32(D); xread_f32(f, ly->ln1_w, D);
        ly->ln1_b = xalloc_f32(D); xread_f32(f, ly->ln1_b, D);

        ly->wq_w = xalloc_i8((size_t)D * D);
        ly->wq_s = read_wq(f, ly->wq_w, (size_t)D * D);
        ly->wq_b = xalloc_f32(D); xread_f32(f, ly->wq_b, D);

        ly->wk_w = xalloc_i8((size_t)D * D);
        ly->wk_s = read_wq(f, ly->wk_w, (size_t)D * D);
        ly->wk_b = xalloc_f32(D); xread_f32(f, ly->wk_b, D);

        ly->wv_w = xalloc_i8((size_t)D * D);
        ly->wv_s = read_wq(f, ly->wv_w, (size_t)D * D);
        ly->wv_b = xalloc_f32(D); xread_f32(f, ly->wv_b, D);

        ly->wo_w = xalloc_i8((size_t)D * D);
        ly->wo_s = read_wq(f, ly->wo_w, (size_t)D * D);
        ly->wo_b = xalloc_f32(D); xread_f32(f, ly->wo_b, D);

        ly->ln2_w = xalloc_f32(D); xread_f32(f, ly->ln2_w, D);
        ly->ln2_b = xalloc_f32(D); xread_f32(f, ly->ln2_b, D);

        ly->w1_w = xalloc_i8((size_t)FF * D);
        ly->w1_s = read_wq(f, ly->w1_w, (size_t)FF * D);
        ly->w1_b = xalloc_f32(FF); xread_f32(f, ly->w1_b, FF);

        ly->w2_w = xalloc_i8((size_t)D * FF);
        ly->w2_s = read_wq(f, ly->w2_w, (size_t)D * FF);
        ly->w2_b = xalloc_f32(D); xread_f32(f, ly->w2_b, D);
    }

    m->ln_f_w = xalloc_f32(D); xread_f32(f, m->ln_f_w, D);
    m->ln_f_b = xalloc_f32(D); xread_f32(f, m->ln_f_b, D);

    fclose(f);

    m->k_cache = (float *)calloc((size_t)N * L * D, sizeof(float));
    m->v_cache = (float *)calloc((size_t)N * L * D, sizeof(float));
    if (!m->k_cache || !m->v_cache) die("out of memory for KV cache");

    return m;
}

/* -------------------------------------------------------------------------
 * Math primitives
 * ------------------------------------------------------------------------- */

/*
 * INT8 linear: y[out] = (W_int8[out × in] @ x[in]) * scale + b[out]
 * W is row-major: W[i * in + j] is the weight for output i, input j.
 * Scale is factored out of the inner loop, which keeps inner work to one
 * cast + one multiply-add per weight — the tightest loop on ARM1176.
 */
static void linear_i8(float *y, const int8_t *W, float scale,
                       const float *b, const float *x, int out, int in)
{
    for (int i = 0; i < out; i++) {
        float acc = b[i];
        const int8_t *row = W + (size_t)i * in;
        for (int j = 0; j < in; j++)
            acc += (float)row[j] * x[j];
        y[i] = acc * scale;
    }
}

/* LayerNorm: y = (x - mean) / sqrt(var + eps) * w + b */
static void layernorm(float *y, const float *x,
                      const float *w, const float *b, int d)
{
    float mean = 0.0f;
    for (int i = 0; i < d; i++) mean += x[i];
    mean /= (float)d;

    float var = 0.0f;
    for (int i = 0; i < d; i++) { float dx = x[i] - mean; var += dx * dx; }
    var /= (float)d;

    float inv = 1.0f / sqrtf(var + 1e-5f);
    for (int i = 0; i < d; i++)
        y[i] = w[i] * (x[i] - mean) * inv + b[i];
}

/* Softmax in-place — subtracts max first for numerical stability */
static void softmax(float *x, int n)
{
    float max = x[0];
    for (int i = 1; i < n; i++) if (x[i] > max) max = x[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) { x[i] = expf(x[i] - max); sum += x[i]; }
    for (int i = 0; i < n; i++) x[i] /= sum;
}

/* GELU — tanh approximation matching PyTorch's default */
static inline float gelu(float x)
{
    return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}

/* -------------------------------------------------------------------------
 * Forward pass — one token at position `pos`
 *
 * Processes token `tok`, writes K[pos] and V[pos] into the KV cache, and
 * attends over all positions 0..pos (causal).  On return, `x` holds the
 * post-residual hidden state and, if `logits` is non-NULL, `logits` holds
 * the unnormalised output scores over the vocabulary.
 *
 * buf_d  must be >= d_model floats.
 * buf_ff must be >= max(d_ff, 3*d_model + ctx_len) floats.
 * ------------------------------------------------------------------------- */
static void forward_token(Model *m, int tok, int pos,
                           float *x, float *buf_d, float *buf_ff,
                           float *logits)
{
    const Config *c  = &m->cfg;
    int D  = c->d_model;
    int H  = c->n_heads;
    int Dh = c->head_dim;
    int FF = c->d_ff;

    /* Embedding: dequantize token embedding + add positional embedding */
    {
        const int8_t *te = m->tok_emb_w + (size_t)tok * D;
        const float  *pe = m->pos_emb   + (size_t)pos * D;
        float s = m->tok_emb_s;
        for (int i = 0; i < D; i++)
            x[i] = (float)te[i] * s + pe[i];
    }

    for (int l = 0; l < c->n_layers; l++) {
        Layer *ly = &m->layers[l];
        float *kc = m->k_cache + (size_t)l * c->ctx_len * D;
        float *vc = m->v_cache + (size_t)l * c->ctx_len * D;

        /* ---- Attention sub-layer ---- */

        /* Pre-norm into buf_d; Q/K/V computed from buf_d */
        layernorm(buf_d, x, ly->ln1_w, ly->ln1_b, D);

        /*
         * Lay Q, K, V side-by-side in buf_ff[0..3D-1].
         * buf_ff[3D..3D+ctx_len-1] holds per-head attention scores.
         * This leaves buf_ff[3D+ctx_len..FF-1] untouched until the FFN.
         */
        float *q      = buf_ff;
        float *k      = buf_ff + D;
        float *v      = buf_ff + 2 * D;
        float *scores = buf_ff + 3 * D;   /* ctx_len floats */

        linear_i8(q, ly->wq_w, ly->wq_s, ly->wq_b, buf_d, D, D);
        linear_i8(k, ly->wk_w, ly->wk_s, ly->wk_b, buf_d, D, D);
        linear_i8(v, ly->wv_w, ly->wv_s, ly->wv_b, buf_d, D, D);

        /* Write K, V for this position into the cache */
        memcpy(kc + (size_t)pos * D, k, D * sizeof(float));
        memcpy(vc + (size_t)pos * D, v, D * sizeof(float));

        /*
         * Multi-head attention: for each head, compute scaled dot-product
         * scores against all cached K[0..pos], softmax, then weighted sum
         * of cached V[0..pos].  Output lands in buf_d (reused as attn_out).
         */
        float inv_scale = 1.0f / sqrtf((float)Dh);
        float *attn_out = buf_d;

        for (int h = 0; h < H; h++) {
            const float *qh  = q + h * Dh;
            float       *outh = attn_out + h * Dh;

            for (int t = 0; t <= pos; t++) {
                const float *kth = kc + (size_t)t * D + h * Dh;
                float dot = 0.0f;
                for (int d = 0; d < Dh; d++) dot += qh[d] * kth[d];
                scores[t] = dot * inv_scale;
            }
            softmax(scores, pos + 1);

            for (int d = 0; d < Dh; d++) outh[d] = 0.0f;
            for (int t = 0; t <= pos; t++) {
                const float *vth = vc + (size_t)t * D + h * Dh;
                float w = scores[t];
                for (int d = 0; d < Dh; d++) outh[d] += w * vth[d];
            }
        }

        /* Output projection + residual.  Reuse v's slot in buf_ff. */
        float *wo_out = v;
        linear_i8(wo_out, ly->wo_w, ly->wo_s, ly->wo_b, attn_out, D, D);
        for (int i = 0; i < D; i++) x[i] += wo_out[i];

        /* ---- FFN sub-layer ---- */

        layernorm(buf_d, x, ly->ln2_w, ly->ln2_b, D);

        /* Up-project D → FF with GELU */
        linear_i8(buf_ff, ly->w1_w, ly->w1_s, ly->w1_b, buf_d, FF, D);
        for (int i = 0; i < FF; i++) buf_ff[i] = gelu(buf_ff[i]);

        /* Down-project FF → D + residual */
        linear_i8(buf_d, ly->w2_w, ly->w2_s, ly->w2_b, buf_ff, D, FF);
        for (int i = 0; i < D; i++) x[i] += buf_d[i];
    }

    /* Final layernorm + output projection (weight-tied to tok_emb) */
    if (logits) {
        layernorm(buf_d, x, m->ln_f_w, m->ln_f_b, D);
        float s = m->tok_emb_s;
        for (int vi = 0; vi < c->vocab; vi++) {
            float acc = 0.0f;
            const int8_t *row = m->tok_emb_w + (size_t)vi * D;
            for (int j = 0; j < D; j++) acc += (float)row[j] * buf_d[j];
            logits[vi] = acc * s;
        }
    }
}

/* -------------------------------------------------------------------------
 * Generation
 * ------------------------------------------------------------------------- */

#ifndef WEIGHTS_FILE
#define WEIGHTS_FILE "weights_neuralpi.bin"
#endif

#define MAX_NEW 20

static void generate(Model *m, const char *prompt, char *out)
{
    const Config *c = &m->cfg;
    int D  = c->d_model;
    int FF = c->d_ff;

    /* Scratch — sized for the worst case of both attention and FFN paths */
    int buf_ff_n = FF > 3 * D + c->ctx_len ? FF : 3 * D + c->ctx_len;
    float *x      = xalloc_f32(D);
    float *buf_d  = xalloc_f32(D);
    float *buf_ff = xalloc_f32(buf_ff_n);
    float *logits = xalloc_f32(c->vocab);

    /* Clear KV cache for this query */
    memset(m->k_cache, 0, (size_t)c->n_layers * c->ctx_len * D * sizeof(float));
    memset(m->v_cache, 0, (size_t)c->n_layers * c->ctx_len * D * sizeof(float));

    /* Build prompt token sequence: [BOS, q0..qn, SEP] */
    int tokens[256];
    int n_tok = 0;
    tokens[n_tok++] = BOS_TOK;
    for (int i = 0; prompt[i] && n_tok < c->ctx_len - MAX_NEW - 1; i++)
        tokens[n_tok++] = encode_char(prompt[i]);
    tokens[n_tok++] = SEP_TOK;

    /*
     * Prefill: feed prompt tokens one at a time to build the KV cache.
     * Skip logit computation for all but the last token.
     */
    int pos = 0;
    for (int i = 0; i < n_tok - 1; i++)
        forward_token(m, tokens[i], pos++, x, buf_d, buf_ff, NULL);
    forward_token(m, tokens[n_tok - 1], pos++, x, buf_d, buf_ff, logits);

    /* Decode: greedy argmax until EOS or budget exhausted */
    int out_len = 0;
    out[0] = '\0';

    for (int step = 0; step < MAX_NEW; step++) {
        int best = 0;
        for (int vi = 1; vi < c->vocab; vi++)
            if (logits[vi] > logits[best]) best = vi;

        if (best == EOS_TOK || pos >= c->ctx_len) break;

        if (best < N_CHARS) {
            out[out_len++] = CHARS[best];
            out[out_len]   = '\0';
        }

        forward_token(m, best, pos++, x, buf_d, buf_ff, logits);
    }

    free(x);
    free(buf_d);
    free(buf_ff);
    free(logits);
}

/* -------------------------------------------------------------------------
 * Banner + REPL
 * ------------------------------------------------------------------------- */

static void print_banner(const Config *c)
{
    printf("================================\n");
    printf("            %s\n", UI_TITLE);
    printf("        Just a Model\n");
#ifdef UI_SIZE_INFO
    printf("        %s\n", UI_SIZE_INFO);
#endif
    printf("  %d layers · d=%d · %d heads · ff=%d\n",
           c->n_layers, c->d_model, c->n_heads, c->d_ff);
    printf("================================\n");
    for (int i = 0; UI_SUBS[i]; i++)
        printf("%s\n", UI_SUBS[i]);
    printf("\nAsk: ");
    for (int i = 0; i < 4 && UI_SUGS[i]; i++) {
        if (i > 0) printf(", ");
        printf("%s", UI_SUGS[i]);
    }
    printf("\n\n");
}

int main(int argc, char *argv[])
{
    const char *weights_path = (argc > 1) ? argv[1] : WEIGHTS_FILE;

    printf("Loading %s ...\n", weights_path);
    fflush(stdout);

    Model *m = load_model(weights_path);
    print_banner(&m->cfg);

    char input[256];
    char output[MAX_NEW + 1];

    while (1) {
        printf("> ");
        fflush(stdout);
        if (!fgets(input, sizeof(input), stdin)) break;

        int len = (int)strlen(input);
        while (len > 0 && (input[len-1] == '\n' || input[len-1] == '\r' ||
                            input[len-1] == ' '))
            input[--len] = '\0';
        if (len == 0) continue;

        generate(m, input, output);
        printf("%s\n\n", output);
    }

    printf("\n");
    return 0;
}

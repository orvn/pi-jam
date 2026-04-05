/*
 * Picojam — ultra-compact generative model (96 → 64 → 45)
 * W1 is 1-bit {-1,+1} packed 8/byte: the most aggressive quantization
 * that still preserves a sign. W2 is INT2 {-2,-1,0,+1} packed 4/byte.
 * No phase biases — EOL is suppressed algorithmically by penalising
 * the EOL logit at early positions, nudging the model to produce at
 * least a few characters before it's allowed to stop.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <ctype.h>
#include "theme.h"

/* Architecture */
#define N_IN     96
#define N_HID    64
#define N_OUT    45
#define EOL_IDX  44
#define MAX_GEN  15
#define MAX_CTX  256

/*
 * Binary weight blob layout.
 * W1 uses 1-bit packing: 8 weights per byte, LSB = weight 0.
 *   Bit set   → weight +1
 *   Bit clear → weight -1
 * W2 uses INT2 packing: 4 weights per byte, same code as compact jam's W1.
 *   Bits 00→0  01→-1  10→+1  11→-2
 */
#define W1_OFF    0       /* 64 * 12 = 768 bytes  */
#define W2_OFF    768     /* 45 * 16 = 720 bytes  */
#define B1_OFF    1488    /* 64 bytes, int8        */
#define B2_OFF    1552    /* 45 bytes, int8        */
#define BLOB_SIZE 1597

#ifndef WEIGHTS_FILE
#define WEIGHTS_FILE "weights_pico.bin"
#endif

/* " abcdefghijklmnopqrstuvwxyz0123456789.-,!?'/" — index 44 = EOL */
static const char CHARSET[N_OUT] =
    " abcdefghijklmnopqrstuvwxyz0123456789.-,!?'/";

static uint8_t blob[BLOB_SIZE];

/* W1: 1-bit, LSB-first within each byte. Bit set → +1, clear → -1. */
static inline int w1_get(int row, int col)
{
    uint8_t b = blob[W1_OFF + row * (N_IN / 8) + col / 8];
    return ((b >> (col & 7)) & 1) ? 1 : -1;
}

/* W2: INT2, codes 00→0  01→-1  10→+1  11→-2, 4 weights per byte */
static inline int w2_get(int row, int col)
{
    static const int8_t dec[4] = {0, -1, 1, -2};
    uint8_t b = blob[W2_OFF + row * (N_HID / 4) + col / 4];
    return dec[(b >> ((col & 3) << 1)) & 3];
}

/*
 * Encode the current context into a 96-element uint8 feature vector.
 * Pico uses a tighter encoding than the compact model — fewer buckets
 * and a 4-bit bag mask — which keeps the weight matrix small enough
 * for 1-bit quantization to remain useful.
 *
 *   Trigram counts [0-63]   — (c0*31 + c1*7 + c2) mod 64
 *   Bag-of-chars  [64-79]  — (char & 0x0F), 16 character classes
 *   Last char     [80-87]  — last generated char & 7 (once gencnt >= 1)
 *   Depth         [88-95]  — gencnt & 7, signals position in response
 *
 * ctx:    lowercase(query) + '>' + lowercase(generated_so_far)
 * gencnt: number of characters produced so far
 */
static void feature_hash(const char *ctx, int gencnt, uint8_t h[N_IN])
{
    memset(h, 0, N_IN);
    int len = (int)strlen(ctx);

    /* Trigram hash [0-63]: 64 buckets (one bit narrower than compact) */
    for (int i = 0; i < len - 2; i++) {
        int c0 = tolower((unsigned char)ctx[i])     & 0x7F;
        int c1 = tolower((unsigned char)ctx[i + 1]) & 0x7F;
        int c2 = tolower((unsigned char)ctx[i + 2]) & 0x7F;
        int idx = (c0 * 31 + c1 * 7 + c2) & 0x3F;
        if (h[idx] < 255) h[idx]++;
    }

    /* Bag-of-chars [64-79]: 4-bit character class (16 buckets) */
    for (int i = 0; i < len; i++) {
        int idx = 64 + (tolower((unsigned char)ctx[i]) & 0x0F);
        if (h[idx] < 255) h[idx]++;
    }

    /* Position features — only meaningful once generation has begun */
    if (gencnt >= 1) {
        const char *sep = strchr(ctx, '>');
        const char *ans = sep ? sep + 1 : ctx;
        int alen = (int)strlen(ans);
        if (alen >= 1) {
            int lc = tolower((unsigned char)ans[alen - 1]);
            h[80 + (lc & 7)]++;         /* last char identity [80-87] */
            h[88 + (gencnt & 7)]++;     /* generation depth   [88-95] */
        }
    }
}

/*
 * Forward pass: features → logits.
 * The EOL logit is penalised at early positions so the model must commit
 * to producing content before it can decide to stop. This is baked into
 * training and must be replicated exactly at inference time.
 */
static void forward(const uint8_t h[N_IN], int pos, int logits[N_OUT])
{
    uint8_t hidden[N_HID];
    const int8_t *bias1 = (const int8_t *)(blob + B1_OFF);
    const int8_t *bias2 = (const int8_t *)(blob + B2_OFF);

    /* Layer 1: input → hidden with ReLU, clamped to [0, 255] */
    for (int r = 0; r < N_HID; r++) {
        int acc = bias1[r];
        for (int c = 0; c < N_IN; c++)
            acc += w1_get(r, c) * (int)h[c];
        hidden[r] = (uint8_t)(acc < 0 ? 0 : acc > 255 ? 255 : acc);
    }

    /* Layer 2: hidden → logits */
    for (int r = 0; r < N_OUT; r++) {
        int acc = bias2[r];
        for (int c = 0; c < N_HID; c++)
            acc += w2_get(r, c) * (int)hidden[c];
        logits[r] = acc;
    }

    /* Algorithmic EOL suppression: matches training-time penalty */
    if      (pos < 3) logits[EOL_IDX] -= 40;
    else if (pos < 8) logits[EOL_IDX] -= 15;
}

/*
 * Generate a response for the given prompt.
 * out receives the lowercase result, null-terminated.
 */
static void generate(const char *prompt, char *out)
{
    char ctx[MAX_CTX];
    uint8_t h[N_IN];
    int logits[N_OUT];

    int plen = (int)strlen(prompt);
    if (plen > MAX_CTX - 2) plen = MAX_CTX - 2;
    for (int i = 0; i < plen; i++)
        ctx[i] = (char)tolower((unsigned char)prompt[i]);
    ctx[plen]     = '>';
    ctx[plen + 1] = '\0';

    int pos = 0;
    out[0] = '\0';

    for (int i = 0; i < MAX_GEN; i++) {
        feature_hash(ctx, pos, h);
        forward(h, pos, logits);

        int best = 0;
        for (int j = 1; j < N_OUT; j++)
            if (logits[j] > logits[best]) best = j;

        if (best == EOL_IDX) break;

        out[pos]     = CHARSET[best];
        out[pos + 1] = '\0';

        int clen = (int)strlen(ctx);
        if (clen < MAX_CTX - 1) {
            ctx[clen]     = CHARSET[best];  /* already lowercase in CHARSET */
            ctx[clen + 1] = '\0';
        }
        pos++;
    }
}

static void print_banner(void)
{
    printf("================================\n");
    printf("            %s\n", UI_TITLE);
    printf("        Just a Model\n");
#ifdef UI_SIZE_INFO
    printf("        %s\n", UI_SIZE_INFO);
#endif
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
    char input[256];
    char output[MAX_GEN + 1];

    FILE *f = fopen(weights_path, "rb");
    if (!f) {
        fprintf(stderr, "error: cannot open weights: %s\n", weights_path);
        return 1;
    }
    size_t n = fread(blob, 1, BLOB_SIZE, f);
    fclose(f);
    if (n != BLOB_SIZE) {
        fprintf(stderr, "error: expected %d bytes, got %zu: %s\n",
                BLOB_SIZE, n, weights_path);
        return 1;
    }

    print_banner();

    while (1) {
        printf("> ");
        fflush(stdout);
        if (!fgets(input, sizeof(input), stdin)) break;

        int len = (int)strlen(input);
        while (len > 0 && (input[len - 1] == '\n' || input[len - 1] == '\r' ||
                            input[len - 1] == ' '))
            input[--len] = '\0';
        if (len == 0) continue;

        generate(input, output);
        printf("%s\n\n", output);
    }

    printf("\n");
    return 0;
}

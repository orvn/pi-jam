/*
 * JAM — compact generative model (192 → 256 → 45)
 * Greedy character-level generation: one forward pass per output character.
 * Three-phase output bias (start / mid / late) shapes response length
 * without adding parameters — the same weights behave differently by position.
 * W1 quantized to INT2 (4/byte), W2 to INT4 (2/byte).
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <ctype.h>
#include "theme.h"

/* Architecture */
#define N_IN     192
#define N_HID    256
#define N_OUT     45
#define EOL_IDX   44
#define MAX_GEN   15
#define MAX_CTX  256

/*
 * Binary weight blob layout — all sections contiguous, no padding.
 * Three separate output bias vectors (B2S/B2M/B2) implement the phase trick:
 * the model sees different output biases depending on how far into the
 * response it is, nudging early chars toward openers and late chars toward
 * endings without changing the learned weights.
 */
#define W1_OFF    0        /* 256 * 48  = 12288 bytes, INT2 packed  */
#define W2_OFF    12288    /* 45  * 128 =  5760 bytes, INT4 packed  */
#define B1_OFF    18048    /* 256 bytes, int8                        */
#define B2_OFF    18304    /* 45  bytes, int8 — late phase           */
#define B2S_OFF   18349    /* 45  bytes, int8 — start phase (pos<3) */
#define B2M_OFF   18394    /* 45  bytes, int8 — mid phase  (pos<8)  */
#define BLOB_SIZE 18439

#ifndef WEIGHTS_FILE
#define WEIGHTS_FILE "weights_b2s.bin"
#endif

/* " ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-,!?'/" — index 44 = EOL */
static const char CHARSET[N_OUT] =
    " ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-,!?'/";

static uint8_t blob[BLOB_SIZE];

/* W1: INT2, codes 0→0  1→-1  2→+1  3→-2, 4 weights per byte */
static inline int w1_get(int row, int col)
{
    static const int8_t dec[4] = {0, -1, 1, -2};
    uint8_t b = blob[W1_OFF + row * (N_IN / 4) + col / 4];
    return dec[(b >> ((col & 3) << 1)) & 3];
}

/* W2: signed nibble, 2 weights per byte (lo=even col, hi=odd col) */
static inline int w2_get(int row, int col)
{
    uint8_t b = blob[W2_OFF + row * (N_HID / 2) + col / 2];
    int v = (col & 1) ? (b >> 4) : (b & 0xF);
    return v > 7 ? v - 16 : v;
}

/*
 * Encode the current context into a 192-element uint8 feature vector.
 * Three overlapping signals give the model its sense of context:
 *   Trigram counts [0-127]   — which letter sequences have appeared;
 *                               coarse syntax and word-shape information.
 *   Bag-of-chars  [128-159]  — which characters are present at all;
 *                               vocabulary and topic signal.
 *   Suffix features[160-191] — last 1-2 generated chars + generation depth;
 *                               lets the model track position in its response.
 *
 * ctx:    lowercase(query) + '>' + lowercase(generated_so_far)
 * gencnt: number of characters produced so far (0 before first char)
 */
static void feature_hash(const char *ctx, int gencnt, uint8_t h[N_IN])
{
    memset(h, 0, N_IN);
    int len = (int)strlen(ctx);

    /* Trigram hash [0-127] */
    for (int i = 0; i < len - 2; i++) {
        int c0 = tolower((unsigned char)ctx[i])     & 0x7F;
        int c1 = tolower((unsigned char)ctx[i + 1]) & 0x7F;
        int c2 = tolower((unsigned char)ctx[i + 2]) & 0x7F;
        int idx = (c0 * 31 + c1 * 7 + c2) & 0x7F;
        if (h[idx] < 255) h[idx]++;
    }

    /* Bag-of-chars [128-159]: (char & 0x1F) offset by 128 */
    for (int i = 0; i < len; i++) {
        int idx = 128 + (tolower((unsigned char)ctx[i]) & 0x1F);
        if (h[idx] < 255) h[idx]++;
    }

    /* Suffix features [160-191]: only once generation has started */
    if (gencnt >= 1) {
        const char *sep = strchr(ctx, '>');
        const char *ans = sep ? sep + 1 : ctx;
        int alen = (int)strlen(ans);
        if (alen >= 1) {
            int lc = tolower((unsigned char)ans[alen - 1]);
            h[160 + (lc & 7)]++;            /* last char & 7  [160-167] */
            h[184 + (gencnt & 7)]++;        /* gencnt & 7     [184-191] */
        }
        if (gencnt >= 2 && alen >= 2) {
            int lc  = tolower((unsigned char)ans[alen - 1]);
            int plc = tolower((unsigned char)ans[alen - 2]);
            h[168 + (plc & 7)]++;           /* prev char & 7  [168-175] */
            h[176 + ((lc - plc) & 7)]++;    /* delta & 7      [176-183] */
        }
    }
}

/*
 * Forward pass: features → logits.
 * phase 0 → B2S (start), 1 → B2M (mid), 2 → B2 (late)
 */
static void forward(const uint8_t h[N_IN], int phase, int logits[N_OUT])
{
    uint8_t hidden[N_HID];
    const int8_t *bias2 = (const int8_t *)(blob +
        (phase == 0 ? B2S_OFF : phase == 1 ? B2M_OFF : B2_OFF));
    const int8_t *bias1 = (const int8_t *)(blob + B1_OFF);

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
}

/*
 * Generate a response for the given prompt.
 * out receives the uppercase result, null-terminated.
 */
static void generate(const char *prompt, char *out)
{
    char ctx[MAX_CTX];
    uint8_t h[N_IN];
    int logits[N_OUT];

    /* Build initial context: lowercase(prompt) + '>' */
    int plen = (int)strlen(prompt);
    if (plen > MAX_CTX - 2) plen = MAX_CTX - 2;
    for (int i = 0; i < plen; i++)
        ctx[i] = (char)tolower((unsigned char)prompt[i]);
    ctx[plen]     = '>';
    ctx[plen + 1] = '\0';

    int pos = 0;
    out[0] = '\0';

    for (int i = 0; i < MAX_GEN; i++) {
        int phase = (pos < 3) ? 0 : (pos < 8) ? 1 : 2;
        feature_hash(ctx, pos, h);
        forward(h, phase, logits);

        int best = 0;
        for (int j = 1; j < N_OUT; j++)
            if (logits[j] > logits[best]) best = j;

        if (best == EOL_IDX) break;

        out[pos]     = CHARSET[best];
        out[pos + 1] = '\0';

        /* Append lowercase generated char to context */
        int clen = (int)strlen(ctx);
        if (clen < MAX_CTX - 1) {
            ctx[clen]     = (char)tolower((unsigned char)CHARSET[best]);
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

        /* Strip trailing whitespace */
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

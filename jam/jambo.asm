; JAMBO - Byte-level generative language model on Atari 800
; Generates text one character at a time. No lookup table.
; Architecture: 192 input features -> 256 hidden (ReLU) -> 45 output (char)
; A. M. Thorn, 2026. twistj.com

.export start
CIOV    = $E456
ICCOM   = $0342
ICBAL   = $0344
ICBAH   = $0345
ICBLL   = $0348
ICBLH   = $0349
PUTCHR  = $0B
GETREC  = $05
COLOR1  = $02C5
COLOR4  = $02C8
EOL     = $9B
CLR     = $7D
POKEY   = $D200
AUDF1   = POKEY + 0
AUDC1   = POKEY + 1
AUDCTL  = POKEY + 8
RANDOM  = POKEY + $0A
CH      = $02FC
COLBK   = $D01A
COLOR2  = $02C6   ; text bg shadow
; Brand colors (jam.ag)

; === Architecture ===
N_IN     = 192     ; 128 tri + 32 bag + 32 suffix
N_HID    = 256
N_HID_LO = N_HID .MOD 256  ; for 8-bit loop counters     ; hidden neurons
N_OUT    = 45      ; charset + EOL
L1_WHALF = 48      ; W1 row bytes (4 x INT2 per byte)
L2_WHALF = 128     ; N_HID / 2      ; N_HID / 2
SEP_CHAR = $3E     ; '>' separator
MAX_GEN  = 15
MAX_INPUT = 16      ; max chars to generate
EOL_IDX  = 44      ; EOL token index

; Weight blob offsets
; W1 is packed INT2: 4 weights per byte, codes 00=0, 01=-1, 10=+1, 11=-2
W1_OFF   = 0       ; 256 * 48 = 12288 bytes
W2_OFF   = 12288   ; 45 * 128 = 5760 packed INT4 bytes
B1_OFF   = 18048   ; 256 bytes
B2_OFF   = 18304   ; 45 bytes
B2S_OFF  = 18349   ; 45 start biases (chars 0-2)
B2M_OFF  = 18394   ; 45 mid biases (chars 3-7)

.segment "ZEROPAGE"
acc_lo: .res 1
acc_hi: .res 1
prod_lo: .res 1
prod_hi: .res 1
act:    .res 1
mag:    .res 1
sh_lo:  .res 1
sh_hi:  .res 1
wptr:   .res 2
aptr:   .res 2
aptr_b: .res 2
bptr:   .res 2
optr:   .res 2
wcnt:   .res 1
ocnt:   .res 1
ninh:   .res 1
wbyte:  .res 1
llen:   .res 1
blo:    .res 1
bhi:    .res 1
besti:  .res 1
; gencnt: MOVED TO BSS (CIOV safe)
nibble_sel: .res 1
pptr:   .res 2
tmp:    .res 1
; lastch, repchr: MOVED TO BSS (CIOV safe)
second: .res 1     ; second-best index
slo:    .res 1     ; second-best value lo
shi:    .res 1     ; second-best value hi
wptr2:  .res 2     ; second weight pointer (bag column)
nibsel2: .res 1    ; nibble selector for bag column

.segment "BSS"
iline:  .res 42
co:     .res 1     ; must survive CIOV
gencnt: .res 1     ; in BSS: CIOV-safe
lastch: .res 1     ; in BSS: CIOV-safe
repchr: .res 1     ; in BSS: CIOV-safe
answer_ok: .res 1   ; 1=good answer, 0=fallback/unsure
ctxlen: .res 1     ; current context length, survives pc/CIOV
tc0:    .res 1     ; trigram temps, survive pc/CIOV
tc1:    .res 1
tc2:    .res 1
tc3:    .res 1     ; suffix hash temp
vbi_cnt: .res 1    ; deferred VBI state, do not keep in OS zero page
vbi_flag: .res 1
old_vbi: .res 2
last_bucket: .res 1
first_turn: .res 1
last_bag: .res 1
inv_len: .res 1    ; ps_inv length must survive pc/CIOV
snd_ctr: .res 1
hint_mode: .res 1
dream_flag: .res 1
dream_chain: .res 1
dream_color: .res 1

ctx:    .res 128   ; query + > + response (expanded)
thash:  .res 128   ; trigram hash buckets
bhash:  .res 64    ; bag-of-chars hash
shash:  .res 32    ; dummy (unused)
hidbuf: .res 256   ; N_HID   ; hidden activations
obuf:   .res 90    ; 45 * 2 byte output
preact: .res 512
dirty_count: .res 1
dirty_off: .res 12
dirty_sgn: .res 12    ; 0=add bucket, 1=sub bucket

; Sparse non-zero list (shared by L1 and L2)
nz_count: .res 1    ; number of non-zero entries
nz_off:  .res 64    ; shared: W1 byte offsets / W2 packed-byte offsets
nz_nib:  .res 64    ; shared: W1 pair index 0..3 / W2 nibble 0..1
nz_val:  .res 64    ; input values
   ; N_HID x 2   ; 192 x 16-bit preactivations

.segment "CODE"
start:
        lda $D301
        ora #$B7         ; B7: OS on, BASIC off, CPU=main RAM (bit2!)
        sta $D301

        ; Clear ALL BSS at $0700-$0BFF (1280 bytes)
        ; Real HW has random RAM after boot/reset!
        ldx #0
        lda #0
@cb:    sta $0700,x
        sta $0800,x
        sta $0900,x
        sta $0A00,x
        sta $0B00,x
        inx
        bne @cb

                lda #$02
        sta COLOR4      ; border: very dark
        lda #$10
        sta COLOR2      ; bg: dark gold (hue for text)
        lda #$0A
        sta COLOR1      ; text lum: bright amber
        jsr stop_sound
        ; Setup VBI cursor blinker
        lda #0
        sta vbi_flag
        lda #8
        sta vbi_cnt
        ; Save old deferred VBI
        lda $0224
        sta old_vbi
        lda $0225
        sta old_vbi+1
        ; Install our VBI
        ldy #<vbi_cursor
        ldx #>vbi_cursor
        lda #7           ; deferred VBI
        jsr $E45C        ; SETVBV

        lda #1
        sta first_turn   ; first turn: clear hash

        lda #CLR
        sta co
        jsr pc
        lda #<ban
        ldx #>ban
        ldy #banl
        jsr ps
        lda #<intro
        ldx #>intro
        ldy #introl
        jsr ps

mloop:
        lda #0
        sta dream_flag
        sta dream_chain
        lda #<prm
        ldx #>prm
        ldy #prml
        jsr ps
        jsr rline_timeout
        bcc got_input
        jsr idle_chatter
got_input:
        ; Strip trailing punctuation (?!.)
@strip: ldx llen
        beq mloop
        dex
        lda iline,x
        cmp #63          ; '?'
        beq @cut
        cmp #33          ; '!'
        beq @cut
        cmp #46          ; '.'
        beq @cut
        jmp @nocut
@cut:   lda #0
        sta iline,x
        stx llen
        jmp @strip
@nocut: lda llen
        beq mloop
        cmp #MAX_INPUT+1
        bcc @lenok
        lda #MAX_INPUT
        sta llen
@lenok:

        ; === Build context: query + '>' ===
        ldx #0
@cpq:   cpx llen
        beq @qd
        lda iline,x
        ; to lowercase
        cmp #65
        bcc @stq
        cmp #91
        bcs @stq
        ora #$20
@stq:   sta ctx,x
        inx
        bne @cpq
@qd:    lda #SEP_CHAR
        sta ctx,x
        inx
        stx ctxlen

        ; === Context-aware hash ===
        lda first_turn
        beq @decay
        ; First turn: clear hash
        jsr thash_clear
        lda #0
        sta first_turn   ; subsequent turns decay
        jmp @dohash
@decay: ; Decay old context (>>1 all buckets)
        jsr thash_clear    ; full clear (clears both thash+bhash)
@dohash:
        jsr thash_hash   ; hash new query INTO thash

        ; === Init rolling trigram state from end of query ===
        ldx ctxlen
        dex
        lda ctx,x        ; ctx[len-1] = separator
        sta tc2
        dex
        lda ctx,x        ; ctx[len-2]
        sta tc1
        dex
        bmi @tc_ok        ; ctxlen < 3? skip tc0
        lda ctx,x
        sta tc0
@tc_ok:

        ; === Generate loop ===
        lda #0
        sta gencnt
        sta repchr       ; repeat counter
        lda #$FF
        sta lastch       ; no previous char
        lda #' '
        sta co
        jsr pc          ; leading space before answer

        jsr start_sound
        lda #1
        sta vbi_flag     ; enable cursor blink (once)

gen_loop:
        jsr thinking_sound
        lda #1
        sta answer_ok
        ; === L1: check first token vs incremental ===
        lda gencnt
        beq @first_l1
        jmp @incr_l1
@first_l1:

        ; --- FIRST TOKEN: sparse L1 with suffix ---
        jsr thash_clear
        jsr thash_hash
        jsr l1_full_sparse_store
        jmp @do_l2

@incr_l1:
        jsr l1_incremental_exact

@do_l2:

        ; === L2: sparse over non-zero hidbuf ===
        jsr build_nz_list_hid
        lda #<(md + W2_OFF)
        sta wptr
        lda #>(md + W2_OFF)
        sta wptr+1
        ; 3-phase bias: B2S(0-2), B2M(3-7), B2(8+)
        lda gencnt
        cmp #8
        bcs @use_b2
        cmp #3
        bcs @use_b2m
        ; chars 0-2: startup
        lda #<(md + B2S_OFF)
        sta bptr
        lda #>(md + B2S_OFF)
        sta bptr+1
        jmp @b2done
@use_b2m:
        ; chars 3-7: mid
        lda #<(md + B2M_OFF)
        sta bptr
        lda #>(md + B2M_OFF)
        sta bptr+1
        jmp @b2done
@use_b2:
        ; chars 8+: cruise
        lda #<(md + B2_OFF)
        sta bptr
        lda #>(md + B2_OFF)
        sta bptr+1
@b2done:
        lda #<obuf
        sta optr
        lda #>obuf
        sta optr+1
        lda #N_OUT
        sta ocnt
        jsr lraw_sparse

        ; === EOL bias: progressive after 12 chars ===
        lda gencnt
        cmp #15
        bcc @no_eol
        ; Progressive: bias = 24 + (gencnt-15) * 16
        sec
        sbc #15          ; A = 0, 1, 2, 3...
        asl              ; *2
        asl              ; *4
        asl              ; *8
        asl              ; *16
        clc
        adc #24          ; 24, 40, 56, 72, 88...
        ; Add to EOL logit at obuf + EOL_IDX * 2
        clc
        adc obuf + EOL_IDX * 2
        sta obuf + EOL_IDX * 2
        lda #0
        adc obuf + EOL_IDX * 2 + 1
        sta obuf + EOL_IDX * 2 + 1
@no_eol:
        ; === Argmax (45 values) ===
        jsr amax45

        ; === Confidence check: first char only ===
        ; margin < 3 -> JUST ASK (prevents garbage start)
        ; No mid-gen check (was cutting valid answers on real HW)
        ; MAX_GEN=15 + repeat-3 detector = sufficient safety
        lda gencnt
        bne @conf_ok
        sec
        lda blo
        sbc slo
        sta tmp
        lda bhi
        sbc shi
        bne @conf_ok        ; high byte != 0 = large margin
        lda tmp
        cmp #3              ; first char threshold (lowered: trained min=2)
        bcs @conf_ok
        jmp gen_unsure      ; -> prints JUST ASK
@conf_ok:
        ; === Check EOL ===
        lda besti
        cmp #EOL_IDX
        bne @not_eol
        jmp gen_done
@not_eol:

        ; === Output character ===
        ldx besti
        lda charset,x
        sta co

        ; === Repeat detection (fallback for unknown) ===
        cmp lastch
        bne @newch
        inc repchr
        lda repchr
        cmp #3           ; 3 same chars in a row = garbage
        bcc @printch_ok
        jmp gen_fallback
@printch_ok:
        jmp @printch
@newch: sta lastch
        lda #1
        sta repchr
@printch:
        jsr stop_sound   ; silence gap before print
        ; === Pico glitch: tight margin -> flash second ===
        sec
        lda blo
        sbc slo
        sta tmp
        lda bhi
        sbc shi
        bne @no_think
        lda tmp
        cmp #8
        bcs @no_think
        lda second
        cmp #EOL_IDX
        beq @no_think
        tax
        lda charset,x
        ora #$80         ; inverse
        sta co
        jsr pc           ; print inverse second-best
        ldx #3           ; wait 3 frames
@twait: lda $14
@tw2:   cmp $14
        beq @tw2
        dex
        bne @twait
        lda #$7E         ; backspace
        sta co
        jsr pc           ; erase
        ldx besti
        lda charset,x
        sta co           ; reload real char
@no_think:
        ; === Dream screensaver: psychedelic colors ===
        lda dream_flag
        beq @no_dinv
        ; Inverse text
        lda co
        ora #$80
        sta co
        ; Cycle colors every char
        lda dream_color
        clc
        adc #$13          ; hue step (prime = no repeat)
        sta dream_color
        sta COLOR4        ; border
        eor #$84
        sta COLOR2        ; background (complementary)
        jsr pc            ; print inverse char
        ; 2-frame delay (pico style)
        lda $14
@dw1:   cmp $14
        beq @dw1
        lda $14
@dw2:   cmp $14
        beq @dw2
        ; Check keypress
        lda CH
        cmp #$FF
        bne @dream_abort
        jmp @after_print
@dream_abort:
        lda #$02
        sta COLOR4
        lda #$10
        sta COLOR2
        lda #$0A
        sta COLOR1
        lda #EOL
        sta co
        jsr pc
        jmp mloop
@no_dinv:
        jsr pc            ; normal print
@after_print:

        ; === Append to context (lowercase) ===
        lda co           ; P2: reuse co (no re-read from charset)
        cmp #65          ; 'A'
        bcc @nolow
        cmp #91          ; 'Z'+1
        bcs @nolow
        ora #$20         ; to lowercase
@nolow: ldx ctxlen
        sta ctx,x
        inx
        stx ctxlen
        inc gencnt
        lda gencnt
        cmp #MAX_GEN
        bcc @notdone
        jmp gen_done
@notdone:
        jmp gen_loop

gen_unsure:
        lda hint_mode
        beq @nohint_skip
        jmp gen_done_hint
@nohint_skip:
        jsr stop_sound
        lda #0
        sta answer_ok
        lda #0
        sta vbi_flag
        ; Pick 1 of 8 unsure messages via RTCLOK
        lda $14
        and #7
        pha
        asl
        tax
        lda us_ptrs,x
        sta aptr
        lda us_ptrs+1,x
        sta aptr+1
        pla
        tax
        lda us_lens,x
        tay
        jsr ps_inv
        lda #EOL
        sta co
        jsr pc
        jmp gen_done

gen_fallback:
        lda hint_mode
        beq @nohint_skip
        jmp gen_done_hint
@nohint_skip:
        jsr stop_sound
        lda #0
        sta answer_ok
        ; Erase printed garbage (repchr-1 chars were printed)
        ldx repchr
        dex
@bksp:  lda #$7E        ; backspace
        sta co
        txa
        pha
        jsr pc
        pla
        tax
        dex
        bne @bksp
        ; New line + leading space
        lda #EOL
        sta co
        jsr pc
        lda #$20
        sta co
        jsr pc
        ; Pick 1 of 3 fallback messages via RTCLOK
        lda $14
        and #$03
        cmp #3
        bcc @fbpick
        lda #0
@fbpick:
        pha              ; save index 0/1/2
        asl
        tax
        lda fb_ptrs,x    ; word-indexed pointer
        sta aptr
        lda fb_ptrs+1,x
        sta aptr+1
        pla              ; restore index 0/1/2
        tax
        lda fb_lens,x    ; byte-indexed length
        tay
        jsr ps_inv

gen_done:
        jsr stop_sound
        lda #0
        sta vbi_flag     ; disable cursor blink
        ; Restore screen byte in case it is inverted
        ldy #0
        lda ($5E),y      ; OLDADR
        and #$7F         ; clear inverse bit
        sta ($5E),y
        lda #EOL
        sta co
        jsr pc

        ; === Dream chaining ===
        lda dream_flag
        bne @is_dreaming
        jmp not_dreaming
@is_dreaming:
        ; Find separator in ctx, copy answer to iline
        ldx #0
@dfs:   lda ctx,x
        cmp #SEP_CHAR
        beq @dfound
        inx
        cpx ctxlen
        bne @dfs
        jmp gen_done_dream
@dfound:
        inx
        ldy #0
@dcpy:  cpx ctxlen
        beq @dcpd
        lda ctx,x
        sta iline,y
        inx
        iny
        cpy #MAX_INPUT
        bcc @dcpy
@dcpd:  lda #0
        sta iline,y
        sty llen
        cpy #2
        bcc gen_done_dream
        inc dream_chain
        lda dream_chain
        cmp #3
        bcs gen_done_dream
        ; Pause ~1 sec, check keypress
        ldx #50
@dpause:
        ; Color cycle during pause
        lda dream_color
        clc
        adc #$07
        sta dream_color
        sta COLOR4
        eor #$84
        sta COLOR2
        lda $14
@dpw:   cmp $14
        beq @dpw
        lda CH
        cmp #$FF
        bne gen_done_dream
        dex
        bne @dpause
        ; Print dream prefix
        lda #'.'+$80
        sta co
        jsr pc
        jsr pc
        jsr pc
        lda #' '
        sta co
        jsr pc
        jmp got_input
gen_done_dream:
        lda #0
        sta dream_flag
        lda #$02
        sta COLOR4
        lda #$10
        sta COLOR2
        lda #$0A
        sta COLOR1
        jmp mloop
not_dreaming:
        ; === Neural hint: re-infer on own answer (~25%) ===
        lda hint_mode
        bne hint_skip         ; no recursion
        lda answer_ok
        beq hint_skip         ; skip after fallback
        lda gencnt
        cmp #2
        bcc hint_skip         ; skip 1-char answers
        lda RANDOM
        and #$03
        bne hint_skip         ; ~25% chance
        ; Copy answer from ctx to start: ctx = "query>answer"
        ; Find separator
        ldx #0
@hfs:   lda ctx,x
        cmp #SEP_CHAR
        beq @hfound
        inx
        cpx ctxlen
        bne @hfs
        jmp hint_skip         ; no separator?!
@hfound:
        inx                  ; skip '>'
        ldy #0
@hcopy: cpx ctxlen
        beq @hend
        lda ctx,x
        sta ctx,y
        inx
        iny
        jmp @hcopy
@hend:  lda #SEP_CHAR
        sta ctx,y
        iny
        sty ctxlen
        ; Setup for generation
        lda #1
        sta hint_mode
        lda #0
        sta gencnt
        sta repchr
        lda #$FF
        sta lastch
        lda #' '
        sta co
        jsr pc               ; leading space
        jsr start_sound
        jmp gen_loop
gen_done_hint:
        jsr stop_sound
        lda #0
        sta hint_mode
        jmp mloop
hint_skip:
        lda #0
        sta hint_mode
        jmp mloop

; =====================================================
; TRIGRAM HASH - Full computation
; Hash all trigrams in ctx[0..ctxlen-1]
; h = (c0*31 + c1*7 + c2) & $7F
; =====================================================
thash_clear:
        ; Clear 192 contiguous bytes: thash(128) + shash(32) + bhash(32)
        ldx #0
        lda #0
@clr:   sta thash,x
        inx
        cpx #192
        bne @clr
        rts

; thash_decay: REMOVED (using full clear now)

thash_hash:
        ; Hash ctx trigrams INTO existing thash (additive)
        lda ctxlen
        cmp #3
        bcc @done

        ldx #0           ; position counter
@loop:  lda ctx,x
        sta tc0
        lda ctx+1,x
        sta tc1
        lda ctx+2,x
        sta tc2
        txa
        pha
        jsr tri_one
        pla
        tax
        inx
        txa
        clc
        adc #2           ; x+2
        cmp ctxlen       ; if x+2 >= ctxlen, done
        bcs @done
        txa              ; restore X (position)
        tax
        jmp @loop
@done:
        ; Bag-of-chars hash: for each char in ctx, bhash[ord(c) & $3F]++
        ldx #0
@bag:   cpx ctxlen
        beq @bdone
        lda ctx,x
        and #$1F         ; mod 32 (test)
        tay
        lda bhash,y
        cmp #255
        beq @bsat
        clc
        adc #1
        sta bhash,y
@bsat:  inx
        jmp @bag
@bdone:
        ; === One-hot suffix: 4 groups x 8 buckets = bhash[32-63] ===
        ; tc-safe (uses only A, X, tmp), no .strip()
        lda gencnt
        beq @sfx_d
        ; Group 1: bhash[32 + (last_char & 7)] ++ 
        ldx ctxlen
        dex
        lda ctx,x           ; A = c[-1] (raw, with spaces)
        sta tmp              ; save c[-1]
        and #7
        clc
        adc #32
        tax
        inc bhash,x
        ; Group 4: bhash[56 + (gencnt & 7)] ++
        lda gencnt
        and #7
        clc
        adc #56
        tax
        inc bhash,x
        ; Groups 2,3 need gencnt >= 2
        lda gencnt
        cmp #2
        bcc @sfx_d
        ; Group 2: bhash[40 + (prev_char & 7)] ++
        ldx ctxlen
        dex
        dex                  ; X = ctxlen-2
        lda ctx,x            ; A = c[-2]
        and #7
        clc
        adc #40
        tax
        inc bhash,x
        ; Group 3: bhash[48 + ((c[-1] - c[-2]) & 7)] ++
        ldx ctxlen
        dex
        dex
        lda tmp              ; A = c[-1]
        sec
        sbc ctx,x            ; A = c[-1] - c[-2]
        and #7
        clc
        adc #48
        tax
        inc bhash,x
@sfx_d: rts

; === Hash one trigram: tc0, tc1, tc2 -> thash[h]++ ===
tri_one:
        ; h = (tc0*31 + tc1*7 + tc2) & $7F
        ; tc0*31 = (tc0<<5) - tc0, low byte only
        lda tc0
        asl
        asl
        asl
        asl
        asl              ; tc0 << 5 (low byte)
        sec
        sbc tc0          ; - tc0 = tc0*31 low byte
        sta acc_lo       ; temp

        ; tc1*7 = (tc1<<3) - tc1
        lda tc1
        asl
        asl
        asl              ; tc1 << 3
        sec
        sbc tc1          ; - tc1 = tc1*7 low byte
        clc
        adc acc_lo       ; + tc0*31
        clc
        adc tc2          ; + tc2

        and #$7F         ; mod 128 = AND $7F
        sta last_bucket  ; save for incremental L1
        tax
        inc thash,x      ; no saturation check needed (max ctx < 80)
        rts

; =====================================================
; L1: ReLU layer (shift=0, clamp 0-255)
; =====================================================
lrelu:
@n:     jsr cneuron2
        ; Add bias
        ldy #0
        lda (bptr),y
        ldx #0
        cmp #$80
        bcc @bp
        ldx #$FF
@bp:    clc
        adc acc_lo
        sta acc_lo
        txa
        adc acc_hi
        sta acc_hi
        ; Save 16-bit preactivation (for incremental L1)
        ldy #0
        lda acc_lo
        sta (pptr),y
        iny
        lda acc_hi
        sta (pptr),y
        clc
        lda pptr
        adc #2
        sta pptr
        bcc @npc
        inc pptr+1
@npc:
        ; shift=0: just ReLU + clamp
        lda acc_hi
        bmi @z           ; negative -> 0
        bne @cl          ; high byte nonzero -> 255
        lda acc_lo       ; high=0: value is low byte
        jmp @st
@cl:    lda #255
        jmp @st
@z:     lda #0
@st:    ldy #0
        sta (optr),y
        inc optr
        bne @no
        inc optr+1
@no:    inc bptr
        bne @nb
        inc bptr+1
@nb:    dec ocnt
        bne @n
        rts

; =====================================================
; L2: raw output layer (16-bit output)
; =====================================================
lraw:
@n:     jsr cneuron4
        ldy #0
        lda (bptr),y
        ldx #0
        cmp #$80
        bcc @bp
        ldx #$FF
@bp:    clc
        adc acc_lo
        sta acc_lo
        txa
        adc acc_hi
        sta acc_hi
        ldy #0
        lda acc_lo
        sta (optr),y
        iny
        lda acc_hi
        sta (optr),y
        clc
        lda optr
        adc #2
        sta optr
        bcc @no
        inc optr+1
@no:    inc bptr
        bne @nb
        inc bptr+1
@nb:    dec ocnt
        bne @n
        rts

; =====================================================
; L2: sparse raw output layer over non-zero hidbuf
; =====================================================
lraw_sparse:
@n:     jsr cneuron_sparse
        ldy #0
        lda (bptr),y
        ldx #0
        cmp #$80
        bcc @bp
        ldx #$FF
@bp:    clc
        adc acc_lo
        sta acc_lo
        txa
        adc acc_hi
        sta acc_hi
        ldy #0
        lda acc_lo
        sta (optr),y
        iny
        lda acc_hi
        sta (optr),y
        clc
        lda optr
        adc #2
        sta optr
        bcc @no
        inc optr+1
@no:    inc bptr
        bne @nb
        inc bptr+1
@nb:    clc
        lda wptr
        adc #L2_WHALF
        sta wptr
        bcc @nw
        inc wptr+1
@nw:    dec ocnt
        bne @n
        rts

; =====================================================
; Neuron core for W1: INT2 MAC
; Codes: 00=0, 01=-1, 10=+1, 11=-2
; =====================================================
cneuron2:
        lda #0
        sta acc_lo
        sta acc_hi
        ldx #0           ; compact W1 always reads contiguous thash+bhash
        lda ninh
        sta wcnt

; ---- Main MAC loop: 4 weights per iteration ----
@p:     ldy #0
        lda (wptr),y
        bne @noskip
        jmp @skip4       ; all 4 weights are zero
@noskip:
        sta wbyte

        ; input 0 / bits 0-1
        lda thash,x
        sta act
        beq @sk0
        lda wbyte
        and #$03
        beq @sk0
        cmp #$01
        beq @sub0
        cmp #$02
        beq @add0
        lda act
        asl
        sta prod_lo
        lda #0
        rol
        sta prod_hi
        sec
        lda acc_lo
        sbc prod_lo
        sta acc_lo
        lda acc_hi
        sbc prod_hi
        sta acc_hi
        jmp @sk0
@sub0:  sec
        lda acc_lo
        sbc act
        sta acc_lo
        lda acc_hi
        sbc #0
        sta acc_hi
        jmp @sk0
@add0:  clc
        lda acc_lo
        adc act
        sta acc_lo
        lda acc_hi
        adc #0
        sta acc_hi
@sk0:
        ; input 1 / bits 2-3
        lda thash+1,x
        sta act
        beq @sk1
        ldy wbyte
        lda twob1,y
        beq @sk1
        cmp #$01
        beq @sub1
        cmp #$02
        beq @add1
        lda act
        asl
        sta prod_lo
        lda #0
        rol
        sta prod_hi
        sec
        lda acc_lo
        sbc prod_lo
        sta acc_lo
        lda acc_hi
        sbc prod_hi
        sta acc_hi
        jmp @sk1
@sub1:  sec
        lda acc_lo
        sbc act
        sta acc_lo
        lda acc_hi
        sbc #0
        sta acc_hi
        jmp @sk1
@add1:  clc
        lda acc_lo
        adc act
        sta acc_lo
        lda acc_hi
        adc #0
        sta acc_hi
@sk1:
        ; input 2 / bits 4-5
        lda thash+2,x
        sta act
        beq @sk2
        ldy wbyte
        lda twob2,y
        beq @sk2
        cmp #$01
        beq @sub2
        cmp #$02
        beq @add2
        lda act
        asl
        sta prod_lo
        lda #0
        rol
        sta prod_hi
        sec
        lda acc_lo
        sbc prod_lo
        sta acc_lo
        lda acc_hi
        sbc prod_hi
        sta acc_hi
        jmp @sk2
@sub2:  sec
        lda acc_lo
        sbc act
        sta acc_lo
        lda acc_hi
        sbc #0
        sta acc_hi
        jmp @sk2
@add2:  clc
        lda acc_lo
        adc act
        sta acc_lo
        lda acc_hi
        adc #0
        sta acc_hi
@sk2:
        ; input 3 / bits 6-7
        lda thash+3,x
        sta act
        beq @sk3
        ldy wbyte
        lda twob3,y
        beq @sk3
        cmp #$01
        beq @sub3
        cmp #$02
        beq @add3
        lda act
        asl
        sta prod_lo
        lda #0
        rol
        sta prod_hi
        sec
        lda acc_lo
        sbc prod_lo
        sta acc_lo
        lda acc_hi
        sbc prod_hi
        sta acc_hi
        jmp @sk3
@sub3:  sec
        lda acc_lo
        sbc act
        sta acc_lo
        lda acc_hi
        sbc #0
        sta acc_hi
        jmp @sk3
@add3:  clc
        lda acc_lo
        adc act
        sta acc_lo
        lda acc_hi
        adc #0
        sta acc_hi
@sk3:
@skip4: inc wptr
        bne @w1
        inc wptr+1
@w1:    txa
        clc
        adc #4
        tax
        dec wcnt
        beq @done2
        jmp @p
@done2:
        rts

; =====================================================
; Neuron core for W2: INT4 MAC
; =====================================================
cneuron4:
        lda #0
        sta acc_lo
        sta acc_hi
        lda aptr_b
        sta aptr
        lda aptr_b+1
        sta aptr+1
        lda ninh
        sta wcnt

; ---- Main MAC loop: 2 weights per iteration ----
@p:     ldy #0
        lda (wptr),y
        sta wbyte
        lda (aptr),y
        sta act

        ; --- LO NIBBLE: fast path for 0,+/-1,+/-2,+/-3 ---
        lda wbyte
        and #$0F
        bne @nzlo          ; w=0: skip (23.6%)
        jmp @sklo
@nzlo:  cmp #8
        bcs @neglo

        ; -- Positive lo: 1-7 --
        cmp #4
        bcs @plo_big       ; w=4-7: rare (1.8%), use mul
        cmp #1
        beq @plo1
        cmp #2
        beq @plo2
        ; w=3: act*3 = act + act<<1
        lda act
        asl
        clc
        adc act
        sta prod_lo
        lda #0
        adc #0
        sta prod_hi
        jmp @addlo
@plo1:  ; w=1: prod = act
        lda act
        sta prod_lo
        lda #0
        sta prod_hi
        jmp @addlo
@plo2:  ; w=2: prod = act<<1
        lda act
        asl
        sta prod_lo
        lda #0
        rol
        sta prod_hi
@addlo: clc
        lda acc_lo
        adc prod_lo
        sta acc_lo
        lda acc_hi
        adc prod_hi
        sta acc_hi
        jmp @sklo
@plo_big:
        jsr mul
        clc
        lda acc_lo
        adc prod_lo
        sta acc_lo
        lda acc_hi
        adc prod_hi
        sta acc_hi
        jmp @sklo

@neglo: ; -- Negative lo: negate, then fast path --
        eor #$0F
        clc
        adc #1             ; now 1-8 absolute value
        cmp #4
        bcs @nlo_big
        cmp #1
        beq @nlo1
        cmp #2
        beq @nlo2
        ; w=3
        lda act
        asl
        clc
        adc act
        sta prod_lo
        lda #0
        adc #0
        sta prod_hi
        jmp @sublo
@nlo1:  lda act
        sta prod_lo
        lda #0
        sta prod_hi
        jmp @sublo
@nlo2:  lda act
        asl
        sta prod_lo
        lda #0
        rol
        sta prod_hi
@sublo: sec
        lda acc_lo
        sbc prod_lo
        sta acc_lo
        lda acc_hi
        sbc prod_hi
        sta acc_hi
        jmp @sklo
@nlo_big:
        jsr mul
        sec
        lda acc_lo
        sbc prod_lo
        sta acc_lo
        lda acc_hi
        sbc prod_hi
        sta acc_hi

@sklo:  inc aptr
        bne @a1
        inc aptr+1
@a1:    ldy #0
        lda (aptr),y
        sta act

        ; --- HI NIBBLE: same fast path ---
        ldx wbyte
        lda hinib,x
        bne @nzhi
        jmp @skhi
@nzhi:  cmp #8
        bcs @neghi

        cmp #4
        bcs @phi_big
        cmp #1
        beq @phi1
        cmp #2
        beq @phi2
        ; w=3
        lda act
        asl
        clc
        adc act
        sta prod_lo
        lda #0
        adc #0
        sta prod_hi
        jmp @addhi
@phi1:  lda act
        sta prod_lo
        lda #0
        sta prod_hi
        jmp @addhi
@phi2:  lda act
        asl
        sta prod_lo
        lda #0
        rol
        sta prod_hi
@addhi: clc
        lda acc_lo
        adc prod_lo
        sta acc_lo
        lda acc_hi
        adc prod_hi
        sta acc_hi
        jmp @skhi
@phi_big:
        jsr mul
        clc
        lda acc_lo
        adc prod_lo
        sta acc_lo
        lda acc_hi
        adc prod_hi
        sta acc_hi
        jmp @skhi

@neghi: eor #$0F
        clc
        adc #1
        cmp #4
        bcs @nhi_big
        cmp #1
        beq @nhi1
        cmp #2
        beq @nhi2
        ; w=3
        lda act
        asl
        clc
        adc act
        sta prod_lo
        lda #0
        adc #0
        sta prod_hi
        jmp @subhi
@nhi1:  lda act
        sta prod_lo
        lda #0
        sta prod_hi
        jmp @subhi
@nhi2:  lda act
        asl
        sta prod_lo
        lda #0
        rol
        sta prod_hi
@subhi: sec
        lda acc_lo
        sbc prod_lo
        sta acc_lo
        lda acc_hi
        sbc prod_hi
        sta acc_hi
        jmp @skhi
@nhi_big:
        jsr mul
        sec
        lda acc_lo
        sbc prod_lo
        sta acc_lo
        lda acc_hi
        sbc prod_hi
        sta acc_hi

@skhi:  inc aptr
        bne @a2
        inc aptr+1
@a2:    inc wptr
        bne @w1
        inc wptr+1
@w1:    dec wcnt
        beq @xd
        jmp @p
@xd:    rts

; =====================================================
; Shift-and-add multiply: mag * act -> prod (16-bit)
; =====================================================
mul:
        sta mag
        lda #0
        sta prod_lo
        sta prod_hi
        lda act
        sta sh_lo
        lda #0
        sta sh_hi
        lsr mag
        bcc @0
        clc
        lda prod_lo
        adc sh_lo
        sta prod_lo
        lda prod_hi
        adc sh_hi
        sta prod_hi
@0:     asl sh_lo
        rol sh_hi
        lsr mag
        bcc @1
        clc
        lda prod_lo
        adc sh_lo
        sta prod_lo
        lda prod_hi
        adc sh_hi
        sta prod_hi
@1:     asl sh_lo
        rol sh_hi
        lsr mag
        bcc @2
        clc
        lda prod_lo
        adc sh_lo
        sta prod_lo
        lda prod_hi
        adc sh_hi
        sta prod_hi
@2:     asl sh_lo
        rol sh_hi
        lsr mag
        bcc @3
        clc
        lda prod_lo
        adc sh_lo
        sta prod_lo
        lda prod_hi
        adc sh_hi
        sta prod_hi
@3:     rts

; =====================================================
; Argmax over 45 values (single loop, fits in page)
; =====================================================
amax45:
        ; Find best AND second-best indices
        lda obuf
        sta blo
        sta slo
        lda obuf+1
        sta bhi
        sta shi
        lda #0
        sta besti
        sta second
        ldx #1
@c:     txa
        asl
        tay
        ; Compare obuf[x] vs best (signed 16-bit)
        sec
        lda obuf,y
        sbc blo
        lda obuf+1,y
        sbc bhi
        bvc @nv
        eor #$80
@nv:    bmi @not_best
        ; New best: old best -> second
        lda besti
        sta second
        lda blo
        sta slo
        lda bhi
        sta shi
        ; Update best
        stx besti
        lda obuf,y
        sta blo
        lda obuf+1,y
        sta bhi
        jmp @next
@not_best:
        ; Check if obuf[x] > second
        sec
        lda obuf,y
        sbc slo
        lda obuf+1,y
        sbc shi
        bvc @nv2
        eor #$80
@nv2:   bmi @next
        ; New second
        stx second
        lda obuf,y
        sta slo
        lda obuf+1,y
        sta shi
@next:  inx
        cpx #N_OUT
        bne @c
        rts

; =====================================================
; I/O routines (from JAM)
; =====================================================
; === TIMEOUT INPUT (from jampico) ===
rline_timeout:
        lda #$FF
        sta CH
        lda RANDOM
        asl
        asl
        sta tmp
        lda #0
        rol
        sta wbyte
        clc
        lda tmp
        adc #<1500
        sta tmp
        lda wbyte
        adc #>1500
        sta wbyte
        lda #0
        sta acc_lo
        sta acc_hi
@wait:  lda CH
        cmp #$FF
        bne @key
        lda $14
@wf:    cmp $14
        beq @wf
        inc acc_lo
        bne @wc
        inc acc_hi
@wc:    lda acc_hi
        cmp wbyte
        bcc @wait
        bne @tout
        lda acc_lo
        cmp tmp
        bcc @wait
@tout:  sec
        rts
@key:   jsr rline
        clc
        rts

; === DREAM: pick random suggestion keyword ===
idle_chatter:
        lda RANDOM
        and #15
        pha
        asl
        tax
        lda drm_ptrs,x
        sta aptr
        lda drm_ptrs+1,x
        sta aptr+1
        pla
        tax
        lda drm_lens,x
        sta llen
        ldy #0
        ldx #0
@dcp:   cpx llen
        beq @dcd
        lda (aptr),y
        cmp #65
        bcc @dst
        cmp #91
        bcs @dst
        ora #$20
@dst:   sta iline,x
        iny
        inx
        jmp @dcp
@dcd:   lda #0
        sta iline,x
        lda #1
        sta dream_flag
        lda #0
        sta dream_chain
        lda RANDOM
        sta dream_color
        ; Print dream prefix
        lda #'.'+$80
        sta co
        jsr pc
        jsr pc
        jsr pc
        lda #' '
        sta co
        jsr pc
        clc
        rts


rline:
        ldx #0
        lda #GETREC
        sta ICCOM,x
        lda #<iline
        sta ICBAL,x
        lda #>iline
        sta ICBAH,x
        lda #40
        sta ICBLL,x
        lda #0
        sta ICBLH,x
        jsr CIOV
        ldx #0
@s:     lda iline,x
        cmp #EOL
        beq @f
        cmp #0
        beq @f
        inx
        cpx #40
        bne @s
@f:     lda #0
        sta iline,x
        stx llen
        rts

pc:     ldx #0
        lda #PUTCHR
        sta ICCOM,x
        lda #<co
        sta ICBAL,x
        lda #>co
        sta ICBAH,x
        lda #1
        sta ICBLL,x
        lda #0
        sta ICBLH,x
        jsr CIOV
        lda #0
        sta $02F0       ; CRSINH=0: cursor blinks after every output
        rts

ps_inv:
        ; Print Y bytes from (aptr) in inverse video
        sty inv_len
        ldy #0
@inv:   lda (aptr),y
        ora #$80         ; set inverse bit
        sta co
        lda aptr         ; CIOV trashes OS zero page, preserve pointer
        pha
        lda aptr+1
        pha
        tya
        pha
        jsr pc
        pla
        tay
        pla
        sta aptr+1
        pla
        sta aptr
        iny
        cpy inv_len
        bne @inv
        rts

ps:     stx @h+1
        pha
        ldx #0
        lda #PUTCHR
        sta ICCOM,x
        pla
        sta ICBAL,x
@h:     lda #0
        sta ICBAH,x
        tya
        sta ICBLL,x
        lda #0
        sta ICBLH,x
        jsr CIOV
        rts

; =====================================================
; POKEY thinking sound (active only during inference)
; =====================================================
start_sound:
        lda dream_flag
        bne snd_off       ; no sound in dreams
        lda #0
        sta snd_ctr
        sta AUDCTL
        lda #$A1        ; pure tone, volume 1
        sta AUDC1
        lda #$60        ; low pitch
        sta AUDF1
        rts

thinking_sound:
        lda dream_flag
        bne @snd_mute
        lda snd_ctr
        lsr
        and #$07        ; slow wobble 0-7
        clc
        adc #$58        ; pitch range $58-$5F (low)
        sta AUDF1
        inc snd_ctr
        rts

@snd_mute:
        rts

snd_off:
stop_sound:
        lda #0
        sta AUDC1
        sta AUDF1
        rts


; =====================================================
; L1 full sparse pass that also stores 16-bit preact for later deltas
; =====================================================
l1_full_sparse_store:
        lda #<(md + W1_OFF)
        sta wptr
        lda #>(md + W1_OFF)
        sta wptr+1
        lda #<(md + B1_OFF)
        sta bptr
        lda #>(md + B1_OFF)
        sta bptr+1
        lda #<hidbuf
        sta optr
        lda #>hidbuf
        sta optr+1
        lda #<preact
        sta pptr
        lda #>preact
        sta pptr+1
        lda #N_HID_LO
        sta ocnt
        jsr build_nz_list
@fl1:   jsr cneuron2_sparse
        ; Add bias
        ldy #0
        lda (bptr),y
        ldx #0
        cmp #$80
        bcc @fl1bp
        ldx #$FF
@fl1bp: clc
        adc acc_lo
        sta acc_lo
        txa
        adc acc_hi
        sta acc_hi
        ; Save 16-bit preactivation
        ldy #0
        lda acc_lo
        sta (pptr),y
        iny
        lda acc_hi
        sta (pptr),y
        clc
        lda pptr
        adc #2
        sta pptr
        bcc @fl1npc
        inc pptr+1
@fl1npc:
        ; ReLU -> hidbuf
        lda acc_hi
        bmi @fl1z
        bne @fl1c
        lda acc_lo
        jmp @fl1s
@fl1c:  lda #255
        jmp @fl1s
@fl1z:  lda #0
@fl1s:  ldy #0
        sta (optr),y
        inc optr
        bne @fl1no
        inc optr+1
@fl1no: inc bptr
        bne @fl1nb
        inc bptr+1
@fl1nb: dec ocnt
        bne @fl1
        rts

; =====================================================
; Exact incremental L1 for compact JAMBO.
; Updates thash+bhash in place, applies only +/- dirty columns to preact,
; then rebuilds hidbuf via ReLU. Must match full thash_hash exactly.
; =====================================================
l1_incremental_exact:
        jsr build_l1_dirty_deltas
        jsr apply_l1_dirty_deltas
        jsr relu_from_preact
        rts

note_dirty_add:
        ldy dirty_count
        cpy #12
        bcs @nda_done
        sta dirty_off,y
        lda #0
        sta dirty_sgn,y
        inc dirty_count
@nda_done:
        rts

note_dirty_sub:
        ldy dirty_count
        cpy #12
        bcs @nds_done
        sta dirty_off,y
        lda #1
        sta dirty_sgn,y
        inc dirty_count
@nds_done:
        rts

build_l1_dirty_deltas:
        lda #0
        sta dirty_count

        ; New trigram at current end: add one bucket
        ldx ctxlen
        dex
        lda ctx,x
        sta tc2
        dex
        lda ctx,x
        sta tc1
        dex
        lda ctx,x
        sta tc0
        jsr tri_one                ; updates thash[last_bucket]++
        lda last_bucket
        jsr note_dirty_add

        ; New bag bucket from the appended char
        ldx ctxlen
        dex
        lda ctx,x
        and #$1F
        tay
        lda bhash,y
        clc
        adc #1
        sta bhash,y
        tya
        clc
        adc #128
        jsr note_dirty_add

        ; Old suffix state belongs to gencnt-1
        lda gencnt
        sec
        sbc #1
        sta tmp
        beq @new_sfx               ; old count 0 => no old suffix buckets

        ; Old group 1: last char of previous state = ctx[len-2]
        ldx ctxlen
        dex
        dex
        lda ctx,x
        sta tc0
        and #7
        tay
        lda bhash+32,y
        sec
        sbc #1
        sta bhash+32,y
        tya
        clc
        adc #160
        jsr note_dirty_sub

        ; Old group 4: generation length bucket for old count
        lda tmp
        and #7
        tay
        lda bhash+56,y
        sec
        sbc #1
        sta bhash+56,y
        tya
        clc
        adc #184
        jsr note_dirty_sub

        lda tmp
        cmp #2
        bcc @new_sfx

        ; Old group 2: previous char of previous state = ctx[len-3]
        ldx ctxlen
        dex
        dex
        dex
        lda ctx,x
        sta tc1
        and #7
        tay
        lda bhash+40,y
        sec
        sbc #1
        sta bhash+40,y
        tya
        clc
        adc #168
        jsr note_dirty_sub

        ; Old group 3: delta(old_last - old_prev)
        lda tc0
        sec
        sbc tc1
        and #7
        tay
        lda bhash+48,y
        sec
        sbc #1
        sta bhash+48,y
        tya
        clc
        adc #176
        jsr note_dirty_sub

@new_sfx:
        ; New group 1: current last char = ctx[len-1]
        ldx ctxlen
        dex
        lda ctx,x
        sta tc2
        and #7
        tay
        lda bhash+32,y
        clc
        adc #1
        sta bhash+32,y
        tya
        clc
        adc #160
        jsr note_dirty_add

        ; New group 4: current generation count
        lda gencnt
        and #7
        tay
        lda bhash+56,y
        clc
        adc #1
        sta bhash+56,y
        tya
        clc
        adc #184
        jsr note_dirty_add

        lda gencnt
        cmp #2
        bcc @dirty_done

        ; New group 2: current previous char = ctx[len-2]
        ldx ctxlen
        dex
        dex
        lda ctx,x
        sta tc1
        and #7
        tay
        lda bhash+40,y
        clc
        adc #1
        sta bhash+40,y
        tya
        clc
        adc #168
        jsr note_dirty_add

        ; New group 3: delta(new_last - new_prev)
        lda tc2
        sec
        sbc tc1
        and #7
        tay
        lda bhash+48,y
        clc
        adc #1
        sta bhash+48,y
        tya
        clc
        adc #176
        jsr note_dirty_add

@dirty_done:
        rts

apply_l1_dirty_deltas:
        lda #0
        sta wcnt
@ald_loop:
        ldx wcnt
        cpx dirty_count
        beq @ald_done
        lda dirty_sgn,x
        sta nibsel2              ; 0=add feature, 1=sub feature
        lda dirty_off,x
        lsr
        lsr
        clc
        adc #<(md + W1_OFF)
        sta wptr
        lda #0
        adc #>(md + W1_OFF)
        sta wptr+1
        lda dirty_off,x
        and #3
        sta nibble_sel
        lda #<preact
        sta optr
        lda #>preact
        sta optr+1
        jsr apply_l1_delta_col
        inc wcnt
        jmp @ald_loop
@ald_done:
        rts

apply_l1_delta_col:
        ldx #0
@adc_loop:
        ldy #0
        lda (wptr),y
        bne @adc_havew
        jmp @adc_skip
@adc_havew:
        sta wbyte
        lda nibble_sel
        beq @adc_p0
        cmp #1
        beq @adc_p1
        cmp #2
        beq @adc_p2
        ldy wbyte
        lda twob3,y
        jmp @adc_got
@adc_p2:
        ldy wbyte
        lda twob2,y
        jmp @adc_got
@adc_p1:
        ldy wbyte
        lda twob1,y
        jmp @adc_got
@adc_p0:
        lda wbyte
        and #$03
@adc_got:
        tay                        ; code 0..3
        bne @adc_havecode
        jmp @adc_skip
@adc_havecode:
        lda nibsel2
        beq @adc_add_bucket
        ; feature delta = -1, reverse signs
        cpy #1
        beq @adc_add1
        cpy #2
        beq @adc_sub1
        jmp @adc_add2
@adc_add_bucket:
        cpy #1
        beq @adc_sub1
        cpy #2
        beq @adc_add1
        jmp @adc_sub2
@adc_add1:
        ldy #0
        clc
        lda (optr),y
        adc #1
        sta (optr),y
        iny
        lda (optr),y
        adc #0
        sta (optr),y
        jmp @adc_skip
@adc_sub1:
        ldy #0
        sec
        lda (optr),y
        sbc #1
        sta (optr),y
        iny
        lda (optr),y
        sbc #0
        sta (optr),y
        jmp @adc_skip
@adc_add2:
        ldy #0
        clc
        lda (optr),y
        adc #2
        sta (optr),y
        iny
        lda (optr),y
        adc #0
        sta (optr),y
        jmp @adc_skip
@adc_sub2:
        ldy #0
        sec
        lda (optr),y
        sbc #2
        sta (optr),y
        iny
        lda (optr),y
        sbc #0
        sta (optr),y
@adc_skip:
        clc
        lda optr
        adc #2
        sta optr
        bcc @adc_np
        inc optr+1
@adc_np:
        clc
        lda wptr
        adc #L1_WHALF
        sta wptr
        bcc @adc_nw
        inc wptr+1
@adc_nw:
        inx
        beq @adc_done
        jmp @adc_loop
@adc_done:
        rts

relu_from_preact:
        lda #<preact
        sta optr
        lda #>preact
        sta optr+1
        ldx #0
@rfp_loop:
        ldy #1
        lda (optr),y
        bmi @rfp_zero
        bne @rfp_max
        dey
        lda (optr),y
        jmp @rfp_store
@rfp_zero:
        lda #0
        jmp @rfp_store
@rfp_max:
        lda #255
@rfp_store:
        sta hidbuf,x
        clc
        lda optr
        adc #2
        sta optr
        bcc @rfp_np
        inc optr+1
@rfp_np:
        inx
        bne @rfp_loop
        rts

; =====================================================
; VBI cursor blinker (deferred VBI)
; Toggles inverse bit at cursor position every 8 frames
; =====================================================
vbi_cursor:
        lda vbi_flag
        beq @vx
        dec vbi_cnt
        bne @vx
        lda #8
        sta vbi_cnt
        ldy #0
        lda ($5E),y      ; OLDADR = screen byte at cursor
        eor #$80         ; toggle inverse bit
        sta ($5E),y
@vx:    jmp (old_vbi)    ; chain to previous deferred VBI

; =====================================================
; RODATA
; =====================================================
; =====================================================
; SPARSE: build non-zero list + sparse cneuron
; =====================================================

; Scan thash+shash+bhash (N_IN=192 bytes) for non-zero entries
; Scan contiguous thash+bhash input buffer (N_IN bytes) for non-zero entries.
; For INT2 W1 we store byte offset = pos >> 2 and pair index = pos & 3.
build_nz_list:
        lda #0
        sta nz_count
        tax
@bn:    lda thash,x
        beq @bns
        ldy nz_count
        cpy #64
        bcs @bnf
        sta nz_val,y
        txa
        lsr
        lsr
        sta nz_off,y
        txa
        and #3
        sta nz_nib,y
        inc nz_count
@bns:   inx
        cpx #N_IN
        bne @bn
@bnf:   rts

; Scan hidbuf (N_HID=256 bytes) for non-zero entries
build_nz_list_hid:
        lda #0
        sta nz_count
        tax
@bh:    lda hidbuf,x
        beq @bhs
        ldy nz_count
        cpy #64
        bcs @bhs
        txa
        lsr
        sta nz_off,y
        txa
        and #1
        sta nz_nib,y
        lda hidbuf,x
        sta nz_val,y
        inc nz_count
@bhs:   inx
        bne @bh
        rts

; Sparse INT2 cneuron for compact W1
; Uses nz_off = byte offset, nz_nib = pair index 0..3, nz_val = activation
cneuron2_sparse:
        lda #0
        sta acc_lo
        sta acc_hi
        sta wcnt
@sl2:   ldx wcnt
        cpx nz_count
        beq @sd2
        ldy nz_off,x
        lda (wptr),y
        beq @s2nxt
        sta wbyte
        ldy nz_nib,x
        beq @s2p0
        cpy #1
        beq @s2p1
        cpy #2
        beq @s2p2
        ldy wbyte
        lda twob3,y
        jmp @s2got
@s2p2:  ldy wbyte
        lda twob2,y
        jmp @s2got
@s2p1:  ldy wbyte
        lda twob1,y
        jmp @s2got
@s2p0:  lda wbyte
        and #$03
@s2got: tay
        beq @s2nxt
        lda nz_val,x
        sta act
        cpy #1
        beq @s2sub
        cpy #2
        beq @s2add
        lda act
        asl
        sta prod_lo
        lda #0
        rol
        sta prod_hi
        sec
        lda acc_lo
        sbc prod_lo
        sta acc_lo
        lda acc_hi
        sbc prod_hi
        sta acc_hi
        jmp @s2nxt
@s2sub: sec
        lda acc_lo
        sbc act
        sta acc_lo
        lda acc_hi
        sbc #0
        sta acc_hi
        jmp @s2nxt
@s2add: clc
        lda acc_lo
        adc act
        sta acc_lo
        lda acc_hi
        adc #0
        sta acc_hi
@s2nxt: inc wcnt
        jmp @sl2
@sd2:   clc
        lda wptr
        adc #L1_WHALF
        sta wptr
        bcc @sd2r
        inc wptr+1
@sd2r:  rts

; Sparse cneuron: MAC using only non-zero inputs
; Reads weights via (wptr),Y with Y = byte offset
cneuron_sparse:
        lda #0
        sta acc_lo
        sta acc_hi
        sta wcnt
@sl:    ldx wcnt
        cpx nz_count
        bne @sc
        jmp @sd
@sc:
        ; Read weight byte at wptr + nz_off[x]
        ldy nz_off,x
        lda (wptr),y
        bne @snx_cont
        jmp @snx
@snx_cont:
        ; Extract nibble
        ldy nz_nib,x
        beq @slo
        tay
        lda hinib,y
@slo:   and #$0F
        bne @snz
        jmp @snx
@snz:
        ; A = unsigned nibble 0-15. Multiply by nz_val[x]
        stx tmp           ; save list index
        sta mag           ; weight nibble
        ldx tmp
        lda nz_val,x
        sta act           ; activation value

        ; Determine sign: positive (0-7) or negative (8-15)
        lda mag
        cmp #8
        bcs @sn

        ; --- Positive weight ---
        ; Product = mag * act
        lda mag
        cmp #1
        beq @sp1
        cmp #2
        beq @sp2
        cmp #3
        beq @sp3
        ; w >= 4: general multiply
        jsr mul
        jmp @spadd
@sp1:   lda act
        sta prod_lo
        lda #0
        sta prod_hi
        jmp @spadd
@sp2:   lda act
        asl
        sta prod_lo
        lda #0
        rol
        sta prod_hi
        jmp @spadd
@sp3:   lda act
        sta prod_lo
        asl
        clc
        adc prod_lo
        sta prod_lo
        lda #0
        adc #0
        sta prod_hi
        jmp @spadd
@spadd: ; Add prod to acc
        clc
        lda acc_lo
        adc prod_lo
        sta acc_lo
        lda acc_hi
        adc prod_hi
        sta acc_hi
        jmp @snx2

@sn:    ; --- Negative weight (8-15) -> abs = 16-mag ---
        lda mag
        eor #$0F
        clc
        adc #1
        sta mag           ; now positive magnitude
        ; Product = mag * act
        lda mag
        cmp #1
        beq @sn1
        cmp #2
        beq @sn2
        cmp #3
        beq @sn3
        jsr mul
        jmp @snsub
@sn1:   lda act
        sta prod_lo
        lda #0
        sta prod_hi
        jmp @snsub
@sn2:   lda act
        asl
        sta prod_lo
        lda #0
        rol
        sta prod_hi
        jmp @snsub
@sn3:   lda act
        sta prod_lo
        asl
        clc
        adc prod_lo
        sta prod_lo
        lda #0
        adc #0
        sta prod_hi
        jmp @snsub
@snsub: ; Sub prod from acc
        sec
        lda acc_lo
        sbc prod_lo
        sta acc_lo
        lda acc_hi
        sbc prod_hi
        sta acc_hi

@snx2:  ldx tmp           ; restore list index
@snx:   inc wcnt
        jmp @sl
@sd:    rts


.segment "RODATA"
; High nibble LUT (same as JAM)
hinib:
.repeat 256, i
        .byte (i >> 4) .mod 16
.endrepeat

twob1:
.repeat 256, i
        .byte (i >> 2) .mod 4
.endrepeat

twob2:
.repeat 256, i
        .byte (i >> 4) .mod 4
.endrepeat

twob3:
.repeat 256, i
        .byte (i >> 6) .mod 4
.endrepeat

; CHARSET: " ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-,!?'/"
; Index 0=space, 1-26=A-Z, 27-36=0-9, 37='.', 38='-', 39=','
; 40='!', 41='?', 42="'", 43='/', 44=EOL
us0:    .byte "JUST ASK"
us0l = * - us0
us1:    .byte "NOT SURE"
us1l = * - us1
us2:    .byte "HMM"
us2l = * - us2
us3:    .byte "WHAT"
us3l = * - us3
us4:    .byte "TRY AGAIN"
us4l = * - us4
us5:    .byte "I JAM"
us5l = * - us5
us6:    .byte "WHATEVER"
us6l = * - us6
us7:    .byte "SO CONFUSING"
us7l = * - us7
us_ptrs: .word us0, us1, us2, us3, us4, us5, us6, us7
us_lens: .byte us0l, us1l, us2l, us3l, us4l, us5l, us6l, us7l
charset:
        .byte " ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        .byte "0123456789.-,!?'/"

; === Theme: auto-generated from training data ===
.include "theme.inc"

fb0:    .byte "TRY SOMETHING ELSE"
fb0l = * - fb0
fb1:    .byte "BEATS ME"
fb1l = * - fb1
fb2:    .byte "NOT IN MY 30 KB"
fb2l = * - fb2
fb_ptrs: .word fb0, fb1, fb2
fb_lens: .byte fb0l, fb1l, fb2l

prm:    .byte "> "
prml = * - prm

; === Weights ===
; sug0-sug7, sug_ptrs, sug_lens: now in theme.inc

md:     .incbin "weights_b2s.bin"

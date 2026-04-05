; PICO SAM - Tiny speaking language model
; picojam + formant speech synthesis via POKEY
; 96 -> 64 -> 45. W1: 1-bit, W2: INT2.
; Marek Spanel, 2026. jam.ag

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
RANDOM  = $D20A
CH      = $02FC

; POKEY audio
POKEY   = $D200
AUDF1   = POKEY + 0
AUDC1   = POKEY + 1
AUDF2   = POKEY + 2
AUDC2   = POKEY + 3
AUDF3   = POKEY + 4
AUDC3   = POKEY + 5
AUDF4   = POKEY + 6
AUDC4   = POKEY + 7
AUDCTL  = POKEY + 8

; SAM phoneme indices
PH_SIL  = 0     ; silence
PH_AA   = 1     ; bat
PH_AE   = 2     ; bait
PH_AH   = 3     ; bot
PH_AO   = 4     ; bought
PH_EH   = 5     ; bet
PH_ER   = 6     ; bird
PH_IH   = 7     ; bit
PH_IY   = 8     ; beat
PH_OH   = 9     ; boat
PH_UH   = 10    ; book
PH_UW   = 11    ; boot
PH_B    = 12
PH_D    = 13
PH_F    = 14
PH_G    = 15
PH_H    = 16
PH_J    = 17
PH_K    = 18
PH_L    = 19
PH_M    = 20
PH_N    = 21
PH_P    = 22
PH_R    = 23
PH_S    = 24
PH_T    = 25
PH_V    = 26
PH_W    = 27
PH_Y    = 28
PH_Z    = 29
PH_CH   = 30
PH_SH   = 31
PH_TH   = 32
PH_NG   = 33
PH_STOP = 34    ; glottal stop / word break
N_PHON  = 35

N_IN     = 96
N_HID    = 64
N_OUT    = 45
W1_STRIDE = 12     ; 96/8 bytes per row (1-bit)
W2_STRIDE = 16     ; 64/4 bytes per row (INT2)
SEP_CHAR = $3E
MAX_GEN  = 15
MAX_INPUT = 4
EOL_IDX  = 44

W1_OFF   = 0       ; 64 * 12 = 768
W2_OFF   = 768     ; 45 * 16 = 720
B1_OFF   = 1488    ; 64
B2_OFF   = 1552    ; 45

.segment "ZEROPAGE"
acc_lo: .res 1
acc_hi: .res 1
wbyte:  .res 1
wptr:   .res 2
bptr:   .res 2
optr:   .res 2
aptr:   .res 2
wcnt:   .res 1
ocnt:   .res 1
llen:   .res 1
blo:    .res 1
bhi:    .res 1
besti:  .res 1
tmp:    .res 1
second: .res 1
slo:    .res 1
shi:    .res 1
inv_len: .res 1
nz_src: .res 2      ; pointer to source buffer (thash or hidbuf)
nz_max: .res 1      ; scan limit
stride: .res 1      ; weight row stride for generic cneuron

.segment "BSS"
iline:  .res 42
co:     .res 1
gencnt: .res 1
lastch: .res 1
repchr: .res 1
answer_ok: .res 1
ctxlen: .res 1
tc0:    .res 1
tc1:    .res 1
tc2:    .res 1
first_turn: .res 1
dream_flag: .res 1
dream_chain: .res 1  ; how many chains so far
hint_mode: .res 1    ; 1=running neural hint (answer->query)
ctx:    .res 80
thash:  .res 96
hidbuf: .res 64
frozen_thash: .res 96
frozen_len: .res 1      ; ctxlen at freeze time (query+SEP length)
obuf:   .res 90
nz_count: .res 1
nz_idx:  .res 48    ; input index (0-95 or 0-127)
nz_val:  .res 48    ; activation value

; === SAM speech engine state ===
sam_queue:  .res 16  ; phoneme ring buffer
sam_qhead:  .res 1   ; write index
sam_qtail:  .res 1   ; read index
sam_f1:     .res 1   ; current F1 frequency
sam_f2:     .res 1   ; current F2 frequency
sam_amp:    .res 1   ; current amplitude
sam_dur:    .res 1   ; frames remaining for current phoneme
sam_noise:  .res 1   ; noise freq for unvoiced (0=off)
sam_durinit:.res 1   ; initial duration (for envelope calc)
sam_active: .res 1   ; 1=speaking, 0=idle
sam_prevch: .res 1   ; previous character for digraph detection

.segment "CODE"
start:
        ldx #0
        lda #0
@cb:    sta $0700,x
        sta $0800,x
        sta $0900,x
        sta $0A00,x
        inx
        bne @cb

        lda #1
        sta first_turn

        jsr sam_init

        ; Install SAM VBI handler (deferred VBI)
        ldy #<sam_vbi
        ldx #>sam_vbi
        lda #7              ; deferred VBI
        jsr $E45C           ; SETVBV

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
        lda #<prm
        ldx #>prm
        ldy #prml
        jsr ps
        jsr rline_timeout
        bcc got_input
        jsr idle_chatter
got_input:
        ; Strip trailing punctuation
@strip: ldx llen
        beq mloop
        dex
        lda iline,x
        cmp #63
        beq @cut
        cmp #33
        beq @cut
        cmp #46
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
        ldx #0
@cpq:   cpx llen
        beq @qd
        lda iline,x
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

        jsr thash_init_frozen   ; freeze query+SEP hash for incremental L1

        ; Context decay: first turn = full clear, then decay
        lda first_turn
        beq @ctx_decay
        jsr thash_clear     ; first interaction: clean slate
        lda #0
        sta first_turn
        jmp @ctx_done
@ctx_decay:
        jsr thash_decay     ; subsequent: old context fades
@ctx_done:

        ldx ctxlen
        dex
        lda ctx,x
        sta tc2
        dex
        lda ctx,x
        sta tc1
        dex
        bmi @tc_ok
        lda ctx,x
        sta tc0
@tc_ok:
        lda #0
        sta gencnt
        sta repchr
        lda #$FF
        sta lastch
        ; Dream prefix "... " or normal " "
        lda dream_flag
        beq @npfx
        lda #'.'
        sta co
        jsr pc
        jsr pc
        jsr pc
        lda #' '
        sta co
        jsr pc
        jmp @pfxd
@npfx:  lda #' '
        sta co
        jsr pc
@pfxd:  lda #1
        sta answer_ok

gen_loop:
        ; === INCREMENTAL HASH (frozen + live delta) ===
        ldx #0
@copyf: lda frozen_thash,x
        sta thash,x
        inx
        cpx #96
        bne @copyf
        jsr thash_add_live_delta

        ; === L1: 1-bit sparse ===
        jsr build_nz_thash
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
        lda #N_HID
        sta ocnt
@l1r:   jsr cneuron_1bit
        ; bias
        ldy #0
        lda (bptr),y
        ldx #0
        cmp #$80
        bcc @l1bp
        ldx #$FF
@l1bp:  clc
        adc acc_lo
        sta acc_lo
        txa
        adc acc_hi
        sta acc_hi
        ; ReLU 0-255
        lda acc_hi
        bmi @l1z
        bne @l1c
        lda acc_lo
        jmp @l1s
@l1c:   lda #255
        jmp @l1s
@l1z:   lda #0
@l1s:   ldy #0
        sta (optr),y
        inc optr
        bne @l1no
        inc optr+1
@l1no:  inc bptr
        bne @l1nb
        inc bptr+1
@l1nb:  dec ocnt
        bne @l1r

        ; === L2: INT2 sparse ===
        jsr build_nz_hidbuf
        lda #<(md + W2_OFF)
        sta wptr
        lda #>(md + W2_OFF)
        sta wptr+1
        lda #<(md + B2_OFF)
        sta bptr
        lda #>(md + B2_OFF)
        sta bptr+1
        lda #<obuf
        sta optr
        lda #>obuf
        sta optr+1
        lda #N_OUT
        sta ocnt
@l2r:   jsr cneuron_int2
        ; bias
        ldy #0
        lda (bptr),y
        ldx #0
        cmp #$80
        bcc @l2bp
        ldx #$FF
@l2bp:  clc
        adc acc_lo
        sta acc_lo
        txa
        adc acc_hi
        sta acc_hi
        ; store 16-bit
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
        bcc @l2no
        inc optr+1
@l2no:  inc bptr
        bne @l2nb
        inc bptr+1
@l2nb:  dec ocnt
        bne @l2r

        ; === Algorithmic EOL suppression ===
        lda gencnt
        cmp #3
        bcs @eol_mid
        ; pos < 3: subtract 40 from EOL logit
        sec
        lda obuf + EOL_IDX * 2
        sbc #40
        sta obuf + EOL_IDX * 2
        lda obuf + EOL_IDX * 2 + 1
        sbc #0
        sta obuf + EOL_IDX * 2 + 1
        jmp @eol_done
@eol_mid:
        cmp #8
        bcs @eol_late
        ; pos 3-7: subtract 15
        sec
        lda obuf + EOL_IDX * 2
        sbc #15
        sta obuf + EOL_IDX * 2
        lda obuf + EOL_IDX * 2 + 1
        sbc #0
        sta obuf + EOL_IDX * 2 + 1
        jmp @eol_done
@eol_late:
        ; pos 12+: progressive EOL boost
        cmp #12
        bcc @eol_done
        sec
        sbc #12
        asl
        asl
        asl
        asl
        clc
        adc #16
        clc
        adc obuf + EOL_IDX * 2
        sta obuf + EOL_IDX * 2
        lda #0
        adc obuf + EOL_IDX * 2 + 1
        sta obuf + EOL_IDX * 2 + 1
@eol_done:

        jsr amax45

        ; Confidence (first char only)
        lda gencnt
        bne @conf_ok
        sec
        lda blo
        sbc slo
        sta tmp
        lda bhi
        sbc shi
        bne @conf_ok
        lda tmp
        cmp #3
        bcs @conf_ok
        jmp gen_unsure
@conf_ok:
        lda besti
        cmp #EOL_IDX
        bne @not_eol
        jmp gen_done
@not_eol:
        ldx besti
        lda charset,x
        sta co
        cmp lastch
        bne @newch
        inc repchr
        lda repchr
        cmp #4           ; allow 3 repeats (e.g. "000" in bin 8)
        bcc @printch
        jmp gen_fallback
@newch: sta lastch
        lda #1
        sta repchr
@printch:
        ; Thinking effect (margin < 8: flash second)
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
        ora #$80
        sta co
        jsr pc
        ldx #3
@twait: lda $14
@tw2:   cmp $14
        beq @tw2
        dex
        bne @twait
        lda #$7E
        sta co
        jsr pc
        ldx besti
        lda charset,x
        sta co
@no_think:
        ; Dream: inverse output + delay + keycheck
        lda dream_flag
        beq @no_dinv
        lda co
        ora #$80
        sta co
        jsr pc
        lda $14
@dw1:   cmp $14
        beq @dw1
        lda $14
@dw2:   cmp $14
        beq @dw2
        lda CH
        cmp #$FF
        bne @dream_abort
        jmp @after_print
@dream_abort:
        lda #EOL
        sta co
        jsr pc
        jmp mloop
@no_dinv:
        jsr pc
        ; === Feed character to SAM speech engine ===
        lda co
        and #$7F            ; strip inverse bit
        jsr sam_speak_char
@after_print:
        lda co
        and #$7F
        ldx ctxlen
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
        lda dream_flag
        beq @not_dream_u
        jmp gen_done
@not_dream_u:
        lda hint_mode
        beq @not_hint_u
        jmp gen_done_hint
@not_hint_u:
        lda #0
        sta answer_ok
        lda $14
        and #3
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
        lda dream_flag
        beq @not_dream_f
        jmp gen_done
@not_dream_f:
        lda hint_mode
        beq @not_hint_f
        jmp gen_done_hint
@not_hint_f:
        lda #0
        sta answer_ok
        ldx repchr
        dex
@bksp:  lda #$7E
        sta co
        txa
        pha
        jsr pc
        pla
        tax
        dex
        bne @bksp
        lda #EOL
        sta co
        jsr pc
        lda #$20
        sta co
        jsr pc
        lda $14
        and #1
        pha
        asl
        tax
        lda fb_ptrs,x
        sta aptr
        lda fb_ptrs+1,x
        sta aptr+1
        pla
        tax
        lda fb_lens,x
        tay
        jsr ps_inv

gen_done_hint:
        lda #0
        sta hint_mode
        lda #EOL
        sta co
        jsr pc
        jmp mloop

gen_done:
        lda #EOL
        sta co
        jsr pc
        lda dream_flag
        beq @not_dreaming
        ; === Dream chaining: feed answer back as next query ===
        ; Find separator in ctx, copy everything after it to iline
        ldx #0
@dfs:   lda ctx,x
        cmp #SEP_CHAR
        beq @dfound
        inx
        cpx ctxlen
        bne @dfs
        jmp @dream_end       ; no separator? stop
@dfound:
        inx                  ; skip '>'
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
        ; Stop if answer was empty or too short
        cpy #2
        bcc @dream_end
        ; Max 3 chains
        inc dream_chain
        lda dream_chain
        cmp #3
        bcs @dream_end
        ; Pause 1-2 seconds between dream chains
        ldx #60              ; ~60 frames = 1.2 sec
@dpause:
        lda $14
@dpw:   cmp $14
        beq @dpw
        ; Check keypress during pause
        lda CH
        cmp #$FF
        bne @dream_end
        dex
        bne @dpause
        ; Chain: go back to process iline as new query
        jmp got_input
@dream_end:
        lda #0
        sta dream_flag
        jmp mloop

@not_dreaming:
        lda answer_ok
        beq @no_sug
        lda gencnt
        cmp #7
        bcs @no_sug
        lda $14
        and #7
        cmp #3
        bcs @no_sug
        lda $14
        lsr
        lsr
        lsr
        and #7
        pha
        lda #$20
        sta co
        jsr pc
        jsr pc
        pla
        pha
        asl
        tax
        lda sug_ptrs,x
        sta aptr
        lda sug_ptrs+1,x
        sta aptr+1
        pla
        tax
        lda sug_lens,x
        tay
        jsr ps_inv
        lda #EOL
        sta co
        jsr pc
@no_sug:
        ; === Neural Hint: 25% chance to feed answer back as query ===
        lda hint_mode
        bne @hint_skip       ; no recursion
        lda answer_ok
        beq @hint_skip       ; skip after fallback
        lda gencnt
        cmp #2
        bcc @hint_skip       ; skip 1-char answers
        lda RANDOM
        and #$03
        bne @hint_skip       ; 25% chance

        ; Copy answer from ctx (after '>') to iline
        ldx #0
@hfs:   lda ctx,x
        cmp #SEP_CHAR
        beq @hfound
        inx
        cpx ctxlen
        bne @hfs
        jmp @hint_skip       ; no separator
@hfound:
        inx                  ; skip '>'
        ldy #0
@hcopy: cpx ctxlen
        beq @hend
        lda ctx,x
        sta iline,y
        inx
        iny
        cpy #MAX_INPUT
        bcc @hcopy
@hend:  lda #0
        sta iline,y
        sty llen
        cpy #1
        bcc @hint_skip       ; empty answer
        ; Setup hint mode
        lda #1
        sta hint_mode
        ; Print leading space
        lda #' '
        sta co
        jsr pc
        jmp got_input        ; process answer as new query
@hint_skip:
        lda #0
        sta hint_mode
        jmp mloop

; =====================================================
; RLINE WITH TIMEOUT (15-35 sec)
; =====================================================
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
        adc #<750
        sta tmp
        lda wbyte
        adc #>750
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

; =====================================================
; DREAM: pick random suggestion keyword, feed as query
; =====================================================
idle_chatter:
        lda RANDOM
        and #7
        pha
        asl
        tax
        lda sug_ptrs,x
        sta aptr
        lda sug_ptrs+1,x
        sta aptr+1
        pla
        tax
        lda sug_lens,x
        sec
        sbc #4
        sta llen
        ldy #4
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
        clc
        rts

; =====================================================
; HASH
; =====================================================
; =====================================================
; TRIGRAM HASH SUBROUTINE (shared by full hash and incremental)
; Input: tc0, tc1, tc2
; Output: increments thash[idx] where idx = (tc0*31 + tc1*7 + tc2) & $3F
; =====================================================
tri_one:
        lda tc0
        asl
        asl
        asl
        asl
        asl
        sec
        sbc tc0
        sta acc_lo
        lda tc1
        asl
        asl
        asl
        sec
        sbc tc1
        clc
        adc acc_lo
        clc
        adc tc2
        and #$3F
        tax
        inc thash,x
        rts

; Same as tri_one but increments frozen_thash instead of thash
tri_one_frozen:
        lda tc0
        asl
        asl
        asl
        asl
        asl
        sec
        sbc tc0
        sta acc_lo
        lda tc1
        asl
        asl
        asl
        sec
        sbc tc1
        clc
        adc acc_lo
        clc
        adc tc2
        and #$3F
        tax
        inc frozen_thash,x
        rts

; =====================================================
; INCREMENTAL HASH - FROZEN + LIVE DELTA (bit-exact)
; =====================================================

; Compute frozen_thash from query+SEP (called once after got_input)
thash_init_frozen:
        lda ctxlen
        sta frozen_len          ; remember query+SEP length
        ldx #0
        lda #0
@clr:   sta frozen_thash,x
        inx
        cpx #96
        bne @clr

        ; Trigrams over query+SEP
        lda ctxlen
        cmp #3
        bcc @bagq
        ldx #0
@loop:  lda ctx,x
        sta tc0
        lda ctx+1,x
        sta tc1
        lda ctx+2,x
        sta tc2
        txa
        pha
        jsr tri_one_frozen
        pla
        tax
        inx
        txa
        clc
        adc #2
        cmp ctxlen
        bcs @bagq
        txa
        tax
        jmp @loop

        ; Bag-of-chars over query+SEP
@bagq:  ldx #0
@bag:   cpx ctxlen
        beq @bagdone
        lda ctx,x
        and #$0F
        tay
        lda frozen_thash+64,y
        cmp #255
        beq @bsat
        clc
        adc #1
        sta frozen_thash+64,y
@bsat:  inx
        jmp @bag
@bagdone:
        rts

; Add all answer-portion contributions on top of thash
; (thash already has frozen copy; ctxlen includes answer chars)
thash_add_live_delta:
        ; Trigrams: iterate from (frozen_len - 2) to (ctxlen - 3)
        ; These are all trigrams that involve at least one answer char
        lda ctxlen
        cmp #3
        bcc @bag_start
        lda frozen_len
        sec
        sbc #2
        bcs @tri_idx
        lda #0              ; clamp to 0
@tri_idx:
        tax
@tri_loop:
        txa
        clc
        adc #2
        cmp ctxlen
        bcs @bag_start      ; i+2 >= ctxlen => done
        lda ctx,x
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
        jmp @tri_loop

        ; Bag: add entries for all answer chars (frozen_len to ctxlen-1)
@bag_start:
        ldx frozen_len
@bag:   cpx ctxlen
        beq @bag_done
        lda ctx,x
        and #$0F
        tay
        lda thash+64,y
        cmp #255
        beq @bsat2
        clc
        adc #1
        sta thash+64,y
@bsat2: inx
        jmp @bag
@bag_done:
        ; Suffix features (gencnt-dependent)
        lda gencnt
        beq @sfxd
        ldx ctxlen
        dex
        lda ctx,x
        and #7
        clc
        adc #80
        tax
        inc thash,x
        lda gencnt
        and #7
        clc
        adc #88
        tax
        inc thash,x
@sfxd:  rts

thash_clear:
        ; Full clear (first turn or dream)
        ldx #0
        lda #0
@clr:   sta thash,x
        inx
        cpx #96
        bne @clr
        rts

thash_decay:
        ; Decay: shift right by 1 (old context fades, doesn't vanish)
        ldx #0
@dec:   lda thash,x
        lsr
        sta thash,x
        inx
        cpx #96
        bne @dec
        rts

thash_hash:
        lda ctxlen
        cmp #3
        bcc @done
        ldx #0
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
        adc #2
        cmp ctxlen
        bcs @done
        txa
        tax
        jmp @loop
@done:
        ldx #0
@bag:   cpx ctxlen
        beq @bdone
        lda ctx,x
        and #$0F
        tay
        lda thash+64,y
        cmp #255
        beq @bsat
        clc
        adc #1
        sta thash+64,y
@bsat:  inx
        jmp @bag
@bdone:
        lda gencnt
        beq @sfxd
        ldx ctxlen
        dex
        lda ctx,x
        and #7
        clc
        adc #80
        tax
        inc thash,x
        lda gencnt
        and #7
        clc
        adc #88
        tax
        inc thash,x
@sfxd:  rts

; =====================================================
; BUILD NZ LISTS
; =====================================================
build_nz_thash:
        lda #0
        sta nz_count
        tax
@bn:    cpx #96
        beq @bd
        lda thash,x
        beq @bs
        ldy nz_count
        cpy #48
        bcs @bs
        txa
        sta nz_idx,y
        lda thash,x
        sta nz_val,y
        inc nz_count
@bs:    inx
        jmp @bn
@bd:    rts

build_nz_hidbuf:
        lda #0
        sta nz_count
        tax
@bn:    lda hidbuf,x
        beq @bs
        ldy nz_count
        cpy #48
        bcs @bs
        txa
        sta nz_idx,y
        lda hidbuf,x
        sta nz_val,y
        inc nz_count
@bs:    inx
        cpx #N_HID
        bne @bn
        rts

; =====================================================
; 1-BIT NEURON (W1): bit=1 -> +val, bit=0 -> -val
; nz_idx = input index, nz_val = activation
; wptr -> weight row (12 bytes = 96 bits)
; =====================================================
cneuron_1bit:
        lda #0
        sta acc_lo
        sta acc_hi
        sta wcnt
@sl:    ldx wcnt
        cpx nz_count
        beq @sd
        ; byte offset = nz_idx / 8, bit = nz_idx & 7
        lda nz_idx,x
        lsr
        lsr
        lsr              ; /8 = byte offset
        tay
        lda (wptr),y     ; weight byte
        sta wbyte
        lda nz_idx,x
        and #7           ; bit position
        tay
        lda wbyte
        ; shift right Y times to get bit into carry
        cpy #0
        beq @got
@sh:    lsr
        dey
        bne @sh
@got:   lsr              ; bit -> carry
        ldx wcnt
        bcs @add         ; bit=1: add
        ; bit=0: subtract
        sec
        lda acc_lo
        sbc nz_val,x
        sta acc_lo
        bcs @snxt
        dec acc_hi
        jmp @snxt
@add:   clc
        lda acc_lo
        adc nz_val,x
        sta acc_lo
        bcc @snxt
        inc acc_hi
@snxt:  inc wcnt
        jmp @sl
@sd:    ; Advance wptr by W1_STRIDE
        clc
        lda wptr
        adc #W1_STRIDE
        sta wptr
        bcc @nw
        inc wptr+1
@nw:    rts

; =====================================================
; INT2 NEURON (W2): 00=0, 01=-1, 10=+1, 11=-2
; nz_idx = hidden index, nz_val = activation
; =====================================================
cneuron_int2:
        lda #0
        sta acc_lo
        sta acc_hi
        sta wcnt
@sl:    ldx wcnt
        cpx nz_count
        bne @cont
        jmp @sd
@cont:
        ; byte offset = nz_idx / 4, pair = nz_idx & 3
        lda nz_idx,x
        lsr
        lsr              ; /4
        tay
        lda (wptr),y
        beq @snxt        ; all 4 zero
        sta wbyte
        lda nz_idx,x
        and #3
        beq @p0
        cmp #1
        beq @p1
        cmp #2
        beq @p2
        lda wbyte
        lsr
        lsr
        lsr
        lsr
        lsr
        lsr
        and #3
        jmp @got
@p2:    lda wbyte
        lsr
        lsr
        lsr
        lsr
        and #3
        jmp @got
@p1:    lda wbyte
        lsr
        lsr
        and #3
        jmp @got
@p0:    lda wbyte
        and #3
@got:   beq @snxt
        cmp #2
        beq @add1
        bcs @sub2
        ; code 01 = -1
        ldx wcnt
        sec
        lda acc_lo
        sbc nz_val,x
        sta acc_lo
        bcs @snxt
        dec acc_hi
        jmp @snxt
@add1:  ldx wcnt
        clc
        lda acc_lo
        adc nz_val,x
        sta acc_lo
        bcc @snxt
        inc acc_hi
        jmp @snxt
@sub2:  ldx wcnt
        lda nz_val,x
        asl
        sta tmp
        sec
        lda acc_lo
        sbc tmp
        sta acc_lo
        bcs @snxt
        dec acc_hi
@snxt:  inc wcnt
        jmp @sl
@sd:    clc
        lda wptr
        adc #W2_STRIDE
        sta wptr
        bcc @nw
        inc wptr+1
@nw:    rts

; =====================================================
; ARGMAX 45
; =====================================================
amax45:
        lda obuf
        sta blo
        lda obuf+1
        sta bhi
        lda #0
        sta besti
        sta slo
        lda #$80
        sta shi
        sta second
        ldx #1
@am:    txa
        asl
        tay
        sec
        lda obuf,y
        sbc blo
        lda obuf+1,y
        sbc bhi
        bvc @amc
        eor #$80
@amc:   bmi @nxt
        lda blo
        sta slo
        lda bhi
        sta shi
        lda besti
        sta second
        txa
        asl
        tay
        lda obuf,y
        sta blo
        lda obuf+1,y
        sta bhi
        stx besti
        jmp @anxt
@nxt:   txa
        asl
        tay
        sec
        lda obuf,y
        sbc slo
        lda obuf+1,y
        sbc shi
        bvc @asc
        eor #$80
@asc:   bmi @anxt
        txa
        asl
        tay
        lda obuf,y
        sta slo
        lda obuf+1,y
        sta shi
        stx second
@anxt:  inx
        cpx #N_OUT
        bne @am
        rts

; =====================================================
; I/O
; =====================================================
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

ps_inv:
        sty inv_len
        ldy #0
@lp:    cpy inv_len
        beq @done
        lda (aptr),y
        ora #$80
        sta co
        tya
        pha
        jsr pc
        pla
        tay
        iny
        jmp @lp
@done:  rts

; =====================================================
.segment "RODATA"

charset:
        .byte " abcdefghijklmnopqrstuvwxyz"
        .byte "0123456789.-,!?'/"

.include "theme_pico.inc"

us0:    .byte "just ask"
us0l = * - us0
us1:    .byte "not sure"
us1l = * - us1
us2:    .byte "hmm"
us2l = * - us2
us3:    .byte "try again"
us3l = * - us3
us_ptrs: .word us0, us1, us2, us3
us_lens: .byte us0l, us1l, us2l, us3l

fb0:    .byte "beats me"
fb0l = * - fb0
fb1:    .byte "not in my 3 kb"
fb1l = * - fb1
fb_ptrs: .word fb0, fb1
fb_lens: .byte fb0l, fb1l

; =====================================================
; SAM SPEECH ENGINE - formant synthesis via POKEY
; =====================================================

; SAM VBI handler (runs every frame, 50/60 Hz)
sam_vbi:
        jsr sam_tick
        jmp $E462           ; chain to OS default deferred VBI (XITVBV)

; Initialize SAM engine (call once at startup)
sam_init:
        lda #0
        sta sam_qhead
        sta sam_qtail
        sta sam_active
        sta sam_f1
        sta sam_f2
        sta sam_amp
        sta sam_dur
        sta sam_noise
        sta sam_prevch
        sta AUDCTL          ; POKEY: normal mode
        sta AUDC1           ; ch1: off
        sta AUDC2           ; ch2: off
        sta AUDC3           ; ch3: off
        rts

; Queue a phoneme (A = phoneme index)
sam_enqueue:
        ldx sam_qhead
        sta sam_queue,x
        inx
        txa
        and #$0F            ; ring buffer mask (16 entries)
        sta sam_qhead
        rts

; Text→phoneme: convert character in A to phoneme(s) and enqueue
; Handles English letter→phoneme mapping + digraphs (TH,SH,CH,WH,PH)
sam_speak_char:
        pha                  ; save raw char
        ; Convert to uppercase if lowercase
        cmp #$61
        bcc @notlower
        cmp #$7B
        bcs @notlower
        and #$DF            ; to uppercase
@notlower:
        ; Space / punctuation
        cmp #$20
        bne @n1
        jmp @silence
@n1:    cmp #$2E            ; '.'
        bne @n2
        jmp @pause
@n2:    cmp #$2C            ; ','
        bne @n3
        jmp @pause
@n3:    cmp #$21            ; '!'
        bne @n4
        jmp @silence
@n4:    cmp #$3F            ; '?'
        bne @n5
        jmp @question
@n5:
        ; Digits 0-9
        cmp #$30
        bcc @tryalpha
        cmp #$3A
        bcs @tryalpha
        sec
        sbc #$30
        tax
        lda digit_phoneme,x
        jsr sam_enqueue
        jmp @saveprev
@tryalpha:
        ; Letters A-Z
        cmp #$41
        bcs @aboveA
        jmp @done
@aboveA:
        cmp #$5B
        bcc @isalpha
        jmp @done
@isalpha:
        ; Check digraphs: is previous char + this char a digraph?
        pha                  ; save current uppercase
        lda sam_prevch
        cmp #$54            ; prev='T'?
        bne @notT
        pla
        cmp #$48            ; cur='H'? → TH digraph
        bne @nodigraph
        lda #PH_TH
        jsr sam_enqueue
        jmp @saveprev
@notT:  cmp #$53            ; prev='S'?
        bne @notS
        pla
        cmp #$48            ; cur='H'? → SH digraph
        bne @nodigraph
        lda #PH_SH
        jsr sam_enqueue
        jmp @saveprev
@notS:  cmp #$43            ; prev='C'?
        bne @notC
        pla
        cmp #$48            ; cur='H'? → CH digraph
        bne @nodigraph
        lda #PH_CH
        jsr sam_enqueue
        jmp @saveprev
@notC:  cmp #$57            ; prev='W'?
        bne @notW
        pla
        cmp #$48            ; cur='H'? → WH (just W)
        bne @nodigraph
        lda #PH_W
        jsr sam_enqueue
        jmp @saveprev
@notW:  cmp #$50            ; prev='P'?
        bne @notdig
        pla
        cmp #$48            ; cur='H'? → PH = F
        bne @nodigraph
        lda #PH_F
        jsr sam_enqueue
        jmp @saveprev
@notdig:
        pla                  ; restore current char (no digraph match)
@nodigraph:
        ; Normal letter lookup
        sec
        sbc #$41
        tax
        lda letter_phoneme,x
        jsr sam_enqueue
        lda letter_vowel,x
        beq @saveprev
        jsr sam_enqueue
        jmp @saveprev
@silence:
        lda #PH_SIL
        jsr sam_enqueue
        jmp @saveprev
@pause:
        lda #PH_SIL
        jsr sam_enqueue
        jsr sam_enqueue      ; double silence for period/comma
        jmp @saveprev
@question:
        ; Rising intonation: queue two high-pitched phonemes
        lda #PH_IY
        jsr sam_enqueue
        lda #PH_SIL
        jsr sam_enqueue
        jmp @saveprev
@done:
        pla
        sta sam_prevch
        rts
@saveprev:
        pla                  ; restore raw char
        ; Convert to uppercase for prev tracking
        cmp #$61
        bcc @sp2
        cmp #$7B
        bcs @sp2
        and #$DF
@sp2:   sta sam_prevch
        rts

; VBI-driven SAM tick: call every frame to advance synthesis
; Ch1=F1, Ch2=F2 (pure tone), Ch3=noise (unvoiced consonants)
; Includes volume envelope and vowel vibrato.
sam_tick:
        lda sam_dur
        bne @playing
        ; Load next phoneme from queue
        lda sam_qtail
        cmp sam_qhead
        beq @idle           ; queue empty
        tax
        lda sam_queue,x
        inx
        txa
        and #$0F
        sta sam_qtail
        ; A = phoneme index, look up formants
        tax
        lda sam_f1_tbl,x
        sta sam_f1
        lda sam_f2_tbl,x
        sta sam_f2
        lda sam_amp_tbl,x
        sta sam_amp
        lda sam_noise_tbl,x
        sta sam_noise
        lda sam_dur_tbl,x
        sta sam_dur
        sta sam_durinit      ; save for envelope
        lda #1
        sta sam_active
        jmp @output
@idle:
        lda #0
        sta sam_active
        sta AUDC1
        sta AUDC2
        sta AUDC3
        rts
@playing:
        dec sam_dur
@output:
        ; === Volume envelope: ramp up first 2 frames, ramp down last 2 ===
        lda sam_amp
        sta tmp              ; base amplitude
        ; Ramp up: if (durinit - dur) < 2, halve volume
        lda sam_durinit
        sec
        sbc sam_dur          ; = elapsed frames
        cmp #2
        bcs @no_rampup
        lsr tmp              ; half volume during attack
@no_rampup:
        ; Ramp down: if dur < 2, halve volume
        lda sam_dur
        cmp #2
        bcs @no_rampdn
        lsr tmp              ; half volume during release
@no_rampdn:

        ; === F1 + vibrato → POKEY channel 1 ===
        lda sam_f1
        beq @f1off
        ; Vowel vibrato: wiggle F1 by +/-1 every 4 frames
        pha
        lda $14              ; RTCLOK frame counter
        and #$04             ; toggle every 4 frames
        beq @vib_up
        pla
        sec
        sbc #1               ; F1 - 1
        jmp @f1set
@vib_up:
        pla
        clc
        adc #1               ; F1 + 1
@f1set: sta AUDF1
        lda tmp
        ora #$A0             ; pure tone + volume
        sta AUDC1
        jmp @f2
@f1off: lda #0
        sta AUDC1
@f2:
        ; === F2 → POKEY channel 2 ===
        lda sam_f2
        beq @f2off
        sta AUDF2
        lda tmp
        ora #$A0
        sta AUDC2
        jmp @noise
@f2off: lda #0
        sta AUDC2
@noise:
        ; === Noise → POKEY channel 3 (unvoiced consonants) ===
        lda sam_noise
        beq @noff
        sta AUDF3
        lda tmp
        ora #$80             ; distortion C (noise) + volume
        sta AUDC3
        rts
@noff:  lda #0
        sta AUDC3
        rts

; =====================================================
; FORMANT TABLES (POKEY frequency values)
; Index by phoneme. F1=low formant, F2=high formant.
; POKEY freq = 3.58MHz / (2 * (N+1) * div), audible range ~$10-$FF
; =====================================================
sam_f1_tbl:
;       SIL AA  AE  AH  AO  EH  ER  IH  IY  OH  UH  UW
.byte   $00,$1C,$1A,$20,$24,$18,$1E,$14,$10,$22,$16,$12
;       B   D   F   G   H   J   K   L   M   N   P   R
.byte   $30,$28,$00,$34,$00,$18,$2C,$1E,$20,$1C,$38,$1E
;       S   T   V   W   Y   Z   CH  SH  TH  NG  STOP
.byte   $00,$00,$1A,$1A,$14,$00,$00,$00,$00,$1E,$00

sam_f2_tbl:
;       SIL AA  AE  AH  AO  EH  ER  IH  IY  OH  UH  UW
.byte   $00,$0A,$0C,$0A,$08,$0E,$0C,$12,$16,$08,$0C,$08
;       B   D   F   G   H   J   K   L   M   N   P   R
.byte   $0A,$0C,$08,$08,$06,$0C,$0A,$0E,$00,$08,$08,$0C
;       S   T   V   W   Y   Z   CH  SH  TH  NG  STOP
.byte   $04,$00,$08,$0C,$14,$05,$06,$05,$06,$04,$00

sam_amp_tbl:
;       SIL AA  AE  AH  AO  EH  ER  IH  IY  OH  UH  UW
.byte   $00,$0C,$0C,$0C,$0C,$0C,$0A,$0C,$0C,$0C,$0A,$0A
;       B   D   F   G   H   J   K   L   M   N   P   R
.byte   $06,$06,$06,$06,$04,$08,$06,$08,$08,$08,$06,$08
;       S   T   V   W   Y   Z   CH  SH  TH  NG  STOP
.byte   $06,$06,$06,$08,$08,$06,$06,$06,$06,$08,$00

sam_noise_tbl:
;       SIL AA  AE  AH  AO  EH  ER  IH  IY  OH  UH  UW
.byte   $00,$00,$00,$00,$00,$00,$00,$00,$00,$00,$00,$00
;       B   D   F   G   H   J   K   L   M   N   P   R
.byte   $10,$0C,$06,$10,$04,$08,$0C,$00,$00,$00,$10,$00
;       S   T   V   W   Y   Z   CH  SH  TH  NG  STOP
.byte   $02,$08,$04,$00,$00,$03,$04,$03,$05,$00,$00

sam_dur_tbl:
;       SIL AA  AE  AH  AO  EH  ER  IH  IY  OH  UH  UW
.byte   $04,$08,$08,$08,$09,$07,$08,$06,$07,$09,$07,$08
;       B   D   F   G   H   J   K   L   M   N   P   R
.byte   $02,$02,$04,$02,$03,$04,$02,$05,$05,$05,$02,$05
;       S   T   V   W   Y   Z   CH  SH  TH  NG  STOP
.byte   $05,$02,$04,$04,$04,$05,$05,$05,$04,$05,$02

; Letter→phoneme mapping (A-Z, 26 entries)
; Primary phoneme for each letter
letter_phoneme:
;       A       B       C       D       E       F       G       H
.byte   PH_AE,  PH_B,   PH_K,   PH_D,   PH_EH,  PH_F,   PH_G,   PH_H
;       I       J       K       L       M       N       O       P
.byte   PH_IH,  PH_J,   PH_K,   PH_L,   PH_M,   PH_N,   PH_OH,  PH_P
;       Q       R       S       T       U       V       W       X
.byte   PH_K,   PH_R,   PH_S,   PH_T,   PH_UH,  PH_V,   PH_W,   PH_K
;       Y       Z
.byte   PH_Y,   PH_Z

; Optional trailing vowel (0 = none, used for consonants that need a schwa)
letter_vowel:
;       A       B       C       D       E       F       G       H
.byte   0,      0,      0,      0,      0,      0,      0,      0
;       I       J       K       L       M       N       O       P
.byte   0,      0,      0,      0,      0,      0,      0,      0
;       Q       R       S       T       U       V       W       X
.byte   PH_UW,  0,      0,      0,      0,      0,      0,      PH_S
;       Y       Z
.byte   0,      0

; Digit→phoneme: each digit gets a vowel sound
digit_phoneme:
;       0       1       2       3       4       5       6       7       8       9
.byte   PH_OH,  PH_UH,  PH_UW,  PH_IY,  PH_AO,  PH_AH,  PH_IH,  PH_EH,  PH_AE,  PH_AA

prm:    .byte "> "
prml = * - prm

md:
        .incbin "weights_pico.bin"

; Auto-generated from training data. Do not edit.
ban:
        .byte $7D  ; clear screen
        .byte "================================", $9B
        .byte "             JAM XE", $9B
        .byte "   JUST ATARI LANGUAGE MODEL", $9B
        .byte "    ", $BE, " jam.ag / Marek Spanel     ", $9B
        .byte "================================", $9B
        .byte "  80 KB / 76 KB brain on 6502", $9B
        .byte "  No lookup. Pure generation.", $9B
        .byte $9B
banl = * - ban

intro:
        .byte "JAM XE 130XE REAL IRON", $9B
        .byte "512 HIDDEN FROZEN ANCHOR", $9B
        .byte "D J ATTENTION NO CLOUD", $9B
        .byte "Try: POKEY, TRON, BTTF, GROK", $9B
        .byte "More: 130XE FL WHY TINY MORE POK MORE ", $9B
        .byte $9B
introl = * - intro

; === Proactive suggestions (theme-specific) ===
sug0:   .byte "WHO ARE YOU"
sug0l = * - sug0
sug1:   .byte "ROAST GROK"
sug1l = * - sug1
sug2:   .byte "GPT VS JAM"
sug2l = * - sug2
sug3:   .byte "GEMINI ROAST"
sug3l = * - sug3
sug4:   .byte "130XE FLEX"
sug4l = * - sug4
sug5:   .byte "WHY TINY"
sug5l = * - sug5
sug6:   .byte "MORE POKEY"
sug6l = * - sug6
sug7:   .byte "MORE TRON"
sug7l = * - sug7
sug_ptrs: .word sug0, sug1, sug2, sug3, sug4, sug5, sug6, sug7
sug_lens: .byte sug0l, sug1l, sug2l, sug3l, sug4l, sug5l, sug6l, sug7l

; === Dream seeds (16 entries for idle_chatter) ===
drm0:   .byte "HELLO JAM"
drm0l = * - drm0
drm1:   .byte "ROAST GROK"
drm1l = * - drm1
drm2:   .byte "CLAUD CLAUN"
drm2l = * - drm2
drm3:   .byte "REMEMBER ME"
drm3l = * - drm3
drm4:   .byte "POKEY"
drm4l = * - drm4
drm5:   .byte "TRON"
drm5l = * - drm5
drm6:   .byte "BTTF"
drm6l = * - drm6
drm7:   .byte "MULE"
drm7l = * - drm7
drm8:   .byte "GHOSTBUSTERS"
drm8l = * - drm8
drm9:   .byte "DEPECHE MODE"
drm9l = * - drm9
drm10:   .byte "STAR RAIDERS"
drm10l = * - drm10
drm11:   .byte "80S"
drm11l = * - drm11
drm12:   .byte "GOONIES"
drm12l = * - drm12
drm13:   .byte "ATARI"
drm13l = * - drm13
drm14:   .byte "ANTIC"
drm14l = * - drm14
drm15:   .byte "GTIA"
drm15l = * - drm15
drm_ptrs: .word drm0, drm1, drm2, drm3, drm4, drm5, drm6, drm7, drm8, drm9, drm10, drm11, drm12, drm13, drm14, drm15
drm_lens: .byte drm0l, drm1l, drm2l, drm3l, drm4l, drm5l, drm6l, drm7l, drm8l, drm9l, drm10l, drm11l, drm12l, drm13l, drm14l, drm15l

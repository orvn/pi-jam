; Auto-generated from training data. Do not edit.
ban:
        .byte "================================", $9B
        .byte "              JAM", $9B
        .byte "        Just Atari Model", $9B
        .byte "    ", $BE, " jam.ag : Marek Spanel", $9B
        .byte "================================", $9B
        .byte "23 KB / 18 KB brain on 6502", $9B
        .byte "No lookup. Pure generation.", $9B
        .byte "Quantized. One char at a time.", $9B
        .byte $9B
banl = * - ban

intro:
        .byte "Atari. Czech. Math. Integers.", $9B
        .byte "18 KB brain. Thinks in chains.", $9B
        .byte "Try: 2+3, COUNT 5, POKEY, LOVE", $9B
        .byte "More: ZORK, STAR RAI, ATARI, ANTIC", $9B
        .byte $9B
introl = * - intro

; === Proactive suggestions (display) ===
sug0:   .byte "TRY 2+3"
sug0l = * - sug0
sug1:   .byte "TRY COUNT 5"
sug1l = * - sug1
sug2:   .byte "TRY BIN 7"
sug2l = * - sug2
sug3:   .byte "TRY REV DOG"
sug3l = * - sug3
sug4:   .byte "TRY ZORK"
sug4l = * - sug4
sug5:   .byte "TRY STAR RAIDER"
sug5l = * - sug5
sug6:   .byte "TRY ATARI"
sug6l = * - sug6
sug7:   .byte "TRY ANTIC"
sug7l = * - sug7
sug_ptrs: .word sug0, sug1, sug2, sug3, sug4, sug5, sug6, sug7
sug_lens: .byte sug0l, sug1l, sug2l, sug3l, sug4l, sug5l, sug6l, sug7l

; === Dream seeds (16 entries for idle_chatter) ===
drm0:   .byte "3+4"
drm0l = * - drm0
drm1:   .byte "COUNT 5"
drm1l = * - drm1
drm2:   .byte "BIN 7"
drm2l = * - drm2
drm3:   .byte "REV DOG"
drm3l = * - drm3
drm4:   .byte "LOVE"
drm4l = * - drm4
drm5:   .byte "42"
drm5l = * - drm5
drm6:   .byte "WHO ARE YOU"
drm6l = * - drm6
drm7:   .byte "DO YOU THINK"
drm7l = * - drm7
drm8:   .byte "HELLO"
drm8l = * - drm8
drm9:   .byte "BEER"
drm9l = * - drm9
drm10:   .byte "NEXT 2 4 6"
drm10l = * - drm10
drm11:   .byte "STAR RAIDERS"
drm11l = * - drm11
drm12:   .byte "RUDE"
drm12l = * - drm12
drm13:   .byte "HOPE"
drm13l = * - drm13
drm14:   .byte "POKEY"
drm14l = * - drm14
drm15:   .byte "SPAM"
drm15l = * - drm15
drm_ptrs: .word drm0, drm1, drm2, drm3, drm4, drm5, drm6, drm7, drm8, drm9, drm10, drm11, drm12, drm13, drm14, drm15
drm_lens: .byte drm0l, drm1l, drm2l, drm3l, drm4l, drm5l, drm6l, drm7l, drm8l, drm9l, drm10l, drm11l, drm12l, drm13l, drm14l, drm15l

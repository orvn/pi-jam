; Auto-generated from training data. Do not edit.
ban:
        .byte "================================", $9B
        .byte "            PICO JAM", $9B
        .byte "        Just Atari Model", $9B
        .byte "    ", $BE, " jam.ag : Marek Spanel", $9B
        .byte "================================", $9B
        .byte "4 KB / 1.5 KB brain on 6502", $9B
        .byte "No lookup. Pure generation.", $9B
        .byte "Quantized. One char at a time.", $9B
        .byte $9B
banl = * - ban

intro:
        .byte "just atari model", $9B
        .byte "1.5 kb. roasts llms. does math.", $9B
        .byte "Ask: 3+4, GPT, CNT5, BIN7", $9B
        .byte "More: 9+9, WHO, 42, HELP", $9B
        .byte $9B
introl = * - intro

; === Proactive suggestions (display) ===
sug0:   .byte "TRY 3+4"
sug0l = * - sug0
sug1:   .byte "TRY GPT"
sug1l = * - sug1
sug2:   .byte "TRY CNT5"
sug2l = * - sug2
sug3:   .byte "TRY BIN7"
sug3l = * - sug3
sug4:   .byte "TRY 9+9"
sug4l = * - sug4
sug5:   .byte "TRY WHO"
sug5l = * - sug5
sug6:   .byte "TRY 42"
sug6l = * - sug6
sug7:   .byte "TRY HELP"
sug7l = * - sug7
sug_ptrs: .word sug0, sug1, sug2, sug3, sug4, sug5, sug6, sug7
sug_lens: .byte sug0l, sug1l, sug2l, sug3l, sug4l, sug5l, sug6l, sug7l

; === Dream seeds (16 entries for idle_chatter) ===
drm0:   .byte "TRY 3+4"
drm0l = * - drm0
drm1:   .byte "TRY GPT"
drm1l = * - drm1
drm2:   .byte "TRY CNT5"
drm2l = * - drm2
drm3:   .byte "TRY BIN7"
drm3l = * - drm3
drm4:   .byte "TRY 9+9"
drm4l = * - drm4
drm5:   .byte "TRY WHO"
drm5l = * - drm5
drm6:   .byte "TRY 42"
drm6l = * - drm6
drm7:   .byte "TRY HELP"
drm7l = * - drm7
drm8:   .byte "TRY 3+4"
drm8l = * - drm8
drm9:   .byte "TRY 3+4"
drm9l = * - drm9
drm10:   .byte "TRY 3+4"
drm10l = * - drm10
drm11:   .byte "TRY 3+4"
drm11l = * - drm11
drm12:   .byte "TRY 3+4"
drm12l = * - drm12
drm13:   .byte "TRY 3+4"
drm13l = * - drm13
drm14:   .byte "TRY 3+4"
drm14l = * - drm14
drm15:   .byte "TRY 3+4"
drm15l = * - drm15
drm_ptrs: .word drm0, drm1, drm2, drm3, drm4, drm5, drm6, drm7, drm8, drm9, drm10, drm11, drm12, drm13, drm14, drm15
drm_lens: .byte drm0l, drm1l, drm2l, drm3l, drm4l, drm5l, drm6l, drm7l, drm8l, drm9l, drm10l, drm11l, drm12l, drm13l, drm14l, drm15l

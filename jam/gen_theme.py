#!/usr/bin/env python3
"""Generate a theme include from @title/@sub/@sug/@dream in training data."""
import sys

def _strip_try_prefix(s):
    s = s.strip()
    if s.upper().startswith('TRY '):
        return s[4:].strip()
    return s

def gen(datafile, outfile='theme.inc'):
    title = 'JAM'
    subs = []
    sugs = []
    dreams = []
    size = '30'
    brain = ''
    with open(datafile) as f:
        for line in f:
            line = line.strip()
            if line.startswith('@title '):
                title = line[7:].strip()
            elif line.startswith('@sub '):
                subs.append(line[5:].strip())
            elif line.startswith('@sug '):
                sugs.append(line[5:].strip()[:15])
            elif line.startswith('@dream '):
                dreams.append(line[7:].strip()[:15])
            elif line.startswith('@size '):
                size = line[6:].strip()
            elif line.startswith('@brain '):
                brain = line[7:].strip()
    if not sugs:
        sugs = ['HELP'] * 8
    while len(sugs) < 8:
        sugs.append(sugs[len(sugs) % len(sugs)])
    sugs = sugs[:8]
    
    # Dream seeds: 16 diverse queries for idle_chatter
    if not dreams:
        dreams = sugs[:]
    while len(dreams) < 16:
        dreams.append(dreams[len(dreams) % len(dreams)])
    dreams = dreams[:16]
    
    hint_words = [_strip_try_prefix(s)[:8] for s in sugs]
    sub_has_prompt = any(sub.lower().startswith(('try:', 'ask:', 'more:')) for sub in subs)
    if sub_has_prompt:
        hint_lines = []
        if len(hint_words) > 4:
            hint_lines.append('More: ' + ', '.join(hint_words[4:8]))
    else:
        hint_lines = ['Ask: ' + ', '.join(hint_words[:4])]
        if len(hint_words) > 4:
            hint_lines.append('More: ' + ', '.join(hint_words[4:8]))
    pad = max(0, (32 - len(title)) // 2)
    padded_title = ' ' * pad + title
    out = []
    out.append('; Auto-generated from training data. Do not edit.')
    out.append('ban:')
    jam_line = 'Just Atari Model'
    copy_line = 'jam.ag : Marek Spanel'
    jam_pad = max(0, (32 - len(jam_line)) // 2)
    copy_pad = max(0, (32 - len(copy_line)) // 2)
    out.append('        .byte "================================", $9B')
    out.append(f'        .byte "{padded_title}", $9B')
    out.append(f'        .byte "{" " * jam_pad}{jam_line}", $9B')
    out.append('        .byte "    ", $BE, " jam.ag : Marek Spanel", $9B')
    out.append('        .byte "================================", $9B')
    if brain:
        out.append(f'        .byte "{size} KB / {brain} KB brain on 6502", $9B')
    else:
        out.append(f'        .byte "{size} KB on 6502 at 1.79 MHz", $9B')
    out.append('        .byte "No lookup. Pure generation.", $9B')
    out.append('        .byte "Quantized. One char at a time.", $9B')
    out.append('        .byte $9B')
    out.append('banl = * - ban')
    out.append('')
    out.append('intro:')
    for sub in subs:
        sub = sub[:38]
        out.append(f'        .byte "{sub}", $9B')
    for hint in hint_lines:
        hint = hint[:38]
        out.append(f'        .byte "{hint}", $9B')
    out.append('        .byte $9B')
    out.append('introl = * - intro')
    out.append('')
    out.append('; === Proactive suggestions (display) ===')
    for i, s in enumerate(sugs):
        out.append(f'sug{i}:   .byte "{s}"')
        out.append(f'sug{i}l = * - sug{i}')
    out.append('sug_ptrs: .word ' + ', '.join(f'sug{i}' for i in range(8)))
    out.append('sug_lens: .byte ' + ', '.join(f'sug{i}l' for i in range(8)))
    out.append('')
    out.append('; === Dream seeds (16 entries for idle_chatter) ===')
    for i, d in enumerate(dreams):
        out.append(f'drm{i}:   .byte "{d}"')
        out.append(f'drm{i}l = * - drm{i}')
    out.append('drm_ptrs: .word ' + ', '.join(f'drm{i}' for i in range(16)))
    out.append('drm_lens: .byte ' + ', '.join(f'drm{i}l' for i in range(16)))
    with open(outfile, 'w') as f:
        f.write('\n'.join(out) + '\n')
    print(f'  {outfile}: "{title}", {len(subs)} sub, {len(sugs)} sug, {len(dreams)} dream')

if __name__ == '__main__':
    datafile = sys.argv[1] if len(sys.argv) > 1 else 'td_animals.txt'
    outfile = sys.argv[2] if len(sys.argv) > 2 else 'theme.inc'
    gen(datafile, outfile)

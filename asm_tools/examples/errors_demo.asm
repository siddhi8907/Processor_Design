; deliberately broken - shows how errors are reported (all at once)
start:  mov   r1, 70000         ; out of range
        ad    r2, r1, r1        ; typo
        add   r2, r1            ; wrong operand count
        mov   r16, 1            ; bad register
        b     strat             ; typo'd label
start:  nop                     ; duplicate label

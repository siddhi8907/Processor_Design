; sum_stack.asm - sums 1..10, uses the stack, call/ret and ld/st
; (set the sp value to match your STACK_BASE)
        mov   sp, 4096          ; initialise stack pointer
        mov   r1, 0             ; sum
        mov   r2, 1             ; i
        mov   r3, 11            ; limit
loop:   add   r1, r1, r2
        add   r2, r2, 1
        cmp   r2, r3
        beq   done              ; stop when i == 11
        b     loop
done:   push  r1                ; save sum on the stack
        call  twice             ; r1 = r1 * 2
        st    r1, -8[sp]
        ld    r7, -8[sp]
        pop   r6                ; r6 = 55
        halt
twice:  add   r1, r1, r1
        ret

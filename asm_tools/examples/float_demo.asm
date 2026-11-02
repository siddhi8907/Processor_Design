; float_demo.asm - IEEE-754 binary32 instructions (F bit = 1)
        fmov  r1, 1.5           ; float immediate (top 17 bits of binary32)
        fmov  r2, 2.25
        fadd  r3, r1, r2        ; 3.75
        fmul  r4, r3, 2.0       ; 7.5
        fcmp  r1, r2            ; sets E / GT using float compare
        bgt   skip
        fneg  r8, r1
skip:   fcvti r5, r4            ; float -> int  (F=1)
        mov   r6, 7
        fcvt  r7, r6            ; int -> float  (F=0)
        fmin  r9, r1, r2
        fabs  r10, r8
        halt

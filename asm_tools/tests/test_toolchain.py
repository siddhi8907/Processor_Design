"""Unit tests for the RISC201 assembler + disassembler.   Run:  python -m unittest -v   (from asm_tools/)"""
import os
import random
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import isa                                                        # noqa: E402
from assembler import AssemblyFailed, assemble_source             # noqa: E402
from disassembler import decode, disassemble_word, disassemble_words  # noqa: E402


def asm(text, **kw):
    return assemble_source(text, **kw).words


class KnownEncodings(unittest.TestCase):
    """Expected values were worked out by hand from the ISA document."""
    CASES = {
        "add r1, r2, r3":     0x00048C000,
        "mov r1, 5":          0x098400005,
        "mov sp, 4096":       0x09B801000,
        "nop":                0x0D0000000,
        "halt":               0x190000000,
        "rte":                0x1A0000000,
        "ret":                0x140000000,
        "ld r1, 8[r2]":       0x0E8480008,
        "push r3":            0x150C00000,
        "fadd r1, r2, 1.0":   0x00C48FE00,   # 1.0f = 0x3F800000 -> top 18 bits 0x0FE00
        "fcvt r1, r2":        0x1F0408000,   # int -> float, F=0
        "fcvti r1, r2":       0x1F4408000,   # float -> int, F=1
        "cmp r1, r2":         0x050048000,
    }

    def test_known(self):
        for text, expected in self.CASES.items():
            with self.subTest(text):
                self.assertEqual(asm(text), [expected], f"{text}: got {asm(text)[0]:09X}")

    def test_modifiers(self):
        self.assertEqual(asm("mov.u r1, 65535"), [0x098400000 | (1 << 16) | 0xFFFF])
        self.assertEqual(asm("movh r1, 0x1234"), [0x098400000 | (2 << 16) | 0x1234])
        self.assertEqual(asm("addu r1, r2, 7"), asm("add.u r1, r2, 7"))

    def test_negative_immediate(self):
        self.assertEqual(asm("mov r1, -1")[0] & 0xFFFF, 0xFFFF)
        self.assertEqual(asm("ld r1, -4[sp]")[0] & 0xFFFF, 0xFFFC)

    def test_register_aliases(self):
        self.assertEqual(asm("mov sp, r1"), asm("mov r14, r1"))
        self.assertEqual(asm("mov RA, R1"), asm("mov r15, r1"))

    def test_word_directive(self):
        self.assertEqual(asm(".word 0x1FFFFFFFF"), [0x1FFFFFFFF])


class Labels(unittest.TestCase):
    def test_forward_and_backward(self):
        w = asm("""
            b fwd          ; 0x00 -> offset +2
            nop
        fwd: b back        ; 0x08 -> offset -2
            nop
        back: nop
        """)
        self.assertEqual(w[0], 0x120000002)
        # b back at 0x08, back = 0x10 -> +2
        self.assertEqual(w[2], 0x120000002)

    def test_backward_offset_negative(self):
        w = asm("top: nop\n b top")
        self.assertEqual(w[1], (0b10010 << 28) | 0xFFFFFFF)      # offset -1

    def test_call_ret(self):
        w = asm("call f\n halt\n f: ret")
        self.assertEqual(w[0], (0b10011 << 28) | 2)

    def test_numeric_branch_is_word_offset(self):
        self.assertEqual(asm("beq 3"), [(0b10000 << 28) | 3])

    def test_equ_and_expressions(self):
        w = asm(".equ N, 4\n .equ M, N*2+1\n mov r1, M\n ld r2, N+4[sp]")
        self.assertEqual(w[0] & 0xFFFF, 9)
        self.assertEqual(w[1] & 0xFFFF, 8)

    def test_label_as_value(self):
        w = asm("nop\n here: mov r1, here")
        self.assertEqual(w[1] & 0xFFFF, 4)

    def test_base_address(self):
        r = assemble_source("x: nop\n mov r1, x", base=0x100)
        self.assertEqual(r.words[1] & 0xFFFF, 0x100)

    def test_labels_on_own_line_and_multiple(self):
        w = asm("a:\n b: nop\n beq a")
        self.assertEqual(w[1], (0b10000 << 28) | (-1 & 0xFFFFFFF))


class Floats(unittest.TestCase):
    def test_float_imm_exact(self):
        r = assemble_source("fmov r1, 1.5")
        self.assertEqual(r.warnings, [])
        self.assertEqual(r.words[0] & 0x3FFFF, isa.float_to_bits32(1.5) >> 14)

    def test_float_imm_truncation_warns(self):
        r = assemble_source("fmov r1, 3.14159")
        self.assertEqual(len(r.warnings), 1)
        with self.assertRaises(AssemblyFailed):
            assemble_source("fmov r1, 3.14159", strict=True)

    def test_float_flags_set(self):
        w = asm("fadd r1, r2, r3")[0]
        f = isa.unpack_word(w)
        self.assertEqual((f.opcode, f.i, f.f), (0, 0, 1))

    def test_integer_literal_in_float_context(self):
        self.assertEqual(asm("fadd r1, r2, 2"), asm("fadd r1, r2, 2.0"))

    def test_raw_bits(self):
        self.assertEqual(asm("fmov r1, 0x3FC00000"), asm("fmov r1, 1.5"))


class Errors(unittest.TestCase):
    def fails(self, text, fragment):
        with self.assertRaises(AssemblyFailed) as cm:
            assemble_source(text)
        self.assertIn(fragment, str(cm.exception))
        return cm.exception

    def test_unknown_mnemonic(self):
        self.fails("addd r1, r2, r3", "unknown instruction")

    def test_bad_register(self):
        self.fails("mov r16, 1", "invalid register")

    def test_range(self):
        self.fails("mov r1, 40000", "out of range")
        self.fails("mov.u r1, -1", "out of range")
        self.fails("ld r1, 99999[r2]", "out of range")

    def test_operand_count(self):
        self.fails("add r1, r2", "expects 3")
        self.fails("halt r1", "expects 0")

    def test_undefined_and_duplicate(self):
        self.fails("b nowhere", "undefined label")
        self.fails("x: nop\nx: nop", "duplicate symbol")

    def test_register_only_ops(self):
        self.fails("abs r1, 5", "register source")
        self.fails("fcvt r1, 5", "register source")

    def test_modifier_with_register(self):
        self.fails("add.u r1, r2, r3", "immediate")

    def test_no_float_form(self):
        self.fails("fand r1, r2, r3", "unknown instruction")
        self.fails("fadd.u r1, r2, r3", "unknown instruction")

    def test_all_errors_reported(self):
        e = self.fails("mov r1, 99999\nfoo\nb zzz", ":2:")
        self.assertEqual(len(e.errors), 3)

    def test_line_numbers(self):
        e = self.fails("nop\nnop\nmov r1, 99999", ":3:")
        self.assertEqual(e.errors[0].line, 3)

    def test_bad_expression(self):
        self.fails("mov r1, 3 +", "incomplete")
        self.fails("mov r1, (2", "missing ')'")
        self.fails("mov r1, 1/0", "division by zero")

    def test_bad_memory_operand(self):
        self.fails("ld r1, r2", "memory operand")


class Disassembly(unittest.TestCase):
    def test_simple(self):
        self.assertEqual(disassemble_word(0x098400005), "mov r1, 5")
        self.assertEqual(disassemble_word(0x190000000), "halt")
        self.assertEqual(disassemble_word(0x0E8480008), "ld r1, 8[r2]")
        self.assertEqual(disassemble_word(0x09B801000), "mov sp, 4096")

    def test_labels(self):
        text = disassemble_words(asm("top: nop\n b top\n beq end\n nop\n end: halt"))
        self.assertIn("L_0000:", text)
        self.assertIn("b L_0000", text)
        self.assertIn("beq L_0010", text)

    def test_reserved_opcode(self):
        d = decode(0b10111 << 28)
        self.assertFalse(d.valid)
        self.assertIn("reserved", d.note)

    def test_noncanonical_becomes_word(self):
        self.assertTrue(disassemble_word(0x0D0000001).startswith(".word"))     # nop with junk bit
        self.assertTrue(disassemble_word((0b00110 << 28) | (1 << 26)).startswith(".word"))  # F=1 on 'and'

    def test_float_text(self):
        self.assertEqual(disassemble_word(asm("fadd r1, r2, 1.0")[0]), "fadd r1, r2, 1.0")
        self.assertEqual(disassemble_word(asm("fcvti r4, r5")[0]), "fcvti r4, r5")

    def test_modifier_text(self):
        self.assertEqual(disassemble_word(asm("mov.h r1, 0x1234")[0]), "mov.h r1, 0x1234")


class RoundTrip(unittest.TestCase):
    PROGRAM = """
    start: add r1, r2, r3
        add r1, r2, -5
        sub.u r4, r5, 65535
        mul.h r4, r5, 0xABCD
        div r1, r1, r2
        mod r1, r1, 7
        cmp r1, r2
        cmp.u r1, 9
        and r3, r3, r4
        or r3, r3, 0x00FF
        not r5, r6
        not r5, 12
        mov r5, r6
        mov.h r5, 0xFFFF
        lsl r1, r1, 3
        lsr r1, r1, r2
        asr r1, r1, 1
        min r1, r2, r3
        max r1, r2, -3
        abs r1, r2
        neg r1, r2
        nop
        ld r1, 4[r2]
        st r1, -4[sp]
        push r7
        pop r7
        beq start
        bgt later
        b start
        call later
    later: ret
        fadd r1, r2, r3
        fsub r1, r2, 0.5
        fmul r1, r2, -2.0
        fdiv r1, r2, 8.0
        fcmp r1, r2
        fcmp r1, 1.0
        fmov r1, r2
        fmov r1, -0.75
        fmin r1, r2, r3
        fmax r1, r2, 100.0
        fabs r1, r2
        fneg r1, r2
        fcvt r1, r2
        fcvti r1, r2
        fmov r3, 0x7FC00000
        fmov r3, 0xFF800000
        rte
        halt
    """

    def test_program_roundtrip(self):
        words = asm(self.PROGRAM)
        text = disassemble_words(words)
        self.assertEqual(asm(text), words)

    def test_roundtrip_without_labels(self):
        words = asm(self.PROGRAM)
        self.assertEqual(asm(disassemble_words(words, use_labels=False)), words)

    def test_annotated_output_still_assembles(self):
        words = asm(self.PROGRAM)
        self.assertEqual(asm(disassemble_words(words, annotate=True)), words)

    def test_hex_and_bin_images(self):
        r = assemble_source(self.PROGRAM)
        self.assertEqual(isa.parse_words("\n".join(r.hex_lines())), r.words)
        self.assertEqual(isa.parse_words("\n".join(r.bin_lines())), r.words)
        self.assertTrue(all(len(h) == 9 for h in r.hex_lines()))
        self.assertTrue(all(len(b) == 33 for b in r.bin_lines()))

    def test_every_random_word_roundtrips(self):
        """assemble(disassemble(w)) == w for ANY 33-bit word (invalid ones become .word)."""
        rng = random.Random(201)
        for _ in range(30000):
            w = rng.getrandbits(33)
            text = disassemble_word(w)
            self.assertEqual(asm(text), [w], f"0x{w:09X} -> '{text}'")

    def test_every_opcode_with_structured_fields(self):
        rng = random.Random(7)
        valid = 0
        for _ in range(30000):
            op = rng.randrange(32)
            w = isa.pack_word(op, rng.getrandbits(1), rng.getrandbits(1), rng.getrandbits(4),
                              rng.getrandbits(4), rng.getrandbits(18) & rng.choice([0x3FFFF, 0x3C000, 0x3FFFF, 0x0FFFF]))
            d = decode(w)
            valid += d.valid
            self.assertEqual(asm(disassemble_word(w)), [w])
        self.assertGreater(valid, 1000)          # make sure we really exercised canonical forms

    def test_all_zero_word_is_add(self):
        self.assertEqual(disassemble_word(0), "add r0, r0, r0")


class ImageParsing(unittest.TestCase):
    def test_formats_and_comments(self):
        text = "098400005 ; mov\n0x190000000\n// c\n\n0b" + format(0x0D0000000, "b") + "\n" + format(0x140000000, "033b")
        self.assertEqual(isa.parse_words(text), [0x098400005, 0x190000000, 0x0D0000000, 0x140000000])

    def test_bad_words(self):
        for bad in ["XYZ", "2000000000", "0x2FFFFFFFF"]:
            with self.assertRaises(ValueError):
                isa.parse_words(bad)

    def test_hex_word_starting_with_0b(self):
        self.assertEqual(isa.parse_words("0B0448000"), [0x0B0448000])

    def test_error_has_line_number(self):
        with self.assertRaises(ValueError) as cm:
            isa.parse_words("098400005\nZZ")
        self.assertIn("line 2", str(cm.exception))


if __name__ == "__main__":
    unittest.main()

#!/usr/bin/env python3
"""
disassembler.py - My RISC201 disassembler: turns those hex/binary words back into assembly text.

Input words can be 8-digit hex (like 4C200005), 0xHEX, 0bBIN or 32-character
binary strings. I made it so it can read one or more per line, and it ignores ; // # comments.

The output is totally re-assemblable. If it finds a word that isn't a perfect canonical encoding of a
valid instruction (like a reserved opcode, junk bits where it should be zero, or F=1 on an
integer-only op), it just spits it out as '.word 0x...' with a comment. 
Basically, assemble(disassemble(x)) == x is guaranteed for every possible 32-bit word.

Library use (when I hook this into the simulator's debugger or pipeline viewer):
    from disassembler import disassemble_word
    disassemble_word(0x4C200005)           # -> 'mov r1, 5'

Command line stuff:
    python disassembler.py prog.hex                 # just prints the assembly
    python disassembler.py prog.hex -o prog.asm     # writes it out to a file
    python disassembler.py --words 4C200005 C8000000
    python disassembler.py prog.hex --annotate      # adds the address + raw word as comments
"""
from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional

try:
    from . import isa
except ImportError:
    import isa


@dataclass
class Decoded:
    word: int
    text: str                      # The actual assembly text (branches use a numeric offset until they get labelled)
    valid: bool                    # If False, I just emit it as a .word
    note: str = ""                 # Why it failed validation or just some extra info
    mnemonic: str = ""
    branch_offset: Optional[int] = None   # Word offset, used for beq/bgt/b/call


def _fmt_float_imm17(imm17: int) -> str:
    # Need to shift the 17-bit immediate back up to get the actual float value
    bits = (imm17 << 15) & 0xFFFFFFFF
    v = isa.bits32_to_float(bits)
    if math.isnan(v) or math.isinf(v):
        return f"0x{bits:08X}"                     # Just keep the raw pattern so the round trip stays exact
    s = f"{v:.9g}"
    if not any(c in s for c in ".eE"):
        s += ".0"
    return s


def _invalid(word: int, note: str) -> Decoded:
    # Helper to just return a raw .word when things don't line up
    return Decoded(word, f".word 0x{word:08X}", False, note)


def decode(word: int) -> Decoded:
    """Decodes a single 32-bit word. Branch offsets stay numeric here (disassemble_word handles the labels)."""
    f = isa.unpack_word(word)
    op = isa.OPS_BY_OPCODE.get(f.opcode)
    if op is None:
        return _invalid(word, f"reserved opcode {f.opcode:05b} (executes as nop)")

    kind, R = op.kind, isa.reg_name
    F = f.f
    name = op.name

    # ---- float naming -----------------------------------------------------
    if F:
        if name == "fcvt":
            name = "fcvti"
        elif op.float_ok:
            name = isa.FLOAT_NAME_OF[name]
        elif kind != isa.K_BRANCH:
            return _invalid(word, f"F=1 is not allowed on '{name}'")

    # ---- per-kind canonical re-encoding ------------------------------------
    # I basically rebuild the instruction from the parsed fields and check if it matches the original word.
    # If it doesn't match, it means there are junk bits somewhere.
    if kind == isa.K_NONE:
        canon = isa.pack_word(op.opcode)
        text = name

    elif kind == isa.K_BRANCH:
        off = isa.sext(f.offset27, 27)
        canon = isa.pack_branch(op.opcode, off)
        return _finish(word, canon, f"{name} {off}", name, off)

    elif kind == isa.K_STACK:
        canon = isa.pack_word(op.opcode, rd=f.rd)
        text = f"{name} {R(f.rd)}"

    elif kind == isa.K_UN2:
        canon = isa.pack_word(op.opcode, 0, F, f.rd, 0, f.rs2 << 13)
        text = f"{name} {R(f.rd)}, {R(f.rs2)}"

    elif kind == isa.K_LDST:
        canon = isa.pack_word(op.opcode, 1, 0, f.rd, f.rs1, f.imm15)
        text = f"{name} {R(f.rd)}, {isa.sext(f.imm15, 15)}[{R(f.rs1)}]"

    else:                                           # Handles ALU3, CMP, and MOV2
        suffix = ""
        if f.i == 0:
            op2 = R(f.rs2)
            rest = f.rs2 << 13
        elif F:
            op2 = _fmt_float_imm17(f.imm17)
            rest = f.imm17
        else:
            if f.mod == 0b11:
                return _invalid(word, "undefined immediate modifier 11")
            suffix = isa.MOD_SUFFIX[f.mod]
            op2 = str(isa.sext(f.imm15, 15)) if f.mod == isa.MOD_NONE else f"0x{f.imm15:04X}"
            rest = (f.mod << 15) | f.imm15
            
        if kind == isa.K_ALU3:
            canon = isa.pack_word(op.opcode, f.i, F, f.rd, f.rs1, rest)
            text = f"{name}{suffix} {R(f.rd)}, {R(f.rs1)}, {op2}"
        elif kind == isa.K_CMP:
            canon = isa.pack_word(op.opcode, f.i, F, 0, f.rs1, rest)
            text = f"{name}{suffix} {R(f.rs1)}, {op2}"
        else:                                       # MOV2
            canon = isa.pack_word(op.opcode, f.i, F, f.rd, 0, rest)
            text = f"{name}{suffix} {R(f.rd)}, {op2}"

    return _finish(word, canon, text, name, None)


def _finish(word: int, canon: int, text: str, name: str, off: Optional[int]) -> Decoded:
    # Final check: if my rebuilt canonical word doesn't match the input, it's invalid.
    if canon != word:
        return _invalid(word, "non-canonical encoding (unused bits are not zero)")
    return Decoded(word, text, True, "", name, off)


def disassemble_word(word: int, pc: int = 0, labels: Optional[Dict[int, str]] = None) -> str:
    """Disassembles a single word. If I pass in a labels dictionary (address->name), it maps the branch targets.
    Without labels, it just shows the signed word offset (which the assembler is fine with anyway)."""
    d = decode(word)
    if d.branch_offset is not None and labels:
        target = pc + d.branch_offset * isa.INSTR_BYTES
        if target in labels:
            return f"{d.mnemonic} {labels[target]}"
    return d.text


def disassemble_words(words: List[int], base: int = 0, use_labels: bool = True,
                      annotate: bool = False) -> str:
    # Decodes a whole block of words and handles the label generation
    decoded = [decode(w) for w in words]
    end = base + len(words) * isa.INSTR_BYTES

    labels: Dict[int, str] = {}
    if use_labels:
        for idx, d in enumerate(decoded):
            if d.branch_offset is not None:
                target = base + idx * isa.INSTR_BYTES + d.branch_offset * isa.INSTR_BYTES
                if base <= target < end:
                    labels[target] = f"L_{target:04X}"

    out: List[str] = []
    for idx, d in enumerate(decoded):
        pc = base + idx * isa.INSTR_BYTES
        if pc in labels:
            out.append(f"{labels[pc]}:")
        text = disassemble_word(d.word, pc, labels)
        comment = []
        if annotate:
            comment.append(f"0x{pc:04X}: {isa.format_hex(d.word)}")
        if d.note:
            comment.append(d.note)
        if d.branch_offset is not None and not (labels and (pc + d.branch_offset * 4) in labels):
            comment.append(f"target 0x{(pc + d.branch_offset * 4) & 0xFFFFFFFF:X}")
        line = f"    {text}"
        if comment:
            line = f"{line:<36}; " + " | ".join(comment)
        out.append(line)
    return "\n".join(out) + ("\n" if out else "")


def load_words(path: str) -> List[int]:
    with open(path, "r", encoding="utf-8") as fh:
        return isa.parse_words(fh.read())


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="RISC201 disassembler (32-bit words -> assembly)")
    ap.add_argument("input", nargs="?", help="hex/binary image file ('-' = stdin)")
    ap.add_argument("--words", nargs="+", metavar="W", help="disassemble these words instead of a file")
    ap.add_argument("-o", "--output", help="write assembly to this file")
    ap.add_argument("--base", type=lambda x: int(x, 0), default=0, help="address of first word (default 0)")
    ap.add_argument("--no-labels", action="store_true", help="show numeric branch offsets, no L_xxxx labels")
    ap.add_argument("--annotate", action="store_true", help="add address and raw word as comments")
    args = ap.parse_args(argv)

    try:
        if args.words:
            words = isa.parse_words("\n".join(args.words))
        elif args.input == "-":
            words = isa.parse_words(sys.stdin.read())
        elif args.input:
            words = load_words(args.input)
        else:
            ap.error("give an input file or --words")
            return 2
    except FileNotFoundError:
        print(f"error: cannot open '{args.input}'", file=sys.stderr)
        return 2
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    text = disassemble_words(words, base=args.base, use_labels=not args.no_labels, annotate=args.annotate)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(text)
        print(f"disassembled {len(words)} word(s) -> {args.output}")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
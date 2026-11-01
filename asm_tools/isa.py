"""
isa.py - RISC201 (SimpleRISC Extended) ISA definition.

Single source of truth shared by the assembler and the disassembler.
Your simulator / control-unit teammates can import this file too, so that
everyone agrees on opcodes and bit positions.

Instruction word = 33 bits (bit 32 = MSB), stored as 9 hex digits.

  Register : opcode[32:28] I[27]=0 F[26] rd[25:22] rs1[21:18] rs2[17:14] 0[13:0]
  Imm  F=0 : opcode        I=1     F=0   rd        rs1        mod[17:16] imm16[15:0]
  Imm  F=1 : opcode        I=1     F=1   rd        rs1        imm18[17:0]  (top 18 bits of binary32)
  Branch   : opcode[32:28] offset[27:0]  (signed, in WORDS; target = PC + offset*4)
  No-operand (nop/ret/halt/rte): opcode, every other bit 0
"""
from __future__ import annotations

import re
import struct
from dataclasses import dataclass
from typing import List, NamedTuple, Optional

# --------------------------------------------------------------------------
# Basic constants
# --------------------------------------------------------------------------
WORD_BITS = 33
WORD_MASK = (1 << WORD_BITS) - 1
HEX_DIGITS = 9
INSTR_BYTES = 4            # PC advances by 4 per instruction (ISA doc)
NUM_REGS = 16
REG_SP = 14                # r14
REG_RA = 15                # r15

# Immediate modifier field, bits [17:16]
MOD_NONE, MOD_U, MOD_H = 0b00, 0b01, 0b10
MOD_SUFFIX = {MOD_NONE: "", MOD_U: ".u", MOD_H: ".h"}

# Operand-shape kinds
K_ALU3 = "alu3"      # rd, rs1, rs2|imm
K_CMP = "cmp"        # rs1, rs2|imm
K_MOV2 = "mov2"      # rd, rs2|imm
K_UN2 = "un2"        # rd, rs2            (register only)
K_LDST = "ldst"      # rd, imm[rs1]
K_STACK = "stack"    # rd
K_BRANCH = "branch"  # label / word offset
K_NONE = "none"      # no operands

MODIFIABLE_KINDS = {K_ALU3, K_CMP, K_MOV2}

SYNTAX = {
    K_ALU3: "rd, rs1, rs2|imm",
    K_CMP: "rs1, rs2|imm",
    K_MOV2: "rd, rs2|imm",
    K_UN2: "rd, rs2",
    K_LDST: "rd, imm[rs1]",
    K_STACK: "rd",
    K_BRANCH: "label",
    K_NONE: "",
}


@dataclass(frozen=True)
class OpInfo:
    name: str
    opcode: int
    kind: str
    float_ok: bool = False      # does an F=1 version exist?


_OPS = [
    OpInfo("add", 0b00000, K_ALU3, True),
    OpInfo("sub", 0b00001, K_ALU3, True),
    OpInfo("mul", 0b00010, K_ALU3, True),
    OpInfo("div", 0b00011, K_ALU3, True),
    OpInfo("mod", 0b00100, K_ALU3),
    OpInfo("cmp", 0b00101, K_CMP, True),
    OpInfo("and", 0b00110, K_ALU3),
    OpInfo("or",  0b00111, K_ALU3),
    OpInfo("not", 0b01000, K_MOV2),
    OpInfo("mov", 0b01001, K_MOV2, True),
    OpInfo("lsl", 0b01010, K_ALU3),
    OpInfo("lsr", 0b01011, K_ALU3),
    OpInfo("asr", 0b01100, K_ALU3),
    OpInfo("nop", 0b01101, K_NONE),
    OpInfo("ld",  0b01110, K_LDST),
    OpInfo("st",  0b01111, K_LDST),
    OpInfo("beq", 0b10000, K_BRANCH),
    OpInfo("bgt", 0b10001, K_BRANCH),
    OpInfo("b",   0b10010, K_BRANCH),
    OpInfo("call", 0b10011, K_BRANCH),
    OpInfo("ret", 0b10100, K_NONE),
    OpInfo("push", 0b10101, K_STACK),
    OpInfo("pop", 0b10110, K_STACK),
    # 10111 and 11000 are free / reserved -> decode as nop
    OpInfo("halt", 0b11001, K_NONE),
    OpInfo("rte", 0b11010, K_NONE),
    OpInfo("min", 0b11011, K_ALU3, True),
    OpInfo("max", 0b11100, K_ALU3, True),
    OpInfo("abs", 0b11101, K_UN2, True),
    OpInfo("neg", 0b11110, K_UN2, True),
    OpInfo("fcvt", 0b11111, K_UN2, True),
]
OPS_BY_NAME = {o.name: o for o in _OPS}
OPS_BY_OPCODE = {o.opcode: o for o in _OPS}
RESERVED_OPCODES = (0b10111, 0b11000)

# Assembler names for the F=1 forms (same opcode as the integer op)
FLOAT_MNEMONICS = {
    "fadd": "add", "fsub": "sub", "fmul": "mul", "fdiv": "div",
    "fcmp": "cmp", "fmov": "mov", "fmin": "min", "fmax": "max",
    "fabs": "abs", "fneg": "neg",
}
FLOAT_NAME_OF = {v: k for k, v in FLOAT_MNEMONICS.items()}
# fcvt: F bit = "source is float".  fcvt = int->float (F=0), fcvti = float->int (F=1)
FCVT_MNEMONICS = {"fcvt": 0, "i2f": 0, "fcvti": 1, "f2i": 1}

ALL_MNEMONICS = sorted(set(OPS_BY_NAME) | set(FLOAT_MNEMONICS) | set(FCVT_MNEMONICS))


def resolve_mnemonic(text: str):
    """Return (OpInfo, F, modifier) for an assembly mnemonic, or None.

    Accepts: add, fadd, fcvt/fcvti, and modifier forms add.u / addu / mov.h / movh.
    """
    m = text.lower()
    if m in FCVT_MNEMONICS:
        return OPS_BY_NAME["fcvt"], FCVT_MNEMONICS[m], MOD_NONE
    if m in OPS_BY_NAME:
        return OPS_BY_NAME[m], 0, MOD_NONE
    if m in FLOAT_MNEMONICS:
        return OPS_BY_NAME[FLOAT_MNEMONICS[m]], 1, MOD_NONE
    for suffix, mod in ((".u", MOD_U), (".h", MOD_H)):
        if m.endswith(suffix):
            base = OPS_BY_NAME.get(m[:-2])
            if base and base.kind in MODIFIABLE_KINDS:
                return base, 0, mod
            return None
    if len(m) > 1 and m[-1] in "uh":
        base = OPS_BY_NAME.get(m[:-1])
        if base and base.kind in MODIFIABLE_KINDS:
            return base, 0, (MOD_U if m[-1] == "u" else MOD_H)
    return None


# --------------------------------------------------------------------------
# Registers
# --------------------------------------------------------------------------
_REG_RE = re.compile(r"^(?:r(\d+)|(sp)|(ra))$", re.I)


def parse_register(tok: str) -> Optional[int]:
    """Return register number, None if `tok` is not register-shaped.
    Raises ValueError for things like r16."""
    m = _REG_RE.match(tok.strip())
    if not m:
        return None
    if m.group(2):
        return REG_SP
    if m.group(3):
        return REG_RA
    n = int(m.group(1))
    if n >= NUM_REGS:
        raise ValueError(f"invalid register '{tok.strip()}' (valid: r0-r15, sp, ra)")
    return n


def reg_name(n: int) -> str:
    return {REG_SP: "sp", REG_RA: "ra"}.get(n, f"r{n}")


# --------------------------------------------------------------------------
# Bit helpers
# --------------------------------------------------------------------------
def sext(value: int, bits: int) -> int:
    value &= (1 << bits) - 1
    return value - (1 << bits) if value >> (bits - 1) else value


def pack_word(opcode: int, i: int = 0, f: int = 0, rd: int = 0, rs1: int = 0, rest: int = 0) -> int:
    return ((opcode << 28) | (i << 27) | (f << 26) | (rd << 22) | (rs1 << 18) | (rest & 0x3FFFF))


def pack_branch(opcode: int, offset: int) -> int:
    return (opcode << 28) | (offset & 0xFFFFFFF)


class Fields(NamedTuple):
    opcode: int
    i: int
    f: int
    rd: int
    rs1: int
    rs2: int
    mod: int
    imm16: int
    imm18: int
    offset28: int


def unpack_word(word: int) -> Fields:
    if not 0 <= word <= WORD_MASK:
        raise ValueError(f"word 0x{word:X} does not fit in {WORD_BITS} bits")
    return Fields(
        opcode=(word >> 28) & 0x1F, i=(word >> 27) & 1, f=(word >> 26) & 1,
        rd=(word >> 22) & 0xF, rs1=(word >> 18) & 0xF, rs2=(word >> 14) & 0xF,
        mod=(word >> 16) & 0x3, imm16=word & 0xFFFF, imm18=word & 0x3FFFF,
        offset28=word & 0xFFFFFFF,
    )


# --------------------------------------------------------------------------
# IEEE-754 binary32 helpers
# --------------------------------------------------------------------------
def float_to_bits32(x: float) -> int:
    try:
        return struct.unpack("<I", struct.pack("<f", x))[0]
    except OverflowError:                      # too big for binary32 -> +/-Inf
        return 0xFF800000 if x < 0 else 0x7F800000


def bits32_to_float(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits & 0xFFFFFFFF))[0]


# --------------------------------------------------------------------------
# Program image text formats (hex = 9 digits/line, bin = 33 chars/line)
# --------------------------------------------------------------------------
def format_hex(word: int) -> str:
    return f"{word:0{HEX_DIGITS}X}"


def format_bin(word: int) -> str:
    return f"{word:0{WORD_BITS}b}"


_COMMENT_RE = re.compile(r";|//|#")


def parse_word_token(tok: str) -> int:
    """Parse one 33-bit word.

    Accepted: 9-digit hex (what the assembler writes), 0xHEX, 0bBIN, or a
    33-character binary string.  A token of exactly 9 hex digits is always hex
    (so '0B0448000' is a hex word, not a 0b-prefixed binary number).
    """
    t = tok.strip().replace("_", "").lower()
    if not t:
        raise ValueError("empty word")
    try:
        if len(t) == HEX_DIGITS and all(c in "0123456789abcdef" for c in t):
            value = int(t, 16)
        elif t.startswith("0x"):
            value = int(t[2:], 16)
        elif t.startswith("0b") and len(t) > 2 and set(t[2:]) <= {"0", "1"}:
            value = int(t[2:], 2)
        elif len(t) == WORD_BITS and set(t) <= {"0", "1"}:
            value = int(t, 2)
        else:
            if len(t) > HEX_DIGITS:
                raise ValueError(f"'{tok}' is too long for a {WORD_BITS}-bit word "
                                 f"(expected up to {HEX_DIGITS} hex digits or {WORD_BITS} binary digits)")
            value = int(t, 16)
    except ValueError as exc:
        if "too long" in str(exc):
            raise
        raise ValueError(f"'{tok}' is not a valid hex/binary word") from None
    if value > WORD_MASK:
        raise ValueError(f"'{tok}' does not fit in {WORD_BITS} bits")
    return value


def parse_words(text: str) -> List[int]:
    """Parse a hex/binary image: one or more words per line, ; // # comments allowed."""
    words: List[int] = []
    for n, line in enumerate(text.splitlines(), 1):
        line = _COMMENT_RE.split(line, 1)[0].strip()
        if not line:
            continue
        for tok in re.split(r"[\s,]+", line):
            if not tok:
                continue
            try:
                words.append(parse_word_token(tok))
            except ValueError as exc:
                raise ValueError(f"line {n}: {exc}") from None
    return words

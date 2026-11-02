"""
isa.py - My RISC201 (SimpleRISC Extended) ISA definition.

This is the single source of truth shared by both the assembler and the disassembler.
The rest of my team working on the simulator or control-unit can just import this file 
so we don't have conflicting opcodes or bit positions.

Instruction word = 32 bits (bit 31 = MSB), stored as 8 hex digits.

  Register : opcode[31:27] I[26]=0 F[25] rd[24:21] rs1[20:17] rs2[16:13] 0[12:0]
  Imm  F=0 : opcode        I=1     F=0   rd        rs1        mod[16:15] imm15[14:0]
  Imm  F=1 : opcode        I=1     F=1   rd        rs1        imm17[16:0]  (top 17 bits of binary32)
  Branch   : opcode[31:27] offset[26:0]  (signed, in WORDS; target = PC + offset*4)
  No-operand (nop/ret/halt/rte): opcode, every other bit is just 0
"""
from __future__ import annotations

import re
import struct
from dataclasses import dataclass
from typing import List, NamedTuple, Optional

# --------------------------------------------------------------------------
# Basic constants: The core architectural parameters
# --------------------------------------------------------------------------
WORD_BITS = 32
WORD_MASK = (1 << WORD_BITS) - 1
HEX_DIGITS = 8
INSTR_BYTES = 4            # PC advances by 4 per instruction (just like the ISA doc says)
NUM_REGS = 16
REG_SP = 14                # r14
REG_RA = 15                # r15

# Immediate modifier field, bits [16:15]
MOD_NONE, MOD_U, MOD_H = 0b00, 0b01, 0b10
MOD_SUFFIX = {MOD_NONE: "", MOD_U: ".u", MOD_H: ".h"}

# Operand-shape kinds: Keeps parsing organized
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
    float_ok: bool = False      # Does an F=1 version exist for this instruction?


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
    # 10111 and 11000 are free / reserved opcodes -> we just decode them as nop
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

# Assembler names for the F=1 float instructions (they share the same opcode as the integer version)
FLOAT_MNEMONICS = {
    "fadd": "add", "fsub": "sub", "fmul": "mul", "fdiv": "div",
    "fcmp": "cmp", "fmov": "mov", "fmin": "min", "fmax": "max",
    "fabs": "abs", "fneg": "neg",
}
FLOAT_NAME_OF = {v: k for k, v in FLOAT_MNEMONICS.items()}
# For fcvt, the F bit means "source is float". So fcvt is int->float (F=0), and fcvti is float->int (F=1)
FCVT_MNEMONICS = {"fcvt": 0, "i2f": 0, "fcvti": 1, "f2i": 1}

ALL_MNEMONICS = sorted(set(OPS_BY_NAME) | set(FLOAT_MNEMONICS) | set(FCVT_MNEMONICS))


def resolve_mnemonic(text: str):
    """Returns a tuple of (OpInfo, F_bit, modifier) for a given assembly mnemonic. Or None if it's junk.

    Handles standard ops (add), float ops (fadd), conversions (fcvt/fcvti), 
    and immediate modifiers (add.u / addu / mov.h / movh).
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
# Registers: Parsing and formatting logic
# --------------------------------------------------------------------------
_REG_RE = re.compile(r"^(?:r(\d+)|(sp)|(ra))$", re.I)


def parse_register(tok: str) -> Optional[int]:
    """Returns the integer register number. If it doesn't look like a register, returns None.
    Throws a ValueError if someone tries to use a non-existent register like r16."""
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
# Bit helpers: Doing the heavy lifting for instruction packing and unpacking
# --------------------------------------------------------------------------
def sext(value: int, bits: int) -> int:
    value &= (1 << bits) - 1
    return value - (1 << bits) if value >> (bits - 1) else value


def pack_word(opcode: int, i: int = 0, f: int = 0, rd: int = 0, rs1: int = 0, rest: int = 0) -> int:
    return ((opcode << 27) | (i << 26) | (f << 25) | (rd << 21) | (rs1 << 17) | (rest & 0x1FFFF))


def pack_branch(opcode: int, offset: int) -> int:
    return (opcode << 27) | (offset & 0x7FFFFFF)


class Fields(NamedTuple):
    opcode: int
    i: int
    f: int
    rd: int
    rs1: int
    rs2: int
    mod: int
    imm15: int
    imm17: int
    offset27: int


def unpack_word(word: int) -> Fields:
    if not 0 <= word <= WORD_MASK:
        raise ValueError(f"word 0x{word:X} does not fit in {WORD_BITS} bits")
    return Fields(
        opcode=(word >> 27) & 0x1F, i=(word >> 26) & 1, f=(word >> 25) & 1,
        rd=(word >> 21) & 0xF, rs1=(word >> 17) & 0xF, rs2=(word >> 13) & 0xF,
        mod=(word >> 15) & 0x3, imm15=word & 0x7FFF, imm17=word & 0x1FFFF,
        offset27=word & 0x7FFFFFF,
    )


# --------------------------------------------------------------------------
# IEEE-754 binary32 helpers: Managing float immediate conversions
# --------------------------------------------------------------------------
def float_to_bits32(x: float) -> int:
    try:
        return struct.unpack("<I", struct.pack("<f", x))[0]
    except OverflowError:                      # If it is too big for binary32, just cap it to +/-Inf
        return 0xFF800000 if x < 0 else 0x7F800000


def bits32_to_float(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits & 0xFFFFFFFF))[0]


# --------------------------------------------------------------------------
# Program image text formats (hex = 8 digits/line, bin = 32 chars/line)
# --------------------------------------------------------------------------
def format_hex(word: int) -> str:
    return f"{word:0{HEX_DIGITS}X}"


def format_bin(word: int) -> str:
    return f"{word:0{WORD_BITS}b}"


_COMMENT_RE = re.compile(r";|//|#")


def parse_word_token(tok: str) -> int:
    """Parses a single 32-bit word token.

    Handles standard 8-digit hex, 0xHEX, 0bBIN, or full 32-character binary strings.
    If the token is exactly 8 hex digits it assumes it's hex, so '0B044800' is a hex word, 
    not a 0b-prefixed binary number.
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
    """Parses an entire hex/binary image block. Allows one or more words per line, and ignores comments."""
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
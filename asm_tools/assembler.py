#!/usr/bin/env python3
"""
assembler.py - My two-pass assembler for the RISC201 COA project (32-bit instructions).

Pass 1: Read every line, collect all the labels and .equ constants, and figure out the memory addresses.
Pass 2: Actually encode every instruction into 32-bit words and resolve those branch labels (PC-relative).

Library use (for when my team hooks this up to the CPU simulator):
    from assembler import assemble_source
    result = assemble_source(text)          # Throws AssemblyFailed if the code is garbage
    result.words                            # list[int], one 32-bit word per instruction

Command line stuff:
    python assembler.py prog.asm                 # -> prog.hex  (8 hex digits / line)
    python assembler.py prog.asm -f bin          # -> prog.bin  (32 binary digits / line)
    python assembler.py prog.asm -l prog.lst -s  # Spits out the listing file + symbol table
"""
from __future__ import annotations

import argparse
import difflib
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

try:
    from . import isa
except ImportError:                      # Fallback just in case I run this directly as a plain script
    import isa

# ==========================================================================
# Diagnostics: Basically how I track errors so I don't lose my mind debugging.
# ==========================================================================


class AsmError(Exception):
    """Just an internal exception for a single error on a source line."""


@dataclass
class Diagnostic:
    level: str            # "error" | "warning"
    line: int
    message: str
    source: str = ""

    def format(self, filename: str = "<input>") -> str:
        out = f"{filename}:{self.line}: {self.level}: {self.message}"
        if self.source.strip():
            out += f"\n    {self.source.strip()}"
        return out


class AssemblyFailed(Exception):
    def __init__(self, diagnostics: List[Diagnostic], filename: str = "<input>"):
        self.diagnostics = diagnostics
        self.filename = filename
        super().__init__(self.__str__())

    @property
    def errors(self) -> List[Diagnostic]:
        return [d for d in self.diagnostics if d.level == "error"]

    def __str__(self) -> str:
        return "\n".join(d.format(self.filename) for d in self.diagnostics)


# ==========================================================================
# Constant-expression evaluator. Handles numbers, symbols, and basic math (+ - * / << >> & | ^ ~ ()).
# Needed this to evaluate the weird .equ directives.
# ==========================================================================


class ExprError(ValueError):
    pass


_NUM = r"0[xX][0-9a-fA-F_]+|0[bB][01_]+|\d[\d_]*"
_TOKEN_RE = re.compile(rf"\s*(?:({_NUM})|([A-Za-z_.][A-Za-z0-9_.$]*)|(<<|>>|[-+*/&|^~()]))")


def _tokenize(expr: str):
    toks = []
    pos = 0
    expr = expr.rstrip()
    while pos < len(expr):
        m = _TOKEN_RE.match(expr, pos)
        if not m:
            raise ExprError(f"unexpected character {expr[pos:].strip()[0]!r} in '{expr}'")
        if m.group(1):
            s = m.group(1).replace("_", "").lower()
            val = int(s[2:], 16) if s.startswith("0x") else int(s[2:], 2) if s.startswith("0b") else int(s, 10)
            toks.append(("num", val))
        elif m.group(2):
            toks.append(("id", m.group(2)))
        else:
            toks.append(("op", m.group(3)))
        pos = m.end()
    return toks


def eval_expr(expr: str, symbols: Dict[str, int]) -> int:
    toks = _tokenize(expr)
    if not toks:
        raise ExprError("missing value")
    pos = 0

    def peek():
        return toks[pos] if pos < len(toks) else (None, None)

    def take():
        nonlocal pos
        t = toks[pos]
        pos += 1
        return t

    def binary(sub, ops, apply):
        left = sub()
        while peek() in [("op", o) for o in ops]:
            op = take()[1]
            left = apply(op, left, sub())
        return left

    def primary():
        kind, val = peek()
        if kind is None:
            raise ExprError(f"incomplete expression '{expr}'")
        take()
        if kind == "num":
            return val
        if kind == "id":
            if val not in symbols:
                raise ExprError(f"undefined symbol '{val}'")
            return symbols[val]
        if val == "(":
            v = parse_or()
            if peek() != ("op", ")"):
                raise ExprError("missing ')'")
            take()
            return v
        raise ExprError(f"unexpected '{val}' in '{expr}'")

    def unary():
        if peek() in (("op", "-"), ("op", "+"), ("op", "~")):
            op = take()[1]
            v = unary()
            return -v if op == "-" else ~v if op == "~" else v
        return primary()

    def div(a, b):
        if b == 0:
            raise ExprError("division by zero in constant expression")
        q = abs(a) // abs(b)
        return q if (a >= 0) == (b >= 0) else -q

    def mul():
        return binary(unary, "*/", lambda o, a, b: a * b if o == "*" else div(a, b))

    def add():
        return binary(mul, "+-", lambda o, a, b: a + b if o == "+" else a - b)

    def shift():
        return binary(add, ("<<", ">>"), lambda o, a, b: a << b if o == "<<" else a >> b)

    def band():
        return binary(shift, "&", lambda o, a, b: a & b)

    def bxor():
        return binary(band, "^", lambda o, a, b: a ^ b)

    def parse_or():
        return binary(bxor, "|", lambda o, a, b: a | b)

    value = parse_or()
    if pos != len(toks):
        raise ExprError(f"unexpected '{toks[pos][1]}' in '{expr}'")
    return value


# ==========================================================================
# Source-line structures: Dataclasses to keep my parsed data organized.
# ==========================================================================

_IDENT_RE = re.compile(r"^[A-Za-z_.][A-Za-z0-9_.$]*$")
_LABEL_RE = re.compile(r"^\s*([A-Za-z_.][A-Za-z0-9_.$]*)\s*:")
_FLOAT_RE = re.compile(
    r"^[+-]?(?:(?:\d[\d_]*\.\d*|\.\d+|\d[\d_]*)(?:[eE][+-]?\d+)?|inf|infinity|nan)$", re.I)
_HEX_RE = re.compile(r"^0[xX][0-9a-fA-F_]+$")
_MEM_RE = re.compile(r"^(?P<off>.*?)\[\s*(?P<reg>[^\]]+?)\s*\]$")


@dataclass
class Symbol:
    name: str
    value: int
    kind: str          # Either a "label" or an "equ" constant
    line: int


@dataclass
class SrcLine:
    lineno: int
    raw: str
    addr: int
    kind: str                       # Can be an actual "instr" or just a raw "word"
    mnemonic: str = ""
    resolved: Optional[tuple] = None
    operands: List[str] = field(default_factory=list)


@dataclass
class ListingItem:
    address: int
    word: int
    line: int
    source: str


@dataclass
class AssemblyResult:
    words: List[int]
    items: List[ListingItem]
    symbols: Dict[str, Symbol]
    warnings: List[Diagnostic]
    base: int = 0
    filename: str = "<input>"

    def hex_lines(self) -> List[str]:
        return [isa.format_hex(w) for w in self.words]

    def bin_lines(self) -> List[str]:
        return [isa.format_bin(w) for w in self.words]

    def listing(self) -> str:
        out = ["ADDRESS   WORD       LINE  SOURCE"]
        for it in self.items:
            out.append(f"{it.address:08X}  {isa.format_hex(it.word)}  {it.line:4d}  {it.source.strip()}")
        return "\n".join(out) + "\n"

    def symbol_table(self) -> str:
        out = ["SYMBOL                           VALUE      KIND"]
        for s in sorted(self.symbols.values(), key=lambda s: (s.value, s.name)):
            out.append(f"{s.name:<32} 0x{s.value & 0xFFFFFFFF:08X} {s.kind}")
        return "\n".join(out) + "\n"


def strip_comment(line: str) -> str:
    return re.split(r";|//|@", line, maxsplit=1)[0]


def split_operands(args: str) -> List[str]:
    """Splits the operands by commas, but ignores commas hiding inside brackets so memory offsets don't break."""
    args = args.strip()
    if not args:
        return []
    parts, depth, cur = [], 0, []
    for ch in args:
        if ch in "[(":
            depth += 1
        elif ch in "])":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    parts.append("".join(cur).strip())
    if any(p == "" for p in parts):
        raise AsmError("empty operand (stray comma?)")
    return parts


# ==========================================================================
# The main Assembler class. Where the magic happens.
# ==========================================================================


class Assembler:
    def __init__(self, base: int = 0, strict: bool = False):
        self.base = base
        self.strict = strict            # If strict is on, warnings become fatal errors. Helpful for the final eval.

    # ---------------------------------------------------------------- driver
    def assemble(self, source: str, filename: str = "<input>") -> AssemblyResult:
        self.filename = filename
        self.symbols: Dict[str, Symbol] = {}
        self.diags: List[Diagnostic] = []
        self.src_lines = source.splitlines()
        self.lines: List[SrcLine] = []

        self._pass1()
        items, words = self._pass2()

        if self.strict:
            for d in self.diags:
                if d.level == "warning":
                    d.level = "error"
        if any(d.level == "error" for d in self.diags):
            raise AssemblyFailed(sorted(self.diags, key=lambda d: d.line), filename)
        warnings = [d for d in self.diags if d.level == "warning"]
        return AssemblyResult(words, items, self.symbols, warnings, self.base, filename)

    # --------------------------------------------------------- diagnostics
    def _error(self, lineno: int, msg: str):
        self.diags.append(Diagnostic("error", lineno, msg, self.src_lines[lineno - 1]))

    def _warn(self, lineno: int, msg: str):
        self.diags.append(Diagnostic("warning", lineno, msg, self.src_lines[lineno - 1]))

    # ---------------------------------------------------------------- PASS 1
    def _pass1(self):
        # Pass 1: Just sweeping through to collect labels, .equ constants, and figure out memory addresses.
        addr = self.base
        for lineno, raw in enumerate(self.src_lines, 1):
            try:
                rest = strip_comment(raw)
                while True:                                  # Grab any leading labels first
                    m = _LABEL_RE.match(rest)
                    if not m:
                        break
                    self._define_label(m.group(1), addr, lineno)
                    rest = rest[m.end():]
                rest = rest.strip()
                if not rest:
                    continue
                head_parts = rest.split(None, 1)
                head = head_parts[0]
                args = head_parts[1] if len(head_parts) > 1 else ""

                if head.startswith("."):
                    addr = self._directive(head.lower(), args, lineno, raw, addr)
                else:
                    resolved = isa.resolve_mnemonic(head)
                    if resolved is None:
                        close = difflib.get_close_matches(head.lower(), isa.ALL_MNEMONICS, n=1)
                        hint = f" (did you mean '{close[0]}'?)" if close else ""
                        raise AsmError(f"unknown instruction '{head}'{hint}")
                    self.lines.append(SrcLine(lineno, raw, addr, "instr", head, resolved,
                                              split_operands(args)))
                    addr += isa.INSTR_BYTES
            except AsmError as exc:
                self._error(lineno, str(exc))

    def _define_label(self, name: str, addr: int, lineno: int):
        try:
            if isa.parse_register(name) is not None:
                raise AsmError(f"'{name}' is a register name and cannot be used as a label")
        except ValueError:
            pass
        if name in self.symbols:
            raise AsmError(f"duplicate symbol '{name}' (first defined on line {self.symbols[name].line})")
        self.symbols[name] = Symbol(name, addr, "label", lineno)

    def _directive(self, name: str, args: str, lineno: int, raw: str, addr: int) -> int:
        if name in (".equ", ".set"):
            parts = args.split(",", 1)
            if len(parts) != 2 or not _IDENT_RE.match(parts[0].strip()):
                raise AsmError(f"usage: {name} NAME, expression")
            sym = parts[0].strip()
            if sym in self.symbols:
                raise AsmError(f"duplicate symbol '{sym}' (first defined on line {self.symbols[sym].line})")
            value = self._eval(parts[1], {k: s.value for k, s in self.symbols.items()})
            self.symbols[sym] = Symbol(sym, value, "equ", lineno)
            return addr
        if name == ".word":
            exprs = split_operands(args)
            if not exprs:
                raise AsmError("usage: .word value[, value...]  (raw 32-bit instruction words)")
            for e in exprs:
                self.lines.append(SrcLine(lineno, raw, addr, "word", ".word", None, [e]))
                addr += isa.INSTR_BYTES
            return addr
        if name in (".text", ".global", ".globl"):
            return addr
        raise AsmError(f"unknown directive '{name}' (supported: .equ .set .word .text .global)")

    # ---------------------------------------------------------------- PASS 2
    def _pass2(self) -> Tuple[List[ListingItem], List[int]]:
        # Pass 2: Actually packing the 32-bit words now that we know where all the labels are.
        items, words = [], []
        for ln in self.lines:
            try:
                word = self._encode_word(ln) if ln.kind == "word" else self._encode(ln)
            except AsmError as exc:
                self._error(ln.lineno, str(exc))
                continue
            words.append(word)
            items.append(ListingItem(ln.addr, word, ln.lineno, ln.raw))
        return items, words

    # -------------------------------------------------------------- helpers
    def _sym_values(self) -> Dict[str, int]:
        return {k: s.value for k, s in self.symbols.items()}

    def _eval(self, expr: str, symbols: Optional[Dict[str, int]] = None) -> int:
        try:
            return eval_expr(expr, self._sym_values() if symbols is None else symbols)
        except ExprError as exc:
            raise AsmError(str(exc)) from None

    @staticmethod
    def _reg(tok: str, what: str = "register") -> int:
        try:
            r = isa.parse_register(tok)
        except ValueError as exc:
            raise AsmError(str(exc)) from None
        if r is None:
            raise AsmError(f"expected a {what} (r0-r15, sp, ra) but got '{tok}'")
        return r

    def _float_imm(self, tok: str, lineno: int) -> int:
        """Returns imm17, which is just the top 17 bits of the binary32 float value."""
        t = tok.strip()
        if _HEX_RE.match(t):                         # Reading the raw binary32 bit pattern
            bits = int(t.replace("_", ""), 16)
            if bits >> 32:
                raise AsmError(f"float bit pattern {t} does not fit in 32 bits")
        else:
            if _FLOAT_RE.match(t):
                value = float(t.replace("_", ""))
            else:
                value = float(self._eval(t))
            bits = isa.float_to_bits32(value)
        
        # We only have 17 bits for the float immediate, so we have to warn if we truncate precision.
        if bits & 0x7FFF:
            kept = isa.bits32_to_float(bits & ~0x7FFF)
            self._warn(lineno, f"float immediate '{t}' needs more than 17 bits; truncated to {kept:.9g} "
                               f"(build full-precision constants with ld)")
        return bits >> 15

    def _op2(self, tok: str, F: int, mod: int, mnemonic: str, lineno: int) -> Tuple[int, int]:
        """Parses the second operand. Figures out if it is a register or immediate and sets the I bit."""
        reg = isa.parse_register(tok) if self._looks_like_reg(tok) else None
        if reg is not None:
            if mod:
                raise AsmError(f"modifier '{isa.MOD_SUFFIX[mod]}' only applies to an immediate operand")
            return 0, reg << 13
        if F:
            return 1, self._float_imm(tok, lineno)
        val = self._eval(tok)
        
        # Checking bounds based on whether it is raw, zero-extended (.u), or upper-half (.h)
        if mod == isa.MOD_NONE:
            lo, hi = -16384, 16383
        else:
            lo, hi = 0, 32767
            
        if not lo <= val <= hi:
            hint = ""
            if mod == isa.MOD_NONE and 16384 <= val <= 32767:
                hint = " - use the .u (zero-extend) or .h (upper half) form, e.g. " \
                       f"'{mnemonic}.u'"
            raise AsmError(f"immediate {val} out of range [{lo}, {hi}]{hint}")
        return 1, (mod << 15) | (val & 0x7FFF)

    @staticmethod
    def _looks_like_reg(tok: str) -> bool:
        return bool(re.match(r"^(r\d+|sp|ra)$", tok.strip(), re.I))

    # ---------------------------------------------------------------- encode
    def _encode_word(self, ln: SrcLine) -> int:
        val = self._eval(ln.operands[0])
        if not 0 <= val <= isa.WORD_MASK:
            raise AsmError(f".word value {val} does not fit in {isa.WORD_BITS} bits")
        return val

    def _encode(self, ln: SrcLine) -> int:
        # This is the giant instruction packer. Figures out the instruction type and packs it into the 32-bit word.
        op, F, mod = ln.resolved
        ops = ln.operands
        kind = op.kind
        name = ln.mnemonic

        def need(n: int):
            if len(ops) != n:
                syntax = isa.SYNTAX[kind]
                raise AsmError(f"'{name}' expects {n} operand(s): {name} {syntax}".rstrip()
                               + f" (got {len(ops)})")

        if kind == isa.K_NONE:
            need(0)
            return isa.pack_word(op.opcode)

        if kind == isa.K_STACK:
            need(1)
            return isa.pack_word(op.opcode, rd=self._reg(ops[0]))

        if kind == isa.K_BRANCH:
            need(1)
            return isa.pack_branch(op.opcode, self._branch_offset(ops[0], ln))

        if kind == isa.K_LDST:
            need(2)
            rd = self._reg(ops[0])
            m = _MEM_RE.match(ops[1])
            if not m:
                raise AsmError(f"memory operand must look like imm[rs1], e.g. 4[r2] or [sp] (got '{ops[1]}')")
            rs1 = self._reg(m.group("reg"), "base register")
            off = self._eval(m.group("off")) if m.group("off").strip() else 0
            if not -16384 <= off <= 16383:
                raise AsmError(f"offset {off} out of range [-16384, 16383]")
            return isa.pack_word(op.opcode, 1, 0, rd, rs1, off & 0x7FFF)

        if kind == isa.K_UN2:                      # For abs, neg, fcvt: register only allowed here.
            need(2)
            rd = self._reg(ops[0])
            if not self._looks_like_reg(ops[1]):
                raise AsmError(f"'{name}' only accepts a register source (got '{ops[1]}')")
            rs2 = self._reg(ops[1])
            return isa.pack_word(op.opcode, 0, F, rd, 0, rs2 << 13)

        if kind == isa.K_CMP:
            need(2)
            rs1 = self._reg(ops[0])
            i, rest = self._op2(ops[1], F, mod, name, ln.lineno)
            return isa.pack_word(op.opcode, i, F, 0, rs1, rest)

        if kind == isa.K_MOV2:
            need(2)
            rd = self._reg(ops[0])
            i, rest = self._op2(ops[1], F, mod, name, ln.lineno)
            return isa.pack_word(op.opcode, i, F, rd, 0, rest)

        if kind == isa.K_ALU3:
            need(3)
            rd = self._reg(ops[0])
            rs1 = self._reg(ops[1])
            i, rest = self._op2(ops[2], F, mod, name, ln.lineno)
            return isa.pack_word(op.opcode, i, F, rd, rs1, rest)

        raise AsmError(f"internal error: unhandled kind {kind}")      # pragma: no cover

    def _branch_offset(self, tok: str, ln: SrcLine) -> int:
        """If it is a label, do the PC-relative math to get the word offset. Otherwise just use the raw offset."""
        tok = tok.strip()
        if _IDENT_RE.match(tok) and not self._looks_like_reg(tok):
            sym = self.symbols.get(tok)
            if sym is None:
                close = difflib.get_close_matches(tok, [s.name for s in self.symbols.values()
                                                        if s.kind == "label"], n=1)
                hint = f" (did you mean '{close[0]}'?)" if close else ""
                raise AsmError(f"undefined label '{tok}'{hint}")
            if sym.kind == "label":
                delta = sym.value - ln.addr
                if delta % isa.INSTR_BYTES:
                    raise AsmError(f"branch target '{tok}' is not word aligned")
                off = delta // isa.INSTR_BYTES
            else:
                off = sym.value
        else:
            off = self._eval(tok)
        if not -(1 << 26) <= off < (1 << 26):
            raise AsmError(f"branch offset {off} does not fit in 27 signed bits")
        return off


# ==========================================================================
# Convenience API + CLI: Stuff to make running this locally easier.
# ==========================================================================


def assemble_source(source: str, base: int = 0, strict: bool = False,
                    filename: str = "<input>") -> AssemblyResult:
    """Assembles the program text. Throws AssemblyFailed if there are errors so I can catch them."""
    return Assembler(base=base, strict=strict).assemble(source, filename)


def assemble_file(path: str, base: int = 0, strict: bool = False) -> AssemblyResult:
    with open(path, "r", encoding="utf-8") as fh:
        return assemble_source(fh.read(), base, strict, filename=path)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="RISC201 two-pass assembler (32-bit instructions)")
    ap.add_argument("source", help="assembly file (.asm / .s)")
    ap.add_argument("-o", "--output", help="output file (default: <source>.hex or .bin)")
    ap.add_argument("-f", "--format", choices=["hex", "bin"], default="hex",
                    help="hex = 8 hex digits/line (default); bin = 32 binary digits/line")
    ap.add_argument("-l", "--listing", metavar="FILE", help="write address/word/source listing")
    ap.add_argument("-s", "--symbols", action="store_true", help="print the symbol table")
    ap.add_argument("--base", type=lambda x: int(x, 0), default=0,
                    help="address of the first instruction (default 0)")
    ap.add_argument("--strict", action="store_true", help="treat warnings as errors")
    ap.add_argument("-q", "--quiet", action="store_true", help="suppress warnings and summary")
    args = ap.parse_args(argv)

    try:
        result = assemble_file(args.source, base=args.base, strict=args.strict)
    except FileNotFoundError:
        print(f"error: cannot open '{args.source}'", file=sys.stderr)
        return 2
    except AssemblyFailed as exc:
        print(exc, file=sys.stderr)
        n = len(exc.errors)
        print(f"\nassembly failed: {n} error{'s' if n != 1 else ''}", file=sys.stderr)
        return 1

    out = args.output or os.path.splitext(args.source)[0] + (".hex" if args.format == "hex" else ".bin")
    lines = result.hex_lines() if args.format == "hex" else result.bin_lines()
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + ("\n" if lines else ""))
    if args.listing:
        with open(args.listing, "w", encoding="utf-8") as fh:
            fh.write(result.listing())
    if not args.quiet:
        for w in result.warnings:
            print(w.format(args.source), file=sys.stderr)
        print(f"assembled {len(result.words)} word(s) -> {out}")
    if args.symbols:
        print(result.symbol_table(), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
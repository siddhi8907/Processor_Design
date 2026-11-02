# RISC201 Assembler & Disassembler

Two-pass assembler and disassembler for the 32-bit RISC201 (SimpleRISC Extended) ISA.
Works for **both** the 4-stage and 6-stage pipelines: the instruction encoding is identical.
Pure Python 3.8+, no dependencies.

| File | Purpose |
|---|---|
| `isa.py` | ISA tables, bit layout, float helpers, hex/bin image parsing. **Shared by everyone.** |
| `assembler.py` | Two-pass assembler + CLI |
| `disassembler.py` | Disassembler + CLI |
| `tests/test_toolchain.py` | 54 unit tests (incl. 30k random-word round trips and new-ISA boundary tests) |
| `examples/` | Sample programs (`sum_stack.asm`, `float_demo.asm`, `errors_demo.asm`) |

## Usage

```bash
python assembler.py examples/sum_stack.asm                # -> sum_stack.hex (8 hex digits / line)
python assembler.py prog.asm -f bin                       # -> prog.bin (32 binary digits / line)
python assembler.py prog.asm -l prog.lst -s               # listing file + symbol table
python assembler.py prog.asm --strict                     # warnings become errors

python disassembler.py prog.hex                           # prints assembly (with L_xxxx labels)
python disassembler.py prog.hex --annotate -o prog.asm    # address + raw word as comments
python disassembler.py --words 4C200005 C8000000        # inline words

python -m unittest discover -s tests -v                   # run the tests
```

### From other modules (simulator / debugger)

```python
from assembler import assemble_source, AssemblyFailed
from disassembler import disassemble_word, load_words

result = assemble_source(open("prog.asm").read())   # raises AssemblyFailed with every error
imem = result.words                                  # list of 32-bit ints, imem[pc // 4]
print(disassemble_word(imem[2]))                     # text for the pipeline viewer / debugger
```
The stack-macro preprocessor can simply return expanded text and feed it to `assemble_source`.

## Assembly syntax

```
label:  mnemonic  op1, op2, op3     ; comment   (also // and @)
```
* Registers `r0`-`r15`, `sp` = r14, `ra` = r15 (case-insensitive).
* Numbers: `10`, `-4`, `0xFF`, `0b1010`. Constant expressions allowed: `N*4+1`, `(1<<8)-1`.
* Directives: `.equ NAME, expr` (or `.set`), `.word value` (raw 32-bit word), `.text`, `.global` (ignored).
* Instruction addresses start at 0 (`--base` to change); PC advances by 4.

| Form | Example |
|---|---|
| 3-operand | `add r1, r2, r3` / `add r1, r2, 10` (add sub mul div mod and or lsl lsr asr min max) |
| compare | `cmp r1, r2` / `cmp r1, 5` |
| move/not | `mov r1, r2` / `mov r1, 100` / `not r1, r2` |
| unary | `abs r1, r2`, `neg r1, r2` (register only) |
| memory | `ld r1, 8[r2]`, `st r1, -4[sp]`, `ld r1, [r2]` |
| stack | `push r1`, `pop r1` |
| branch | `beq L`, `bgt L`, `b L`, `call L` (label -> PC-relative word offset); a plain number is a raw word offset |
| no operand | `nop`, `ret`, `halt`, `rte` |

**Immediate modifiers** (bits [16:15]): none = `00` signed 15-bit (-16384..16383); `.u` = `01` zero-extended (0..32767);
`.h` = `10` imm15 << 16 (0..32767). `ld`/`st` offsets are signed 15-bit (-16384..16383). Both `mov.h` and `movh` spellings work. Register form + modifier is an error.

**Float (F = 1)** - same opcode as the integer op, F bit set:
`fadd fsub fmul fdiv fcmp fmov fmin fmax fabs fneg`.
* Float immediates (`fadd r1, r2, 2.5`, `fmov r1, 1.5`) store the **top 17 bits** (sign, 8 exponent, 8 mantissa) of the binary32 value.
  If the value needs more bits the assembler **warns and truncates** (use `ld` for full-precision constants).
  A `0x...` literal in a float instruction is a raw binary32 bit pattern (`fmov r1, 0x3FC00000`).
* `fcvt rd, rs2` = int -> float (F=0). `fcvti rd, rs2` = float -> int (F=1). Aliases: `i2f`, `f2i`.

## Encoding summary (bit 31 = MSB)

```
Register : opcode[31:27] I=0 F rd[24:21] rs1[20:17] rs2[16:13] 0[12:0]
Imm F=0  : opcode I=1 F=0 rd rs1  mod[16:15] imm15[14:0]
Imm F=1  : opcode I=1 F=1 rd rs1  imm17[16:0]   (value = {imm17, 15'b0})
Branch   : opcode[31:27] offset[26:0]   target = PC + offset*4   (27-bit signed)
```
Unused fields (e.g. `rd` in `cmp`, `rs1` in `mov`) are assembled as 0.

## What changed from the previous (33-bit) version
| | Old | New (SimpleRISC_Extended_ISA_1) |
|---|---|---|
| Word width | 33 bits (9 hex digits) | **32 bits (8 hex digits)** |
| Integer immediate | 16 bits | **15 bits** (-16384..16383; `.u`/`.h` 0..32767) |
| Float immediate | top 18 bits | **top 17 bits** ({imm17, 15'b0}) |
| ld/st offset | 16-bit signed | **15-bit signed** |
| Branch offset | 28-bit signed words | **27-bit signed words** |
| Field positions | opcode[32:28] ... | opcode[31:27], I[26], F[25], rd[24:21], rs1[20:17], rs2[16:13] |

Opcodes, mnemonics, register names and assembly syntax are unchanged, so existing `.asm` files still
assemble as long as their immediates fit in 15 bits. The 4-stage and 6-stage pipeline documents use the
same encoding, so one toolchain serves both.

## Design decisions to confirm with your team / TA
1. Modifier codes `u=01`, `h=10` (ISA doc only says "modifier (u/h)"). `h` = imm15 << 16, so bits 15 and 31 cannot be set by one `mov` (as the ISA doc notes).
2. Branch offset unit = **words** (target = PC + offset << 2, as in the pipeline docs).
3. `fcvt` / `fcvti` naming for the two directions.
4. Float immediate truncation (warning) instead of rounding.
5. Reserved opcodes `10111`, `11000` are disassembled as `.word` (they execute as `nop`).
6. Image format: one word per line, 8 upper-case hex digits (or 32 binary digits).

## Disassembler guarantee
`assemble(disassemble(w)) == w` for **every** 32-bit word. Anything that is not a canonical valid
instruction (reserved opcode, stray bits in unused fields, F=1 on `and`, ...) is printed as
`.word 0x...` with an explanatory comment.

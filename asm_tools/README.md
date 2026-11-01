# RISC201 Assembler & Disassembler

Two-pass assembler and disassembler for the 33-bit RISC201 (SimpleRISC Extended) ISA.
Works for **both** the 4-stage and 6-stage pipelines: the instruction encoding is identical.
Pure Python 3.8+, no dependencies.

| File | Purpose |
|---|---|
| `isa.py` | ISA tables, bit layout, float helpers, hex/bin image parsing. **Shared by everyone.** |
| `assembler.py` | Two-pass assembler + CLI |
| `disassembler.py` | Disassembler + CLI |
| `tests/test_toolchain.py` | 47 unit tests (incl. 30k random-word round trips) |
| `examples/` | Sample programs (`sum_stack.asm`, `float_demo.asm`, `errors_demo.asm`) |

## Usage

```bash
python assembler.py examples/sum_stack.asm                # -> sum_stack.hex (9 hex digits / line)
python assembler.py prog.asm -f bin                       # -> prog.bin (33 binary digits / line)
python assembler.py prog.asm -l prog.lst -s               # listing file + symbol table
python assembler.py prog.asm --strict                     # warnings become errors

python disassembler.py prog.hex                           # prints assembly (with L_xxxx labels)
python disassembler.py prog.hex --annotate -o prog.asm    # address + raw word as comments
python disassembler.py --words 098400005 190000000        # inline words

python -m unittest discover -s tests -v                   # run the tests
```

### From other modules (simulator / debugger)

```python
from assembler import assemble_source, AssemblyFailed
from disassembler import disassemble_word, load_words

result = assemble_source(open("prog.asm").read())   # raises AssemblyFailed with every error
imem = result.words                                  # list of 33-bit ints, imem[pc // 4]
print(disassemble_word(imem[2]))                     # text for the pipeline viewer / debugger
```
The stack-macro preprocessor can simply return expanded text and feed it to `assemble_source`.

## Assembly syntax

```
label:  mnemonic  op1, op2, op3     ; comment   (also // and @)
```
* Registers `r0`-`r15`, `sp` = r14, `ra` = r15 (case-insensitive).
* Numbers: `10`, `-4`, `0xFF`, `0b1010`. Constant expressions allowed: `N*4+1`, `(1<<8)-1`.
* Directives: `.equ NAME, expr` (or `.set`), `.word value` (raw 33-bit word), `.text`, `.global` (ignored).
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

**Immediate modifiers** (bits [17:16]): none = `00` signed 16-bit (-32768..32767); `.u` = `01` zero-extended (0..65535);
`.h` = `10` upper half (0..65535). Both `mov.h` and `movh` spellings work. Register form + modifier is an error.

**Float (F = 1)** - same opcode as the integer op, F bit set:
`fadd fsub fmul fdiv fcmp fmov fmin fmax fabs fneg`.
* Float immediates (`fadd r1, r2, 2.5`, `fmov r1, 1.5`) store the **top 18 bits** of the binary32 value.
  If the value needs more bits the assembler **warns and truncates** (use `ld` for full-precision constants).
  A `0x...` literal in a float instruction is a raw binary32 bit pattern (`fmov r1, 0x3FC00000`).
* `fcvt rd, rs2` = int -> float (F=0). `fcvti rd, rs2` = float -> int (F=1). Aliases: `i2f`, `f2i`.

## Encoding summary (bit 32 = MSB)

```
Register : opcode[32:28] I=0 F rd[25:22] rs1[21:18] rs2[17:14] 0[13:0]
Imm F=0  : opcode I=1 F=0 rd rs1  mod[17:16] imm16[15:0]
Imm F=1  : opcode I=1 F=1 rd rs1  imm18[17:0]
Branch   : opcode[32:28] offset[27:0]   target = PC + offset*4
```
Unused fields (e.g. `rd` in `cmp`, `rs1` in `mov`) are assembled as 0.

## Design decisions to confirm with your team / TA
1. Modifier codes `u=01`, `h=10` (ISA doc only says "modifier (u/h)").
2. Branch offset unit = **words** (target = PC + offset << 2, as in the pipeline docs).
3. `fcvt` / `fcvti` naming for the two directions.
4. Float immediate truncation (warning) instead of rounding.
5. Reserved opcodes `10111`, `11000` are disassembled as `.word` (they execute as `nop`).
6. Image format: one word per line, 9 upper-case hex digits (or 33 binary digits).

## Disassembler guarantee
`assemble(disassemble(w)) == w` for **every** 33-bit word. Anything that is not a canonical valid
instruction (reserved opcode, stray bits in unused fields, F=1 on `and`, ...) is printed as
`.word 0x...` with an explanatory comment.

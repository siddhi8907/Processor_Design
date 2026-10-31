"""
RISC-201 Processor Simulation Framework
Module: ALU & FPU Unit (CSC-201 SimpleRISC Extended)
"""

import struct

MASK32 = 0xFFFFFFFF
SIGN32 = 0x80000000


def to_signed(val: int) -> int:
    val &= MASK32
    return val - 0x100000000 if val & SIGN32 else val


def to_unsigned(val: int) -> int:
    return val & MASK32


# -------------------------------------------------------------------------
# 1. ADDER IMPLEMENTATIONS
# -------------------------------------------------------------------------
def ripple_carry_adder_32(a: int, b: int, cin: int = 0):
    a &= MASK32
    b &= MASK32
    s = 0
    c = cin
    for i in range(32):
        ai = (a >> i) & 1
        bi = (b >> i) & 1
        sum_bit = ai ^ bi ^ c
        c = (ai & bi) | (bi & c) | (ai & c)
        s |= (sum_bit << i)
    return s & MASK32, c


def kogge_stone_adder_32(a: int, b: int, cin: int = 0):
    """
    32-bit Kogge-Stone logarithmic prefix network (5 levels).
    Returns (sum_32, cout, overflow_flag).
    """
    a &= MASK32
    b &= MASK32

    # Level 0: Bitwise generate and propagate
    g = [((a >> i) & 1) & ((b >> i) & 1) for i in range(32)]
    p = [((a >> i) & 1) ^ ((b >> i) & 1) for i in range(32)]

    # Incorporate Cin into bit 0
    g[0] = g[0] | (p[0] & cin)
    p[0] = p[0] & (~cin & 1)

    # Prefix stages: offsets 1, 2, 4, 8, 16
    for step in [1, 2, 4, 8, 16]:
        next_g = list(g)
        next_p = list(p)
        for i in range(step, 32):
            next_g[i] = g[i] | (p[i] & g[i - step])
            next_p[i] = p[i] & p[i - step]
        g, p = next_g, next_p

    # Compute carries
    c = [cin] + [g[i] for i in range(31)]
    cout = g[31]

    # Final sum
    raw_p = [((a >> i) & 1) ^ ((b >> i) & 1) for i in range(32)]
    s = 0
    for i in range(32):
        s |= ((raw_p[i] ^ c[i]) << i)

    # V = Overflow flag for signed addition
    c31 = c[31]
    v = cout ^ c31
    return s & MASK32, cout, v


# -------------------------------------------------------------------------
# 2. MULTIPLIER (Booth Radix-4)
# -------------------------------------------------------------------------
def booth_radix4_mul_32(a: int, b: int):
    """
    Radix-4 Booth's Multiplier (16 steps).
    Computes low 32 bits (identical for signed & unsigned).
    """
    m = to_signed(a)
    r = to_signed(b) & MASK32
    # Extend with phantom bit b_{-1} = 0
    padded_r = (r << 1)
    acc = 0

    for i in range(16):
        triplet = (padded_r >> (2 * i)) & 0b111
        if triplet in (0b001, 0b010):
            d = 1
        elif triplet == 0b011:
            d = 2
        elif triplet == 0b100:
            d = -2
        elif triplet in (0b101, 0b110):
            d = -1
        else:
            d = 0

        acc += (d * m) << (2 * i)

    return acc & MASK32


# -------------------------------------------------------------------------
# 3. DIVIDER (Non-Restoring Division)
# -------------------------------------------------------------------------
def non_restoring_div_mod_32(dividend: int, divisor: int):
    """
    32-step Non-Restoring Divider returning (quotient, remainder).
    """
    if divisor == 0:
        return MASK32, dividend & MASK32  # Spec: div -> 0xFFFFFFFF, mod -> dividend

    # Handle INT_MIN / -1 overflow edge case
    if dividend == 0x80000000 and divisor == MASK32:
        return 0x80000000, 0

    sign_d = -1 if (dividend & SIGN32) else 1
    sign_v = -1 if (divisor & SIGN32) else 1

    d_mag = abs(to_signed(dividend))
    v_mag = abs(to_signed(divisor))

    # 32-bit register [A, Q]
    A = 0
    Q = d_mag

    for _ in range(32):
        # Shift [A, Q] left by 1
        A = ((A << 1) | ((Q >> 31) & 1)) & 0x1FFFFFFFF
        Q = (Q << 1) & MASK32

        if A & 0x100000000:  # A < 0
            A = (A + v_mag) & 0x1FFFFFFFF
        else:  # A >= 0
            A = (A - v_mag) & 0x1FFFFFFFF

        if not (A & 0x100000000):  # A >= 0
            Q |= 1

    # Final correction step
    if A & 0x100000000:
        A = (A + v_mag) & 0x1FFFFFFFF

    quotient = Q
    remainder = A & MASK32

    # Fix signs
    if sign_d != sign_v:
        quotient = -quotient
    if sign_d < 0:
        remainder = -remainder

    return quotient & MASK32, remainder & MASK32


# -------------------------------------------------------------------------
# 4. BARREL SHIFTER
# -------------------------------------------------------------------------
def barrel_shift_32(val: int, amount: int, mode: str = "lsl") -> int:
    val &= MASK32
    shamt = amount & 0x1F  # Only low 5 bits used
    if mode == "lsl":
        return (val << shamt) & MASK32
    elif mode == "lsr":
        return (val >> shamt) & MASK32
    elif mode == "asr":
        signed_val = to_signed(val)
        return (signed_val >> shamt) & MASK32
    return val


# -------------------------------------------------------------------------
# 5. FLOATING POINT UNIT (IEEE-754 Single Precision / binary32)
# -------------------------------------------------------------------------
def bits_to_float(b: int) -> float:
    return struct.unpack('>f', struct.pack('>I', b & MASK32))[0]


def float_to_bits(f: float) -> int:
    return struct.unpack('>I', struct.pack('>f', f))[0]


class FloatUnit:
    @staticmethod
    def is_nan(bits: int) -> bool:
        exp = (bits >> 23) & 0xFF
        frac = bits & 0x7FFFFF
        return exp == 0xFF and frac != 0

    @staticmethod
    def fadd(a_bits: int, b_bits: int) -> int:
        fa, fb = bits_to_float(a_bits), bits_to_float(b_bits)
        return float_to_bits(fa + fb)

    @staticmethod
    def fsub(a_bits: int, b_bits: int) -> int:
        fa, fb = bits_to_float(a_bits), bits_to_float(b_bits)
        return float_to_bits(fa - fb)

    @staticmethod
    def fmul(a_bits: int, b_bits: int) -> int:
        fa, fb = bits_to_float(a_bits), bits_to_float(b_bits)
        return float_to_bits(fa * fb)

    @staticmethod
    def fdiv(a_bits: int, b_bits: int) -> int:
        fa, fb = bits_to_float(a_bits), bits_to_float(b_bits)
        if fb == 0.0:
            if fa == 0.0:
                return 0x7FC00000  # NaN
            return 0x7F800000 if fa > 0 else 0xFF800000  # +/- Inf
        return float_to_bits(fa / fb)

    @staticmethod
    def fcmp(a_bits: int, b_bits: int):
        if FloatUnit.is_nan(a_bits) or FloatUnit.is_nan(b_bits):
            return 0, 0  # Unordered: flags.E = 0, flags.GT = 0
        fa, fb = bits_to_float(a_bits), bits_to_float(b_bits)
        return (1 if fa == fb else 0), (1 if fa > fb else 0)

    @staticmethod
    def fmin(a_bits: int, b_bits: int) -> int:
        if FloatUnit.is_nan(a_bits): return b_bits
        if FloatUnit.is_nan(b_bits): return a_bits
        return a_bits if bits_to_float(a_bits) < bits_to_float(b_bits) else b_bits

    @staticmethod
    def fmax(a_bits: int, b_bits: int) -> int:
        if FloatUnit.is_nan(a_bits): return b_bits
        if FloatUnit.is_nan(b_bits): return a_bits
        return a_bits if bits_to_float(a_bits) > bits_to_float(b_bits) else b_bits

    @staticmethod
    def fabs(bits: int) -> int:
        return bits & 0x7FFFFFFF  # Clear sign bit

    @staticmethod
    def fneg(bits: int) -> int:
        return bits ^ SIGN32  # Toggle sign bit


# -------------------------------------------------------------------------
# 6. MAIN RISC-201 ALU CLASS
# -------------------------------------------------------------------------
class RISC201_ALU:
    def __init__(self, adder_mode="kogge"):
        self.adder_mode = adder_mode

    def _add(self, a: int, b: int, cin: int = 0):
        if self.adder_mode == "kogge":
            return kogge_stone_adder_32(a, b, cin)
        s, cout = ripple_carry_adder_32(a, b, cin)
        v = (((a ^ s) & (b ^ s)) >> 31) & 1
        return s, cout, v

    def _sub(self, a: int, b: int):
        # A - B = A + (~B) + 1
        return self._add(a, (~b) & MASK32, 1)

    def execute(self, opcode: int, op1: int, op2: int, is_float: bool = False, is_fcvt: bool = False):
        """
        Executes an ALU operation.
        Returns: (result_32bit, flag_E, flag_GT, latency_cycles)
        """
        op1 &= MASK32
        op2 &= MASK32
        res = 0
        flag_e = 0
        flag_gt = 0
        latency = 1

        # 1. Type Conversion instruction (fcvt, opcode 11111)
        if is_fcvt or opcode == 0b11111:
            latency = 2
            if not is_float:
                # F = 0: int -> float (round-to-nearest-even)
                res = float_to_bits(float(to_signed(op2)))
            else:
                # F = 1: float -> int (truncate toward zero, clamp on overflow)
                fval = bits_to_float(op2)
                if FloatUnit.is_nan(op2):
                    res = 0
                else:
                    ival = int(fval)
                    ival = max(-0x80000000, min(0x7FFFFFFF, ival))
                    res = ival & MASK32
            return res, 0, 0, latency

        # 2. Floating-Point Operations (is_float == True)
        if is_float:
            if opcode == 0b00000:    # fadd
                return FloatUnit.fadd(op1, op2), 0, 0, 3
            elif opcode == 0b00001:  # fsub
                return FloatUnit.fsub(op1, op2), 0, 0, 3
            elif opcode == 0b00010:  # fmul
                return FloatUnit.fmul(op1, op2), 0, 0, 3
            elif opcode == 0b00011:  # fdiv
                return FloatUnit.fdiv(op1, op2), 0, 0, 10
            elif opcode == 0b00101:  # fcmp
                fe, fgt = FloatUnit.fcmp(op1, op2)
                return 0, fe, fgt, 1
            elif opcode == 0b01001:  # mov (float immediate / copy)
                return op2, 0, 0, 1
            elif opcode == 0b11011:  # fmin
                return FloatUnit.fmin(op1, op2), 0, 0, 1
            elif opcode == 0b11100:  # fmax
                return FloatUnit.fmax(op1, op2), 0, 0, 1
            elif opcode == 0b11101:  # fabs
                return FloatUnit.fabs(op2), 0, 0, 1
            elif opcode == 0b11110:  # fneg
                return FloatUnit.fneg(op2), 0, 0, 1

        # 3. Integer Operations (is_float == False)
        if opcode == 0b00000:    # add / ld / st / push / pop address calc
            res, _, _ = self._add(op1, op2)
            latency = 1

        elif opcode == 0b00001:  # sub
            res, _, _ = self._sub(op1, op2)
            latency = 1

        elif opcode == 0b00010:  # mul
            res = booth_radix4_mul_32(op1, op2)
            latency = 2

        elif opcode == 0b00011:  # div
            res, _ = non_restoring_div_mod_32(op1, op2)
            latency = 8

        elif opcode == 0b00100:  # mod
            _, res = non_restoring_div_mod_32(op1, op2)
            latency = 8

        elif opcode == 0b00101:  # cmp
            diff, _, v = self._sub(op1, op2)
            n = (diff >> 31) & 1
            is_less = n ^ v
            flag_e = 1 if diff == 0 else 0
            flag_gt = 1 if (diff != 0 and not is_less) else 0
            res = 0
            latency = 1

        elif opcode == 0b00110:  # and
            res = op1 & op2
        elif opcode == 0b00111:  # or
            res = op1 | op2
        elif opcode == 0b01000:  # not
            res = (~op2) & MASK32
        elif opcode == 0b01001:  # mov
            res = op2
        elif opcode == 0b01010:  # lsl
            res = barrel_shift_32(op1, op2, "lsl")
        elif opcode == 0b01011:  # lsr
            res = barrel_shift_32(op1, op2, "lsr")
        elif opcode == 0b01100:  # asr
            res = barrel_shift_32(op1, op2, "asr")

        elif opcode == 0b11011:  # min (signed)
            s1, s2 = to_signed(op1), to_signed(op2)
            res = op1 if s1 < s2 else op2

        elif opcode == 0b11100:  # max (signed)
            s1, s2 = to_signed(op1), to_signed(op2)
            res = op1 if s1 > s2 else op2

        elif opcode == 0b11101:  # abs
            res = 0x80000000 if op2 == 0x80000000 else (abs(to_signed(op2)) & MASK32)

        elif opcode == 0b11110:  # neg
            res, _, _ = self._sub(0, op2)

        return res & MASK32, flag_e, flag_gt, latency
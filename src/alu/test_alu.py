import unittest
import struct
from src.alu.alu import RISC201_ALU, float_to_bits, bits_to_float, to_signed

class TestRISC201ALU(unittest.TestCase):
    def setUp(self):
        self.alu_kogge = RISC201_ALU(adder_mode="kogge")
        self.alu_ripple = RISC201_ALU(adder_mode="ripple")

    # --- 1. Arithmetic & Adder Tests ---
    def test_add_sub(self):
        # 15 + 27 = 42
        res, e, gt, lat = self.alu_kogge.execute(0b00000, 15, 27)
        self.assertEqual(res, 42)
        self.assertEqual(lat, 1)

        # 42 - 15 = 27
        res, e, gt, _ = self.alu_kogge.execute(0b00001, 42, 15)
        self.assertEqual(res, 27)

        # Adder equivalence check: Kogge-Stone vs Ripple-Carry
        test_pairs = [(0xFFFFFFFF, 1), (12345678, 87654321), (0x7FFFFFFF, 1), (0x80000000, 0xFFFFFFFF)]
        for a, b in test_pairs:
            res_k, _, _, _ = self.alu_kogge.execute(0b00000, a, b)
            res_r, _, _, _ = self.alu_ripple.execute(0b00000, a, b)
            self.assertEqual(res_k, res_r, f"Mismatch on {hex(a)} + {hex(b)}")

    # --- 2. Compare & Architectural Flags ---
    def test_cmp_flags(self):
        # Equal
        _, e, gt, _ = self.alu_kogge.execute(0b00101, 50, 50)
        self.assertEqual((e, gt), (1, 0))

        # Greater than (positive)
        _, e, gt, _ = self.alu_kogge.execute(0b00101, 100, 25)
        self.assertEqual((e, gt), (0, 1))

        # Less than (signed: -5 < 2)
        neg5 = (-5) & 0xFFFFFFFF
        _, e, gt, _ = self.alu_kogge.execute(0b00101, neg5, 2)
        self.assertEqual((e, gt), (0, 0))

    # --- 3. Multiplier & Divider Edge Cases ---
    def test_mul_booth(self):
        # Simple positive
        res, _, _, lat = self.alu_kogge.execute(0b00010, 12, 13)
        self.assertEqual(res, 156)
        self.assertEqual(lat, 2)

        # Signed multiply: -6 * 7 = -42
        neg6 = (-6) & 0xFFFFFFFF
        res, _, _, _ = self.alu_kogge.execute(0b00010, neg6, 7)
        self.assertEqual(to_signed(res), -42)

    def test_div_mod_spec_edge_cases(self):
        # 17 / 5 = 3, 17 % 5 = 2
        q, _, _, lat = self.alu_kogge.execute(0b00011, 17, 5)
        r, _, _, _ = self.alu_kogge.execute(0b00100, 17, 5)
        self.assertEqual((q, r), (3, 2))
        self.assertEqual(lat, 8)

        # Spec: Division by zero -> div returns 0xFFFFFFFF, mod returns dividend
        q_zero, _, _, _ = self.alu_kogge.execute(0b00011, 42, 0)
        r_zero, _, _, _ = self.alu_kogge.execute(0b00100, 42, 0)
        self.assertEqual(q_zero, 0xFFFFFFFF)
        self.assertEqual(r_zero, 42)

        # Spec: INT_MIN / -1 -> returns INT_MIN
        int_min = 0x80000000
        neg_one = 0xFFFFFFFF
        q_ovf, _, _, _ = self.alu_kogge.execute(0b00011, int_min, neg_one)
        self.assertEqual(q_ovf, int_min)

    # --- 4. Neg, Abs, Min, Max Boundary Tests ---
    def test_neg_abs_boundary(self):
        # Spec edge case: 0x80000000 has no positive twin -> returns 0x80000000
        res_neg, _, _, _ = self.alu_kogge.execute(0b11110, 0, 0x80000000)
        res_abs, _, _, _ = self.alu_kogge.execute(0b11101, 0, 0x80000000)
        self.assertEqual(res_neg, 0x80000000)
        self.assertEqual(res_abs, 0x80000000)

        # Normal neg/abs
        neg10 = (-10) & 0xFFFFFFFF
        res_abs2, _, _, _ = self.alu_kogge.execute(0b11101, 0, neg10)
        self.assertEqual(res_abs2, 10)

    # --- 5. Floating Point Unit (binary32) ---
    def test_float_operations(self):
        # 1.5 + 0.25 = 1.75
        f1_5 = float_to_bits(1.5)
        f0_25 = float_to_bits(0.25)
        res_bits, _, _, lat = self.alu_kogge.execute(0b00000, f1_5, f0_25, is_float=True)
        self.assertAlmostEqual(bits_to_float(res_bits), 1.75)
        self.assertEqual(lat, 3)

        # Float cmp: NaN unordered test (both flags must be 0)
        nan_bits = 0x7FC00000
        _, fe, fgt, _ = self.alu_kogge.execute(0b00101, nan_bits, f1_5, is_float=True)
        self.assertEqual((fe, fgt), (0, 0))

    # --- 6. Converter (fcvt) ---
    def test_fcvt(self):
        # F=0: int to float (42 -> 42.0)
        res, _, _, lat = self.alu_kogge.execute(0b11111, 0, 42, is_float=False, is_fcvt=True)
        self.assertAlmostEqual(bits_to_float(res), 42.0)
        self.assertEqual(lat, 2)

        # F=1: float to int (3.75 -> 3, truncate towards zero)
        f3_75 = float_to_bits(3.75)
        res, _, _, _ = self.alu_kogge.execute(0b11111, 0, f3_75, is_float=True, is_fcvt=True)
        self.assertEqual(res, 3)

if __name__ == '__main__':
    unittest.main()
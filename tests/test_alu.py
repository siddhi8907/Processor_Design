import unittest
import sys
import os

#unittest allows us to check different components in isolation
#so we can verify all edge cases

# Include src/ in the Python import path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.alu.alu import ALU


class TestRISC201ALU(unittest.TestCase):

    def setUp(self):
        self.alu = ALU()

    def test_arithmetic_operations(self):
        """Verifies basic arithmetic and 32-bit overflow wrapping"""
        # Basic Addition
        self.assertEqual(self.alu.execute(ALU.OP_ADD, 15, 25), 40)
        # Overflow Wrap (0xFFFFFFFF + 1 = 0)
        self.assertEqual(self.alu.execute(ALU.OP_ADD, 0xFFFFFFFF, 1), 0)

        # Basic Subtraction
        self.assertEqual(self.alu.execute(ALU.OP_SUB, 50, 20), 30)
        # Underflow Wrap (0 - 1 = 0xFFFFFFFF)
        self.assertEqual(self.alu.execute(ALU.OP_SUB, 0, 1), 0xFFFFFFFF)

        # Multiplication
        self.assertEqual(self.alu.execute(ALU.OP_MUL, 12, 10), 120)

        # Division and Modulo
        self.assertEqual(self.alu.execute(ALU.OP_DIV, 100, 4), 25)
        self.assertEqual(self.alu.execute(ALU.OP_MOD, 100, 3), 1)

        # Division by zero safety check
        with self.assertRaises(ZeroDivisionError):
            self.alu.execute(ALU.OP_DIV, 10, 0)

    def test_cmp_and_flags(self):
        """Verifies flags.E and flags.GT updates with signed comparisons[cite: 2]."""
        # Case 1: Equal values
        self.alu.execute(ALU.OP_CMP, 100, 100)
        self.assertEqual(self.alu.flags.E, 1)
        self.assertEqual(self.alu.flags.GT, 0)

        # Case 2: Greater Than (Positive)
        self.alu.execute(ALU.OP_CMP, 100, 50)
        self.assertEqual(self.alu.flags.E, 0)
        self.assertEqual(self.alu.flags.GT, 1)

        # Case 3: Less Than (Positive)
        self.alu.execute(ALU.OP_CMP, 30, 80)
        self.assertEqual(self.alu.flags.E, 0)
        self.assertEqual(self.alu.flags.GT, 0)

        # Case 4: Signed Comparison (-5 vs +2)
        # -5 in 32-bit 2's complement is 0xFFFFFFFB
        self.alu.execute(ALU.OP_CMP, 0xFFFFFFFB, 2)
        self.assertEqual(self.alu.flags.E, 0)
        self.assertEqual(self.alu.flags.GT, 0)  # -5 is NOT > 2

    def test_logical_operations(self):
        """Verifies bitwise AND, OR, NOT, and MOV"""
        self.assertEqual(self.alu.execute(ALU.OP_AND, 0b1100, 0b1010), 0b1000)
        self.assertEqual(self.alu.execute(ALU.OP_OR, 0b1100, 0b1010), 0b1110)
        self.assertEqual(self.alu.execute(ALU.OP_NOT, 0, 0x00000000), 0xFFFFFFFF)
        self.assertEqual(self.alu.execute(ALU.OP_MOV, 0, 0x12345678), 0x12345678)

    def test_shift_operations(self):
        """Verifies LSL, LSR, and sign-preserving ASR"""
        # Logical Shift Left
        self.assertEqual(self.alu.execute(ALU.OP_LSL, 1, 4), 16)

        # Logical Shift Right
        self.assertEqual(self.alu.execute(ALU.OP_LSR, 16, 4), 1)

        # Arithmetic Shift Right (-16 >> 2 = -4)
        # -16 in 32-bit hex = 0xFFFFFFF0; -4 in 32-bit hex = 0xFFFFFFFC
        result_asr = self.alu.execute(ALU.OP_ASR, 0xFFFFFFF0, 2)
        self.assertEqual(result_asr, 0xFFFFFFFC)


if __name__ == "__main__":
    unittest.main()
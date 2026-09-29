from enum import Enum

class AdderAlgorithm(Enum):
    RIPPLE_CARRY = "ripple_carry"       # O(N) gate delay
    CARRY_LOOKAHEAD = "carry_lookahead" # O(log N) gate delay

class ALUFlags:
    """Stores internal hardware flags """
    def __init__(self):
        self.E = 0   # Equal flag (written exclusively by CMP)
        self.GT = 0  # Greater Than flag (written exclusively by CMP)

    def __repr__(self):
        return f"ALUFlags(E={self.E}, GT={self.GT})"
    
    #__init__ sets up the initial values when we create a new object of this class
    #__repr__ helps us by giving a standard form of output when we try to print any object of this class


class ALU:
    """RISC201 Arithmetic Logic Unit driving math, shift, and logic operations."""

    # Opcode Definitions
    OP_ADD = 0b00000  # 0
    OP_SUB = 0b00001  # 1
    OP_MUL = 0b00010  # 2
    OP_DIV = 0b00011  # 3
    OP_MOD = 0b00100  # 4
    OP_CMP = 0b00101  # 5
    OP_AND = 0b00110  # 6
    OP_OR  = 0b00111  # 7
    OP_NOT = 0b01000  # 8
    OP_MOV = 0b01001  # 9
    OP_LSL = 0b01010  # 10
    OP_LSR = 0b01011  # 11
    OP_ASR = 0b01100  # 12

    # Gate Delay Constants (in nanoseconds) for Hardware Evaluation
    GATE_DELAY_AND_OR = 0.1  # ns
    GATE_DELAY_XOR    = 0.2  # ns

    def __init__(self, adder_algo: AdderAlgorithm = AdderAlgorithm.CARRY_LOOKAHEAD):
        self.flags = ALUFlags()
        self.adder_algo = adder_algo
        self.last_execution_delay_ns = 0.0

    @staticmethod
    def _to_signed_32(val: int) -> int:
        """Helper to convert an unsigned 32-bit integer into a signed 32-bit integer."""
        val = val & 0xFFFFFFFF
        return val - 0x100000000 if val & 0x80000000 else val
    
    #so basically in val & 0x80000000 we chekc the sign bit if it's 0 we just return the normal positive val 
    #otherwise we convert it into it's 2's complement



    def _simulate_adder_delay(self) -> float:
        """Calculates simulated hardware propagation delay based on selected algorithm."""
        if self.adder_algo == AdderAlgorithm.RIPPLE_CARRY:
            # 32 Full Adders in series: 32 * (2 * XOR delay + 1 * AND/OR delay)
            return 32 * (2 * self.GATE_DELAY_XOR + self.GATE_DELAY_AND_OR) # ~16.0 ns
        else:
            # Carry-Lookahead Adder (CLA): 4-level gate logic tree
            return (4 * self.GATE_DELAY_AND_OR) + (2 * self.GATE_DELAY_XOR) # ~0.8 ns





    def execute(self, opcode: int, operand_a: int, operand_b: int = 0) -> int:
        """
        Executes an operation given a 5-bit opcode and two 32-bit integer inputs.
        Returns a 32-bit unsigned integer result and updates internal flags if opcode is CMP[cite: 2].
        """
        # Clamp inputs to 32-bit unsigned boundaries
        a = operand_a & 0xFFFFFFFF
        b = operand_b & 0xFFFFFFFF

        # Reset execution delay for this cycle
        self.last_execution_delay_ns = 0.1  # Baseline logic delay

        #Using & 0xFFFFFFFF ensures that every ALU output behaves strictly as a 32-bit unsigned integer as python allows it to overflow so we need to keep a check
        if opcode == self.OP_ADD:
            self.last_execution_delay_ns = self._simulate_adder_delay()
            return (a + b) & 0xFFFFFFFF
        
        elif opcode == self.OP_SUB:
            self.last_execution_delay_ns = self._simulate_adder_delay() + self.GATE_DELAY_XOR
            return (a - b) & 0xFFFFFFFF
        
        elif opcode == self.OP_MUL:
            # Array Multiplier latency model
            self.last_execution_delay_ns = self._simulate_adder_delay() * 4
            return (a * b) & 0xFFFFFFFF
        
        elif opcode == self.OP_DIV:
            if b == 0:
                raise ZeroDivisionError("ALU Error: Division by zero")
            self.last_execution_delay_ns = self._simulate_adder_delay() * 8
            return (a // b) & 0xFFFFFFFF
        
        elif opcode == self.OP_MOD:
            if b == 0:
                raise ZeroDivisionError("ALU Error: Modulo by zero")
            self.last_execution_delay_ns = self._simulate_adder_delay() * 8
            return (a % b) & 0xFFFFFFFF

        # 2. Comparison (CMP updates internal flags, does not write to a destination register)
        elif opcode == self.OP_CMP:
            signed_a = self._to_signed_32(a)
            signed_b = self._to_signed_32(b)

            self.flags.E = 1 if signed_a == signed_b else 0
            self.flags.GT = 1 if signed_a > signed_b else 0
            self.last_execution_delay_ns = self._simulate_adder_delay()
            return 0  # No return register value used for CMP

        # 3. Bitwise Logical Operations
        elif opcode == self.OP_AND:
            self.last_execution_delay_ns = self.GATE_DELAY_AND_OR
            return (a & b) & 0xFFFFFFFF
        
        elif opcode == self.OP_OR:
            self.last_execution_delay_ns = self.GATE_DELAY_AND_OR
            return (a | b) & 0xFFFFFFFF
        
        elif opcode == self.OP_NOT:
            self.last_execution_delay_ns = self.GATE_DELAY_AND_OR
            # Format: not rd, (rs2/imm) -> bitwise invert second operand
            return (~b) & 0xFFFFFFFF
        
        elif opcode == self.OP_MOV:
            self.last_execution_delay_ns = 0.05
            # Format: mov rd, (rs2/imm) -> pass-through second operand
            return b & 0xFFFFFFFF

        # 4. Shift Operations
        elif opcode == self.OP_LSL:
            shift_amt = b & 0x1F  # Restrict shift to 5 bits (0-31)

            self.last_execution_delay_ns = 0.4  # Barrel shifter delay
            return (a << shift_amt) & 0xFFFFFFFF
        
        elif opcode == self.OP_LSR:
            shift_amt = b & 0x1F
            self.last_execution_delay_ns = 0.4
            return (a >> shift_amt) & 0xFFFFFFFF
        
        elif opcode == self.OP_ASR:
            shift_amt = b & 0x1F
            signed_a = self._to_signed_32(a)
            res = signed_a >> shift_amt
            self.last_execution_delay_ns = 0.5
            return res & 0xFFFFFFFF

        else:
            raise ValueError(f"ALU Error: Invalid opcode 0b{bin(opcode)[2:].zfill(5)}")
        return 0
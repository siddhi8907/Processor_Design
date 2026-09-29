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

    def __init__(self):
        self.flags = ALUFlags()

    @staticmethod
    def _to_signed_32(val: int) -> int:
        """Helper to convert an unsigned 32-bit integer into a signed 32-bit integer."""
        val = val & 0xFFFFFFFF
        return val - 0x100000000 if val & 0x80000000 else val
    
    #so basically in val & 0x80000000 we chekc the sign bit if it's 0 we just return the normal positive val 
    #otherwise we convert it into it's 2's complement

    def execute(self, opcode: int, operand_a: int, operand_b: int = 0) -> int:
        """
        Executes an operation given a 5-bit opcode and two 32-bit integer inputs.
        Returns a 32-bit unsigned integer result and updates internal flags if opcode is CMP[cite: 2].
        """
        # Clamp inputs to 32-bit unsigned boundaries
        a = operand_a & 0xFFFFFFFF
        b = operand_b & 0xFFFFFFFF

        #Using & 0xFFFFFFFF ensures that every ALU output behaves strictly as a 32-bit unsigned integer as python allows it to overflow so we need to keep a check
        if opcode == self.OP_ADD:
            return (a + b) & 0xFFFFFFFF
        
        elif opcode == self.OP_SUB:
            return (a - b) & 0xFFFFFFFF
        
        elif opcode == self.OP_MUL:
            return (a * b) & 0xFFFFFFFF
        
        elif opcode == self.OP_DIV:
            if b == 0:
                raise ZeroDivisionError("ALU Error: Division by zero")
            return (a // b) & 0xFFFFFFFF
        
        elif opcode == self.OP_MOD:
            if b == 0:
                raise ZeroDivisionError("ALU Error: Modulo by zero")
            return (a % b) & 0xFFFFFFFF

        # 2. Comparison (CMP updates internal flags, does not write to a destination register)
        elif opcode == self.OP_CMP:
            signed_a = self._to_signed_32(a)
            signed_b = self._to_signed_32(b)

            self.flags.E = 1 if signed_a == signed_b else 0
            self.flags.GT = 1 if signed_a > signed_b else 0
            return 0  # No return register value used for CMP

        # 3. Bitwise Logical Operations
        elif opcode == self.OP_AND:
            return (a & b) & 0xFFFFFFFF
        
        elif opcode == self.OP_OR:
            return (a | b) & 0xFFFFFFFF
        
        elif opcode == self.OP_NOT:
            # Format: not rd, (rs2/imm) -> bitwise invert second operand
            return (~b) & 0xFFFFFFFF
        
        elif opcode == self.OP_MOV:
            # Format: mov rd, (rs2/imm) -> pass-through second operand
            return b & 0xFFFFFFFF

        # 4. Shift Operations
        elif opcode == self.OP_LSL:
            shift_amt = b & 0x1F  # Restrict shift to 5 bits (0-31)
            return (a << shift_amt) & 0xFFFFFFFF
        
        elif opcode == self.OP_LSR:
            shift_amt = b & 0x1F
            return (a >> shift_amt) & 0xFFFFFFFF
        
        elif opcode == self.OP_ASR:
            shift_amt = b & 0x1F
            signed_a = self._to_signed_32(a)
            res = signed_a >> shift_amt
            return res & 0xFFFFFFFF

        else:
            raise ValueError(f"ALU Error: Invalid opcode 0b{bin(opcode)[2:].zfill(5)}")
        return 0
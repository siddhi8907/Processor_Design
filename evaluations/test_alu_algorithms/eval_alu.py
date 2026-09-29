

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.alu.alu import ALU, AdderAlgorithm

def run_evaluation():
    
    print("      RISC201 ALU ALGORITHM EVALUATION REPORT (Task 5)    ")
   

    alu_rca = ALU(adder_algo=AdderAlgorithm.RIPPLE_CARRY)
    alu_cla = ALU(adder_algo=AdderAlgorithm.CARRY_LOOKAHEAD)

    operand_a = 0x12345678
    operand_b = 0x87654321

    print(f"Test Operand A: 0x{operand_a:08X}")
    print(f"Test Operand B: 0x{operand_b:08X}\n")

    # 1. Addition Delay Comparison
    res_rca = alu_rca.execute(ALU.OP_ADD, operand_a, operand_b)
    delay_rca = alu_rca.last_execution_delay_ns

    res_cla = alu_cla.execute(ALU.OP_ADD, operand_a, operand_b)
    delay_cla = alu_cla.last_execution_delay_ns

    print(f"[ADDITION RESULT]: 0x{res_rca:08X}")
    print(f"  - Ripple-Carry Adder Latency:   {delay_rca:.2f} ns")
    print(f"  - Carry-Lookahead Adder Latency: {delay_cla:.2f} ns")
    print(f"  - Speedup Factor:                {delay_rca / delay_cla:.2f}x faster\n")

    # 2. Multiplication Delay Comparison
    alu_rca.execute(ALU.OP_MUL, 1000, 500)
    delay_mul_rca = alu_rca.last_execution_delay_ns

    alu_cla.execute(ALU.OP_MUL, 1000, 500)
    delay_mul_cla = alu_cla.last_execution_delay_ns

    print("[MULTIPLICATION LATENCY]:")
    print(f"  - RCA-Based Multiplier Latency: {delay_mul_rca:.2f} ns")
    print(f"  - CLA-Based Multiplier Latency: {delay_mul_cla:.2f} ns\n")

    

if __name__ == "__main__":
    run_evaluation()
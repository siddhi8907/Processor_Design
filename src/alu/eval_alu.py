import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

import time
import random
from alu import RISC201_ALU

def run_adder_benchmark(num_trials=50_000):
    print("=" * 65)
    print(" 1. ADDER EVALUATION: KOGGE-STONE VS. RIPPLE-CARRY (32-BIT)")
    print("=" * 65)
    alu_kogge = RISC201_ALU(adder_mode="kogge")
    alu_ripple = RISC201_ALU(adder_mode="ripple")

    pairs = [(random.randint(0, 0xFFFFFFFF), random.randint(0, 0xFFFFFFFF)) for _ in range(num_trials)]

    # Software runtime
    t0 = time.perf_counter()
    for a, b in pairs:
        alu_kogge.execute(0b00000, a, b)
    t_kogge = time.perf_counter() - t0

    t0 = time.perf_counter()
    for a, b in pairs:
        alu_ripple.execute(0b00000, a, b)
    t_ripple = time.perf_counter() - t0

    # Hardware theoretical models
    rca_gate_levels = 32 * 2       # ~2 gate delays per full adder carry
    ksa_gate_levels = 1 + (5 * 2)  # 1 generate/propagate level + 5 prefix stages * 2
    theoretical_speedup = rca_gate_levels / ksa_gate_levels

    print(f"Test vectors executed         : {num_trials:,}")
    print(f"Software Sim Time (Kogge-Stone): {t_kogge:.4f} s (interpreted list allocations)")
    print(f"Software Sim Time (Ripple-Carry): {t_ripple:.4f} s (flat loop)")
    print("-" * 65)
    print("HARDWARE CRITICAL PATH & COMPLEXITY ANALYSIS:")
    print(f"  * Ripple-Carry Adder (RCA)  : O(n) delay      => 32 carry stages (~{rca_gate_levels} gate delays)")
    print(f"  * Kogge-Stone Adder (KSA)   : O(log2 n) delay => 5 prefix stages (~{ksa_gate_levels} gate delays)")
    print(f"  * Theoretical Hardware Gain : ~{theoretical_speedup:.2f}x critical path speedup")
    print("  * Architectural Rationale   : KSA is chosen to prevent clock elongation in EX_MA/EX stage.")
    print("=" * 65 + "\n")

def print_latency_summary():
    print("=" * 65)
    print(" 2. RISC-201 ALU EXECUTION CYCLES & PIPELINE STALL MAPPING")
    print("=" * 65)
    alu = RISC201_ALU()
    samples = [
        ("add / sub / address calc", 0b00000, False, "1 cycle  (single-cycle EX)"),
        ("cmp / min / max / abs / neg", 0b00101, False, "1 cycle  (flags written in EX)"),
        ("mul (Booth Radix-4)", 0b00010, False, "2 cycles (asserts exBusy = 1 cycle)"),
        ("div / mod (Non-Restoring)", 0b00011, False, "8 cycles (asserts exBusy = 7 cycles)"),
        ("fadd / fsub", 0b00000, True, "3 cycles (asserts exBusy = 2 cycles)"),
        ("fmul", 0b00010, True, "3 cycles (asserts exBusy = 2 cycles)"),
        ("fdiv (Digit Recurrence)", 0b00011, True, "10 cycles (asserts exBusy = 9 cycles)"),
        ("fcvt (int <-> float)", 0b11111, False, "2 cycles (asserts exBusy = 1 cycle)"),
    ]
    print(f"{'Operation':<30} | {'EX Latency':<10} | {'Pipeline Impact'}")
    print("-" * 65)
    for name, op, is_fl, impact in samples:
        _, _, _, lat = alu.execute(op, 1, 1, is_float=is_fl, is_fcvt=(op == 0b11111))
        print(f"{name:<30} | {lat:<10} | {impact}")
    print("=" * 65)

if __name__ == "__main__":
    run_adder_benchmark(50_000)
    print_latency_summary()
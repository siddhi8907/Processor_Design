#!/usr/bin/env python3
"""
pipeline_sim.py - verifies Pipeline_4Stage.docx and Pipeline_6Stage.docx

Contains:
  1. 32-bit encoder/decoder for the extended SimpleRISC ISA (float bit F, fcvt)
  2. Reference model (one instruction at a time, no pipeline)
  3. Cycle-accurate 4-stage and 6-stage pipelines written as the docs describe
     (forwarding, load-use stall, multi-cycle EX stall, flushes, halt, rte, exceptions)
  4. Tests: encoding, directed programs (values + cycle counts), random programs
     compared against the reference model, and "mutation" runs where a hazard
     feature is switched off to prove the tests can catch bugs.

Run:  python3 pipeline_sim.py
"""
import math
import random
import struct
import sys

M32 = 0xFFFFFFFF
STACK_BASE, STACK_LIMIT = 0x2000, 0x1F00
EXEC_VECTOR = 0x400
SP, RA, DBASE = 14, 15, 13           # r14 = SP, r15 = ra, r13 = data base (test convention)

# ------------------------------------------------------------------ ISA tables
OPC = {'add': 0, 'sub': 1, 'mul': 2, 'div': 3, 'mod': 4, 'cmp': 5, 'and': 6, 'or': 7,
       'not': 8, 'mov': 9, 'lsl': 10, 'lsr': 11, 'asr': 12, 'nop': 13, 'ld': 14, 'st': 15,
       'beq': 16, 'bgt': 17, 'b': 18, 'call': 19, 'ret': 20, 'push': 21, 'pop': 22,
       'halt': 25, 'rte': 26, 'min': 27, 'max': 28, 'abs': 29, 'neg': 30, 'fcvt': 31}
OPN = {v: k for k, v in OPC.items()}          # 23 and 24 (old lsbase/lslim) are absent -> nop
BR = ('b', 'beq', 'bgt', 'call')
ADD3 = ('add', 'sub', 'mul', 'div', 'mod', 'and', 'or', 'lsl', 'lsr', 'asr', 'min', 'max')
UN = ('not', 'mov', 'abs', 'neg', 'fcvt')
FLOAT_OK = ('add', 'sub', 'mul', 'div', 'cmp', 'mov', 'min', 'max', 'abs', 'neg')


def s32(x):
    x &= M32
    return x - (1 << 32) if x >> 31 else x


def sext(x, n):
    x &= (1 << n) - 1
    return x - (1 << n) if x >> (n - 1) else x


# ------------------------------------------------------------------ encode / decode (32 bits)
def A(op, rd=0, rs1=0, rs2=0, imm=None, F=0, mod=0, off=0):
    """Tiny assembler: returns an instruction dict."""
    return dict(op=op, rd=rd, rs1=rs1, rs2=rs2, imm=imm, F=F, mod=mod, off=off,
                I=0 if imm is None else 1)


def encode(i):
    op = OPC[i['op']]
    if i['op'] in BR:
        return (op << 27) | (i['off'] & 0x7FFFFFF)                    # offset[26:0]
    w = (op << 27) | (i['I'] << 26) | (i['F'] << 25) | (i['rd'] << 21) | (i['rs1'] << 17)
    if i['I']:
        w |= (i['imm'] & 0x1FFFF) if i['F'] else ((i['mod'] << 15) | (i['imm'] & 0x7FFF))
    else:
        w |= i['rs2'] << 13
    assert w < (1 << 32)
    return w


def decode(w):
    op = OPN.get((w >> 27) & 31, 'nop')
    if op in BR:
        return dict(op=op, I=0, F=0, rd=0, rs1=0, rs2=0, imm=0, mod=0, off=sext(w, 27))
    I, F = (w >> 26) & 1, (w >> 25) & 1
    d = dict(op=op, I=I, F=F, rd=(w >> 21) & 15, rs1=(w >> 17) & 15, rs2=(w >> 13) & 15,
             imm=0, mod=0, off=0)
    if I:
        if F:
            d['imm'] = w & 0x1FFFF                                      # imm17
        else:
            d['mod'], d['imm'] = (w >> 15) & 3, w & 0x7FFF              # modifier[16:15], imm15
    return d


# ------------------------------------------------------------------ float helpers (binary32, FTZ)
def b2f(bits):
    bits &= M32
    if ((bits >> 23) & 0xFF) == 0 and (bits & 0x7FFFFF):
        return math.copysign(0.0, -1.0 if bits >> 31 else 1.0)
    return struct.unpack('>f', struct.pack('>I', bits))[0]


def f2b(x):
    if x != x:
        return 0x7FC00000
    try:
        bits = struct.unpack('>I', struct.pack('>f', x))[0]
    except OverflowError:
        return 0xFF800000 if x < 0 else 0x7F800000
    if ((bits >> 23) & 0xFF) == 0 and (bits & 0x7FFFFF):
        return bits & 0x80000000
    return bits


def fimm(x):
    """float -> imm17 field (top 17 bits of the binary32 pattern)"""
    return f2b(x) >> 15


def falu(op, a, b):
    if op == 'mov': return b
    if op == 'abs': return b & 0x7FFFFFFF
    if op == 'neg': return b ^ 0x80000000
    x, y = b2f(a), b2f(b)
    if op == 'add': return f2b(x + y)
    if op == 'sub': return f2b(x - y)
    if op == 'mul': return f2b(x * y)
    if op == 'div':
        if y == 0:
            if x == 0 or x != x: return 0x7FC00000
            return f2b(math.copysign(math.inf, x) * math.copysign(1.0, y))
        return f2b(x / y)
    if op == 'min':
        if x != x: return b & M32
        if y != y: return a & M32
        return a if x < y else b
    if op == 'max':
        if x != x: return b & M32
        if y != y: return a & M32
        return a if x > y else b
    raise ValueError(op)


def alu(op, F, a, b):
    a &= M32; b &= M32
    if F:
        return falu(op, a, b)
    sa, sb = s32(a), s32(b)
    if op == 'add': return (a + b) & M32
    if op == 'sub': return (a - b) & M32
    if op == 'mul': return (a * b) & M32
    if op == 'div':
        if sb == 0: return M32
        if sa == -(1 << 31) and sb == -1: return a
        q = abs(sa) // abs(sb)
        return (q if (sa < 0) == (sb < 0) else -q) & M32
    if op == 'mod':
        if sb == 0: return a
        if sa == -(1 << 31) and sb == -1: return 0
        q = abs(sa) // abs(sb)
        q = q if (sa < 0) == (sb < 0) else -q
        return (sa - q * sb) & M32
    if op == 'and': return a & b
    if op == 'or': return a | b
    if op == 'not': return ~b & M32
    if op == 'mov': return b
    if op == 'lsl': return (a << (b & 31)) & M32
    if op == 'lsr': return a >> (b & 31)
    if op == 'asr': return (sa >> (b & 31)) & M32
    if op == 'min': return a if sa < sb else b
    if op == 'max': return a if sa > sb else b
    if op == 'abs': return abs(sb) & M32
    if op == 'neg': return (-sb) & M32
    raise ValueError(op)


def compare(F, a, b):
    if F:
        x, y = b2f(a), b2f(b)
        return (0, 0) if (x != x or y != y) else (int(x == y), int(x > y))
    return (int(s32(a) == s32(b)), int(s32(a) > s32(b)))


def fcvt(F, b):
    if F == 0:
        return f2b(float(s32(b)))
    x = b2f(b)
    if x != x: return 0
    if x >= 2.0 ** 31: return 0x7FFFFFFF
    if x <= -2.0 ** 31: return 0x80000000
    return int(x) & M32


def latency(u):
    op, F = u['op'], u['F']
    if op == 'fcvt': return 2
    if F:
        return {'add': 3, 'sub': 3, 'mul': 3, 'div': 10}.get(op, 1)
    return {'mul': 2, 'div': 8, 'mod': 8}.get(op, 1)


# ------------------------------------------------------------------ micro-op (what the decoder/MCU produces)
def make_uop(word, pc):
    i = decode(word)
    op = i['op']
    if op not in FLOAT_OK and op != 'fcvt':
        i['F'] = 0                                   # control unit forces isFloat = 0
    u = dict(i, pc=pc, cyc=0, use1=False, use2=False, s1=0, s2=0, wb1=False, d1=0, wb2=False,
             res=0, ldres=0, addr=None, sdata=None, op1=0, op2=0)
    I = i['I']
    if op in ADD3:
        u.update(use1=True, s1=i['rs1'], use2=not I, s2=i['rs2'], wb1=True, d1=i['rd'])
    elif op == 'cmp':
        u.update(use1=True, s1=i['rs1'], use2=not I, s2=i['rs2'])
    elif op in UN:
        u.update(use2=not I, s2=i['rs2'], wb1=True, d1=i['rd'])
    elif op == 'ld':
        u.update(use1=True, s1=i['rs1'], wb1=True, d1=i['rd'])
    elif op == 'st':
        u.update(use1=True, s1=i['rs1'], use2=True, s2=i['rd'])
    elif op == 'push':
        u.update(use1=True, s1=SP, use2=True, s2=i['rd'], wb2=True)
    elif op == 'pop':
        u.update(use1=True, s1=SP, wb1=True, d1=i['rd'], wb2=True)
    elif op == 'call':
        u.update(wb1=True, d1=RA)
    elif op == 'ret':
        u.update(use1=True, s1=RA)
    # immediate unit
    if I and i['F']:
        u['immv'] = (i['imm'] << 15) & M32
    elif I:
        v = i['imm']
        u['immv'] = {0: sext(v, 15) & M32, 1: v, 2: (v << 16) & M32}.get(i['mod'], sext(v, 15) & M32)
        if op in ('ld', 'st'):
            u['immv'] = sext(v, 15) & M32
    else:
        u['immv'] = 0
    u['btarget'] = (pc + (i['off'] << 2)) & M32 if op in BR else 0
    return u


def execute(u, A_, B_, flags):
    """EX-stage function. A_ = op1 (after forwarding), B_ = op2 (after forwarding)."""
    op, F, I, imm = u['op'], u['F'], u['I'], u['immv']
    r = dict(res=0, addr=None, sdata=None, fl=None, taken=False, target=0, exc=False)
    if op in BR:
        r['taken'] = op in ('b', 'call') or (op == 'beq' and flags[0]) or (op == 'bgt' and flags[1])
        r['target'] = u['btarget']
        if op == 'call': r['res'] = (u['pc'] + 4) & M32
    elif op == 'ret':
        r['taken'], r['target'] = True, A_
    elif op == 'push':
        ns = (A_ - 4) & M32
        r.update(res=ns, addr=ns, sdata=B_, exc=ns < STACK_LIMIT)
    elif op == 'pop':
        ns = (A_ + 4) & M32
        r.update(res=ns, addr=A_, exc=ns > STACK_BASE)
    elif op == 'ld':
        r['addr'] = r['res'] = (A_ + imm) & M32
    elif op == 'st':
        r.update(addr=(A_ + imm) & M32, sdata=B_)
    elif op == 'cmp':
        r['fl'] = compare(F, A_, imm if I else B_)
    elif op == 'fcvt':
        r['res'] = fcvt(F, B_)
    elif op in ('nop', 'halt', 'rte'):
        pass
    else:
        r['res'] = alu(op, F, A_, imm if I else B_)
    return r


# ------------------------------------------------------------------ reference model
class Ref:
    def __init__(self, words):
        self.im = words
        self.regs = [0] * 16
        self.regs[SP], self.regs[DBASE] = STACK_BASE, 0x100
        self.dm, self.flags, self.epc, self.cause = {}, (0, 0), 0, 0
        self.retired, self.halted = [], False
        self.exlog = []

    def run(self, maxn=100000):
        pc, n = 0, 0
        while not self.halted and n < maxn:
            n += 1
            idx = pc // 4
            w = self.im[idx] if idx < len(self.im) else encode(A('nop'))
            u = make_uop(w, pc)
            a = self.regs[u['s1']] if u['use1'] else 0
            b = self.regs[u['s2']] if u['use2'] else 0
            r = execute(u, a, b, self.flags)
            ev = 'exc' if r['exc'] else 'taken' if r['taken'] else 'rte' if u['op'] == 'rte' else ''
            self.exlog.append((pc, a, b, ev, u))
            if r['exc']:
                self.epc, self.cause, pc = pc, 1, EXEC_VECTOR
                continue
            if r['fl'] is not None: self.flags = r['fl']
            if u['op'] in ('st', 'push'): self.dm[r['addr']] = r['sdata'] & M32
            if u['wb2']: self.regs[SP] = r['res']
            if u['wb1']:
                self.regs[u['d1']] = (self.dm.get(r['addr'], 0) if u['op'] in ('ld', 'pop') else r['res']) & M32
            self.retired.append(pc)
            if u['op'] == 'halt': self.halted = True
            pc = r['target'] if r['taken'] else (self.epc if u['op'] == 'rte' else pc + 4)
        return self

    def state(self):
        return (tuple(self.regs), tuple(sorted(self.dm.items())), self.flags, self.epc, self.cause,
                tuple(self.retired), self.halted)


# ------------------------------------------------------------------ pipelines
# (instrumented: every cycle can be traced, and hazard-unit faults can be injected with mut={...})
class Pipe:
    NOP = encode(A('nop'))

    def __init__(self, words, fwd=True, bypass=True, stall_lu=True, latch_ops=True, p1_wins=True,
                 mut=(), trace=False):
        self.im, self.fwd, self.bypass, self.stall_lu, self.latch_ops = words, fwd, bypass, stall_lu, latch_ops
        self.p1_wins = p1_wins
        self.mut = set(mut)               # injected hazard-unit faults (see hazard_check.py)
        self.tr, self.trace, self.exlog = trace, [], []
        self._lu2 = False
        self.regs = [0] * 16
        self.regs[SP], self.regs[DBASE] = STACK_BASE, 0x100
        self.dm, self.flags, self.epc, self.cause = {}, (0, 0), 0, 0
        self.pc, self.haltPending, self.halted = 0, False, False
        self.retired, self.cycles = [], 0

    def lat(self, u):
        n = latency(u)
        return n - 1 if ('lat_minus1' in self.mut and n > 1) else n

    def fetch(self, pc):
        i = pc // 4
        return (self.im[i] if i < len(self.im) else self.NOP, pc)

    def run(self, maxc=200000):
        while not self.halted and self.cycles < maxc:
            self.cycles += 1
            self.step()
        return self

    def state(self):
        return (tuple(self.regs), tuple(sorted(self.dm.items())), self.flags, self.epc, self.cause,
                tuple(self.retired), self.halted)

    # shared helpers ---------------------------------------------------------
    def write_back(self, W1, W2):
        order = (W2, W1) if self.p1_wins else (W1, W2)   # later write wins: port 1 over port 2
        for w in order:
            if w: self.regs[w[0]] = w[1] & M32

    def bypass_read(self, u, W1, W2):
        """OF-stage register read with RW bypass. Records which source supplied each operand in u['byp']."""
        u = dict(u)
        byp = {}
        for k in (1, 2):
            if not u['use%d' % k]:
                u['op%d' % k] = 0
                byp[k] = 'unused'
                continue
            ad = u['s%d' % k]
            v, tag = self.regs[ad], 'reg'
            if self.bypass:
                for w, t in (((W2, 'W2'), (W1, 'W1')) if self.p1_wins else ((W1, 'W1'), (W2, 'W2'))):
                    if w and w[0] == ad and not (w[2] == 'call' and 'no_call_link_fwd' in self.mut): v, tag = w[1], t
            u['op%d' % k] = v
            byp[k] = tag
        u['byp'] = byp
        return u


class Pipe4(Pipe):
    """IF, OF, EX_MA, RW"""
    S = 4

    def __init__(self, words, **o):
        super().__init__(words, **o)
        self.IF_OF = self.OF_EX = self.EX_RW = None

    def step(self):
        rec = dict(c=self.cycles, pc=self.pc, IF_OF=self.IF_OF, OF_EX=self.OF_EX, EX_RW=self.EX_RW,
                   hp=self.haltPending) if self.tr else None
        # ---- RW
        W1 = W2 = None
        r = self.EX_RW
        halting = False
        if r:
            if r['wb1']:
                d = r['ldres'] if r['op'] in ('ld', 'pop') else ((r['pc'] + 4) if r['op'] == 'call' else r['res'])
                W1 = (r['d1'], d & M32, r['op'])
            if r['wb2']: W2 = (SP, r['res'], r['op'])
            self.retired.append(r['pc'])
            halting = r['op'] == 'halt'

        # ---- EX_MA
        u = self.OF_EX
        busy, redirect, newEX_RW, newOF_EX_hold = False, None, None, None
        newflags = None
        fs = {}
        ev = ''
        if u:
            def f(k):
                if not u['use%d' % k]:
                    fs[k] = 'unused'
                    return 0
                ad, cur = u['s%d' % k], u['op%d' % k]
                if self.fwd:
                    p2 = W2 and not (W1 and W1[0] == SP)           # pop sp: port 1 wins, port 2 not used
                    ws = ((W1, 'W1'), (W2 if p2 else None, 'W2')) if self.p1_wins else ((W2, 'W2'), (W1, 'W1'))
                    for w, t in ws:
                        if t == 'W2' and 'no_sp_fwd' in self.mut: continue
                        if w and w[0] == ad and not (w[2] == 'call' and 'no_call_link_fwd' in self.mut):
                            fs[k] = t
                            return w[1]
                fs[k] = 'reg'
                return cur
            a, b = f(1), f(2)
            if u['cyc'] < self.lat(u) - 1:
                busy = True
                h = dict(u, cyc=u['cyc'] + 1)
                if self.latch_ops: h['op1'], h['op2'] = a, b      # keep forwarded operands
                newOF_EX_hold = h
            else:
                x = execute(u, a, b, self.flags)
                ev = 'exc' if x['exc'] else 'taken' if x['taken'] else 'rte' if u['op'] == 'rte' else ''
                self.exlog.append((u['pc'], a, b, ev, u))
                if x['exc']:
                    self.epc, self.cause, redirect = u['pc'], 1, EXEC_VECTOR
                else:
                    if x['fl'] is not None: newflags = x['fl']
                    if u['op'] in ('st', 'push'): self.dm[x['addr']] = x['sdata'] & M32
                    ld = self.dm.get(x['addr'], 0) if u['op'] in ('ld', 'pop') else 0
                    if x['taken']: redirect = x['target']
                    newEX_RW = dict(u, res=x['res'], ldres=ld)

        # ---- OF / IF
        fekill = False
        if busy:
            nIF_OF, nOF_EX = self.IF_OF, newOF_EX_hold
            npc = self.pc + 4 if 'busy_no_hold_pc' in self.mut else self.pc
        elif redirect is not None:
            nIF_OF, nOF_EX, npc = None, None, redirect
            self.haltPending = False
        else:
            nOF_EX, rte, hp = None, False, False
            if self.IF_OF:
                w, pc = self.IF_OF
                v = self.bypass_read(make_uop(w, pc), W1, W2)
                rte, hp = v['op'] == 'rte', v['op'] == 'halt'
                nOF_EX = v
            if hp: self.haltPending = True
            if rte:
                nIF_OF, npc = None, self.epc
            elif self.haltPending:
                nIF_OF, npc = None, self.pc
            else:
                nIF_OF, npc = self.fetch(self.pc), self.pc + 4
            fekill = rte or self.haltPending
        # ---- commit state at clock edge
        self.write_back(W1, W2)
        if newflags is not None: self.flags = newflags
        self.IF_OF, self.OF_EX, self.EX_RW, self.pc = nIF_OF, nOF_EX, newEX_RW, npc & M32
        if halting: self.halted = True
        if rec is not None:
            rec.update(busy=busy, stallLU=False, redirect=redirect, fs=fs, fekill=fekill, ev=ev,
                       nxt=dict(pc=self.pc, IF_OF=self.IF_OF, OF_EX=self.OF_EX, EX_RW=self.EX_RW))
            self.trace.append(rec)


class Pipe6(Pipe):
    """IF, ID, OF, EX, MA, RW"""
    S = 6

    def __init__(self, words, **o):
        super().__init__(words, **o)
        self.IF_ID = self.ID_OF = self.OF_EX = self.EX_MA = self.MA_RW = None

    def step(self):
        rec = dict(c=self.cycles, pc=self.pc, IF_ID=self.IF_ID, ID_OF=self.ID_OF, OF_EX=self.OF_EX,
                   EX_MA=self.EX_MA, MA_RW=self.MA_RW, hp=self.haltPending) if self.tr else None
        # ---- RW
        W1 = W2 = None
        r = self.MA_RW
        halting = False
        if r:
            if r['wb1']: W1 = (r['d1'], (r['ldres'] if r['op'] in ('ld', 'pop') else r['res']) & M32, r['op'])
            if r['wb2']: W2 = (SP, r['res'], r['op'])
            self.retired.append(r['pc'])
            halting = r['op'] == 'halt'

        # ---- MA
        nMA_RW = None
        m = self.EX_MA
        if m:
            m = dict(m)
            if m['op'] in ('ld', 'pop'): m['ldres'] = self.dm.get(m['addr'], 0)
            if m['op'] in ('st', 'push'): self.dm[m['addr']] = m['sdata'] & M32
            nMA_RW = m

        # ---- EX
        u = self.OF_EX
        busy, redirect, nEX_MA, hold, newflags = False, None, None, None, None
        fs = {}
        ev = ''
        if u:
            e = self.EX_MA                                        # instruction now in MA

            def f(k):
                if not u['use%d' % k]:
                    fs[k] = 'unused'
                    return 0
                ad, cur = u['s%d' % k], u['op%d' % k]
                if self.fwd:
                    def ma_fwd():
                        if e:
                            if e['wb1'] and e['op'] not in ('ld', 'pop') and e['d1'] == ad and not (e['op'] == 'call' and 'no_call_link_fwd' in self.mut): return ('MA1', e['res'])
                            if e['wb2'] and ad == SP and not (e['wb1'] and e['d1'] == SP): return ('MA2', e['res'])
                        return None

                    def rw_fwd():
                        p2 = W2 and not (W1 and W1[0] == SP)       # pop sp: port 1 wins, port 2 not used
                        ws = ((W1, 'W1'), (W2 if p2 else None, 'W2')) if self.p1_wins else ((W2, 'W2'), (W1, 'W1'))
                        for w, t in ws:
                            if w and w[0] == ad and not (w[2] == 'call' and 'no_call_link_fwd' in self.mut): return (t, w[1])
                        return None
                    for fn in ((rw_fwd, ma_fwd) if 'fwd_old_first' in self.mut else (ma_fwd, rw_fwd)):
                        hit = fn()
                        if hit and not (hit[0] in ('MA2', 'W2') and 'no_sp_fwd' in self.mut):
                            fs[k] = hit[0]
                            return hit[1]
                fs[k] = 'reg'
                return cur
            a, b = f(1), f(2)
            if u['cyc'] < self.lat(u) - 1:
                busy = True
                hold = dict(u, cyc=u['cyc'] + 1)
                if self.latch_ops: hold['op1'], hold['op2'] = a, b
            else:
                x = execute(u, a, b, self.flags)
                ev = 'exc' if x['exc'] else 'taken' if x['taken'] else 'rte' if u['op'] == 'rte' else ''
                self.exlog.append((u['pc'], a, b, ev, u))
                if x['exc']:
                    self.epc, self.cause, redirect = u['pc'], 1, EXEC_VECTOR
                else:
                    if x['fl'] is not None: newflags = x['fl']
                    if x['taken']: redirect = x['target']
                    nEX_MA = dict(u, res=x['res'], addr=x['addr'], sdata=x['sdata'])

        # ---- load-use check (OF stage, looks at the instruction in EX)
        stallLU = False
        d = self.ID_OF
        if self.stall_lu and d and u and not busy and u['wb1'] and u['op'] in (('ld',) if 'lu_ld_only' in self.mut else ('ld', 'pop')):
            ig = 'lu_ignore_use' in self.mut                      # no useSrc gating: compares raw address fields
            m1 = (d['use1'] or ig) and (d['s1'] if d['use1'] else d['rs1']) == u['d1']
            m2 = (d['use2'] or ig) and (d['s2'] if d['use2'] else d['rs2']) == u['d1']
            if 'lu_src1_only' in self.mut: m2 = False
            if 'lu_src2_only' in self.mut: m1 = False
            stallLU = m1 or m2
        if 'lu_two_cycles' in self.mut and not busy and redirect is None:
            if stallLU: self._lu2 = True
            elif self._lu2: stallLU, self._lu2 = True, False

        # ---- ID / IF and latch updates
        fekill = False
        if busy:
            nIF_ID, nID_OF, nOF_EX = self.IF_ID, self.ID_OF, hold
            npc = self.pc + 4 if 'busy_no_hold_pc' in self.mut else self.pc
        elif redirect is not None:
            nIF_ID = nID_OF = nOF_EX = None
            npc = redirect
            self.haltPending = False
        elif stallLU:
            nIF_ID, nID_OF, npc = self.IF_ID, self.ID_OF, (self.pc + 4 if 'lu_no_hold_pc' in self.mut else self.pc)
            nOF_EX = self.bypass_read(self.ID_OF, W1, W2) if 'lu_no_bubble' in self.mut else None
        else:
            nOF_EX = self.bypass_read(self.ID_OF, W1, W2) if self.ID_OF else None
            nID_OF, rte, hp = None, False, False
            if self.IF_ID:
                w, pc = self.IF_ID
                nID_OF = make_uop(w, pc)
                rte, hp = nID_OF['op'] == 'rte', nID_OF['op'] == 'halt'
            if hp: self.haltPending = True
            if rte:
                nIF_ID, npc = None, self.epc
            elif self.haltPending:
                nIF_ID, npc = None, self.pc
            else:
                nIF_ID, npc = self.fetch(self.pc), self.pc + 4
            fekill = rte or self.haltPending
        self.write_back(W1, W2)
        if newflags is not None: self.flags = newflags
        self.IF_ID, self.ID_OF, self.OF_EX, self.EX_MA, self.MA_RW = nIF_ID, nID_OF, nOF_EX, nEX_MA, nMA_RW
        self.pc = npc & M32
        if halting: self.halted = True
        if rec is not None:
            rec.update(busy=busy, stallLU=stallLU, redirect=redirect, fs=fs, fekill=fekill, ev=ev,
                       nxt=dict(pc=self.pc, IF_ID=self.IF_ID, ID_OF=self.ID_OF, OF_EX=self.OF_EX,
                                EX_MA=self.EX_MA, MA_RW=self.MA_RW))
            self.trace.append(rec)



# ------------------------------------------------------------------ program helpers
def assemble(main, subs=(), handler=None):
    """main (ends with halt), subroutines after it, exception handler at EXEC_VECTOR."""
    words = [encode(i) for i in main] + [encode(i) for i in subs]
    if handler is None:
        handler = [A('mov', SP, imm=(STACK_BASE + STACK_LIMIT) // 2), A('rte')]
    words += [encode(A('nop'))] * (EXEC_VECTOR // 4 - len(words))
    assert len(words) <= EXEC_VECTOR // 4
    words += [encode(i) for i in handler]
    return words


def compare_run(words, label, **opt):
    ref = Ref(words).run()
    out = []
    for cls in (Pipe4, Pipe6):
        p = cls(words, **opt).run()
        out.append((cls.__name__, p, p.state() == ref.state()))
    return ref, out


# ------------------------------------------------------------------ tests
def test_encoding():
    rnd = random.Random(1)
    assert len(set(OPC.values())) == len(OPC) and max(OPC.values()) <= 31
    for _ in range(5000):
        op = rnd.choice(list(OPC))
        if op in BR:
            i = A(op, off=rnd.randint(-(1 << 26), (1 << 26) - 1))
        else:
            F = rnd.randint(0, 1) if op in FLOAT_OK else 0
            if rnd.random() < .5 and op != 'fcvt':
                imm = rnd.randint(0, (1 << 17) - 1) if F else rnd.randint(0, 0x7FFF)
                i = A(op, rnd.randint(0, 15), rnd.randint(0, 15), 0, imm=imm, F=F, mod=0 if F else rnd.randint(0, 2))
            else:
                i = A(op, rnd.randint(0, 15), rnd.randint(0, 15), rnd.randint(0, 15), F=F)
        w = encode(i)
        assert w < (1 << 32)
        d = decode(w)
        assert d['op'] == i['op'] and d['F'] == i['F'] and d['I'] == i['I']
        if i['op'] in BR: assert d['off'] == i['off']
        else:
            assert d['rd'] == i['rd'] and d['rs1'] == i['rs1']
            if i['I']: assert d['imm'] == (i['imm'] & (0x1FFFF if i['F'] else 0x7FFF)) and d['mod'] == (0 if i['F'] else i['mod'])
            else: assert d['rs2'] == i['rs2']
    assert decode((23 << 27))['op'] == 'nop' and decode((24 << 27))['op'] == 'nop'   # removed lsbase/lslim
    print('  encoding: 5000 random instructions round-trip in 32 bits; opcodes 23/24 decode as nop')


def expect(cond, msg):
    if not cond:
        print('  FAIL:', msg)
        sys.exit(1)


def directed():
    n_ok = 0

    def run(name, main, subs=(), handler=None, regs=None, cycles=None):
        nonlocal n_ok
        words = assemble(main, subs, handler)
        ref, outs = compare_run(words, name)
        for nm, p, same in outs:
            expect(same, '%s: %s differs from reference' % (name, nm))
        if regs:
            for r, v in regs.items():
                expect(ref.regs[r] == v & M32, '%s: reference r%d = %#x, expected %#x' % (name, r, ref.regs[r], v & M32))
        if cycles:
            for nm, p, _ in outs:
                expect(p.cycles == cycles[p.S], '%s: %s took %d cycles, expected %d' % (name, nm, p.cycles, cycles[p.S]))
        n_ok += 1
        return ref, outs

    H = A('halt')
    # RAW at distance 1..4
    run('raw-distances', [A('mov', 1, imm=5), A('add', 2, 1, imm=3), A('add', 3, 2, 1), A('add', 4, 3, 1),
                          A('add', 5, 4, 1), A('add', 6, 5, 1), H],
        regs={2: 8, 3: 13, 4: 18, 5: 23, 6: 28}, cycles={4: 7 + 3, 6: 7 + 5})
    # straight line cycle count
    run('straight-line', [A('mov', 1, imm=1)] * 5 + [H], cycles={4: 6 + 3, 6: 6 + 5})
    # load-use: 4-stage no stall, 6-stage 1 stall
    run('load-use', [A('mov', 1, imm=77), A('st', 1, DBASE, imm=0), A('ld', 2, DBASE, imm=0), A('add', 3, 2, 2), H],
        regs={2: 77, 3: 154}, cycles={4: 5 + 3, 6: 5 + 5 + 1})
    run('load-use-dist2', [A('mov', 1, imm=9), A('st', 1, DBASE, imm=4), A('ld', 2, DBASE, imm=4), A('nop'), A('add', 3, 2, 2), H],
        regs={3: 18}, cycles={4: 6 + 3, 6: 6 + 5})
    # push/pop chains (SP forwarding, pop->use)
    run('push-push-pop', [A('mov', 1, imm=11), A('mov', 2, imm=22), A('push', 1), A('push', 2), A('pop', 3), A('add', 4, 3, 3),
                          A('pop', 5), H],
        regs={3: 22, 4: 44, 5: 11, SP: STACK_BASE}, cycles={4: 8 + 3, 6: 8 + 5 + 1})
    # call / ret with ra forwarding
    run('call-ret', [A('mov', 1, imm=3), A('call', off=3), A('add', 3, 2, 1), H, A('add', 2, 1, imm=10), A('ret')],
        regs={2: 13, 3: 16})
    # cmp -> branch directly after, taken and not taken
    run('beq-taken', [A('mov', 1, imm=4), A('cmp', 0, 1, 1), A('beq', off=2), A('mov', 5, imm=99), H],
        regs={5: 0}, cycles={4: 4 + 3 + 2, 6: 4 + 5 + 3})
    run('bgt-not-taken', [A('mov', 1, imm=4), A('cmp', 0, 1, 1), A('bgt', off=2), A('mov', 5, imm=99), H],
        regs={5: 99}, cycles={4: 5 + 3, 6: 5 + 5})
    # wrong-path store must not happen
    ref, _ = run('wrong-path-store', [A('mov', 1, imm=1), A('b', off=3), A('st', 1, DBASE, imm=0), A('push', 1), H],
                 regs={SP: STACK_BASE})
    expect(not ref.dm, 'wrong-path store/push leaked into memory')
    # halt on the wrong path must be killed
    run('wrong-path-halt', [A('mov', 1, imm=1), A('b', off=3), A('halt'), A('mov', 2, imm=7), A('mov', 3, imm=8), H],
        regs={2: 0, 3: 8})
    # multi-cycle EX with operands that must be latched (producer one/two ahead of a mul)
    run('mul-latch', [A('mov', 1, imm=6), A('mov', 2, imm=7), A('mul', 3, 1, 2), A('add', 4, 3, 1), A('add', 5, 4, 3), H],
        regs={3: 42, 4: 48, 5: 90}, cycles={4: 6 + 3 + 1, 6: 6 + 5 + 1})
    run('div-chain', [A('mov', 1, imm=100), A('mov', 2, imm=7), A('div', 3, 1, 2), A('mod', 4, 1, 2), A('div', 5, 3, 3), H],
        regs={3: 14, 4: 2, 5: 1}, cycles={4: 6 + 3 + 7 + 7 + 7, 6: 6 + 5 + 7 + 7 + 7})
    run('div-by-zero', [A('mov', 1, imm=9), A('mov', 2, imm=0), A('div', 3, 1, 2), A('mod', 4, 1, 2), H], regs={3: M32, 4: 9})
    # floats
    run('float-basic', [A('mov', 1, imm=fimm(1.5), F=1), A('mov', 2, imm=fimm(2.25), F=1), A('add', 3, 1, 2, F=1),
                        A('mul', 4, 3, 2, F=1), A('fcvt', 5, 0, 3, F=1), A('mov', 6, imm=7), A('fcvt', 7, 0, 6, F=0),
                        A('sub', 8, 7, 1, F=1), A('div', 9, 4, 1, F=1), H],
        regs={3: f2b(3.75), 4: f2b(8.4375), 5: 3, 7: f2b(7.0), 8: f2b(5.5), 9: f2b(5.625)})
    run('float-cmp-branch', [A('mov', 1, imm=fimm(2.0), F=1), A('mov', 2, imm=fimm(1.0), F=1), A('cmp', 0, 1, 2, F=1),
                             A('bgt', off=2), A('mov', 7, imm=111), A('mov', 8, imm=222), H], regs={7: 0, 8: 222})
    run('float-nan-unordered', [A('mov', 1, imm=fimm(1.0), F=1), A('mov', 2, imm=fimm(0.0), F=1), A('div', 3, 2, 2, F=1),
                                A('cmp', 0, 1, 3, F=1), A('beq', off=2), A('bgt', off=2), A('mov', 7, imm=5), H], regs={7: 5})
    run('float-minmax-absneg', [A('mov', 1, imm=fimm(-3.0), F=1), A('mov', 2, imm=fimm(2.0), F=1), A('min', 3, 1, 2, F=1),
                                A('max', 4, 1, 2, F=1), A('abs', 5, 0, 1, F=1), A('neg', 6, 0, 2, F=1), H],
        regs={3: f2b(-3.0), 4: f2b(2.0), 5: f2b(3.0), 6: f2b(-2.0)})
    run('int-misc', [A('mov', 1, imm=-8), A('abs', 2, 0, 1), A('neg', 3, 0, 2), A('min', 4, 1, 2), A('max', 5, 1, 2),
                     A('asr', 6, 1, imm=1), A('lsr', 7, 1, imm=28), A('not', 8, 0, 2), H],
        regs={2: 8, 3: -8, 4: -8, 5: 8, 6: -4, 7: 0xF, 8: ~8})
    # exception: push below the limit, handler resets SP, rte re-executes the push
    ref, outs = run('exception-push', [A('mov', SP, imm=STACK_LIMIT), A('mov', 1, imm=5), A('push', 1), H],
                    regs={SP: (STACK_BASE + STACK_LIMIT) // 2 - 4}, cycles={4: 12 + 1, 6: 15 + 1})
    expect(ref.cause == 1 and ref.epc == 8, 'exception-push: epc/cause wrong')
    run('exception-pop', [A('pop', 1), H], regs={SP: (STACK_BASE + STACK_LIMIT) // 2 + 4})
    # halt waits for older instructions
    run('halt-drains', [A('mov', 1, imm=2), A('mul', 2, 1, 1), A('mul', 3, 2, 2), H], regs={2: 4, 3: 16})
    # pop sp: port 1 (loaded value) must win over port 2 (SP+4); SP read at distance 1, 2, 3 and 4
    SPV = 0x1F80
    pre = [A('mov', 1, imm=SPV), A('push', 1), A('pop', SP)]
    for dist in range(0, 4):
        run('pop-sp-dist%d' % (dist + 1), pre + [A('nop')] * dist + [A('add', 5, SP, imm=0), H], regs={SP: SPV, 5: SPV})
    run('pop-sp-then-push', pre + [A('mov', 2, imm=77), A('push', 2), A('pop', 3), H], regs={SP: SPV, 3: 77})
    # exceptions and redirects can never coincide with a busy multi-cycle EX: those instructions are 1 cycle
    for op in BR + ('ret', 'push', 'pop', 'rte', 'halt'):
        for F in (0, 1):
            expect(latency(make_uop(encode(A(op, F=F) if op not in BR else A(op, off=1)), 0)) == 1,
                   '%s must be single-cycle' % op)
    print('  directed: %d programs match the reference (values and cycle counts)' % n_ok)
    print('  invariant: branch/call/ret/push/pop/rte/halt are always 1 cycle, so exBusy never overlaps a redirect or exception')


def gen_program(rnd):
    pool = [1, 2, 3, 4, 5]
    def src():
        return rnd.choice(pool) if rnd.random() < .75 else rnd.choice(list(range(0, 14)) + [SP, RA])
    def dst():
        return rnd.choice(pool) if rnd.random() < .7 else rnd.randint(0, 12)
    def small(): return rnd.choice([0, 1, 2, 3, 7, 100, 0x7FFF, 0xFFFF])
    def body_inst(allow_stack=True):
        k = rnd.random()
        if k < .30:
            op = rnd.choice(ADD3)
            F = 1 if op in FLOAT_OK and rnd.random() < .3 else 0
            if rnd.random() < .4:
                return A(op, dst(), src(), imm=(rnd.randint(0, 0x1FFFF) if F else small()), F=F, mod=0 if F else rnd.randint(0, 2))
            return A(op, dst(), src(), src(), F=F)
        if k < .42:
            op = rnd.choice(UN)
            if op == 'fcvt': return A('fcvt', dst(), 0, src(), F=rnd.randint(0, 1))
            F = 1 if op in FLOAT_OK and rnd.random() < .3 else 0
            if rnd.random() < .4:
                return A(op, dst(), 0, imm=(fimm(rnd.choice([0.5, -2.0, 3.25, 1e10, 0.0])) if F else small()), F=F, mod=0 if F else rnd.randint(0, 2))
            return A(op, dst(), 0, src(), F=F)
        if k < .52:
            F = 1 if rnd.random() < .3 else 0
            return A('cmp', 0, src(), src(), F=F)
        if k < .64:
            base = DBASE if rnd.random() < .9 else src()
            return A(rnd.choice(['ld', 'ld', 'st']), dst() if True else 0, base, imm=rnd.randrange(0, 64, 4))
        if k < .76 and allow_stack:
            return A('push', src()) if rnd.random() < .5 else A('pop', dst() if rnd.random() < .9 else SP)
        if k < .80:
            return A('nop')
        return A('add', dst(), src(), src())
    n = rnd.randint(8, 45)
    main = [A('mov', r, imm=rnd.choice([1, 2, 3, 5, 9, 0x1234])) for r in (1, 2, 3)]
    main += [A('mov', 4, imm=fimm(1.5), F=1), A('mov', 5, imm=fimm(-2.5), F=1)]
    subs = []
    nsub = 2
    body = []
    for _ in range(n):
        k = rnd.random()
        if k < .10: body.append(('br', rnd.choice(['b', 'beq', 'bgt'])))
        elif k < .16: body.append(('call', rnd.randint(0, nsub - 1)))
        else: body.append(('i', body_inst()))
    total = len(main) + len(body)           # index of final halt = total
    # place subroutines after the halt
    sub_start = []
    pos = total + 1
    for _ in range(nsub):
        sub_start.append(pos)
        sub = [body_inst() for _ in range(rnd.randint(2, 6))] + [A('ret')]
        subs += sub
        pos += len(sub)
    out = list(main)
    for j, (kind, v) in enumerate(body):
        idx = len(out)
        if kind == 'i': out.append(v)
        elif kind == 'br':
            tgt = min(idx + rnd.randint(1, 4), total)
            out.append(A(v, off=tgt - idx))
        else: out.append(A('call', off=sub_start[v] - idx))
    out.append(A('halt'))
    return out, subs


def random_tests(count, seed, **opt):
    rnd = random.Random(seed)
    bad = 0
    stats = dict(retired=0, cyc4=0, cyc6=0, exc=0)
    first = None
    for t in range(count):
        main, subs = gen_program(rnd)
        words = assemble(main, subs)
        ref = Ref(words).run()
        assert ref.halted
        r4, r6 = Pipe4(words, **opt).run(), Pipe6(words, **opt).run()
        ok4, ok6 = r4.state() == ref.state(), r6.state() == ref.state()
        if not (ok4 and ok6):
            bad += 1
            if first is None: first = (t, ok4, ok6)
        stats['retired'] += len(ref.retired); stats['cyc4'] += r4.cycles; stats['cyc6'] += r6.cycles
        stats['exc'] += ref.cause
    return bad, stats, first


def main():
    print('1. Encoding'); test_encoding()
    print('2. Directed tests'); directed()
    N = 4000
    print('3. Random programs (%d programs, compared with the reference model)' % N)
    bad, st, first = random_tests(N, seed=2024)
    expect(bad == 0, 'random tests: %d mismatches, first at program %s' % (bad, first))
    print('  random: all %d programs match on both pipelines (%d instructions retired, %d programs raised an exception)'
          % (N, st['retired'], st['exc']))
    print('  average CPI: 4-stage %.2f, 6-stage %.2f' % (st['cyc4'] / st['retired'], st['cyc6'] / st['retired']))
    print('4. Mutation checks (each feature switched off must make the tests FAIL)')
    for name, opt in [('no forwarding', dict(fwd=False)), ('no OF register bypass', dict(bypass=False)),
                      ('no load-use stall (6-stage)', dict(stall_lu=False)),
                      ('busy EX does not latch forwarded operands', dict(latch_ops=False)),
                      ('pop sp: port 2 wins instead of port 1', dict(p1_wins=False))]:
        bad, _, _ = random_tests(600, seed=7, **opt)
        expect(bad > 0, 'mutation "%s" was NOT detected' % name)
        print('  %-45s detected (%d of 600 programs fail)' % (name, bad))
    print('ALL CHECKS PASSED')


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
hazard_check.py - independent verification of the RAW-hazard / stall / freeze logic of
Pipeline_4Stage and Pipeline_6Stage (uses the instrumented pipeline_sim.py).

The checks do NOT trust the pipeline's own hazard signals. For every cycle they re-derive,
from first principles (what is really in flight and when each result becomes available):

  [operand ] every instruction executes with exactly the operand values of the one-at-a-time
             reference model (per instruction, not just final state), and the same
             branch / exception / rte outcome
  [path    ] each operand comes from the right place: youngest in-flight producer wins
             (EX_MA latch / RW bus W1,W2 / OF bypass / register file)
  [nostall ] an instruction never reaches EX while the data it needs does not exist yet
  [LU      ] load-use stall happens in exactly the cycles where it is needed (no more, no less)
  [busy    ] multi-cycle freeze lasts exactly latency-1 cycles (latencies re-typed from doc sec. 8)
  [freeze  ] during a stall PC / IF / ID / OF hold, a bubble enters the next stage, the stages
             ahead keep draining; during normal flow everything advances; on a redirect younger
             stages are killed and PC = target
  [cycles  ] total cycle count equals an analytic model (no extra or missing bubbles anywhere)
  [state   ] final registers / memory / flags / epc / cause / retire order match the reference

Run:  python3 hazard_check.py
"""
import random
import sys
from collections import Counter, defaultdict

import pipeline_sim as ps
from pipeline_sim import (A, Ref, Pipe4, Pipe6, make_uop, assemble, fimm, SP, RA, DBASE, M32,
                          STACK_BASE, STACK_LIMIT)

# ----------------------------------------------------------------------------- spec (typed from the docs)
def spec_latency(op, F):
    """doc section 8 'EX latency'"""
    if op == 'fcvt': return 2
    if F: return {'add': 3, 'sub': 3, 'mul': 3, 'div': 10}.get(op, 1)
    return {'mul': 2, 'div': 8, 'mod': 8}.get(op, 1)


def writes(u):
    """registers an instruction writes -> stage at whose END the value exists ('EX' or 'MA').
    ld/pop read memory in MA; everything else is known at the end of EX. Port 1 beats port 2 (pop sp)."""
    w = {}
    if u['wb2']: w[SP] = 'EX'
    if u['wb1']: w[u['d1']] = 'MA' if u['op'] in ('ld', 'pop') else 'EX'
    return w


def srcs(u):
    return [u['s%d' % k] for k in (1, 2) if u['use%d' % k]]


def port(u, r):
    return 'W1' if (u['wb1'] and u['d1'] == r) else 'W2'


def pcof(x):
    if x is None: return None
    return x[1] if isinstance(x, tuple) else x['pc']


def is_lu_pair(prod, cons, S):
    """does the dynamic pair (prod, cons) need a load-use bubble?"""
    if S != 6: return False
    w = writes(prod)
    return any(w.get(r) == 'MA' for r in srcs(cons))


# ----------------------------------------------------------------------------- analytic cycle model
def predict_cycles(exlog, S):
    """Rigid pipeline: EX final cycle of instr i is L_i. Gaps between consecutive instructions:
       after taken branch / exception: K bubbles (K = stages ahead of EX), after rte: 1, load-use: 1."""
    K, post = (2, 1) if S == 4 else (3, 2)
    L = None
    prev = None
    for (pc, a, b, ev, u) in exlog:
        if prev is None:
            E = 1 + K
        else:
            pev, pu = prev[3], prev[4]
            gap = K if pev in ('taken', 'exc') else 1 if pev == 'rte' else 1 if is_lu_pair(pu, u, S) else 0
            E = L + 1 + gap
        L = E + spec_latency(u['op'], u['F']) - 1
        prev = (pc, a, b, ev, u)
    return L + post


# ----------------------------------------------------------------------------- per-cycle checks
def view(rec, S):
    if S == 4:
        raw = rec['IF_OF']
        return dict(OF=make_uop(*raw) if raw else None, EX=rec['OF_EX'], MA=None, RW=rec['EX_RW'])
    return dict(OF=rec['ID_OF'], EX=rec['OF_EX'], MA=rec['EX_MA'], RW=rec['MA_RW'])


def check_cycles(p, S, err):
    T = p.trace
    for i, rec in enumerate(T):
        v = view(rec, S)
        X, C, MA, RW = v['EX'], v['OF'], v['MA'], v['RW']
        c = rec['c']
        # ---- [busy]
        exp_busy = bool(X) and X['cyc'] < spec_latency(X['op'], X['F']) - 1
        if exp_busy != rec['busy']:
            err('busy', 'cycle %d: EX=%s %s cyc=%d busy=%s expected %s' % (c, X and X['op'], X and hex(X['pc']), X and X['cyc'], rec['busy'], exp_busy))
        # ---- [path] / [nostall] at the first EX cycle of an instruction
        if X and X['cyc'] == 0:
            prev = T[i - 1] if i else None
            for k in (1, 2):
                if not X['use%d' % k]:
                    continue
                r = X['s%d' % k]
                exp = 'reg'
                for stage, P in (('MA', MA), ('RW', RW)):
                    if P and r in writes(P):
                        if stage == 'MA':
                            if writes(P)[r] == 'MA':
                                exp = 'NODATA'
                            else:
                                exp = 'MA1' if (P['wb1'] and P['d1'] == r) else 'MA2'
                        else:
                            exp = port(P, r)
                        break
                if exp == 'NODATA':
                    err('nostall', 'cycle %d: %s@%#x reached EX needing r%d while producer %s@%#x is a load still in MA (data not ready)'
                        % (c, X['op'], X['pc'], r, MA['op'], MA['pc']))
                elif rec['fs'].get(k) != exp:
                    err('path', 'cycle %d: %s@%#x operand %d (r%d) took %s, expected %s' % (c, X['op'], X['pc'], k, r, rec['fs'].get(k), exp))
                # OF bypass: judged from the cycle in which X was in OF (previous cycle)
                if prev is not None and S in (4, 6):
                    pv = view(prev, S)
                    if pv['OF'] and pv['OF']['pc'] == X['pc']:
                        expb = port(pv['RW'], r) if (pv['RW'] and r in writes(pv['RW'])) else 'reg'
                        if X['byp'].get(k) != expb:
                            err('path', 'cycle %d: %s@%#x OF read of r%d used %s, expected %s' % (c, X['op'], X['pc'], r, X['byp'].get(k), expb))
        # ---- [LU]
        if S == 6 and C and X and not rec['busy'] and rec['redirect'] is None:
            wX = writes(X)
            exp_lu = any(wX.get(r) == 'MA' for r in srcs(C))
            if exp_lu != rec['stallLU']:
                err('LU', 'cycle %d: OF=%s@%#x EX=%s@%#x stallLU=%s expected %s' % (c, C['op'], C['pc'], X['op'], X['pc'], rec['stallLU'], exp_lu))
        if S == 4 and rec['stallLU']:
            err('LU', 'cycle %d: 4-stage must never load-use stall' % c)
        # ---- [freeze]   (a stallLU asserted in the same cycle as a redirect is overridden: exception/branch > stall)
        lu_eff = rec['stallLU'] and rec['redirect'] is None
        n = rec['nxt']
        bad = []
        if S == 4:
            fr, nfr = rec['IF_OF'], n['IF_OF']
            nEX, nPOST = n['OF_EX'], n['EX_RW']
            nOF = nfr
        else:
            fr, nfr = rec['IF_ID'], n['IF_ID']
            nEX, nMA, nPOST = n['OF_EX'], n['EX_MA'], n['MA_RW']
            nOF = n['ID_OF']
        # generic checks, written per case
        if rec['busy']:
            if n['pc'] != rec['pc']: bad.append('PC moved during busy')
            if nfr is not fr: bad.append('IF latch changed during busy')
            if S == 6 and nOF is not rec['ID_OF']: bad.append('ID_OF changed during busy')
            if pcof(nEX) != pcof(X) or nEX['cyc'] != X['cyc'] + 1: bad.append('EX instruction not held')
            if S == 4:
                if nPOST is not None: bad.append('no bubble into EX_RW')
            else:
                if nMA is not None: bad.append('no bubble into EX_MA')
                if pcof(nPOST) != pcof(MA): bad.append('MA/RW did not keep draining')
        elif lu_eff:
            if n['pc'] != rec['pc']: bad.append('PC moved during LU stall')
            if nfr is not fr or nOF is not rec['ID_OF']: bad.append('IF/ID/OF not held during LU stall')
            if nEX is not None: bad.append('no bubble into OF_EX')
            if pcof(nMA) != pcof(X): bad.append('EX did not advance into MA')
            if pcof(nPOST) != pcof(MA): bad.append('MA did not advance into RW')
        elif rec['redirect'] is not None:
            if nfr is not None or nOF is not None or nEX is not None: bad.append('younger stages not killed on redirect')
            if n['pc'] != rec['redirect'] & M32: bad.append('PC != redirect target')
            exc = rec['ev'] == 'exc'
            tail = nPOST if S == 4 else nMA
            if exc and tail is not None: bad.append('faulting instruction not turned into a bubble')
            if not exc and pcof(tail) != pcof(X): bad.append('branch did not advance')
            if S == 6 and pcof(nPOST) != pcof(MA): bad.append('MA did not advance into RW')
        else:
            if S == 4:
                if pcof(nPOST) != pcof(X): bad.append('EX_MA did not advance')
                if pcof(nEX) != (pcof(rec['IF_OF'])): bad.append('IF_OF did not advance')
            else:
                if pcof(nMA) != pcof(X): bad.append('EX did not advance into MA')
                if pcof(nPOST) != pcof(MA): bad.append('MA did not advance into RW')
                if pcof(nEX) != pcof(rec['ID_OF']): bad.append('OF did not advance into EX')
                if pcof(nOF) != pcof(rec['IF_ID']): bad.append('ID did not advance into OF')
            if not rec['fekill'] and not rec['hp'] and fr is not None and n['pc'] != rec['pc'] + 4:
                bad.append('PC did not advance by 4')
        for b in bad:
            err('freeze', 'cycle %d: %s' % (c, b))


# ----------------------------------------------------------------------------- one run, all checks
CATS = ['operand', 'path', 'nostall', 'LU', 'busy', 'freeze', 'cycles', 'state', 'hang']


def check_program(words, cls, spec=None, keep=3):
    """returns (Counter of failing categories, dict category->first messages, pipe, ref)"""
    ref = Ref(words).run()
    p = cls(words, trace=True, **(spec or {})).run(maxc=4000)
    cnt, msgs = Counter(), defaultdict(list)

    def err(cat, m):
        cnt[cat] += 1
        if len(msgs[cat]) < keep: msgs[cat].append(m)
    if not p.halted:
        err('hang', 'did not halt in 4000 cycles')
        return cnt, msgs, p, ref
    # [operand] per dynamic instruction
    a, b = ref.exlog, p.exlog
    if len(a) != len(b):
        err('operand', 'executed %d instructions, reference executed %d' % (len(b), len(a)))
    for j, (x, y) in enumerate(zip(a, b)):
        if x[:4] != y[:4]:
            err('operand', 'instr #%d %s@%#x: pipeline (a=%#x b=%#x ev=%r) vs reference (a=%#x b=%#x ev=%r)'
                % (j, x[4]['op'], x[0], y[1], y[2], y[3], x[1], x[2], x[3]))
            break
    check_cycles(p, cls.S, err)
    if p.cycles != predict_cycles(ref.exlog, cls.S):
        err('cycles', 'took %d cycles, analytic model says %d' % (p.cycles, predict_cycles(ref.exlog, cls.S)))
    if p.state() != ref.state():
        err('state', 'final architectural state differs from reference')
    return cnt, msgs, p, ref


# ----------------------------------------------------------------------------- directed hazard matrix
def prefix():
    return [A('mov', 2, imm=0x100), A('mov', 3, imm=4), A('mov', 5, imm=7), A('mov', 12, imm=0x1F80),
            A('mov', 6, imm=55), A('st', 12, DBASE, imm=0), A('st', 6, DBASE, imm=8),
            A('mov', 14, imm=0x1F90), A('push', 12), A('pop', 14)] + [A('nop')] * 5      # SP = 0x1F80, mem[SP]=0


DATA_PROD = {
    'alu  add r1': lambda: A('add', 1, 2, 3),
    'mul  (2cyc)': lambda: A('mul', 1, 2, 3),
    'div  (8cyc)': lambda: A('div', 1, 2, 3),
    'fdiv (10cyc)': lambda: A('div', 1, 2, 3, F=1),
    'fcvt (2cyc)': lambda: A('fcvt', 1, 0, 2, F=0),
    'ld   r1': lambda: A('ld', 1, DBASE, imm=0),
    'pop  r1': lambda: A('pop', 1),
}
SP_PROD = {
    'push': lambda: A('push', 6),
    'pop r6': lambda: A('pop', 6),
    'add sp': lambda: A('add', SP, 12, imm=0),
    'pop sp': lambda: A('pop', SP),
    'ld sp': lambda: A('ld', SP, DBASE, imm=0),
}
DATA_CONS = {
    'rs1': lambda: A('add', 4, 1, 5),
    'rs2': lambda: A('add', 4, 5, 1),
    'both': lambda: A('add', 4, 1, 1),
    'st-data': lambda: A('st', 1, DBASE, imm=4),
    'ld-base': lambda: A('ld', 4, 1, imm=0),
    'push-data': lambda: A('push', 1),
    'cmp': lambda: A('cmp', 0, 1, 5),
    'fcvt-rs2': lambda: A('fcvt', 4, 0, 1, F=0),
    'mov-rs2': lambda: A('mov', 4, 0, 1),
}
SP_CONS = {
    'push': lambda: A('push', 5),
    'pop': lambda: A('pop', 7),
    'add rs1=sp': lambda: A('add', 7, SP, 5),
    'ld base=sp': lambda: A('ld', 7, SP, imm=0),
}
# instructions that must NOT see a hazard although their raw bit-fields name r1 (useSrc gating)
ALIAS_CONS = {
    'imm aliases rs2': lambda: A('add', 4, 5, imm=1 << 13),                 # rs2 field = 1 but I=1 -> not a source
    'nop with fields': lambda: A('nop', 1, 1, 1),
    'reserved opcode': lambda: ps.encode(A('nop')) | (23 << 27) | (1 << 17) | (1 << 13),
    'mov: rs1 field=1': lambda: A('mov', 4, 1, 5),                          # mov reads only rs2
    'not: rs1 field=1': lambda: A('not', 4, 1, 5),
    'st: rs2 field=1': lambda: A('st', 5, DBASE, imm=1 << 13),              # st has no rs2: imm bits alias r1
}


def build(prod, cons, dist):
    body = prefix()
    pp = len(body)
    body.append(prod())
    body += [A('nop')] * (dist - 1)
    cp = len(body)
    c = cons()
    body.append(c)
    body += [A('nop')] * 6 + [A('halt')]
    words = []
    for ins in body:
        words.append(ins if isinstance(ins, int) else ps.encode(ins))
    words += [ps.encode(A('nop'))] * (ps.EXEC_VECTOR // 4 - len(words))
    words += [ps.encode(i) for i in (A('mov', SP, imm=(STACK_BASE + STACK_LIMIT) // 2), A('rte'))]
    return words, pp * 4, cp * 4


def classify(p, cons_pc, k=1):
    """path actually used by the consumer for operand k, plus total freeze cycles"""
    path = '?'
    for rec in p.trace:
        X = rec['OF_EX']
        if X and X['pc'] == cons_pc and X['cyc'] == 0:
            t = rec['fs'].get(k)
            if t in ('MA1', 'MA2'): path = 'fwd EX/MA latch'
            elif t in ('W1', 'W2'): path = 'fwd RW bus'
            elif X['byp'].get(k) in ('W1', 'W2'): path = 'OF bypass'
            else: path = 'reg file'
            break
    lu = sum(1 for r in p.trace if r['stallLU'] and r['redirect'] is None)
    bz = sum(1 for r in p.trace if r['busy'])
    return path, lu, bz


def directed_matrix(spec=None):
    total, fails = 0, []
    tables = {S: defaultdict(dict) for S in (4, 6)}

    def one(group, prod_name, prod, cons_name, cons, dist):
        nonlocal total
        words, ppc, cpc = build(prod, cons, dist)
        for cls in (Pipe4, Pipe6):
            cnt, msgs, p, ref = check_program(words, cls, spec)
            total += 1
            if cnt:
                fails.append((cls.S, prod_name, cons_name, dist, dict(cnt), {k: v[:1] for k, v in msgs.items()}))
            if spec is None and cons_name in ('rs1', 'add rs1=sp') and not cnt:
                tables[cls.S][(group, prod_name)][dist] = classify(p, cpc, 1)

    for dist in range(1, 6):
        for pn, pf in DATA_PROD.items():
            for cn, cf in DATA_CONS.items(): one('data', pn, pf, cn, cf, dist)
            for cn, cf in ALIAS_CONS.items(): one('data', pn, pf, 'ALIAS ' + cn, cf, dist)
        for pn, pf in SP_PROD.items():
            for cn, cf in SP_CONS.items(): one('sp', pn, pf, cn, cf, dist)
    return total, fails, tables


# ----------------------------------------------------------------------------- random hazard-dense programs
def gen_dense(rnd):
    P = [1, 2, 3]
    r = lambda: rnd.choice(P)
    anyr = lambda: SP if rnd.random() < .05 else r()

    def inst():
        k = rnd.random()
        if k < .18: return A('ld', r(), DBASE, imm=rnd.randrange(0, 16, 4))
        if k < .25: return A('st', r(), DBASE, imm=rnd.randrange(0, 16, 4))
        if k < .33: return A('pop', anyr())
        if k < .41: return A('push', r())
        if k < .60:
            op = rnd.choice(['add', 'sub', 'and', 'or', 'lsl', 'min', 'max'])
            return A(op, r(), r(), r()) if rnd.random() < .7 else A(op, r(), r(), imm=rnd.choice([0, 1, 2, 3, 7]))
        if k < .66: return A(rnd.choice(['mul', 'div', 'mod']), r(), r(), r())
        if k < .71: return A(rnd.choice(['add', 'mul', 'div']), r(), r(), r(), F=1)
        if k < .75: return A('fcvt', r(), 0, r(), F=rnd.randint(0, 1))
        if k < .79: return A('cmp', 0, r(), r(), F=rnd.randint(0, 1))
        if k < .83: return A(rnd.choice(['mov', 'not', 'neg', 'abs']), r(), 0, r())
        if k < .86: return A('ld', r(), r(), imm=0)
        if k < .88: return A('st', r(), r(), imm=0)
        if k < .90: return A('add', SP, SP, imm=rnd.choice([0, 0, (-4) & 0x7FFF, 4]))
        if k < .93: return A('add', r(), SP, imm=rnd.choice([0, 4]))
        return A('nop')
    main = [A('mov', i, imm=rnd.choice([1, 2, 3, 0x104, 0x108, 5])) for i in (1, 2, 3)]
    nsub = 2
    n = rnd.randint(12, 40)
    body = []
    for _ in range(n):
        k = rnd.random()
        if k < .09: body.append(('br', rnd.choice(['b', 'beq', 'bgt'])))
        elif k < .13: body.append(('call', rnd.randint(0, nsub - 1)))
        else: body.append(('i', inst()))
    total = len(main) + len(body)
    subs, sub_start, pos = [], [], total + 1
    for _ in range(nsub):
        sub_start.append(pos)
        sub = [inst() for _ in range(rnd.randint(2, 5))] + [A('ret')]
        subs += sub
        pos += len(sub)
    out = list(main)
    for kind, v in body:
        idx = len(out)
        if kind == 'i': out.append(v)
        elif kind == 'br': out.append(A(v, off=min(idx + rnd.randint(1, 4), total) - idx))
        else: out.append(A('call', off=sub_start[v] - idx))
    out.append(A('halt'))
    return out, subs


def random_campaign(n, seed, spec=None, pipes=(Pipe4, Pipe6)):
    rnd = random.Random(seed)
    fail = {c.S: Counter() for c in pipes}
    first = {c.S: {} for c in pipes}
    cover = {c.S: Counter() for c in pipes}
    for _ in range(n):
        main, subs = gen_dense(rnd)
        words = assemble(main, subs)
        for cls in pipes:
            cnt, msgs, p, ref = check_program(words, cls, spec)
            for c in cnt:
                fail[cls.S][c] += 1
                first[cls.S].setdefault(c, msgs[c][0])
            if not spec:
                cv = cover[cls.S]
                cv['programs'] += 1
                cv['instructions executed'] += len(ref.exlog)
                cv['cycles'] += p.cycles
                for rec in p.trace:
                    cv['load-use stall cycles'] += rec['stallLU'] and rec['redirect'] is None
                    cv['stallLU asserted together with exception (overridden)'] += rec['stallLU'] and rec['redirect'] is not None
                    cv['multi-cycle freeze cycles'] += rec['busy']
                    X = rec['OF_EX']
                    if X and X['cyc'] == 0:
                        for k in (1, 2):
                            t = rec['fs'].get(k)
                            if t in ('MA1', 'MA2', 'W1', 'W2'): cv['EX forward: ' + t] += 1
                            if X['byp'].get(k) in ('W1', 'W2'): cv['OF bypass: ' + X['byp'][k]] += 1
                            if t in ('MA1', 'MA2', 'W1', 'W2') and X['s%d' % k] == SP: cv['  of which SP (r14)'] += 1
                            if t in ('MA1', 'MA2', 'W1', 'W2') and X['s%d' % k] == RA: cv['  of which ra (r15)'] += 1
                        if rec['fs'].get(1) in ('W2', 'MA2') or rec['fs'].get(2) in ('W2', 'MA2'): cv['port-2 (SP) forwards'] += 1
                for (pc, a_, b_, ev, u) in ref.exlog:
                    if ev: cv['events: ' + ev] += 1
    return fail, first, cover


MUTS = [  # (name, constructor-flags, pipelines, must_be_caught)
    ('no forwarding at all', dict(fwd=False), (4, 6), True),
    ('no OF register-file bypass', dict(bypass=False), (4, 6), True),
    ('no load-use stall', dict(stall_lu=False), (6,), True),
    ('load-use stall forgets pop (ld only)', dict(mut={'lu_ld_only'}), (6,), True),
    ('load-use stall checks only src1', dict(mut={'lu_src1_only'}), (6,), True),
    ('load-use stall checks only src2', dict(mut={'lu_src2_only'}), (6,), True),
    ('load-use stall without useSrc gating (false stalls)', dict(mut={'lu_ignore_use'}), (6,), True),
    ('load-use stall lasts 2 cycles', dict(mut={'lu_two_cycles'}), (6,), True),
    ('load-use stall does not freeze PC', dict(mut={'lu_no_hold_pc'}), (6,), True),
    ('load-use stall does not insert bubble', dict(mut={'lu_no_bubble'}), (6,), True),
    ('forwarding priority: older producer beats younger', dict(mut={'fwd_old_first'}), (6,), True),
    ('no port-2 (SP) forwarding', dict(mut={'no_sp_fwd'}), (4, 6), True),
    ('pop sp: port 2 wins over port 1', dict(p1_wins=False), (4, 6), True),
    ('multi-cycle EX does not latch operands', dict(latch_ops=False), (4, 6), True),
    ('multi-cycle EX does not freeze PC', dict(mut={'busy_no_hold_pc'}), (4, 6), True),
    ('multi-cycle EX releases 1 cycle early', dict(mut={'lat_minus1'}), (4, 6), True),
    ('call link (ra) never forwarded/bypassed', dict(mut={'no_call_link_fwd'}), (4, 6), False),
]


# ----------------------------------------------------------------------------- extra directed hazard cases
def prog(body, subs=()):
    return assemble(body, subs)


def directed_extra(spec=None, quiet=False):
    """named hand-written hazard programs; returns (n, failures, notes)"""
    H = A('halt')
    cases = []
    # 1) faulting pop with a dependent instruction right behind it (stallLU asserted AND exception in the same cycle)
    cases.append(('faulting pop r1 ; add r2,r1,r1 (consumer right behind)', prog([A('pop', 1), A('add', 2, 1, 1), H])))
    cases.append(('faulting pop r1 ; push r1 (reads r1 and SP)', prog([A('mov', 1, imm=3), A('pop', 1), A('push', 1), H])))
    cases.append(('faulting pop sp ; add r2,sp,0', prog([A('pop', SP), A('add', 2, SP, imm=0), H])))
    cases.append(('faulting push after ld (ld r1 ; push r1 with SP at limit)', prog([A('mov', SP, imm=STACK_LIMIT), A('ld', 1, DBASE, imm=0), A('push', 1), H])))
    # 2) flags: cmp -> branch at distance 0..3, double cmp, wrong-path cmp, cmp behind a load-use stall
    for d in range(4):
        for br, eq in (('beq', 1), ('bgt', 0)):
            r2 = 4 if eq else 2
            cases.append(('cmp ; %d nop ; %s' % (d, br), prog([A('mov', 1, imm=4), A('mov', 2, imm=r2), A('cmp', 0, 1, 2)] + [A('nop')] * d + [A(br, off=2), A('mov', 5, imm=99), H])))
    cases.append(('cmp ; cmp ; beq uses 2nd flags', prog([A('mov', 1, imm=4), A('mov', 2, imm=9), A('cmp', 0, 1, 1), A('cmp', 0, 1, 2), A('beq', off=2), A('mov', 5, imm=99), H])))
    cases.append(('wrong-path cmp must not touch flags', prog([A('mov', 1, imm=4), A('mov', 2, imm=9), A('cmp', 0, 1, 2), A('b', off=3), A('cmp', 0, 1, 1), A('cmp', 0, 1, 1), A('beq', off=2), A('mov', 5, imm=99), H])))
    cases.append(('ld ; cmp on loaded reg (stall) ; beq', prog([A('mov', 1, imm=4), A('st', 1, DBASE, imm=0), A('ld', 2, DBASE, imm=0), A('cmp', 0, 2, 1), A('beq', off=2), A('mov', 5, imm=99), H])))
    cases.append(('mul ; cmp on mul result ; beq (busy then flags)', prog([A('mov', 1, imm=4), A('mul', 2, 1, 1), A('cmp', 0, 2, 2), A('beq', off=2), A('mov', 5, imm=99), H])))
    # 3) memory RAW (single memory port in order: no hazard expected)
    for d in range(4):
        cases.append(('st ; %d nop ; ld same address' % d, prog([A('mov', 1, imm=0x55), A('st', 1, DBASE, imm=4)] + [A('nop')] * d + [A('ld', 2, DBASE, imm=4), A('add', 3, 2, 2), H])))
        cases.append(('push ; %d nop ; pop' % d, prog([A('mov', SP, imm=0x1F80), A('mov', 1, imm=0x66), A('push', 1)] + [A('nop')] * d + [A('pop', 2), A('add', 3, 2, 2), H])))
    cases.append(('push ; ld r2,0[sp] (SP forwarded as base, memory just written)', prog([A('mov', SP, imm=0x1F80), A('mov', 1, imm=0x77), A('push', 1), A('ld', 2, SP, imm=0), A('add', 3, 2, 2), H])))
    # 4) ret reading ra produced by ordinary instructions at distance 1..4 (generic forwarding on r15)
    for d in range(1, 5):
        tgt = (1 + 1 + (d - 1) + 1 + 3) * 4              # index of the final halt (see layout below)
        cases.append(('mov r15 ; %d nop ; ret' % (d - 1), prog([A('mov', 6, imm=tgt), A('mov', RA, 0, 6)] + [A('nop')] * (d - 1) + [A('ret'), A('mov', 7, imm=1), A('mov', 7, imm=2), A('mov', 7, imm=3), H])))
        cases.append(('ld r15 ; %d nop ; ret' % (d - 1), prog([A('mov', 6, imm=tgt), A('st', 6, DBASE, imm=0), A('ld', RA, DBASE, imm=0)] + [A('nop')] * (d - 1) + [A('ret'), A('mov', 7, imm=1), A('mov', 7, imm=2), A('mov', 7, imm=3), H])))
    # 5) call then use of ra right after the call returns / inside the subroutine
    cases.append(('call ; sub: ret immediately', prog([A('call', off=3), A('mov', 5, imm=1), H, A('ret')])))
    cases.append(('call ; sub: add r2,ra,0 ; ret', prog([A('call', off=3), A('mov', 5, imm=1), H, A('add', 2, RA, imm=0), A('ret')])))
    # 6) halt behind hazards: older instructions must complete
    cases.append(('ld ; consumer ; halt', prog([A('mov', 1, imm=9), A('st', 1, DBASE, imm=0), A('ld', 2, DBASE, imm=0), A('add', 3, 2, 2), H])))
    cases.append(('div ; halt', prog([A('mov', 1, imm=100), A('mov', 2, imm=7), A('div', 3, 1, 2), H])))
    cases.append(('div ; dependent add ; halt', prog([A('mov', 1, imm=100), A('mov', 2, imm=7), A('div', 3, 1, 2), A('add', 4, 3, 3), H])))
    cases.append(('fdiv ; dependent fadd ; dependent st ; halt', prog([A('mov', 1, imm=fimm(3.0), F=1), A('mov', 2, imm=fimm(2.0), F=1), A('div', 3, 1, 2, F=1), A('add', 4, 3, 3, F=1), A('st', 4, DBASE, imm=0), H])))
    # 7) back-to-back writers of the same register: youngest must win (WAW + RAW)
    cases.append(('mov r1 ; mov r1 ; mov r1 ; add r2,r1,r1', prog([A('mov', 1, imm=1), A('mov', 1, imm=2), A('mov', 1, imm=3), A('add', 2, 1, 1), H])))
    cases.append(('ld r1 ; mov r1 ; add r2,r1 (younger ALU beats older load, no stall)', prog([A('mov', 3, imm=5), A('st', 3, DBASE, imm=0), A('ld', 1, DBASE, imm=0), A('mov', 1, imm=8), A('add', 2, 1, 1), H])))
    cases.append(('mov r1 ; ld r1 ; add r2,r1 (younger load beats older ALU, stall)', prog([A('mov', 3, imm=5), A('st', 3, DBASE, imm=0), A('mov', 1, imm=8), A('ld', 1, DBASE, imm=0), A('add', 2, 1, 1), H])))
    cases.append(('add sp ; pop ; push (youngest SP writer = pop port 2)', prog([A('mov', 12, imm=0x1F80), A('add', SP, 12, imm=0), A('pop', 1), A('push', 1), A('pop', 2), H])))
    cases.append(('pop r1 ; pop r2 ; push r1 ; push r2 chain', prog([A('mov', SP, imm=0x1F80), A('mov', 4, imm=11), A('push', 4), A('push', 4), A('pop', 1), A('pop', 2), A('push', 1), A('push', 2), A('pop', 3), H])))
    fails, notes = [], Counter()
    for name, words in cases:
        for cls in (Pipe4, Pipe6):
            cnt, msgs, p, ref = check_program(words, cls, spec)
            if cnt: fails.append((cls.S, name, dict(cnt), {k: v[:1] for k, v in msgs.items()}))
    return len(cases) * 2, fails, notes


def main():
    ok = True
    print('=' * 78)
    print('A. DIRECTED HAZARD MATRIX  (producer x consumer x distance 1..5, both pipelines)')
    print('=' * 78)
    total, fails, tables = directed_matrix()
    print('  %d runs, each checked by [operand][path][nostall][LU][busy][freeze][cycles][state]: %d failures' % (total, len(fails)))
    for f in fails[:10]: print('   FAIL', f)
    ok &= not fails
    for S in (4, 6):
        print('\n  %d-STAGE: path the consumer used for its rs1 operand; LU=load-use bubble, BZ=multi-cycle freeze cycles' % S)
        print('  %-14s' % 'producer' + ''.join('%-24s' % ('distance %d' % d) for d in range(1, 6)))
        order = lambda k: (k[0] != 'data', list(DATA_PROD).index(k[1]) if k[0] == 'data' else list(SP_PROD).index(k[1]))
        for key in sorted(tables[S], key=order):
            row = tables[S][key]
            cells = []
            for d in range(1, 6):
                path, lu, bz = row[d]
                cells.append('%-24s' % (path + (' +LU%d' % lu if lu else '') + (' +BZ%d' % bz if bz else '')))
            print('  %-14s' % key[1] + ''.join(cells))

    print('\n' + '=' * 78)
    print('A2. HAND-WRITTEN HAZARD CASES (faulting pop + consumer, flags, memory RAW, ret/ra, halt, WAW)')
    print('=' * 78)
    n, f2, notes = directed_extra()
    print('  %d runs: %d failures' % (n, len(f2)))
    for f in f2[:10]: print('   FAIL', f)
    ok &= not f2

    print('\n' + '=' * 78)
    print('B. RANDOM HAZARD-DENSE PROGRAMS (3 live registers; loads, stack ops, mul/div/float, branches, calls, exceptions)')
    print('=' * 78)
    N = 3000
    fail, first, cover = random_campaign(N, seed=11)
    for S in (4, 6):
        bad = sum(fail[S].values())
        print('  %d-stage: %d programs, violations: %d %s' % (S, N, bad, dict(fail[S]) if bad else ''))
        ok &= not bad
        for c, m in first[S].items(): print('     first', c, ':', m)
    print('\n  coverage (how often each hazard path was really exercised):')
    keys = sorted(set(cover[4]) | set(cover[6]))
    print('  %-58s %9s %9s' % ('', '4-stage', '6-stage'))
    for k in keys: print('  %-58s %9d %9d' % (k, cover[4][k], cover[6][k]))

    print('\n' + '=' * 78)
    print('C. FAULT INJECTION: break one hazard feature at a time; the checks must catch it')
    print('   random = programs failing out of 400, by detector;  matrix/hand = failing runs of A / A2')
    print('=' * 78)
    for name, spec, ss, must in MUTS:
        pipes = tuple(c for c in (Pipe4, Pipe6) if c.S in ss)
        fail, first, _ = random_campaign(400, 5, spec, pipes)
        mt, mf, _ = directed_matrix(spec)
        et, ef, _ = directed_extra(spec)
        for S in sorted(fail):
            cats = ', '.join('%s:%d' % (c, fail[S][c]) for c in CATS if fail[S][c])
            dm = sum(1 for f in mf if f[0] == S) 
            de = sum(1 for f in ef if f[0] == S)
            caught = bool(fail[S]) or dm or de
            if must: ok &= bool(caught)
            tag = 'CAUGHT' if caught else ('survives' if not must else 'MISSED!')
            print('  %-52s %d-stage %-8s random[%s] matrix:%d hand:%d' % (name, S, tag, cats or '-', dm, de))
    print('\n' + ('ALL HAZARD CHECKS PASSED' if ok else 'SOME CHECKS FAILED'))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())

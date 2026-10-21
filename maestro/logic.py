"""
Theory Checker: a knowledge-based agent for music theory.
Facts are extracted from a melody, rules are Horn clauses with variables over note positions,
forward chaining derives violations with proofs, and a truth-table model checker shows why
inference beats enumeration.
"""
import itertools
import re
import time
from .theory import scale_pcs, midi_to_name

RULES = {
    "Western tonal": [
        {"name": "R1 Large leap", "scope": "local",
         "premises": [("Leap({i})", False)],
         "conclusion": "Violation[Leap larger than 7 semitones]({i})"},
        {"name": "R2 Leading tone resolves", "scope": "local",
         "premises": [("LeadingTone({i})", False), ("Tonic({j})", True), ("Last({i})", True)],
         "conclusion": "Violation[Leading tone not resolved to tonic]({i})"},
        {"name": "R3 Leap recovered (up)", "scope": "local",
         "premises": [("Leap({i})", False), ("Up({i})", False), ("Up({j})", False)],
         "conclusion": "Violation[Leap not recovered by contrary motion]({i})"},
        {"name": "R3 Leap recovered (down)", "scope": "local",
         "premises": [("Leap({i})", False), ("Down({i})", False), ("Down({j})", False)],
         "conclusion": "Violation[Leap not recovered by contrary motion]({i})"},
        {"name": "R4 No tritone leaps", "scope": "local",
         "premises": [("Tritone({i})", False)],
         "conclusion": "Violation[Tritone leap]({i})"},
        {"name": "R5 No static repetition", "scope": "local",
         "premises": [("Repeat3({i})", False)],
         "conclusion": "Violation[Same note three times in a row]({i})"},
        {"name": "R6 Chord tone on strong beat", "scope": "local",
         "premises": [("StrongBeat({i})", False), ("ChordTone({i})", True)],
         "conclusion": "Violation[Non-chord tone on a strong beat]({i})"},
        {"name": "R7 Stay in key", "scope": "local",
         "premises": [("Outside({i})", False)],
         "conclusion": "Violation[Note outside the key]({i})"},
        {"name": "R8 End on tonic", "scope": "global",
         "premises": [("EndsOnTonic", True)],
         "conclusion": "Violation[Does not end on the tonic]"},
    ],
    "Hindustani raga": [
        {"name": "H1 No varjit swara", "scope": "local",
         "premises": [("Outside({i})", False)],
         "conclusion": "Violation[Varjit (forbidden) swara]({i})"},
        {"name": "H2 Aroha / avaroha flow", "scope": "local",
         "premises": [("Jump({i})", False)],
         "conclusion": "Violation[Jump larger than 5 semitones breaks the flow]({i})"},
        {"name": "H3 No static repetition", "scope": "local",
         "premises": [("Repeat3({i})", False)],
         "conclusion": "Violation[Same swara three times in a row]({i})"},
        {"name": "H4 Nyasa on Sa", "scope": "global",
         "premises": [("EndsOnTonic", True)],
         "conclusion": "Violation[Does not rest on Sa]"},
        {"name": "H5 Vadi emphasis", "scope": "global",
         "premises": [("VadiEmphasised", True)],
         "conclusion": "Violation[Vadi is not emphasised]"},
        {"name": "H6 Samvadi presence", "scope": "global",
         "premises": [("SamvadiPresent", True)],
         "conclusion": "Violation[Samvadi never appears]"},
    ],
}


def melody_facts(notes, ctx):
    """Perception: turn a melody into a set of ground atomic facts."""
    seq = [(m, d) for m, d in notes if m is not None]
    p = [m for m, _ in seq]
    n = len(p)
    pcs = scale_pcs(ctx.root, ctx.scale_name)
    tonic = ctx.tonic_pc
    chord_pcs = {pcs[k] for k in (0, 2, 4) if k < len(pcs)}
    vadi_pc, sam_pc = pcs[ctx.vadi], pcs[ctx.samvadi]
    facts, onset, vadi_t, sam_seen = set(), 0.0, 0.0, False
    total = sum(d for _, d in seq) or 1.0
    for i, (m, d) in enumerate(seq):
        c = m % 12
        if c == tonic:
            facts.add(f"Tonic({i})")
        if c == (tonic - 1) % 12:
            facts.add(f"LeadingTone({i})")
        if c not in pcs:
            facts.add(f"Outside({i})")
        if c in chord_pcs:
            facts.add(f"ChordTone({i})")
        if abs(onset % 2) < 1e-9:
            facts.add(f"StrongBeat({i})")
        if i == n - 1:
            facts.add(f"Last({i})")
        if i >= 2 and p[i] == p[i - 1] == p[i - 2]:
            facts.add(f"Repeat3({i})")
        if c == vadi_pc:
            vadi_t += d
        if c == sam_pc:
            sam_seen = True
        onset += d
    for i in range(n - 1):
        dlt = p[i + 1] - p[i]
        facts.add(f"Up({i})" if dlt > 0 else f"Down({i})" if dlt < 0 else f"Same({i})")
        if abs(dlt) > 7:
            facts.add(f"Leap({i})")
        if abs(dlt) > 5:
            facts.add(f"Jump({i})")
        if abs(dlt) == 6:
            facts.add(f"Tritone({i})")
    if p and p[-1] % 12 == tonic:
        facts.add("EndsOnTonic")
    if vadi_t / total >= 0.15:
        facts.add("VadiEmphasised")
    if sam_seen:
        facts.add("SamvadiPresent")
    return facts, n


def forward_chain(facts, rules, n):
    """Inference: ground each rule over note positions and fire it when all premises hold.
    Negated premises use negation-as-failure (closed world). Returns derived facts and proofs."""
    derived, proofs = set(facts), {}
    changed = True
    while changed:
        changed = False
        for r in rules:
            positions = range(n) if r["scope"] == "local" else [None]
            for i in positions:
                env = {"i": i, "j": (i + 1) if i is not None else None}
                prem = [(t.format(**env), neg) for t, neg in r["premises"]]
                if all((lit in derived) != neg for lit, neg in prem):
                    concl = r["conclusion"].format(**env)
                    if concl not in derived:
                        derived.add(concl)
                        proofs[concl] = {"rule": r["name"], "premises": prem}
                        changed = True
    return derived, proofs


def check(notes, ctx, rule_set):
    t0 = time.perf_counter()
    facts, n = melody_facts(notes, ctx)
    derived, proofs = forward_chain(facts, RULES[rule_set], n)
    violations = []
    for concl in sorted(c for c in derived if c.startswith("Violation")):
        name = re.search(r"\[(.*?)\]", concl).group(1)
        m = re.search(r"\((\d+)\)$", concl)
        idx = int(m.group(1)) if m else None
        pr = proofs[concl]
        prem_str = " ∧ ".join(("¬" if neg else "") + lit for lit, neg in pr["premises"])
        violations.append({"rule": pr["rule"], "violation": name, "index": idx,
                           "position": (idx + 1) if idx is not None else "whole melody",
                           "proof": f"{concl}  ⇐  {pr['rule']}:  {prem_str}"})
    violations.sort(key=lambda v: (v["index"] is None, v["index"] if v["index"] is not None else 0))
    return {"facts": facts, "n": n, "derived": derived, "proofs": proofs,
            "violations": violations, "seconds": time.perf_counter() - t0}


def model_check(facts, k):
    """Truth-table entailment (TT-ENTAILS) for 'the first k notes contain a violation' under
    simplified R1 and R2. Enumerates every model over the relevant symbols."""
    atoms = [f"{p}({i})" for i in range(k) for p in ("Leap", "LeadingTone", "Tonic")] + [f"Tonic({k})"]
    r1 = [([(f"Leap({i})", False)], f"V_leap({i})") for i in range(k)]
    r2 = [([(f"LeadingTone({i})", False), (f"Tonic({i + 1})", True)], f"V_lt({i})") for i in range(k)]
    rules = r1 + r2
    concl = [c for _, c in rules]
    symbols = atoms + concl
    truth = {a: (a in facts) for a in atoms}

    def kb_true(model):
        if any(model[a] != truth[a] for a in atoms):
            return False
        for prem, c in rules:
            if all(model[lit] != neg for lit, neg in prem) and not model[c]:
                return False
        return True

    t0 = time.perf_counter()
    models = kb_models = 0
    entailed = True
    for vals in itertools.product((False, True), repeat=len(symbols)):
        model = dict(zip(symbols, vals))
        models += 1
        if kb_true(model):
            kb_models += 1
            if not any(model[c] for c in concl):
                entailed = False
    return {"k": k, "symbols": len(symbols), "models": models, "kb_models": kb_models,
            "entailed": entailed, "seconds": time.perf_counter() - t0}


def label(midi):
    return midi_to_name(midi)
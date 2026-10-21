"""
Composer Agent: a goal-based agent whose goal is 'zero rule violations', with fitness as the
utility tie-breaker. Percept = facts and violations, action = change one note, environment = the melody.
"""
from .logic import check
from .evolve import FITNESS, HOLD
from .theory import midi_to_name


def encode(notes, ctx):
    """Melody -> genome, so the Stage 3 fitness functions can score it."""
    genome = []
    for m, d in notes:
        steps = max(1, int(round(d / 0.5)))
        if m is None:
            genome += [HOLD] * steps
        else:
            nearest = min(ctx.pitches, key=lambda x: abs(x - m))
            genome.append(ctx.pitches.index(nearest))
            genome += [HOLD] * (steps - 1)
    return genome


def run_agent(notes, ctx, rule_set, max_steps=12):
    fit_fn = FITNESS[rule_set]
    melody = [[m, d] for m, d in notes]

    def evaluate(mel):
        res = check(mel, ctx, rule_set)
        return res, fit_fn(encode(mel, ctx), ctx)[0]

    res, fit = evaluate(melody)
    trace = []
    for step in range(1, max_steps + 1):
        if not res["violations"]:
            trace.append({"Step": step, "Percept": "0 violations", "Focus": "—",
                          "Action": "Goal reached: accept melody", "Result": f"fitness {fit:.1f}"})
            break
        v = res["violations"][0]
        sounded = [k for k, (m, _) in enumerate(melody) if m is not None]
        if v["index"] is not None and v["index"] < len(sounded):
            targets = [sounded[v["index"]]]
            if v["index"] + 1 < len(sounded):
                targets.append(sounded[v["index"] + 1])
        elif "Vadi" in v["violation"] or "Samvadi" in v["violation"]:
            targets = [max(sounded, key=lambda k: melody[k][1])]
        else:
            targets = [sounded[-1]]

        best = None
        for ti in targets:
            for pitch in ctx.pitches:
                if pitch == melody[ti][0]:
                    continue
                trial = [list(x) for x in melody]
                trial[ti][0] = pitch
                r2, f2 = evaluate(trial)
                key = (len(r2["violations"]), -f2)
                if best is None or key < best[0]:
                    best = (key, trial, r2, f2, ti, pitch)

        current = (len(res["violations"]), -fit)
        if best and best[0] < current:
            old = midi_to_name(melody[best[4]][0])
            melody, res, fit = best[1], best[2], best[3]
            trace.append({"Step": step, "Percept": f"{current[0]} violations", "Focus": v["violation"],
                          "Action": f"Change note {best[4] + 1}: {old} → {midi_to_name(best[5])}",
                          "Result": f"{len(res['violations'])} violations · fitness {fit:.1f}"})
        else:
            trace.append({"Step": step, "Percept": f"{current[0]} violations", "Focus": v["violation"],
                          "Action": "No single-note change improves the melody: stop",
                          "Result": f"{current[0]} violations · fitness {fit:.1f}"})
            break
    return melody, trace, res, fit
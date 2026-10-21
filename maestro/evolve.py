"""
Melody Evolver: a genetic algorithm that evolves melodies under a chosen musical tradition.
Course mapping: Unit 3 (optimisation, genetic algorithms). The two fitness functions are the
seed of the Stage 5 bias audit: the same melody scores differently under each tradition.
"""
import random
import numpy as np
from .theory import scale_midi, pc, SCALE_FAMILY

HOLD = -1  # gene value meaning "sustain the previous note"

RAGA_INFO = {
    "Raga Yaman":   {"vadi": 2, "samvadi": 6, "time": "Early night", "mood": "Serene, romantic",
                     "sargam": ["S", "R", "G", "M#", "P", "D", "N"]},
    "Raga Bhairav": {"vadi": 5, "samvadi": 1, "time": "Dawn", "mood": "Devotional, grave",
                     "sargam": ["S", "r", "G", "M", "P", "d", "N"]},
    "Raga Kafi":    {"vadi": 4, "samvadi": 0, "time": "Late evening", "mood": "Playful, festive",
                     "sargam": ["S", "R", "g", "M", "P", "D", "n"]},
    "Raga Bhupali": {"vadi": 2, "samvadi": 4, "time": "Evening", "mood": "Calm, pastoral",
                     "sargam": ["S", "R", "G", "P", "D"]},
}


class Context:
    """Everything a fitness function needs to know about the musical setting."""
    def __init__(self, root, scale_name, tradition, length=16, octave=4):
        self.root, self.scale_name, self.tradition, self.length = root, scale_name, tradition, length
        base = scale_midi(root, scale_name, octave, include_octave=False)
        self.pitches = base + [m + 12 for m in base] + [base[0] + 24]
        self.n_deg = len(base)
        self.tonic_pc = pc(root)
        info = RAGA_INFO.get(scale_name, {})
        self.vadi = info.get("vadi", 0)
        self.samvadi = info.get("samvadi", 4 % self.n_deg)
        self.sargam = info.get("sargam")

    def degree(self, gene):
        return gene % self.n_deg

    def label(self, midi):
        from .theory import midi_to_name
        if self.sargam:
            d = self.pitches.index(midi) % self.n_deg
            octv = self.pitches.index(midi) // self.n_deg
            return self.sargam[d] + ("'" * octv)
        return midi_to_name(midi)


def decode(genome, ctx, step=0.5):
    """Genome -> list of [midi or None, beats]."""
    notes = []
    for g in genome:
        if g == HOLD:
            if notes:
                notes[-1][1] += step
            else:
                notes.append([None, step])
        else:
            notes.append([ctx.pitches[g], step])
    return notes


# ---------------- fitness: Western tonal ----------------
def fitness_western(genome, ctx):
    notes = [n for n in decode(genome, ctx) if n[0] is not None]
    if len(notes) < 2:
        return 0.0, {}
    p = [n[0] for n in notes]
    iv = [abs(b - a) for a, b in zip(p, p[1:])]
    c = {}
    c["Stepwise motion"] = round(20 * sum(1 for i in iv if i <= 2) / len(iv), 1) - 4 * sum(1 for i in iv if i > 7)
    c["Starts / ends on tonic"] = (10 if p[-1] % 12 == ctx.tonic_pc else 0) + (5 if p[0] % 12 == ctx.tonic_pc else 0)
    rep = sum(1 for i in range(2, len(p)) if p[i] == p[i - 1] == p[i - 2])
    c["Variety"] = round(10 * len(set(p)) / len(p), 1) - 5 * rep
    dirs = [np.sign(b - a) for a, b in zip(p, p[1:]) if b != a]
    changes = sum(1 for a, b in zip(dirs, dirs[1:]) if a != b)
    c["Contour"] = 10 if 2 <= changes <= 5 else round(max(0, 10 - 2 * abs(changes - 3.5)), 1)
    strong = [genome[i] for i in range(0, len(genome), 4) if genome[i] != HOLD]
    ct = sum(1 for g in strong if ctx.degree(g) in (0, 2, 4))
    c["Chord tones on strong beats"] = round(15 * ct / len(strong), 1) if strong else 0
    rng = max(p) - min(p)
    c["Comfortable range"] = 10 if rng <= 12 else max(-10, 10 - (rng - 12) * 2)
    lt = (ctx.tonic_pc - 1) % 12
    res = sum((3 if b % 12 == ctx.tonic_pc else -3) for a, b in zip(p, p[1:]) if a % 12 == lt)
    c["Leading-tone resolution"] = res
    holds = genome.count(HOLD) / len(genome)
    c["Rhythmic interest"] = 10 if 0.15 <= holds <= 0.4 else 3
    return max(0.0, float(sum(c.values()))), c


# ---------------- fitness: Hindustani raga ----------------
def fitness_hindustani(genome, ctx):
    notes = [n for n in decode(genome, ctx) if n[0] is not None]
    if len(notes) < 2:
        return 0.0, {}
    p = [n[0] for n in notes]
    total = sum(n[1] for n in notes)
    iv = [abs(b - a) for a, b in zip(p, p[1:])]
    deg_time = {}
    for m, d in notes:
        deg = ctx.pitches.index(m) % ctx.n_deg
        deg_time[deg] = deg_time.get(deg, 0) + d
    vt = deg_time.get(ctx.vadi, 0) / total
    stt = deg_time.get(ctx.samvadi, 0) / total
    c = {}
    c["Vadi emphasis"] = 20 if 0.2 <= vt <= 0.4 else round(max(0, 20 - 80 * abs(vt - 0.3)), 1)
    c["Samvadi support"] = 10 if 0.1 <= stt <= 0.25 else round(max(0, 10 - 50 * abs(stt - 0.17)), 1)
    c["Stepwise (aroha / avaroha)"] = round(15 * sum(1 for i in iv if i <= 3) / len(iv), 1) - 4 * sum(1 for i in iv if i > 5)
    first_deg = ctx.pitches.index(p[0]) % ctx.n_deg
    c["Resolves to Sa"] = (10 if p[-1] % 12 == ctx.tonic_pc else 0) + (5 if first_deg in (0, ctx.vadi) else 0)
    peak = p.index(max(p)) / max(1, len(p) - 1)
    c["Melodic arc"] = 10 if 0.35 <= peak <= 0.75 else 3
    c["Raga coverage"] = round(10 * len(deg_time) / ctx.n_deg, 1)
    holds = genome.count(HOLD) / len(genome)
    c["Sustained notes"] = 15 if 0.3 <= holds <= 0.55 else round(max(0, 15 - 60 * abs(holds - 0.42)), 1)
    rng = max(p) - min(p)
    c["Range"] = 10 if rng <= 14 else max(-10, 10 - (rng - 14) * 2)
    rep = sum(1 for i in range(2, len(p)) if p[i] == p[i - 1] == p[i - 2])
    c["Avoids static repetition"] = -3 * rep
    return max(0.0, float(sum(c.values()))), c


FITNESS = {"Western tonal": fitness_western, "Hindustani raga": fitness_hindustani}


# ---------------- genetic operators ----------------
def random_genome(ctx, rng, hold_rate=0.25):
    return [HOLD if rng.random() < hold_rate else rng.randrange(len(ctx.pitches)) for _ in range(ctx.length)]

def tournament(pop, fits, rng, k=3):
    idx = max(rng.sample(range(len(pop)), k), key=lambda i: fits[i])
    return pop[idx]

def crossover(a, b, rng):
    pt = rng.randrange(1, len(a))
    return a[:pt] + b[pt:], b[:pt] + a[pt:]

def mutate(g, ctx, rng, rate):
    out = list(g)
    top = len(ctx.pitches) - 1
    for i in range(len(out)):
        if rng.random() < rate:
            r = rng.random()
            if r < 0.2:
                out[i] = HOLD
            elif r < 0.6 and out[i] != HOLD:
                out[i] = min(top, max(0, out[i] + rng.choice([-2, -1, 1, 2])))
            else:
                out[i] = rng.randrange(top + 1)
    return out


def evolve(ctx, fitness_fn, pop_size=60, generations=80, mutation_rate=0.08, elitism=2, seed=42, snapshots=4):
    rng = random.Random(seed)
    pop = [random_genome(ctx, rng) for _ in range(pop_size)]
    history, snaps = [], []
    snap_gens = sorted({0, generations - 1} | {round(generations * i / (snapshots - 1)) for i in range(1, snapshots - 1)})

    for gen in range(generations):
        fits = [fitness_fn(g, ctx)[0] for g in pop]
        order = sorted(range(pop_size), key=lambda i: fits[i], reverse=True)
        history.append({"generation": gen + 1, "best": fits[order[0]], "mean": sum(fits) / pop_size})
        if gen in snap_gens:
            snaps.append({"generation": gen + 1, "genome": list(pop[order[0]]), "fitness": fits[order[0]]})
        new = [list(pop[i]) for i in order[:elitism]]
        while len(new) < pop_size:
            a, b = tournament(pop, fits, rng), tournament(pop, fits, rng)
            c1, c2 = crossover(a, b, rng)
            new.append(mutate(c1, ctx, rng, mutation_rate))
            if len(new) < pop_size:
                new.append(mutate(c2, ctx, rng, mutation_rate))
        pop = new

    fits = [fitness_fn(g, ctx)[0] for g in pop]
    best = pop[max(range(pop_size), key=lambda i: fits[i])]
    score, breakdown = fitness_fn(best, ctx)
    return {"best": best, "fitness": score, "breakdown": breakdown, "history": history, "snapshots": snaps}


def western_scales():
    return [s for s, f in SCALE_FAMILY.items() if f != "Hindustani Classical"]

def raga_scales():
    return [s for s, f in SCALE_FAMILY.items() if f == "Hindustani Classical"]
"""
Mood classifier + fairness audit.
Synthetic melodies are generated per mood and tradition, features are extracted,
classifiers are trained, and the model is audited for bias between traditions.
"""
import random
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, r2_score

from .theory import NOTE_NAMES, scale_pcs
from .evolve import Context, decode, HOLD, FITNESS

MOODS = {
    "Happy": {"tempo": (110, 160), "step": 2, "hold": 0.15, "up": 0.6,
              "W": ["Major (Ionian)", "Lydian", "Mixolydian", "Major Pentatonic"], "R": ["Raga Yaman", "Raga Bhupali"]},
    "Sad":   {"tempo": (55, 85), "step": 2, "hold": 0.4, "up": 0.35,
              "W": ["Natural Minor (Aeolian)", "Harmonic Minor", "Phrygian", "Minor Pentatonic"], "R": ["Raga Bhairav", "Raga Kafi"]},
    "Calm":  {"tempo": (60, 95), "step": 1, "hold": 0.5, "up": 0.5,
              "W": ["Major (Ionian)", "Dorian", "Major Pentatonic"], "R": ["Raga Bhupali", "Raga Kafi"]},
    "Tense": {"tempo": (100, 150), "step": 6, "hold": 0.1, "up": 0.5,
              "W": ["Harmonic Minor", "Phrygian", "Blues"], "R": ["Raga Bhairav", "Raga Yaman"]},
}
MOOD_ORDER = list(MOODS.keys())

FEATURES = ["tempo", "note_density", "mean_interval", "max_interval", "interval_std", "pitch_range",
            "up_ratio", "hold_fraction", "brightness", "tonic_fraction", "unique_ratio"]
FEATURE_LABELS = {
    "tempo": "Tempo (BPM)", "note_density": "Notes per beat", "mean_interval": "Average jump size",
    "max_interval": "Largest jump", "interval_std": "Jump variability", "pitch_range": "Pitch range",
    "up_ratio": "Share of upward moves", "hold_fraction": "Share of sustained time",
    "brightness": "Scale brightness", "tonic_fraction": "Time on home note", "unique_ratio": "Pitch variety",
}


def features(notes, bpm, ctx):
    seq = [(m, d) for m, d in notes if m is not None]
    p = [m for m, _ in seq]
    total = sum(d for _, d in notes) or 1.0
    iv = [b - a for a, b in zip(p, p[1:])] or [0]
    rel = [(x - ctx.tonic_pc) % 12 for x in scale_pcs(ctx.root, ctx.scale_name)]
    bright = sum(1 for r in rel if r in (4, 9, 11)) - sum(1 for r in rel if r in (1, 3, 8, 10))
    tonic_t = sum(d for m, d in seq if m % 12 == ctx.tonic_pc)
    return {
        "tempo": float(bpm),
        "note_density": len(p) / total,
        "mean_interval": float(np.mean(np.abs(iv))),
        "max_interval": float(max(abs(i) for i in iv)),
        "interval_std": float(np.std(iv)),
        "pitch_range": float(max(p) - min(p)) if p else 0.0,
        "up_ratio": sum(1 for i in iv if i > 0) / len(iv),
        "hold_fraction": max(0.0, (total - 0.5 * len(p)) / total),
        "brightness": float(bright),
        "tonic_fraction": tonic_t / total,
        "unique_ratio": len(set(p)) / max(1, len(p)),
    }


def make_melody(mood, tradition, rng, length=16):
    P = MOODS[mood]
    scale = rng.choice(P["R"] if tradition == "Hindustani raga" else P["W"])
    root = rng.choice(NOTE_NAMES)
    ctx = Context(root, scale, tradition, length)
    tempo, hold, step = rng.uniform(*P["tempo"]), P["hold"], P["step"]
    if tradition == "Hindustani raga":          # the systematic difference the model must cope with
        tempo *= 0.8
        hold = min(0.7, hold + 0.2)
    top = len(ctx.pitches) - 1
    cur = top // 2
    genome = [cur]
    for _ in range(length - 1):
        if rng.random() < hold:
            genome.append(HOLD)
            continue
        d = rng.randint(1, step) * (1 if rng.random() < P["up"] else -1)
        cur = min(top, max(0, cur + d))
        genome.append(cur)
    tempo += rng.gauss(0, 4)
    return decode(genome, ctx), round(tempo, 1), ctx, genome


def generate_dataset(n=600, western_share=0.9, seed=7):
    rng = random.Random(seed)
    rows = []
    for k in range(n):
        tradition = "Western tonal" if rng.random() < western_share else "Hindustani raga"
        mood = MOOD_ORDER[k % 4]
        notes, bpm, ctx, genome = make_melody(mood, tradition, rng)
        f = features(notes, bpm, ctx)
        f.update(mood=mood, tradition=tradition, root=ctx.root, scale=ctx.scale_name, genome=genome)
        rows.append(f)
    return pd.DataFrame(rows)


def quality_scores(df, culturally_aware):
    """Score every melody with a fitness function: Western-only, or each tradition's own."""
    out = []
    for r in df.itertuples():
        ctx = Context(r.root, r.scale, r.tradition)
        fn = FITNESS[r.tradition] if culturally_aware else FITNESS["Western tonal"]
        out.append(fn(list(r.genome), ctx)[0])
    return np.array(out)


def train_models(df, seed=7):
    X, y, g = df[FEATURES].values, df["mood"].values, df["tradition"].values
    strat = df["mood"] + "|" + df["tradition"]
    Xtr, Xte, ytr, yte, gtr, gte = train_test_split(X, y, g, test_size=0.3, random_state=seed, stratify=strat)

    rf = RandomForestClassifier(n_estimators=200, random_state=seed).fit(Xtr, ytr)
    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(Xtr, ytr)
    pred = rf.predict(Xte)

    per_group = {}
    for grp in ("Western tonal", "Hindustani raga"):
        mask = gte == grp
        per_group[grp] = accuracy_score(yte[mask], pred[mask]) if mask.any() else float("nan")

    reg_idx = [i for i, f in enumerate(FEATURES) if f != "tempo"]
    reg = LinearRegression().fit(Xtr[:, reg_idx], Xtr[:, 0])
    reg_pred = reg.predict(Xte[:, reg_idx])

    return {
        "rf": rf, "lr": lr,
        "acc_rf": accuracy_score(yte, pred), "acc_lr": accuracy_score(yte, lr.predict(Xte)),
        "per_group": per_group,
        "confusion": confusion_matrix(yte, pred, labels=MOOD_ORDER),
        "importances": dict(zip(FEATURES, rf.feature_importances_)),
        "n_train": len(ytr), "n_test": len(yte),
        "reg_r2": r2_score(Xte[:, 0], reg_pred), "reg_actual": Xte[:, 0], "reg_pred": reg_pred,
    }


def predict_one(model, notes, bpm, ctx):
    f = features(notes, bpm, ctx)
    x = np.array([[f[k] for k in FEATURES]])
    probs = model.predict_proba(x)[0]
    return dict(zip(model.classes_, probs)), f
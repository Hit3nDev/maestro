"""
Music theory core + Music Ontology (knowledge graph).
Course mapping: Unit 4 (ontologies, semantic networks, knowledge engineering).
"""
import networkx as nx

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Scale patterns as semitone offsets from the root
SCALES = {
    "Major (Ionian)":          [0, 2, 4, 5, 7, 9, 11],
    "Natural Minor (Aeolian)": [0, 2, 3, 5, 7, 8, 10],
    "Harmonic Minor":          [0, 2, 3, 5, 7, 8, 11],
    "Dorian":                  [0, 2, 3, 5, 7, 9, 10],
    "Phrygian":                [0, 1, 3, 5, 7, 8, 10],
    "Lydian":                  [0, 2, 4, 6, 7, 9, 11],
    "Mixolydian":              [0, 2, 4, 5, 7, 9, 10],
    "Major Pentatonic":        [0, 2, 4, 7, 9],
    "Minor Pentatonic":        [0, 3, 5, 7, 10],
    "Blues":                   [0, 3, 5, 6, 7, 10],
    "Raga Bhairav":            [0, 1, 4, 5, 7, 8, 11],
    "Raga Kafi":               [0, 2, 3, 5, 7, 9, 10],
    "Raga Yaman":              [0, 2, 4, 6, 7, 9, 11],
    "Raga Bhupali":            [0, 2, 4, 7, 9],
}

SCALE_FAMILY = {
    "Major (Ionian)": "Western Tonal", "Natural Minor (Aeolian)": "Western Tonal",
    "Harmonic Minor": "Western Tonal", "Dorian": "Western Modal", "Phrygian": "Western Modal",
    "Lydian": "Western Modal", "Mixolydian": "Western Modal",
    "Major Pentatonic": "Pentatonic & Folk", "Minor Pentatonic": "Pentatonic & Folk",
    "Blues": "Pentatonic & Folk",
    "Raga Bhairav": "Hindustani Classical", "Raga Kafi": "Hindustani Classical",
    "Raga Yaman": "Hindustani Classical", "Raga Bhupali": "Hindustani Classical",
}

CHORDS = {
    "maj": [0, 4, 7], "min": [0, 3, 7], "dim": [0, 3, 6], "aug": [0, 4, 8],
    "sus4": [0, 5, 7], "maj7": [0, 4, 7, 11], "min7": [0, 3, 7, 10], "dom7": [0, 4, 7, 10],
}
CHORD_LABEL = {"maj": "", "min": "m", "dim": "dim", "aug": "aug",
               "sus4": "sus4", "maj7": "maj7", "min7": "m7", "dom7": "7", "other": "?"}

INTERVALS = {
    0: "Unison", 1: "Minor 2nd", 2: "Major 2nd", 3: "Minor 3rd", 4: "Major 3rd",
    5: "Perfect 4th", 6: "Tritone", 7: "Perfect 5th", 8: "Minor 6th", 9: "Major 6th",
    10: "Minor 7th", 11: "Major 7th", 12: "Octave",
}

GENRES = {
    "Pop": ["Major (Ionian)", "Natural Minor (Aeolian)", "Major Pentatonic"],
    "Blues / Rock": ["Blues", "Minor Pentatonic", "Mixolydian"],
    "Jazz": ["Dorian", "Lydian", "Mixolydian", "Harmonic Minor"],
    "Hindustani Classical": ["Raga Bhairav", "Raga Kafi", "Raga Yaman", "Raga Bhupali"],
    "Film Score": ["Harmonic Minor", "Phrygian", "Lydian"],
}

ROMAN = ["I", "II", "III", "IV", "V", "VI", "VII"]


# ---------- basic conversions ----------
def pc(name):
    return NOTE_NAMES.index(name)

def note_to_midi(name, octave):
    return 12 * (octave + 1) + pc(name)

def midi_to_name(m):
    return f"{NOTE_NAMES[m % 12]}{m // 12 - 1}"

def midi_to_freq(m):
    return 440.0 * 2 ** ((m - 69) / 12)


# ---------- scales ----------
def scale_pcs(root, scale_name):
    return [(pc(root) + i) % 12 for i in SCALES[scale_name]]

def scale_midi(root, scale_name, octave=4, include_octave=True):
    base = note_to_midi(root, octave)
    notes = [base + i for i in SCALES[scale_name]]
    if include_octave:
        notes.append(base + 12)
    return notes


# ---------- chords ----------
def chord_pcs(root, quality):
    return [(pc(root) + i) % 12 for i in CHORDS[quality]]

def chord_node(root, quality):
    """Name used inside the ontology graph (avoids clashing with note names)."""
    return f"{root} {quality}"

def chord_display(root, quality):
    return root + CHORD_LABEL.get(quality, "?")

def pcs_to_midi(pcs, octave=4):
    """Stack pitch classes upward starting at the given octave."""
    out = [note_to_midi(NOTE_NAMES[pcs[0]], octave)]
    for p in pcs[1:]:
        prev = out[-1]
        cand = prev + ((p - prev) % 12)
        if cand == prev:
            cand += 12
        out.append(cand)
    return out

def diatonic_chords(root, scale_name):
    """Triads built by stacking thirds on each degree of a 7-note scale."""
    pcs = scale_pcs(root, scale_name)
    if len(pcs) != 7:
        return []
    table = {(4, 7): "maj", (3, 7): "min", (3, 6): "dim", (4, 8): "aug"}
    out = []
    for i in range(7):
        r, third, fifth = pcs[i], pcs[(i + 2) % 7], pcs[(i + 4) % 7]
        quality = table.get(((third - r) % 12, (fifth - r) % 12), "other")
        rn = ROMAN[i]
        if quality in ("min", "dim"):
            rn = rn.lower()
        if quality == "dim":
            rn += "°"
        if quality == "aug":
            rn += "+"
        out.append({
            "degree": rn,
            "name": NOTE_NAMES[r] + CHORD_LABEL.get(quality, "?"),
            "root": NOTE_NAMES[r],
            "quality": quality,
            "pcs": [r, third, fifth],
        })
    return out


# ---------- ontology / semantic network ----------
def build_ontology():
    """
    Directed knowledge graph.
    Node kinds: note, scale_type, scale, family, chord_type, chord, interval, genre
    Edge relations: has_note, is_a, belongs_to, contains_interval, uses_scale
    """
    G = nx.DiGraph()
    for n in NOTE_NAMES:
        G.add_node(n, kind="note")
    for fam in set(SCALE_FAMILY.values()):
        G.add_node(fam, kind="family")

    for s in SCALES:
        G.add_node(s, kind="scale_type")
        G.add_edge(s, SCALE_FAMILY[s], rel="belongs_to")
        for root in NOTE_NAMES:
            inst = f"{root} {s}"
            G.add_node(inst, kind="scale", root=root, scale_type=s)
            G.add_edge(inst, s, rel="is_a")
            for p in scale_pcs(root, s):
                G.add_edge(inst, NOTE_NAMES[p], rel="has_note")

    for q, pattern in CHORDS.items():
        G.add_node(q, kind="chord_type")
        for iv in pattern[1:]:
            G.add_node(INTERVALS[iv], kind="interval")
            G.add_edge(q, INTERVALS[iv], rel="contains_interval")
        for root in NOTE_NAMES:
            inst = chord_node(root, q)
            G.add_node(inst, kind="chord", root=root, quality=q)
            G.add_edge(inst, q, rel="is_a")
            for p in chord_pcs(root, q):
                G.add_edge(inst, NOTE_NAMES[p], rel="has_note")

    for g, scales in GENRES.items():
        G.add_node(g, kind="genre")
        for s in scales:
            G.add_edge(g, s, rel="uses_scale")
    return G


def notes_of(G, node):
    return {v for v in G.successors(node) if G[node][v]["rel"] == "has_note"}

def scales_containing_chord(G, chord_inst):
    """Query: which scales (all roots) contain every note of this chord?"""
    cn = notes_of(G, chord_inst)
    out = []
    for n, d in G.nodes(data=True):
        if d.get("kind") == "scale" and cn <= notes_of(G, n):
            out.append({"scale": n, "type": d["scale_type"], "family": SCALE_FAMILY[d["scale_type"]]})
    return out

def chords_sharing_notes(G, chord_inst, k=2):
    """Query: which other chords share at least k notes with this chord?"""
    cn = notes_of(G, chord_inst)
    out = []
    for n, d in G.nodes(data=True):
        if d.get("kind") == "chord" and n != chord_inst:
            shared = cn & notes_of(G, n)
            if len(shared) >= k:
                out.append({"chord": chord_display(d["root"], d["quality"]),
                            "shared": len(shared),
                            "notes": ", ".join(sorted(shared, key=NOTE_NAMES.index))})
    out.sort(key=lambda r: -r["shared"])
    return out

def chord_neighbourhood(G, chord_inst, max_scales=6):
    """Small subgraph for drawing: the chord, its notes, its type, a few containing scales."""
    nodes = {chord_inst} | notes_of(G, chord_inst)
    nodes |= {v for v in G.successors(chord_inst) if G[chord_inst][v]["rel"] == "is_a"}
    for row in scales_containing_chord(G, chord_inst)[:max_scales]:
        nodes.add(row["scale"])
    return G.subgraph(nodes).copy()
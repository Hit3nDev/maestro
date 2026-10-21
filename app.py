import re
import streamlit as st
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

from maestro import theory, audio, search, ui, evolve, logic, agent

st.set_page_config(page_title="Maestro", page_icon="🎼", layout="wide")
ui.inject_css()
ui.brand()

STAGES = [
    "Foundations · Sound & Ontology",
    "Chord Pathfinder · Search",
    "Melody Evolver · Genetic Algorithm",
    "Theory Checker · Logic & Agent",
    "Mood & Ethics · ML Audit",
]
page = st.sidebar.radio("Modules", STAGES, label_visibility="collapsed")
st.sidebar.markdown("---")
st.sidebar.caption("A transparent, culturally configurable AI composer. Built on search, logic, genetic algorithms and machine learning.")

INSTRUMENTS = ["piano", "sine", "triangle", "square"]

PROGRESSIONS = {
    "I – IV – V – I (classic)": [0, 3, 4, 0],
    "I – V – vi – IV (pop)": [0, 4, 5, 3],
    "ii – V – I (jazz)": [1, 4, 0],
    "I – vi – IV – V (50s)": [0, 5, 3, 4],
    "i – VI – III – VII (minor pop)": [0, 5, 2, 6],
}

KIND_COLOUR = {"chord": ui.PALETTE["rose"], "note": ui.PALETTE["teal"], "chord_type": ui.PALETTE["gold"],
               "scale": ui.PALETTE["violet2"], "scale_type": "#264653", "family": "#9b5de5",
               "genre": "#f15bb5", "interval": "#777"}

DEMO_MELODY = "C4 E4 B4 F4 G4 D5 D5 D5 F4 E4 B3 C4"


@st.cache_resource
def get_ontology():
    return theory.build_ontology()


@st.cache_resource
def get_chord_graph():
    return search.build_chord_graph(max_cost=4)


# ---------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------
def melody_audio(notes, bpm, waveform, drone_midi=None):
    samples = audio.render_melody([n[0] for n in notes], [n[1] for n in notes], bpm, waveform).copy()
    if drone_midi is not None and len(samples):
        dur = len(samples) / audio.SR
        drone = audio.synth_tone(theory.midi_to_freq(drone_midi), dur, "sine", 0.10) \
              + audio.synth_tone(theory.midi_to_freq(drone_midi + 7), dur, "sine", 0.05)
        n = min(len(samples), len(drone))
        samples[:n] += drone[:n]
    return audio.to_wav_bytes(samples)


def safe_label(ctx, m):
    try:
        return ctx.label(m)
    except (ValueError, IndexError):
        return theory.midi_to_name(m)


def piano_roll(notes, ctx, title, bad=None, height=3.2):
    bad = bad or set()
    fig, ax = plt.subplots(figsize=(9, height))
    ui.style_axes(fig, ax, title)
    t, k = 0.0, 0
    for m, d in notes:
        if m is not None:
            colour = ui.PALETTE["rose"] if k in bad else ui.PALETTE["gold"]
            ax.add_patch(plt.Rectangle((t, m - 0.42), d - 0.06, 0.84, color=colour, alpha=.95))
            k += 1
        t += d
    pitches = sorted(set(ctx.pitches) | {m for m, _ in notes if m is not None})
    ax.set_yticks(pitches)
    ax.set_yticklabels([safe_label(ctx, m) for m in pitches], fontsize=7)
    ax.set_ylim(min(pitches) - 1, max(pitches) + 1)
    ax.set_xlim(0, max(t, 1))
    ax.set_xlabel("beats")
    for b in range(0, int(t) + 1, 4):
        ax.axvline(b, color=ui.PALETTE["line"], lw=.8)
    return fig


def parse_notes(text):
    notes = []
    for tok in text.replace(",", " ").split():
        m = re.fullmatch(r"([A-Ga-g][#b]?)(\d)", tok.strip())
        if not m:
            raise ValueError(f"Can't read '{tok}'. Use names like C4, F#4, Bb3.")
        name, octv = m.group(1), int(m.group(2))
        name = name[0].upper() + name[1:]
        flats = {"Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#", "Cb": "B", "Fb": "E"}
        name = flats.get(name, name)
        if name not in theory.NOTE_NAMES:
            raise ValueError(f"Unknown note '{tok}'.")
        notes.append([theory.note_to_midi(name, octv), 1.0])
    if len(notes) < 2:
        raise ValueError("Enter at least two notes.")
    return notes


# =====================================================================
# STAGE 1
# =====================================================================
def stage1():
    ui.hero("Knowledge representation", "Foundations",
            "The sound engine and the music ontology that every later module builds on.")
    tab1, tab2, tab3 = st.tabs(["Scale Player", "Chord Lab", "Music Ontology"])

    with tab1:
        c1, c2, c3 = st.columns(3)
        root = c1.selectbox("Root note", theory.NOTE_NAMES, index=0, key="s_root")
        scale = c2.selectbox("Scale / Raga", list(theory.SCALES.keys()), key="s_scale")
        octave = c3.selectbox("Octave", [3, 4, 5], index=1, key="s_oct")
        c4, c5 = st.columns(2)
        bpm = c4.slider("Tempo (BPM)", 60, 200, 120, key="s_bpm")
        waveform = c5.selectbox("Instrument", INSTRUMENTS, key="s_wave")

        notes = theory.scale_midi(root, scale, octave)
        ui.chips([theory.midi_to_name(m) for m in notes])
        st.caption(f"Family: {theory.SCALE_FAMILY[scale]} · Semitone pattern: {theory.SCALES[scale]}")
        steps = [{"From": theory.midi_to_name(a), "To": theory.midi_to_name(b),
                  "Interval": theory.INTERVALS[b - a]} for a, b in zip(notes, notes[1:])]
        with st.expander("Interval steps"):
            st.dataframe(pd.DataFrame(steps), hide_index=True, width="stretch")
        if st.button("▶  Play scale (up and down)"):
            seq = notes + notes[-2::-1]
            st.audio(audio.to_wav_bytes(audio.render_melody(seq, [0.5] * len(seq), bpm, waveform)), format="audio/wav")

    with tab2:
        c1, c2, c3 = st.columns(3)
        root = c1.selectbox("Key root", theory.NOTE_NAMES, index=0, key="c_root")
        seven = [s for s in theory.SCALES if len(theory.SCALES[s]) == 7]
        scale = c2.selectbox("Key scale (7-note)", seven, key="c_scale")
        octave = c3.selectbox("Octave", [3, 4, 5], index=1, key="c_oct")
        c4, c5 = st.columns(2)
        bpm = c4.slider("Tempo (BPM)", 60, 200, 100, key="c_bpm")
        waveform = c5.selectbox("Instrument", INSTRUMENTS, key="c_wave")

        chords = theory.diatonic_chords(root, scale)
        st.subheader(f"Diatonic chords of {root} {scale}")
        cols = st.columns(7)
        for col, ch in zip(cols, chords):
            col.markdown(f"**{ch['degree']}**  \n{ch['name']}")
            col.caption(", ".join(theory.NOTE_NAMES[p] for p in ch["pcs"]))
            midi = theory.pcs_to_midi(ch["pcs"], octave)
            col.audio(audio.to_wav_bytes(audio.render_chords([midi], [1.5], bpm, waveform)), format="audio/wav")

        st.subheader("Progression")
        prog_name = st.selectbox("Progression", list(PROGRESSIONS.keys()))
        degrees = PROGRESSIONS[prog_name]
        ui.path_line([chords[d]["name"] for d in degrees])
        if st.button("▶  Play progression"):
            seq = [theory.pcs_to_midi(chords[d]["pcs"], octave) for d in degrees] * 2
            st.audio(audio.to_wav_bytes(audio.render_chords(seq, [2.0] * len(seq), bpm, waveform)), format="audio/wav")

    with tab3:
        G = get_ontology()
        kinds = pd.Series([d["kind"] for _, d in G.nodes(data=True)]).value_counts()
        m1, m2, m3 = st.columns(3)
        m1.metric("Nodes", G.number_of_nodes())
        m2.metric("Edges", G.number_of_edges())
        m3.metric("Concept kinds", len(kinds))

        chord_nodes = sorted(n for n, d in G.nodes(data=True) if d["kind"] == "chord")
        default = chord_nodes.index("C maj") if "C maj" in chord_nodes else 0
        sel = st.selectbox("Pick a chord to query", chord_nodes, index=default)
        d = G.nodes[sel]
        ui.chips([theory.chord_display(d['root'], d['quality'])] +
                 sorted(theory.notes_of(G, sel), key=theory.NOTE_NAMES.index))

        left, right = st.columns(2)
        with left:
            st.markdown("**Query 1 — scales containing every note of this chord**")
            st.dataframe(pd.DataFrame(theory.scales_containing_chord(G, sel)), hide_index=True,
                         width="stretch", height=300)
        with right:
            k = st.slider("Query 2 — chords sharing at least k notes", 1, 3, 2)
            st.dataframe(pd.DataFrame(theory.chords_sharing_notes(G, sel, k)), hide_index=True,
                         width="stretch", height=300)

        st.markdown("**Semantic network around this chord**")
        H = theory.chord_neighbourhood(G, sel)
        fig, ax = plt.subplots(figsize=(9, 5))
        fig.patch.set_alpha(0)
        ax.set_facecolor("none")
        pos = nx.spring_layout(H, seed=42, k=0.9)
        colours = [KIND_COLOUR.get(H.nodes[n]["kind"], "#ccc") for n in H.nodes]
        nx.draw_networkx_nodes(H, pos, node_color=colours, node_size=900, ax=ax, alpha=.95)
        nx.draw_networkx_labels(H, pos, font_size=7, font_color="#111", ax=ax)
        nx.draw_networkx_edges(H, pos, arrows=True, edge_color="#6b6480", ax=ax)
        nx.draw_networkx_edge_labels(H, pos, {(u, v): e["rel"] for u, v, e in H.edges(data=True)},
                                     font_size=6, font_color=ui.PALETTE["gold2"], bbox=dict(alpha=0), ax=ax)
        ax.axis("off")
        st.pyplot(fig)
        plt.close(fig)

        with st.expander("Concepts used"):
            st.markdown("""
- **Ontology / semantic network:** nodes are concepts (notes, chords, scales, genres); edges are typed relations
  (`has_note`, `is_a`, `belongs_to`, `uses_scale`). Queries are traversals over these relations.
- **Knowledge engineering:** the tables in `theory.py` are hand-encoded domain knowledge; `build_ontology()` turns them into a graph.
""")


# =====================================================================
# STAGE 2
# =====================================================================
def draw_chord_graph(adj, meta, result, start, goal):
    pos = search.positions(meta)
    fig, ax = plt.subplots(figsize=(8.5, 8.5))
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")
    ax.axis("off")
    ax.set_aspect("equal")

    for r, label in [(3.0, "Major"), (2.05, "Minor"), (1.2, "Dominant 7")]:
        ax.add_patch(plt.Circle((0, 0), r, fill=False, color=ui.PALETTE["line"], lw=.8, ls="--"))
        ax.text(0.05, r + 0.12, label, color=ui.PALETTE["muted"], fontsize=7, ha="left")

    for a in adj:
        for b in adj[a]:
            if a < b:
                ax.plot([pos[a][0], pos[b][0]], [pos[a][1], pos[b][1]], color="#3a3450", lw=.6, alpha=.55, zorder=1)

    expanded = set(result["order"]) if result else set()
    path = result["path"] if result else []
    pathset = set(path)
    for a, b in zip(path, path[1:]):
        ax.plot([pos[a][0], pos[b][0]], [pos[a][1], pos[b][1]], color=ui.PALETTE["gold"], lw=3.2, zorder=3,
                solid_capstyle="round")

    for n, (x, y) in pos.items():
        if n in pathset:
            fc, ec, s, tc, fw = ui.PALETTE["gold2"], ui.PALETTE["gold"], 460, "#15120a", "bold"
        elif n in expanded:
            fc, ec, s, tc, fw = ui.PALETTE["violet"], ui.PALETTE["violet2"], 330, ui.PALETTE["ink"], "normal"
        else:
            fc, ec, s, tc, fw = ui.PALETTE["panel2"], "#3a3450", 260, ui.PALETTE["muted"], "normal"
        ax.scatter(x, y, s=s, c=fc, edgecolors=ec, linewidths=1.2, zorder=4)
        ax.text(x, y, n, ha="center", va="center", fontsize=7.2, color=tc, fontweight=fw, zorder=5)

    ax.set_xlim(-3.6, 3.6)
    ax.set_ylim(-3.6, 3.6)
    ax.text(-3.5, -3.45, "●  path", color=ui.PALETTE["gold2"], fontsize=8)
    ax.text(-2.6, -3.45, "●  expanded", color=ui.PALETTE["violet2"], fontsize=8)
    ax.text(-1.4, -3.45, "●  untouched", color=ui.PALETTE["muted"], fontsize=8)
    return fig


def stage2():
    ui.hero("Search", "Chord Pathfinder",
            "Find the smoothest road from one chord to another. Four search strategies compete over a 36-chord harmonic graph, and you can hear the path each one finds.")
    nodes, adj, meta = get_chord_graph()
    names = list(nodes.keys())

    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
    start = c1.selectbox("Start chord", names, index=names.index("C"))
    goal = c2.selectbox("Goal chord", names, index=names.index("F#"))
    bpm = c3.slider("Tempo (BPM)", 60, 180, 90)
    waveform = c4.selectbox("Instrument", INSTRUMENTS, key="p_wave")

    ui.card("How the graph is built",
            "Nodes are the 12 major, 12 minor and 12 dominant-7th chords. Two chords are connected when the total "
            "voice-leading movement between them is 4 semitones or less, and that movement is the edge cost. "
            "A* uses the number of notes that still differ from the goal as its heuristic: each differing note must "
            "move at least one semitone, so the heuristic never overestimates.")

    results = {name: fn(adj, start, goal, nodes=nodes) for name, fn in search.ALGORITHMS.items()}
    found = {k: v for k, v in results.items() if v}
    if not found:
        st.warning("No path found between these chords.")
        return
    best_cost = min(v["cost"] for v in found.values())

    cols = st.columns(4)
    for col, (name, r) in zip(cols, results.items()):
        if r:
            col.metric(name.replace(" Search", ""), f"cost {r['cost']}", f"{r['expanded']} nodes expanded",
                       delta_color="off")
        else:
            col.metric(name, "—", "no path")

    rows = [{"Algorithm": k, "Path": " → ".join(v["path"]), "Hops": v["hops"], "Total cost": v["cost"],
             "Nodes expanded": v["expanded"], "Max frontier": v["frontier_max"],
             "Optimal": "✓" if v["cost"] == best_cost else "✗"} for k, v in found.items()]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    st.subheader("Visualise one strategy")
    chosen = st.radio("Strategy", list(found.keys()), horizontal=True, label_visibility="collapsed")
    r = found[chosen]
    gcol, icol = st.columns([1.25, 1])
    with gcol:
        fig = draw_chord_graph(adj, meta, r, start, goal)
        st.pyplot(fig)
        plt.close(fig)
    with icol:
        ui.path_line(r["path"])
        st.caption(f"{r['hops']} hops · total voice-leading cost {r['cost']} · {r['expanded']} nodes expanded")
        if st.button("▶  Hear this path"):
            seq = [theory.pcs_to_midi(nodes[n], 4) for n in r["path"]]
            st.audio(audio.to_wav_bytes(audio.render_chords(seq, [2.0] * len(seq), bpm, waveform)), format="audio/wav")

        st.markdown("**Nodes expanded, by strategy**")
        fig2, ax2 = plt.subplots(figsize=(5, 2.6))
        ui.style_axes(fig2, ax2)
        labels = [k.replace(" Search", "") for k in found]
        vals = [v["expanded"] for v in found.values()]
        colours = [ui.PALETTE["gold"] if k == chosen else ui.PALETTE["violet"] for k in found]
        ax2.barh(labels, vals, color=colours)
        ax2.invert_yaxis()
        ax2.set_xlabel("nodes expanded")
        for i, v in enumerate(vals):
            ax2.text(v + .3, i, str(v), va="center", color=ui.PALETTE["ink"], fontsize=8)
        st.pyplot(fig2)
        plt.close(fig2)

        with st.expander("Expansion order"):
            st.write(" → ".join(r["order"]))

    with st.expander("Strategy properties"):
        st.dataframe(pd.DataFrame(search.PROPERTIES), hide_index=True, width="stretch")
        st.markdown("""
- **b** = branching factor, **d** = depth of the shallowest goal, **m** = maximum depth, **C\\*** = optimal cost, **ε** = smallest edge cost.
- BFS finds the fewest *hops*, but hops are not cost: UCS and A* find the smoothest *voice leading*.
- A* and UCS reach the same cost; A* usually expands fewer nodes. That gap is the value of a good heuristic.
- DFS finds *a* path quickly but it is often long and musically awkward: completeness without optimality.
""")


# =====================================================================
# STAGE 3
# =====================================================================
def stage3():
    ui.hero("Optimisation", "Melody Evolver",
            "Random noise becomes a melody through selection, crossover and mutation. Pick a tradition, "
            "watch fitness climb, and listen to generation one against the final generation.")

    c1, c2, c3 = st.columns([1.1, 1, 1])
    tradition = c1.radio("Tradition (fitness function)", list(evolve.FITNESS.keys()), horizontal=True)
    scale_opts = evolve.raga_scales() if tradition == "Hindustani raga" else evolve.western_scales()
    root = c2.selectbox("Root / Sa", theory.NOTE_NAMES, index=0, key="e_root")
    scale = c3.selectbox("Scale / Raga", scale_opts, key="e_scale_" + tradition)

    if scale in evolve.RAGA_INFO:
        info = evolve.RAGA_INFO[scale]
        ui.card(f"{scale}",
                f"Vadi (king note): <b>{info['sargam'][info['vadi']]}</b> · Samvadi (minister): <b>{info['sargam'][info['samvadi']]}</b> · "
                f"Time: {info['time']} · Mood: {info['mood']}. The fitness function rewards dwelling on the vadi, "
                f"sustained notes, stepwise movement and resolving to Sa.")
    else:
        ui.card("Western tonal fitness",
                "Rewards stepwise motion, chord tones on strong beats, a clear contour, variety, a comfortable range, "
                "leading-tone resolution and starting/ending on the tonic.")

    with st.expander("Genetic algorithm parameters", expanded=False):
        p1, p2, p3, p4, p5 = st.columns(5)
        length = p1.selectbox("Melody length (8th notes)", [16, 24, 32], index=0)
        pop_size = p2.slider("Population", 20, 150, 60, step=10)
        generations = p3.slider("Generations", 10, 200, 80, step=10)
        mutation = p4.slider("Mutation rate", 0.01, 0.30, 0.08, step=0.01)
        seed = p5.number_input("Random seed", 0, 9999, 42)
    a1, a2 = st.columns(2)
    bpm = a1.slider("Tempo (BPM)", 50, 180, 96 if tradition == "Western tonal" else 72, key="e_bpm")
    waveform = a2.selectbox("Instrument", INSTRUMENTS, key="e_wave")

    if st.button("✦  Evolve a melody"):
        ctx = evolve.Context(root, scale, tradition, length=length)
        with st.spinner("Evolving…"):
            result = evolve.evolve(ctx, evolve.FITNESS[tradition], pop_size, generations, mutation, seed=int(seed))
        st.session_state["evo"] = {"ctx": ctx, "result": result, "tradition": tradition}

    if "evo" not in st.session_state:
        st.info("Set your options and press Evolve.")
        return

    ctx, result, trad = st.session_state["evo"]["ctx"], st.session_state["evo"]["result"], st.session_state["evo"]["tradition"]
    hist = pd.DataFrame(result["history"])
    other = [t for t in evolve.FITNESS if t != trad][0]
    cross_score, _ = evolve.FITNESS[other](result["best"], ctx)
    drone = ctx.pitches[0] - 12 if trad == "Hindustani raga" else None

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Generation 1 best", f"{hist['best'].iloc[0]:.1f}")
    m2.metric("Final best", f"{result['fitness']:.1f}", f"+{result['fitness'] - hist['best'].iloc[0]:.1f}", delta_color="off")
    m3.metric(f"Scored by {trad}", f"{result['fitness']:.0f}")
    m4.metric(f"Same melody scored by {other}", f"{cross_score:.0f}", "the gap is the ethics story", delta_color="off")

    g1, g2 = st.columns([1.2, 1])
    with g1:
        fig, ax = plt.subplots(figsize=(7, 3.4))
        ui.style_axes(fig, ax, "Convergence")
        ax.plot(hist["generation"], hist["best"], color=ui.PALETTE["gold"], lw=2.2, label="best")
        ax.plot(hist["generation"], hist["mean"], color=ui.PALETTE["violet2"], lw=1.4, label="population mean")
        ax.fill_between(hist["generation"], hist["mean"], hist["best"], color=ui.PALETTE["gold"], alpha=.08)
        ax.set_xlabel("generation")
        ax.set_ylabel("fitness")
        ax.legend(frameon=False, labelcolor=ui.PALETTE["ink"], fontsize=8)
        st.pyplot(fig)
        plt.close(fig)
    with g2:
        st.markdown("**Fitness breakdown of the final melody**")
        bd = pd.DataFrame([{"Component": k, "Score": v} for k, v in result["breakdown"].items()])
        st.dataframe(bd, hide_index=True, width="stretch", height=330)

    st.subheader("Listen to evolution")
    cols = st.columns(len(result["snapshots"]))
    for col, snap in zip(cols, result["snapshots"]):
        col.markdown(f"**Generation {snap['generation']}**  \nfitness {snap['fitness']:.1f}")
        col.audio(melody_audio(evolve.decode(snap["genome"], ctx), bpm, waveform, drone), format="audio/wav")

    final_notes = evolve.decode(result["best"], ctx)
    fig = piano_roll(final_notes, ctx, f"Final melody · {ctx.root} {ctx.scale_name}")
    st.pyplot(fig)
    plt.close(fig)
    ui.chips([safe_label(ctx, m) for m, _ in final_notes if m is not None])
    st.caption("This melody is now available in the Theory Checker.")

    with st.expander("How the genetic algorithm works here"):
        st.markdown(f"""
- **Genome:** {ctx.length} genes. Each gene is an index into the {len(ctx.pitches)} available scale notes, or `HOLD` to sustain the previous note.
- **Fitness:** a weighted sum of musical criteria (table above). Changing the tradition changes the criteria, not the algorithm.
- **Selection:** tournament of 3. **Crossover:** one-point. **Mutation:** per-gene; a mutated gene becomes a hold, nudges by a step or two, or re-randomises.
- **Elitism:** the top 2 survive unchanged so the best fitness never decreases.
- **Why it matters for ethics:** the metric on the far right scores the *same* melody under the other tradition's rules. A fitness function is a value judgement encoded as arithmetic.
""")


# =====================================================================
# STAGE 4
# =====================================================================
def violations_panel(title, res, notes, ctx, bpm, waveform, key):
    st.markdown(f"### {title}")
    n_v = len(res["violations"])
    m1, m2 = st.columns(2)
    m1.metric("Violations", n_v)
    m2.metric("Inference time", f"{res['seconds'] * 1000:.2f} ms")
    if n_v:
        df = pd.DataFrame([{"Rule": v["rule"], "Position": v["position"], "Violation": v["violation"]}
                           for v in res["violations"]])
        st.dataframe(df, hide_index=True, width="stretch", height=min(260, 45 + 36 * n_v))
    else:
        st.success("No violations. The melody satisfies every rule in this set.")
    bad = {v["index"] for v in res["violations"] if v["index"] is not None}
    fig = piano_roll(notes, ctx, f"{ctx.root} {ctx.scale_name}", bad=bad, height=2.6)
    st.pyplot(fig)
    plt.close(fig)
    with st.expander("Proofs (why each violation holds)"):
        if n_v:
            for v in res["violations"]:
                st.code(v["proof"], language=None)
        else:
            st.write("Nothing to prove.")
    with st.expander(f"Perceived facts ({len(res['facts'])})"):
        ui.chips(sorted(res["facts"]))


def stage4():
    ui.hero("Logic & agents", "Theory Checker & Composer Agent",
            "A knowledge-based agent perceives a melody as logical facts, fires rules to derive violations with proofs, "
            "and then repairs the melody until the rules are satisfied. Two rule sets, Western and Hindustani, judge the same notes.")

    src_opts = ["Demo melody (deliberate mistakes)", "Type your own notes"]
    if "evo" in st.session_state:
        src_opts.insert(0, "Melody from the Evolver")
    c1, c2, c3, c4 = st.columns([1.3, 1, 1, 1])
    source = c1.radio("Melody source", src_opts, horizontal=False)
    root = c2.selectbox("Key root / Sa", theory.NOTE_NAMES, index=0, key="l_root")
    wscale = c3.selectbox("Western key", evolve.western_scales(), index=0, key="l_wscale")
    raga = c4.selectbox("Raga", evolve.raga_scales(), index=0, key="l_raga")

    if source == "Melody from the Evolver":
        evo = st.session_state["evo"]
        notes = [list(n) for n in evolve.decode(evo["result"]["best"], evo["ctx"])]
        st.caption(f"Loaded the evolved {evo['ctx'].root} {evo['ctx'].scale_name} melody ({evo['tradition']}).")
    elif source == "Type your own notes":
        text = st.text_input("Notes (name + octave, space separated)", DEMO_MELODY)
        try:
            notes = parse_notes(text)
        except ValueError as e:
            st.error(str(e))
            return
    else:
        notes = parse_notes(DEMO_MELODY)
        ui.chips(DEMO_MELODY.split())

    a1, a2 = st.columns(2)
    bpm = a1.slider("Tempo (BPM)", 50, 180, 100, key="l_bpm")
    waveform = a2.selectbox("Instrument", INSTRUMENTS, key="l_wave")
    st.audio(melody_audio(notes, bpm, waveform), format="audio/wav")

    ctx_w = evolve.Context(root, wscale, "Western tonal")
    ctx_r = evolve.Context(root, raga, "Hindustani raga")
    res_w = logic.check(notes, ctx_w, "Western tonal")
    res_r = logic.check(notes, ctx_r, "Hindustani raga")

    tab1, tab2, tab3 = st.tabs(["Rule check & proofs", "Model checking vs inference", "Composer agent"])

    with tab1:
        left, right = st.columns(2)
        with left:
            violations_panel("Western tonal rules", res_w, notes, ctx_w, bpm, waveform, "w")
        with right:
            violations_panel("Hindustani raga rules", res_r, notes, ctx_r, bpm, waveform, "r")
        ui.card("Same notes, different verdicts",
                f"The Western rule set found <b>{len(res_w['violations'])}</b> violations and the raga rule set found "
                f"<b>{len(res_r['violations'])}</b>. Neither is wrong: each encodes a different tradition's grammar. "
                "A music AI that ships with only one of these rule sets silently declares the other tradition incorrect.")
        with st.expander("Concepts used"):
            st.markdown("""
- **Knowledge-based agent:** perceive → TELL facts to the knowledge base → ASK what follows → act.
- **Propositional facts** like `LeadingTone(3)` are ground atoms; rules like `LeadingTone(i) ∧ ¬Tonic(i+1) ∧ ¬Last(i) → Violation(i)` are
  first-order Horn clauses with a variable over note positions, grounded before inference (propositionalisation).
- **Forward chaining** fires every rule whose premises are satisfied until nothing new is derived; each derived fact keeps its proof.
- Negated premises use **negation as failure** (closed-world assumption): what is not known to be true is treated as false.
""")

    with tab2:
        ui.card("Why not just enumerate every possible world?",
                "Truth-table entailment (TT-ENTAILS) checks whether 'a violation exists' is true in every model of the knowledge base. "
                "It is sound and complete, but it must enumerate 2ⁿ models for n symbols. Forward chaining over the full melody took "
                f"<b>{res_w['seconds'] * 1000:.2f} ms</b>. Watch what enumeration costs for just the first few notes.")
        kmax = st.slider("Notes to model-check (first k notes, simplified rules R1 and R2)", 1, 3, 3)
        if st.button("Run model checker"):
            rows = [logic.model_check(res_w["facts"], k) for k in range(1, kmax + 1)]
            st.session_state["mc"] = rows
        if "mc" in st.session_state:
            rows = st.session_state["mc"]
            df = pd.DataFrame([{"k notes": r["k"], "Symbols": r["symbols"], "Models enumerated": r["models"],
                                "Models where KB is true": r["kb_models"],
                                "KB ⊨ 'violation exists'": "Yes" if r["entailed"] else "No",
                                "Time (ms)": round(r["seconds"] * 1000, 2)} for r in rows])
            st.dataframe(df, hide_index=True, width="stretch")
            fig, ax = plt.subplots(figsize=(7, 2.8))
            ui.style_axes(fig, ax, "Models enumerated grows as 2ⁿ")
            ax.bar([str(r["symbols"]) for r in rows], [r["models"] for r in rows], color=ui.PALETTE["gold"])
            ax.set_yscale("log")
            ax.set_xlabel("number of symbols")
            ax.set_ylabel("models (log)")
            st.pyplot(fig)
            plt.close(fig)
            fc_count = sum(1 for v in res_w["violations"] if v["index"] is not None and v["index"] < kmax
                           and ("Leap larger" in v["violation"] or "Leading tone" in v["violation"]))
            st.caption(f"Cross-check: forward chaining found {fc_count} matching violation(s) in the first {kmax} notes, "
                       f"so the two methods {'agree' if (fc_count > 0) == rows[-1]['entailed'] else 'disagree (R2 is simplified here)'}.")

    with tab3:
        rule_set = st.radio("Rule set the agent must satisfy", list(logic.RULES.keys()), horizontal=True)
        ctx = ctx_w if rule_set == "Western tonal" else ctx_r
        max_steps = st.slider("Maximum actions", 3, 20, 12)
        if st.button("✦  Let the agent repair the melody"):
            with st.spinner("Agent working…"):
                fixed, trace, res_f, fit_f = agent.run_agent(notes, ctx, rule_set, max_steps)
            st.session_state["agent"] = {"fixed": fixed, "trace": trace, "res": res_f, "fit": fit_f,
                                         "rule_set": rule_set, "before": [list(n) for n in notes]}
        if "agent" in st.session_state and st.session_state["agent"]["rule_set"] == rule_set:
            A = st.session_state["agent"]
            before = logic.check(A["before"], ctx, rule_set)
            m1, m2, m3 = st.columns(3)
            m1.metric("Violations before", len(before["violations"]))
            m2.metric("Violations after", len(A["res"]["violations"]))
            m3.metric("Fitness after", f"{A['fit']:.1f}")
            st.dataframe(pd.DataFrame(A["trace"]), hide_index=True, width="stretch")
            b1, b2 = st.columns(2)
            drone = ctx.pitches[0] - 12 if rule_set == "Hindustani raga" else None
            with b1:
                st.markdown("**Before**")
                st.audio(melody_audio(A["before"], bpm, waveform, drone), format="audio/wav")
            with b2:
                st.markdown("**After**")
                st.audio(melody_audio(A["fixed"], bpm, waveform, drone), format="audio/wav")
            bad = {v["index"] for v in A["res"]["violations"] if v["index"] is not None}
            fig = piano_roll(A["fixed"], ctx, "Repaired melody", bad=bad)
            st.pyplot(fig)
            plt.close(fig)
        with st.expander("Agent design"):
            st.markdown("""
- **Type:** goal-based agent with a utility tie-breaker. **Goal:** zero violations. **Utility:** the tradition's fitness score.
- **Percept:** the facts and violations derived by the Theory Checker. **Actions:** change one note to any pitch of the scale.
- **Policy:** focus on the first violation, try every candidate pitch at the offending position (and the next), pick the action
  that minimises violations and then maximises fitness. Stop when the goal is reached or no single action improves the state.
- **Environment:** fully observable, deterministic, discrete, single-agent, and sequential (each repair changes future percepts).
""")


# =====================================================================
def placeholder(title, desc):
    ui.hero("Coming next", title, desc)


if page == STAGES[0]:
    stage1()
elif page == STAGES[1]:
    stage2()
elif page == STAGES[2]:
    stage3()
elif page == STAGES[3]:
    stage4()
else:
    placeholder("Mood & Ethics", "ML mood classifier plus the cultural-bias audit that gives the project its problem statement.")
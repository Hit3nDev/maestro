import streamlit as st
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx

from maestro import theory, audio, search, ui

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


@st.cache_resource
def get_ontology():
    return theory.build_ontology()


@st.cache_resource
def get_chord_graph():
    return search.build_chord_graph(max_cost=4)


# =====================================================================
# STAGE 1
# =====================================================================
def stage1():
    ui.hero("Unit 4 · Knowledge representation", "Foundations",
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
            st.dataframe(pd.DataFrame(steps), hide_index=True, use_container_width=True)
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
                         use_container_width=True, height=300)
        with right:
            k = st.slider("Query 2 — chords sharing at least k notes", 1, 3, 2)
            st.dataframe(pd.DataFrame(theory.chords_sharing_notes(G, sel, k)), hide_index=True,
                         use_container_width=True, height=300)

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

        with st.expander("How this maps to the course"):
            st.markdown("""
- **Ontology / semantic network (Unit 4):** nodes are concepts (notes, chords, scales, genres); edges are typed relations
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
    ui.hero("Unit 2 · Problem-solving agents", "Chord Pathfinder",
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
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

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

    with st.expander("Strategy properties (viva notes)"):
        st.dataframe(pd.DataFrame(search.PROPERTIES), hide_index=True, use_container_width=True)
        st.markdown("""
- **b** = branching factor, **d** = depth of the shallowest goal, **m** = maximum depth, **C\\*** = optimal cost, **ε** = smallest edge cost.
- BFS finds the fewest *hops*, but hops are not cost: UCS and A* find the smoothest *voice leading*.
- A* and UCS reach the same cost; A* usually expands fewer nodes. That gap is the value of a good heuristic.
- DFS finds *a* path quickly but it is often long and musically awkward: a textbook example of completeness without optimality.
""")


# =====================================================================
def placeholder(title, desc):
    ui.hero("Coming next", title, desc)


if page == STAGES[0]:
    stage1()
elif page == STAGES[1]:
    stage2()
elif page == STAGES[2]:
    placeholder("Melody Evolver", "A genetic algorithm that evolves melodies you can listen to, generation by generation.")
elif page == STAGES[3]:
    placeholder("Theory Checker & Composer Agent", "Propositional and first-order rule engine with proofs, under Western or Hindustani rule sets, driven by a goal-based agent.")
else:
    placeholder("Mood & Ethics", "ML mood classifier plus the cultural-bias audit that gives the project its problem statement.")
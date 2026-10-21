"""Visual theme for Maestro: injected CSS, layout helpers, matplotlib palette."""
import streamlit as st

PALETTE = {
    "bg": "#0b0a10", "panel": "#15131d", "panel2": "#1c1927", "line": "#2a2638",
    "ink": "#ece7dd", "muted": "#9a93a8", "gold": "#d4af6a", "gold2": "#f0d9a5",
    "violet": "#6b5ca5", "violet2": "#8f7fd1", "teal": "#2a9d8f", "rose": "#e76f51",
}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,500;0,600;0,700;1,500&family=Manrope:wght@300;400;500;600;700&display=swap');

:root {
  --bg:#0b0a10; --panel:#15131d; --panel2:#1c1927; --line:#2a2638;
  --ink:#ece7dd; --muted:#9a93a8; --gold:#d4af6a; --gold2:#f0d9a5;
}
html, body, [class*="css"], .stMarkdown, .stText, p, li, label, div { font-family: 'Manrope', sans-serif; }
.stApp {
  background:
    radial-gradient(1100px 600px at 8% -10%, #2b1f44 0%, transparent 60%),
    radial-gradient(900px 500px at 100% 0%, #3b2a1d 0%, transparent 55%),
    radial-gradient(700px 500px at 50% 110%, #1a1430 0%, transparent 60%),
    var(--bg);
  color: var(--ink);
}
header[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }

h1, h2, h3 { font-family: 'Cormorant Garamond', serif !important; font-weight: 600 !important; color: var(--gold2) !important; letter-spacing: .01em; }
h2 { font-size: 1.9rem !important; margin-top: 1.2rem !important; }
h3 { font-size: 1.45rem !important; }

section[data-testid="stSidebar"] { background: linear-gradient(180deg, #130f1c 0%, #0b0a10 100%); border-right: 1px solid var(--line); }
section[data-testid="stSidebar"] * { color: var(--ink); }
section[data-testid="stSidebar"] .stRadio label { padding: .35rem .2rem; border-radius: 8px; }
section[data-testid="stSidebar"] .stRadio label:hover { background: var(--panel2); }

.stTabs [data-baseweb="tab-list"] { gap: .4rem; border-bottom: 1px solid var(--line); }
.stTabs [data-baseweb="tab"] { background: transparent; color: var(--muted); padding: .6rem 1.1rem; border-radius: 10px 10px 0 0; font-weight: 500; }
.stTabs [aria-selected="true"] { color: var(--gold2) !important; background: var(--panel) !important; border-bottom: 2px solid var(--gold) !important; }

.stButton > button {
  background: linear-gradient(135deg, #d6b26a 0%, #8f6d2e 100%); color: #15120a; border: none; border-radius: 999px;
  padding: .55rem 1.4rem; font-weight: 700; letter-spacing: .03em; box-shadow: 0 8px 24px rgba(212,175,106,.22);
  transition: transform .12s ease, filter .12s ease;
}
.stButton > button:hover { filter: brightness(1.08); transform: translateY(-1px); color: #15120a; }
.stButton > button:active { transform: translateY(0); }

div[data-testid="stMetric"] { background: linear-gradient(180deg, var(--panel), #110f18); border: 1px solid var(--line); border-radius: 16px; padding: .9rem 1.1rem; }
div[data-testid="stMetricLabel"] p { color: var(--muted) !important; font-size: .8rem; letter-spacing: .08em; text-transform: uppercase; }
div[data-testid="stMetricValue"] { color: var(--gold2); font-family: 'Cormorant Garamond', serif; font-size: 2rem; }
div[data-testid="stMetricDelta"] { color: var(--muted) !important; }
div[data-testid="stMetricDelta"] svg { display: none; }

div[data-testid="stDataFrame"], div[data-testid="stExpander"] { border: 1px solid var(--line); border-radius: 14px; background: var(--panel); }
div[data-testid="stExpander"] summary { color: var(--gold2); font-weight: 600; }

.stSelectbox div[data-baseweb="select"] > div, .stTextInput input, .stNumberInput input {
  background: var(--panel2) !important; border: 1px solid var(--line) !important; border-radius: 10px !important; color: var(--ink) !important;
}
.stSelectbox label, .stSlider label, .stRadio label, .stTextInput label { color: var(--muted) !important; font-size: .82rem; letter-spacing: .06em; text-transform: uppercase; }
div[data-baseweb="slider"] div[role="slider"] { background: var(--gold) !important; border: 2px solid var(--gold2) !important; }

audio { width: 100%; border-radius: 999px; filter: sepia(.45) saturate(.7) hue-rotate(-12deg) brightness(.92); }

.m-hero { padding: 1.4rem 0 .6rem; }
.m-hero h1 { font-size: 3.3rem; margin: 0; line-height: 1.05; }
.m-hero p { color: var(--muted); font-size: 1.08rem; margin: .5rem 0 0; max-width: 60ch; }
.m-eyebrow { font-size: .72rem; letter-spacing: .28em; text-transform: uppercase; color: var(--gold); margin-bottom: .3rem; }
.m-card { background: linear-gradient(180deg, var(--panel), #110f18); border: 1px solid var(--line); border-radius: 16px; padding: 1rem 1.25rem; margin: .5rem 0 .8rem; }
.m-card h4 { font-family: 'Cormorant Garamond', serif; color: var(--gold2); margin: 0 0 .35rem; font-size: 1.35rem; }
.m-card p { color: var(--muted); margin: 0; font-size: .93rem; line-height: 1.55; }
.m-chip { display: inline-block; background: var(--panel2); border: 1px solid var(--line); color: var(--gold2); border-radius: 999px; padding: .18rem .75rem; margin: .15rem .2rem .15rem 0; font-size: .85rem; }
.m-path { font-family: 'Cormorant Garamond', serif; font-size: 1.6rem; color: var(--gold2); letter-spacing: .02em; padding: .4rem 0; }
.m-path span { color: var(--muted); font-size: 1.1rem; margin: 0 .5rem; }
.m-brand { font-family: 'Cormorant Garamond', serif; font-size: 2rem; color: var(--gold2); margin-bottom: 0; }
.m-sub { color: var(--muted); font-size: .85rem; margin-top: -.3rem; margin-bottom: 1rem; }
hr { border-color: var(--line) !important; }
</style>
"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def brand():
    st.sidebar.markdown('<div class="m-eyebrow">AI composer</div><div class="m-brand">Maestro</div>'
                        '<div class="m-sub">Search · Logic · Evolution · Learning</div>', unsafe_allow_html=True)


def hero(eyebrow, title, subtitle):
    st.markdown(f'<div class="m-hero"><div class="m-eyebrow">{eyebrow}</div><h1>{title}</h1><p>{subtitle}</p></div>',
                unsafe_allow_html=True)


def card(title, body):
    st.markdown(f'<div class="m-card"><h4>{title}</h4><p>{body}</p></div>', unsafe_allow_html=True)


def chips(items):
    st.markdown("".join(f'<span class="m-chip">{i}</span>' for i in items), unsafe_allow_html=True)


def path_line(path):
    st.markdown('<div class="m-path">' + '<span>→</span>'.join(path) + '</div>', unsafe_allow_html=True)


def style_axes(fig, ax, title=None):
    fig.patch.set_alpha(0)
    ax.set_facecolor(PALETTE["panel"])
    for s in ax.spines.values():
        s.set_color(PALETTE["line"])
    ax.tick_params(colors=PALETTE["muted"], labelsize=8)
    ax.xaxis.label.set_color(PALETTE["muted"])
    ax.yaxis.label.set_color(PALETTE["muted"])
    ax.grid(color=PALETTE["line"], alpha=.5, linewidth=.6)
    if title:
        ax.set_title(title, color=PALETTE["gold2"], fontsize=12, loc="left", pad=10)
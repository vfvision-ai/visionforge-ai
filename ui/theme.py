"""
Global theme, CSS injection, and re-usable styled components.
Import apply_theme() once in app.py; import helpers in each page.
"""

import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# MASTER CSS
# ─────────────────────────────────────────────────────────────────────────────
_CSS = """
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Root Variables ── */
:root {
  --bg-primary:    #0d0d1a;
  --bg-secondary:  #12121f;
  --bg-card:       rgba(255,255,255,0.04);
  --bg-card-hover: rgba(255,255,255,0.07);
  --border:        rgba(255,255,255,0.08);
  --border-accent: rgba(108,99,255,0.4);

  --accent:        #6c63ff;
  --accent2:       #00d4aa;
  --accent3:       #ff6b6b;
  --accent4:       #ffd93d;

  --text-primary:  #f0f0f8;
  --text-secondary:#a0a0b8;
  --text-muted:    #606080;

  --gradient-hero: linear-gradient(135deg, #6c63ff 0%, #3ec6e0 50%, #00d4aa 100%);
  --gradient-card: linear-gradient(135deg, rgba(108,99,255,0.15) 0%, rgba(0,212,170,0.08) 100%);
  --gradient-warm: linear-gradient(135deg, #ff6b6b 0%, #ffd93d 100%);
  --shadow-card:   0 4px 24px rgba(0,0,0,0.5);
  --shadow-glow:   0 0 30px rgba(108,99,255,0.25);

  --radius-sm:  8px;
  --radius-md:  12px;
  --radius-lg:  18px;
  --radius-xl:  24px;
}

/* ── Global App Background ── */
.stApp, .main {
  background: var(--bg-primary) !important;
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
  color: var(--text-primary) !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0f0f1e 0%, #131325 100%) !important;
  border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text-primary) !important; }
[data-testid="stSidebarNav"] { display: none !important; }

/* ── Sidebar navigation buttons — flat link style ── */
[data-testid="stSidebar"] .stButton > button {
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
  text-align: left !important;
  justify-content: flex-start !important;
  padding: 0.18rem 0.6rem !important;
  font-size: 0.76rem !important;
  font-weight: 500 !important;
  color: #7070a0 !important;
  border-radius: 6px !important;
  min-height: 0 !important;
  height: auto !important;
  line-height: 1.4 !important;
  letter-spacing: 0.04em !important;
  text-transform: uppercase !important;
  transition: color 0.15s, background 0.15s !important;
  width: 100% !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
  background: rgba(108,99,255,0.1) !important;
  color: #a89cff !important;
}
[data-testid="stSidebar"] .stButton > button:focus:not(:active) {
  box-shadow: none !important;
  border: none !important;
}

/* ── Block‑container padding ── */
.block-container {
  padding: 1.5rem 2rem 3rem !important;
  max-width: 100% !important;
}

/* ── All text elements ── */
h1, h2, h3, h4, h5, p, li, label, div {
  font-family: 'Inter', sans-serif !important;
}
h1 { font-size: 2rem !important; font-weight: 800 !important; }
h2 { font-size: 1.45rem !important; font-weight: 700 !important; }
h3 { font-size: 1.15rem !important; font-weight: 600 !important; }

/* ── Streamlit metric ── */
[data-testid="stMetricValue"] {
  font-size: 1.8rem !important;
  font-weight: 800 !important;
  color: var(--text-primary) !important;
}
[data-testid="stMetricLabel"] {
  font-size: 0.78rem !important;
  font-weight: 500 !important;
  color: var(--text-secondary) !important;
  text-transform: uppercase !important;
  letter-spacing: 0.05em !important;
}
[data-testid="stMetricDelta"] { font-size:0.82rem !important; }

/* ── Cards (via st.markdown) ── */
.cv-card {
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  padding: 1.4rem 1.6rem;
  margin-bottom: 1rem;
  transition: all 0.25s ease;
  box-shadow: var(--shadow-card);
}
.cv-card:hover {
  background: var(--bg-card-hover);
  border-color: var(--border-accent);
  transform: translateY(-2px);
  box-shadow: var(--shadow-glow);
}
.cv-card-gradient {
  background: var(--gradient-card);
  border: 1px solid var(--border-accent);
}

/* Task selection cards */
.task-card {
  background: rgba(255,255,255,0.03);
  border: 1px solid rgba(255,255,255,0.07);
  border-radius: var(--radius-xl);
  padding: 2rem 1.5rem;
  text-align: center;
  cursor: pointer;
  transition: all 0.3s cubic-bezier(0.4,0,0.2,1);
  height: 100%;
}
.task-card:hover {
  background: rgba(108,99,255,0.12);
  border-color: rgba(108,99,255,0.5);
  transform: translateY(-4px);
  box-shadow: 0 12px 40px rgba(108,99,255,0.3);
}
.task-card .icon {
  font-size: 3rem;
  margin-bottom: 0.8rem;
  display: block;
}
.task-card h3 {
  font-size: 1.3rem !important;
  font-weight: 700 !important;
  margin-bottom: 0.5rem;
  background: var(--gradient-hero);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}
.task-card p { color: var(--text-secondary) !important; font-size:0.88rem !important; }
.task-pill {
  display: inline-block;
  background: rgba(108,99,255,0.15);
  border: 1px solid rgba(108,99,255,0.3);
  border-radius: 999px;
  padding: 0.18rem 0.7rem;
  font-size: 0.75rem;
  color: #a89cff !important;
  margin: 0.2rem;
}

/* ── Hero banner ── */
.hero-banner {
  background: var(--gradient-hero);
  border-radius: var(--radius-xl);
  padding: 2.5rem 2.5rem 2rem;
  margin-bottom: 2rem;
  position: relative;
  overflow: hidden;
}
.hero-banner::before {
  content: '';
  position: absolute; inset: 0;
  background: radial-gradient(ellipse at 70% 50%, rgba(255,255,255,0.1) 0%, transparent 60%);
}
.hero-banner h1 {
  font-size: 2.4rem !important;
  font-weight: 900 !important;
  color: #fff !important;
  margin-bottom: 0.4rem;
}
.hero-banner p { color: rgba(255,255,255,0.85) !important; font-size:1.05rem !important; }
.hero-badge {
  display: inline-flex; align-items: center; gap: 0.35rem;
  background: rgba(255,255,255,0.2);
  border: 1px solid rgba(255,255,255,0.3);
  border-radius: 999px;
  padding: 0.3rem 0.9rem;
  font-size: 0.82rem;
  color: #fff !important;
  font-weight: 600;
  margin-right: 0.5rem;
}

/* ── Section headers ── */
.section-header {
  display: flex; align-items: center; gap: 0.7rem;
  margin-bottom: 1.2rem;
  padding-bottom: 0.6rem;
  border-bottom: 1px solid var(--border);
}
.section-header .icon { font-size: 1.3rem; }
.section-header h2 {
  font-size: 1.2rem !important;
  font-weight: 700 !important;
  color: var(--text-primary) !important;
  margin: 0 !important;
}

/* ── Stat row ── */
.stat-row {
  display: flex; gap: 1rem; flex-wrap: wrap; margin-bottom: 1rem;
}
.stat-box {
  flex: 1; min-width: 120px;
  background: var(--bg-card);
  border: 1px solid var(--border);
  border-radius: var(--radius-md);
  padding: 1rem;
  text-align: center;
}
.stat-box .val {
  font-size: 1.6rem; font-weight: 800; 
  background: var(--gradient-hero);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  background-clip: text;
}
.stat-box .lbl { font-size:0.75rem; color: var(--text-secondary) !important;
  text-transform: uppercase; letter-spacing:0.06em; margin-top:0.2rem; }

/* ── Tag / badge ── */
.badge {
  display: inline-block;
  border-radius: 999px;
  padding: 0.2rem 0.75rem;
  font-size: 0.74rem;
  font-weight: 600;
  letter-spacing:0.03em;
}
.badge-purple { background: rgba(108,99,255,0.18); color: #a89cff !important; border:1px solid rgba(108,99,255,0.3); }
.badge-teal   { background: rgba(0,212,170,0.15);  color: #00d4aa !important; border:1px solid rgba(0,212,170,0.3); }
.badge-red    { background: rgba(255,107,107,0.15); color: #ff6b6b !important; border:1px solid rgba(255,107,107,0.3); }
.badge-yellow { background: rgba(255,217,61,0.15);  color: #ffd93d !important; border:1px solid rgba(255,217,61,0.3); }

/* ── Step indicator ── */
.step-track { display: flex; align-items:center; gap:0; margin-bottom:2rem; }
.step-item  { display:flex; flex-direction:column; align-items:center; flex:1; position:relative; }
.step-circle {
  width:36px; height:36px; border-radius:50%; display:flex; align-items:center; justify-content:center;
  font-size:0.85rem; font-weight:700; border:2px solid;
  z-index:1; position:relative; background: var(--bg-primary);
}
.step-circle.done   { background:#1a3a2a; border-color: var(--accent2); color: var(--accent2) !important; }
.step-circle.active { background:rgba(108,99,255,0.2); border-color:var(--accent); color:var(--accent) !important;
  box-shadow: 0 0 16px rgba(108,99,255,0.5); }
.step-circle.todo   { background: var(--bg-card); border-color: var(--border); color: var(--text-muted) !important; }
.step-label { font-size:0.7rem; margin-top:0.35rem; text-align:center; color:var(--text-secondary) !important; font-weight:500; }
.step-line  { flex:1; height:2px; background: var(--border); margin:0 -4px; position:relative; top:-8px; }
.step-line.done { background: var(--accent2); }

/* ── Buttons ── */
.stButton > button {
  background: linear-gradient(135deg, #6c63ff, #3ec6e0) !important;
  color: #fff !important;
  border: none !important;
  border-radius: var(--radius-md) !important;
  font-weight: 600 !important;
  font-family: 'Inter', sans-serif !important;
  padding: 0.55rem 1.4rem !important;
  transition: all 0.2s ease !important;
  box-shadow: 0 4px 15px rgba(108,99,255,0.35) !important;
  font-size: 0.9rem !important;
}
.stButton > button:hover {
  transform: translateY(-1px) !important;
  box-shadow: 0 6px 22px rgba(108,99,255,0.55) !important;
}
.stButton > button[kind="secondary"] {
  background: var(--bg-card) !important;
  border: 1px solid var(--border) !important;
  box-shadow: none !important;
  color: var(--text-primary) !important;
}

/* ── Input widgets ── */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea,
.stSelectbox > div > div,
.stNumberInput > div > div > input {
  background-color: rgba(255,255,255,0.05) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-sm) !important;
  color: var(--text-primary) !important;
  font-family: 'Inter', sans-serif !important;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 2px rgba(108,99,255,0.2) !important;
}

/* ── Slider ── */
.stSlider > div > div > div { background: rgba(255,255,255,0.1) !important; }
.stSlider > div > div > div > div { background: var(--gradient-hero) !important; }

/* ── Checkbox ── */
.stCheckbox > label > div:first-child {
  border-color: var(--border) !important;
  background: var(--bg-card) !important;
}

/* ── Radio ── */
.stRadio > div { gap: 0.5rem !important; }
.stRadio label {
  background: var(--bg-card) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-sm) !important;
  padding: 0.4rem 0.9rem !important;
  transition: all 0.2s !important;
  cursor: pointer;
}
.stRadio label:hover { border-color: var(--accent) !important; }

/* ── Progress bar ── */
.stProgress > div > div > div {
  background: var(--gradient-hero) !important;
  border-radius: 999px !important;
}
.stProgress > div > div {
  background: rgba(255,255,255,0.08) !important;
  border-radius: 999px !important;
  height: 8px !important;
}

/* ── Alerts ── */
.stSuccess > div { background: rgba(0,212,170,0.12) !important; border: 1px solid rgba(0,212,170,0.3) !important; color: var(--text-primary) !important; border-radius: var(--radius-md) !important; }
.stInfo    > div { background: rgba(108,99,255,0.1) !important;  border: 1px solid rgba(108,99,255,0.25) !important; color: var(--text-primary) !important; border-radius: var(--radius-md) !important; }
.stWarning > div { background: rgba(255,217,61,0.1) !important;  border: 1px solid rgba(255,217,61,0.3) !important;  color: var(--text-primary) !important; border-radius: var(--radius-md) !important; }
.stError   > div { background: rgba(255,107,107,0.1) !important; border: 1px solid rgba(255,107,107,0.3) !important; color: var(--text-primary) !important; border-radius: var(--radius-md) !important; }

/* ── Dataframe / Table ── */
[data-testid="stDataFrameContainer"] {
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-md) !important;
  overflow: hidden;
}
.stDataFrame thead th {
  background: rgba(108,99,255,0.15) !important;
  color: var(--text-primary) !important;
  font-weight: 600 !important;
}

/* ── Expander ── */
.streamlit-expanderHeader {
  background: var(--bg-card) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-md) !important;
  color: var(--text-primary) !important;
}
.streamlit-expanderContent {
  background: rgba(255,255,255,0.02) !important;
  border: 1px solid var(--border) !important;
  border-top: none !important;
  border-radius: 0 0 var(--radius-md) var(--radius-md) !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
  background: var(--bg-card) !important;
  border-radius: var(--radius-md) !important;
  gap: 0.25rem !important;
  padding: 0.3rem !important;
  border: 1px solid var(--border) !important;
}
.stTabs [data-baseweb="tab"] {
  border-radius: var(--radius-sm) !important;
  color: var(--text-secondary) !important;
  font-weight: 600 !important;
  padding: 0.45rem 1.2rem !important;
  transition: all 0.2s !important;
}
.stTabs [aria-selected="true"] {
  background: linear-gradient(135deg,#6c63ff,#3ec6e0) !important;
  color: #fff !important;
  box-shadow: 0 2px 12px rgba(108,99,255,0.4) !important;
}

/* ── Divider ── */
hr {
  border: none !important;
  border-top: 1px solid var(--border) !important;
  margin: 1.5rem 0 !important;
}

/* ── Code blocks ── */
code, .stCodeBlock { background: rgba(255,255,255,0.06) !important; border-radius: var(--radius-sm) !important; }

/* ── File uploader ── */
[data-testid="stFileUploadDropzone"] {
  background: rgba(108,99,255,0.06) !important;
  border: 2px dashed rgba(108,99,255,0.35) !important;
  border-radius: var(--radius-lg) !important;
}

/* ── Sidebar nav item ── */
.sidebar-nav-item {
  display: flex; align-items: center; gap: 0.6rem;
  padding: 0.6rem 0.9rem;
  border-radius: var(--radius-sm);
  margin-bottom: 0.25rem;
  font-size: 0.9rem; font-weight: 500;
  color: var(--text-secondary) !important;
  transition: all 0.2s;
  cursor: pointer;
}
.sidebar-nav-item.active {
  background: rgba(108,99,255,0.18) !important;
  color: var(--accent) !important;
  border-left: 3px solid var(--accent);
}
.sidebar-nav-item.done {
  color: var(--accent2) !important;
}
.sidebar-nav-item.locked { opacity: 0.4; cursor: not-allowed; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.12); border-radius: 999px; }
::-webkit-scrollbar-thumb:hover { background: rgba(255,255,255,0.2); }

/* ── Page title (top bar) ── */
header[data-testid="stHeader"] {
  background: rgba(13,13,26,0.95) !important;
  border-bottom: 1px solid var(--border) !important;
  backdrop-filter: blur(10px) !important;
}

/* ── Animated gradient text ── */
.gradient-text {
  background: var(--gradient-hero);
  -webkit-background-clip: text; -webkit-text-fill-color: transparent;
  background-clip: text;
  font-weight: 800;
}

/* ── Spinner ── */
.stSpinner > div { border-top-color: var(--accent) !important; }

/* ── Toast / notification ── */
[data-testid="stNotificationContentSuccess"] { background: rgba(0,212,170,0.1) !important; }
[data-testid="stNotificationContentError"]   { background: rgba(255,107,107,0.1) !important; }

/* ── Hide streamlit branding ── */
#MainMenu, footer, [data-testid="stDecoration"] { display: none !important; }
</style>
"""


def apply_theme() -> None:
    """Inject the master CSS into the Streamlit app. Call once from app.py."""
    st.markdown(_CSS, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# COMPONENT HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def hero(title: str, subtitle: str = "", badges: list = None) -> None:
    """Render a full-width gradient hero banner."""
    badge_html = ""
    for b in (badges or []):
        badge_html += f'<span class="hero-badge">{b}</span>'
    st.markdown(f"""
    <div class="hero-banner">
      <h1>{title}</h1>
      {f'<p>{subtitle}</p>' if subtitle else ''}
      {f'<div style="margin-top:1rem">{badge_html}</div>' if badge_html else ''}
    </div>
    """, unsafe_allow_html=True)


def section(icon: str, title: str) -> None:
    """Styled section header."""
    st.markdown(f"""
    <div class="section-header">
      <span class="icon">{icon}</span>
      <h2>{title}</h2>
    </div>
    """, unsafe_allow_html=True)


def card(content_html: str, gradient: bool = False) -> None:
    """Render an HTML card block."""
    cls = "cv-card cv-card-gradient" if gradient else "cv-card"
    st.markdown(f'<div class="{cls}">{content_html}</div>', unsafe_allow_html=True)


def stat_row(stats: list) -> None:
    """
    Render a horizontal row of stat boxes.
    stats: list of (value, label) or (value, label, badge_class)
    """
    boxes = ""
    for s in stats:
        val, lbl = s[0], s[1]
        boxes += f'<div class="stat-box"><div class="val">{val}</div><div class="lbl">{lbl}</div></div>'
    st.markdown(f'<div class="stat-row">{boxes}</div>', unsafe_allow_html=True)


def badge(text: str, color: str = "purple") -> str:
    """Return inline HTML badge."""
    return f'<span class="badge badge-{color}">{text}</span>'


def step_tracker(steps: list, current: int) -> None:
    """
    Render a horizontal step tracker.
    steps: list of step name strings
    current: 0-based index of the active step
    """
    items = ""
    for i, name in enumerate(steps):
        if i < current:
            cls, sym = "done", "✓"
        elif i == current:
            cls, sym = "active", str(i + 1)
        else:
            cls, sym = "todo", str(i + 1)
        line_cls = "step-line done" if i < current else "step-line"
        line_html = f'<div class="{line_cls}"></div>' if i < len(steps) - 1 else ""
        items += f"""
        <div class="step-item">
          <div class="step-circle {cls}">{sym}</div>
          <span class="step-label">{name}</span>
        </div>
        {line_html}"""
    st.markdown(f'<div class="step-track">{items}</div>', unsafe_allow_html=True)


def gradient_text(text: str, tag: str = "h2") -> None:
    st.markdown(f'<{tag} class="gradient-text">{text}</{tag}>', unsafe_allow_html=True)


def info_card(icon: str, title: str, body: str, color: str = "purple") -> None:
    accent_map = {"purple": "#6c63ff", "teal": "#00d4aa", "red": "#ff6b6b", "yellow": "#ffd93d"}
    c = accent_map.get(color, "#6c63ff")
    st.markdown(f"""
    <div class="cv-card" style="border-left:3px solid {c};">
      <div style="display:flex;align-items:flex-start;gap:0.8rem;">
        <span style="font-size:1.6rem;line-height:1">{icon}</span>
        <div>
          <div style="font-weight:700;font-size:1rem;color:var(--text-primary);margin-bottom:0.3rem">{title}</div>
          <div style="font-size:0.88rem;color:var(--text-secondary);line-height:1.5">{body}</div>
        </div>
      </div>
    </div>""", unsafe_allow_html=True)


def task_card_html(icon: str, title: str, desc: str, pills: list) -> str:
    pill_html = "".join(f'<span class="task-pill">{p}</span>' for p in pills)
    return f"""
    <div class="task-card">
      <span class="icon">{icon}</span>
      <h3>{title}</h3>
      <p>{desc}</p>
      <div style="margin-top:1rem">{pill_html}</div>
    </div>"""


def metric_card(label: str, value: str, delta: str = "", icon: str = "",
                color: str = "purple") -> None:
    accent_map = {"purple": "var(--accent)", "teal": "var(--accent2)",
                  "red": "var(--accent3)", "yellow": "var(--accent4)"}
    c = accent_map.get(color, "var(--accent)")
    delta_html = (f'<div style="font-size:0.8rem;color:{c};margin-top:0.2rem">{delta}</div>'
                  if delta else "")
    st.markdown(f"""
    <div class="cv-card" style="text-align:center;padding:1.2rem;">
      {f'<div style="font-size:1.8rem;margin-bottom:0.4rem">{icon}</div>' if icon else ''}
      <div style="font-size:0.72rem;text-transform:uppercase;letter-spacing:.06em;
           color:var(--text-secondary);font-weight:600;margin-bottom:.3rem">{label}</div>
      <div style="font-size:1.9rem;font-weight:900;color:{c}">{value}</div>
      {delta_html}
    </div>""", unsafe_allow_html=True)


def progress_card(label: str, pct: float, color: str = "#6c63ff") -> None:
    w = max(0, min(100, pct))
    st.markdown(f"""
    <div class="cv-card" style="padding:1rem 1.4rem;">
      <div style="display:flex;justify-content:space-between;margin-bottom:.5rem">
        <span style="font-size:.875rem;font-weight:600;color:var(--text-primary)">{label}</span>
        <span style="font-size:.875rem;font-weight:700;color:{color}">{pct:.1f}%</span>
      </div>
      <div style="height:8px;background:rgba(255,255,255,.08);border-radius:999px;overflow:hidden">
        <div style="height:100%;width:{w}%;background:{color};border-radius:999px;
             transition:width .4s ease"></div>
      </div>
    </div>""", unsafe_allow_html=True)

import datetime
import hmac
import html
import random

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from pipeline import (build, WEIGHTS, CLUSTER_FEATURES, compute_scores,
                      add_risk_flags, MASTER_SQL, REQUIRED, FILES, FIELDS,
                      next_student_id, save_student, reset_database,
                      get_interventions, save_intervention, update_intervention_status,
                      delete_intervention, get_career_recommendations,
                      calculate_goal_plan, simulate_campus_policy,
                      get_scenario_data, generate_advisor_response)

# ---------- Page setup (must be the first Streamlit command) ----------
st.set_page_config(page_title="Smart Campus Analytics | AI Student Success", page_icon="🎓", layout="wide")

# ---------- Demo accounts (synthetic data only) ----------
FACULTY_USER = "faculty"
FACULTY_PASS = "faculty@123"

RISK_COLORS = {"High": "#EF4444", "Medium": "#F59E0B", "Low": "#10B981"}
px.defaults.color_discrete_sequence = ["#4F46E5", "#0EA5E9", "#10B981",
                                       "#F59E0B", "#EF4444", "#8B5CF6"]

# =====================================================
# THEME (light / dark) with a toggle button
# =====================================================
THEMES = {
    "light": dict(bg="#F5F7FF", card="#FFFFFF", soft="#EEF2FF", text="#1E293B",
                  muted="#64748B", border="#E0E7FF", accent="#4F46E5", field="#FFFFFF"),
    "dark":  dict(bg="#0B1020", card="#141B33", soft="#1B2442", text="#E5E7EB",
                  muted="#9CA3AF", border="#2B3560", accent="#818CF8", field="#1B2442"),
}

if "dark" not in st.session_state:
    st.session_state["dark"] = False
DARK = st.session_state["dark"]
T = THEMES["dark" if DARK else "light"]

STATIC_CSS = """
@keyframes fadeUp{from{opacity:0;transform:translateY(14px)}to{opacity:1;transform:none}}
@keyframes shift{0%{background-position:0% 50%}50%{background-position:100% 50%}100%{background-position:0% 50%}}
@keyframes float{0%,100%{transform:translateY(0)}50%{transform:translateY(-6px)}}
@keyframes grow{from{transform:scaleX(0)}to{transform:scaleX(1)}}
@keyframes pop{0%{transform:scale(.92);opacity:0}100%{transform:scale(1);opacity:1}}

.stApp{background:var(--bg);}
header[data-testid="stHeader"]{background:transparent;}
section[data-testid="stSidebar"]{background:var(--card);border-right:1px solid var(--border);}
.stApp h1,.stApp h2,.stApp h3,.stApp h4{color:var(--text);}
[data-testid="stMarkdownContainer"] p,[data-testid="stMarkdownContainer"] li,
[data-testid="stWidgetLabel"] p{color:var(--text);}
[data-testid="stCaptionContainer"]{color:var(--muted);}
[data-testid="stMetricLabel"] *{color:var(--muted);}
[data-testid="stMetricValue"]{color:var(--text);}

/* hero banner */
.hero{background:linear-gradient(120deg,#4F46E5,#7C3AED,#0EA5E9);background-size:200% 200%;
animation:shift 10s ease infinite,fadeUp .6s ease both;padding:22px 28px;border-radius:18px;
margin-bottom:16px;box-shadow:0 10px 30px rgba(79,70,229,.22);}
.hero-t{color:#fff;font-size:1.95rem;font-weight:800;display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:10px;}
.hero-s{color:#fff;opacity:.94;margin-top:4px;font-size:0.95rem;}
.float{display:inline-block;animation:float 3s ease-in-out infinite;}
.hero-pill{background:rgba(255,255,255,0.22);backdrop-filter:blur(6px);color:#fff;padding:4px 14px;border-radius:999px;font-size:0.85rem;font-weight:600;}

/* section titles */
.sec{font-size:1.2rem;font-weight:700;color:var(--text);margin:20px 0 10px;
padding-bottom:5px;border-bottom:2px solid var(--accent);display:inline-block;}

/* info cards */
.mrow{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:14px;
align-items:stretch;margin:6px 0 14px;}
.mcard{background:var(--card);border:1px solid var(--border);border-left:5px solid var(--accent);
border-radius:12px;padding:14px 16px;box-shadow:0 2px 8px rgba(79,70,229,.08);
animation:fadeUp .7s ease both;transition:transform .2s,box-shadow .2s;}
.mcard:hover{transform:translateY(-4px);box-shadow:0 8px 20px rgba(79,70,229,.18);}
.ml{color:var(--muted);font-size:.88rem;}
.mv{color:var(--text);font-size:1.65rem;font-weight:700;line-height:1.25;margin-top:2px;overflow-wrap:anywhere;}
.mn{color:var(--muted);font-size:.84rem;line-height:1.4;margin-top:8px;}
.md{display:inline-block;margin-top:8px;padding:2px 10px;border-radius:999px;font-size:.82rem;font-weight:600;}
.md.up{background:rgba(16,185,129,.15);color:#10B981;}
.md.down{background:rgba(239,68,68,.15);color:#EF4444;}

/* stat cards */
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(145px,1fr));gap:12px;margin:6px 0 10px;}
.stat{background:var(--card);border:1px solid var(--border);border-radius:14px;padding:12px 14px;
text-align:center;animation:pop .6s ease both;transition:transform .2s,box-shadow .2s;}
.stat:hover{transform:translateY(-3px);box-shadow:0 10px 22px rgba(79,70,229,.18);}
.stat-ic{font-size:1.4rem;}
.stat-v{font-size:1.6rem;font-weight:800;color:var(--accent);}
.stat-l{color:var(--muted);font-size:.82rem;}

/* tip / feature cards */
.tips{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:12px;}
.tip{display:flex;gap:12px;background:var(--card);border:1px solid var(--border);
border-left:4px solid var(--accent);border-radius:12px;padding:12px 14px;
animation:fadeUp .5s ease both;transition:transform .2s,box-shadow .2s;}
.tip:hover{transform:translateY(-3px);box-shadow:0 8px 20px rgba(79,70,229,.18);}
.tip-ic{font-size:1.6rem;}
.tip-t{font-weight:700;color:var(--text);}
.tip-x{color:var(--muted);font-size:.88rem;margin-top:2px;}

/* score formula bars */
.bar-row{display:flex;align-items:center;gap:10px;margin:8px 0;}
.bar-n{width:95px;color:var(--text);font-size:.88rem;font-weight:500;}
.track{flex:1;background:var(--soft);border-radius:8px;height:12px;overflow:hidden;}
.fill{height:100%;background:linear-gradient(90deg,#4F46E5,#0EA5E9);border-radius:8px;
transform-origin:left;animation:grow 1s ease both;}
.bar-v{width:42px;text-align:right;color:var(--muted);font-weight:700;font-size:0.88rem;}

/* career cards */
.career-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(270px,1fr));gap:14px;margin:10px 0;}
.career-card{background:var(--card);border:1px solid var(--border);border-top:4px solid var(--accent);
border-radius:12px;padding:16px;box-shadow:0 2px 8px rgba(79,70,229,.06);transition:transform .2s,box-shadow .2s;}
.career-card:hover{transform:translateY(-3px);box-shadow:0 8px 18px rgba(79,70,229,.16);}
.career-header{display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;}
.career-title{font-weight:700;font-size:1.02rem;color:var(--text);}
.career-badge{background:rgba(79,70,229,.12);color:var(--accent);padding:3px 10px;border-radius:999px;font-weight:700;font-size:0.85rem;}

/* badges */
.badge{display:inline-block;padding:3px 10px;border-radius:999px;font-size:0.8rem;font-weight:600;}
.badge-red{background:rgba(239,68,68,.15);color:#EF4444;}
.badge-amber{background:rgba(245,158,11,.15);color:#F59E0B;}
.badge-green{background:rgba(16,185,129,.15);color:#10B981;}
.badge-blue{background:rgba(14,165,233,.15);color:#0EA5E9;}
.badge-purple{background:rgba(139,92,246,.15);color:#8B5CF6;}

/* step roadmap card */
.step-card{background:var(--card);border:1px solid var(--border);border-left:4px solid #10B981;border-radius:10px;padding:12px 14px;margin:8px 0;}

/* tip of the day + estimator result */
.daytip{background:var(--soft);border:1px dashed var(--accent);border-radius:14px;
padding:14px 16px;margin:10px 0;animation:pop .5s ease both;color:var(--text);}
.score{text-align:center;background:var(--card);border:1px solid var(--border);
border-radius:14px;padding:12px;margin:8px 0;animation:pop .4s ease both;}
.score-v{font-size:2.5rem;font-weight:800;}
.score-l{color:var(--muted);font-size:0.9rem;}

/* Streamlit widgets */
div[data-testid="stMetric"]{background:var(--card);border:1px solid var(--border);
border-left:5px solid var(--accent);border-radius:12px;padding:12px 14px;
box-shadow:0 2px 8px rgba(79,70,229,.08);animation:fadeUp .7s ease both;
transition:transform .2s,box-shadow .2s;}
div[data-testid="stMetric"]:hover{transform:translateY(-3px);box-shadow:0 8px 18px rgba(79,70,229,.18);}
div[data-testid="stForm"]{background:var(--card);border:1px solid var(--border);border-radius:16px;}
div[data-testid="stExpander"]{background:var(--card);border:1px solid var(--border);border-radius:12px;}
div[data-testid="stExpander"] summary p,div[data-testid="stExpander"] summary span{color:var(--text);}
div.stButton>button,div[data-testid="stFormSubmitButton"]>button,div.stDownloadButton>button{
background:var(--card);color:var(--text);border:1px solid var(--border);border-radius:10px;
transition:transform .15s,box-shadow .15s,border-color .15s;}
div.stButton>button:hover,div[data-testid="stFormSubmitButton"]>button:hover,
div.stDownloadButton>button:hover{transform:translateY(-2px);border-color:var(--accent);
color:var(--accent);box-shadow:0 6px 14px rgba(79,70,229,.2);}
button[data-baseweb="tab"] p{color:var(--muted);}
button[data-baseweb="tab"][aria-selected="true"] p{color:var(--accent);font-weight:700;}
div[data-baseweb="input"]>div,div[data-baseweb="base-input"],div[data-baseweb="select"]>div{
background:var(--field)!important;border-color:var(--border)!important;}
input,textarea{color:var(--text)!important;-webkit-text-fill-color:var(--text)!important;}
div[data-baseweb="select"] *{color:var(--text);}
div[data-baseweb="popover"] ul,div[data-baseweb="popover"] li{background:var(--card);color:var(--text);}
[data-testid="stTable"] table{background:var(--card);}
[data-testid="stTable"] th,[data-testid="stTable"] td{color:var(--text);border-color:var(--border);}
"""

DARK_EXTRA = """
div[data-testid="stDataFrame"]{filter:invert(.92) hue-rotate(180deg);}
div[data-testid="stAlert"] p{color:var(--text);}
"""


def inject_css():
    root = ":root{" + "".join(f"--{k}:{v};" for k, v in T.items()) + "}"
    css = root + STATIC_CSS + (DARK_EXTRA if DARK else "")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def flip_theme():
    st.session_state["dark"] = not st.session_state["dark"]


def show(fig):
    """Draw a Plotly chart that matches the current theme."""
    fig.update_layout(
        template="plotly_dark" if DARK else "plotly_white",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=T["text"]))
    st.plotly_chart(fig, use_container_width=True, theme=None)


# =====================================================
# HTML & Visualization Helpers
# =====================================================
def hero(title, subtitle, icon="🎓", pill=None):
    pill_html = f'<div class="hero-pill">{html.escape(pill)}</div>' if pill else ""
    st.markdown(
        f'<div class="hero"><div class="hero-t"><div><span class="float">{icon}</span> '
        f'{title}</div>{pill_html}</div><div class="hero-s">{subtitle}</div></div>',
        unsafe_allow_html=True)


def section(text):
    st.markdown(f'<div class="sec">{text}</div>', unsafe_allow_html=True)


def info_cards(cards):
    out = []
    for c in cards:
        note = f'<div class="mn">{html.escape(c["note"])}</div>' if c.get("note") else ""
        delta = (f'<div class="md {c.get("tone", "up")}">{html.escape(c["delta"])}</div>'
                 if c.get("delta") else "")
        out.append(f'<div class="mcard"><div class="ml">{html.escape(c["label"])}</div>'
                   f'<div class="mv">{html.escape(str(c["value"]))}</div>{delta}{note}</div>')
    return f'<div class="mrow">{"".join(out)}</div>'


def stat_cards(items):
    cards = "".join(
        f'<div class="stat"><div class="stat-ic">{i}</div>'
        f'<div class="stat-v">{v}</div><div class="stat-l">{l}</div></div>'
        for i, v, l in items)
    return f'<div class="stats">{cards}</div>'


def tip_grid(items):
    cards = "".join(
        f'<div class="tip"><div class="tip-ic">{e}</div>'
        f'<div><div class="tip-t">{t}</div><div class="tip-x">{x}</div></div></div>'
        for e, t, x in items)
    return f'<div class="tips">{cards}</div>'


def formula_bars():
    top = max(WEIGHTS.values())
    rows = "".join(
        f'<div class="bar-row"><span class="bar-n">{k.title()}</span>'
        f'<div class="track"><div class="fill" style="width:{v / top * 100:.0f}%;'
        f'animation-delay:{i * 0.1:.1f}s"></div></div>'
        f'<span class="bar-v">{v * 100:.0f}%</span></div>'
        for i, (k, v) in enumerate(sorted(WEIGHTS.items(), key=lambda kv: -kv[1])))
    return rows


def make_radar_chart(student_row, class_avg, top10_avg=None):
    """Generates 7-dimension holistic radar chart comparing student against benchmarks."""
    categories = ["Academic", "Attendance", "LMS", "Engagement", "Placement", "Skills", "Feedback"]
    keys = ["academic", "attendance", "lms", "engagement", "placement", "skills", "feedback"]

    student_vals = [float(student_row[k]) for k in keys]
    avg_vals = [float(class_avg[k]) for k in keys]

    categories_closed = categories + [categories[0]]
    student_closed = student_vals + [student_vals[0]]
    avg_closed = avg_vals + [avg_vals[0]]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=avg_closed,
        theta=categories_closed,
        name="Cohort Average",
        line=dict(color="#94A3B8", dash="dash", width=2),
        hoverinfo="r+theta"
    ))

    if top10_avg is not None:
        top_vals = [float(top10_avg[k]) for k in keys]
        top_closed = top_vals + [top_vals[0]]
        fig.add_trace(go.Scatterpolar(
            r=top_closed,
            theta=categories_closed,
            name="Top 10% Honors Benchmark",
            line=dict(color="#10B981", dash="dot", width=2),
            hoverinfo="r+theta"
        ))

    fig.add_trace(go.Scatterpolar(
        r=student_closed,
        theta=categories_closed,
        fill="toself",
        name=student_row["name"],
        line=dict(color="#6366F1", width=3),
        fillcolor="rgba(99, 102, 241, 0.28)",
        hoverinfo="r+theta"
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], color="#64748B"),
            bgcolor="rgba(0,0,0,0)"
        ),
        showlegend=True,
        margin=dict(l=40, r=40, t=30, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5)
    )
    return fig


def make_quadrant_matrix(data):
    """Draws 2x2 Academic vs Placement Quadrant Matrix with strategic annotations."""
    fig = px.scatter(
        data,
        x="academic",
        y="placement",
        color="risk_level",
        size="success_score",
        hover_name="name",
        hover_data=["student_id", "department", "cgpa", "coding", "success_score", "ml_placement_prob"],
        color_discrete_map=RISK_COLORS,
        labels={"academic": "Academic Score (CGPA & Backlogs)", "placement": "Placement Score (Coding, Aptitude, Mock)"},
        title="Campus Strategic Quadrant: Academic Strength vs Placement Readiness"
    )
    fig.add_hline(y=50, line_dash="dash", line_color="#94A3B8", annotation_text="Placement Cutoff (50)")
    fig.add_vline(x=70, line_dash="dash", line_color="#94A3B8", annotation_text="Academic Proficiency (70)")

    fig.add_annotation(x=85, y=85, text="🌟 Honors & Placement Star", showarrow=False, font=dict(color="#10B981", size=11))
    fig.add_annotation(x=85, y=30, text="⚠️ Academic Strong, Needs Coding Clinic", showarrow=False, font=dict(color="#F59E0B", size=11))
    fig.add_annotation(x=45, y=85, text="💡 Placement Ready, Academic Remedial", showarrow=False, font=dict(color="#3B82F6", size=11))
    fig.add_annotation(x=45, y=30, text="🚨 Urgent Double Intervention", showarrow=False, font=dict(color="#EF4444", size=11))

    fig.update_layout(margin=dict(l=20, r=20, t=50, b=20))
    return fig


def make_feature_importance_plot(ml_metadata):
    """Visualizes Scikit-Learn Random Forest feature importances for explainable AI."""
    place_imp = ml_metadata.get("placement_model", {}).get("feature_importances", {})
    risk_imp = ml_metadata.get("risk_model", {}).get("feature_importances", {})

    p_df = pd.DataFrame(list(place_imp.items())[:6], columns=["Feature", "Importance"])
    p_df["Model"] = "Placement Predictor (90.7% CV Acc)"

    r_df = pd.DataFrame(list(risk_imp.items())[:6], columns=["Feature", "Importance"])
    r_df["Model"] = "Academic Distress Detector (99.7% CV Acc)"

    combined = pd.concat([p_df, r_df])
    fig = px.bar(
        combined,
        x="Importance",
        y="Feature",
        color="Model",
        barmode="group",
        orientation="h",
        title="Explainable AI: Key Signals Driving Machine Learning Predictions",
        color_discrete_map={
            "Placement Predictor (90.7% CV Acc)": "#4F46E5",
            "Academic Distress Detector (99.7% CV Acc)": "#EF4444"
        }
    )
    fig.update_layout(yaxis=dict(autorange="reversed"), margin=dict(l=20, r=20, t=50, b=20))
    return fig


def make_student_diagnostic_card(s, career_paths):
    """Generates an executive HTML Diagnostic Card for download/printing."""
    top_career = career_paths[0] if career_paths else {}
    html_card = f"""
    <div style="font-family:system-ui,-apple-system,sans-serif; border:2px solid #4F46E5; border-radius:12px; padding:24px; max-width:680px; background:#ffffff; color:#1E293B;">
        <div style="display:flex; justify-content:space-between; border-bottom:2px solid #EEF2FF; padding-bottom:12px; margin-bottom:16px;">
            <div>
                <h2 style="margin:0; color:#4F46E5;">Smart Campus Diagnostic Card</h2>
                <div style="font-size:14px; color:#64748B;">Student: <b>{s['name']}</b> ({s['student_id']}) · Department of {s['department']}</div>
            </div>
            <div style="text-align:right;">
                <div style="font-size:26px; font-weight:800; color:{RISK_COLORS.get(s['risk_level'], '#4F46E5')}">{float(s['success_score']):.1f}</div>
                <div style="font-size:12px; color:#64748B;">Success Score ({s['risk_level']} Risk)</div>
            </div>
        </div>
        <div style="display:grid; grid-template-columns:1fr 1fr; gap:12px; margin-bottom:16px; font-size:13px;">
            <div style="background:#F8FAFC; padding:12px; border-radius:8px;">
                <b>Academic Profile:</b><br/>
                • CGPA: <b>{s['cgpa']}</b> | Backlogs: <b>{s['backlogs']}</b><br/>
                • Attendance: <b>{s['attendance_pct']}%</b> | LMS: <b>{s['assignment_completion_pct']}%</b>
            </div>
            <div style="background:#F8FAFC; padding:12px; border-radius:8px;">
                <b>Placement Profile:</b><br/>
                • Coding: <b>{s['coding']}/100</b> | Aptitude: <b>{s['aptitude']}/100</b><br/>
                • Mock Interview: <b>{s['mock_interview']}/100</b> | ML Ready: <b>{s.get('ml_placement_prob', 0)}%</b>
            </div>
        </div>
        <div style="background:#EEF2FF; border-left:4px solid #4F46E5; padding:12px 14px; border-radius:6px; margin-bottom:14px; font-size:13px;">
            <b>Assigned Segment:</b> {s['segment']}<br/>
            <b>Recommended Action:</b> {s['suggested_action']}<br/>
            <b>Top Career Match:</b> {top_career.get('icon', '')} {top_career.get('title', '')} ({top_career.get('match_score', 0)}% match)
        </div>
        <div style="font-size:11px; color:#94A3B8; text-align:center; border-top:1px solid #E2E8F0; padding-top:8px;">
            Generated by Smart Campus Analytics · Student Success Platform · Confidential
        </div>
    </div>
    """
    return html_card


# =====================================================
# Real Names Generator & Demo Accounts
# =====================================================
FIRST_NAMES = [
    "Aarav", "Vivaan", "Aditya", "Arjun", "Sai", "Karthik", "Rohan", "Rahul",
    "Vikram", "Nikhil", "Harsha", "Teja", "Pranav", "Manoj", "Suresh", "Ramesh",
    "Anil", "Kiran", "Varun", "Charan", "Naveen", "Praveen", "Ajay", "Vijay",
    "Mahesh", "Ganesh", "Sandeep", "Dinesh", "Abhishek", "Yashwanth", "Bharath",
    "Tarun", "Uday", "Srinivas", "Venkat", "Ananya", "Priya", "Sneha", "Divya",
    "Lakshmi", "Meghana", "Pooja", "Kavya", "Swathi", "Bhavana", "Harini",
    "Nandini", "Sravani", "Anusha", "Keerthi", "Mounika", "Shreya", "Ishita",
    "Neha", "Riya", "Sowmya", "Deepika", "Vaishnavi", "Tanvi", "Jyothi",
]
LAST_NAMES = [
    "Reddy", "Sharma", "Kumar", "Rao", "Naidu", "Patel", "Singh", "Gupta",
    "Verma", "Iyer", "Nair", "Das", "Chowdary", "Goud", "Yadav", "Mehta",
    "Joshi", "Kulkarni", "Menon", "Pillai", "Shetty", "Bhat", "Chauhan", "Jain",
    "Agarwal", "Prasad", "Varma", "Raju", "Murthy", "Sastry", "Babu", "Mishra",
    "Pandey", "Thakur", "Banerjee", "Mukherjee", "Desai", "Kapoor", "Bose", "Hegde",
]


def give_real_names(data):
    names = data["name"].astype(str)
    todo = (names.str.match(r"^Student \d+$")
            | ((data["student_id"] == "S0001") & (names == "Ravi")))
    if todo.mean() <= 0.5:
        return data
    ids = sorted(data["student_id"])
    rng = np.random.RandomState(7)
    combos = [f"{f} {l}" for f in FIRST_NAMES for l in LAST_NAMES]
    picked = rng.choice(len(combos), len(ids), replace=False)
    mapping = dict(zip(ids, [combos[i] for i in picked]))
    if "S0001" in mapping:
        mapping["S0001"] = "Ravi Kumar"
    data = data.copy()
    data.loc[todo, "name"] = data.loc[todo, "student_id"].map(mapping)
    return data


@st.cache_data
def get_data(sources=None):
    data, report = build(with_report=True, sources=sources)
    return give_real_names(data), report


# Load base data
custom_sources = st.session_state.get("custom_sources")
df_base, clean_report = get_data(custom_sources)

# Scenario selector state
if "scenario" not in st.session_state:
    st.session_state["scenario"] = "Standard Cohort"

# Apply scenario perturbation if not standard
if st.session_state["scenario"] != "Standard Cohort":
    df = get_scenario_data(st.session_state["scenario"], df_base)
else:
    df = df_base

name_map = dict(zip(df["student_id"], df["name"]))
RAW_COLS = ["cgpa", "backlogs", "attendance_pct", "logins_per_week",
            "assignment_completion_pct", "events", "clubs", "hackathons",
            "certifications", "aptitude", "coding", "mock_interview",
            "technical", "softskill", "satisfaction"]


def estimate_score(cgpa, attendance, coding, assignments):
    row = df[RAW_COLS].median().to_frame().T
    row["cgpa"] = cgpa
    row["attendance_pct"] = attendance
    row["coding"] = coding
    row["assignment_completion_pct"] = assignments
    r = add_risk_flags(compute_scores(row)).iloc[0]
    return float(r["success_score"]), str(r["risk_level"]), bool(r["placement_risk"])


# =====================================================
# LOGIN & DEMO ACCOUNTS
# =====================================================
def same(a, b):
    return hmac.compare_digest(a.encode(), b.encode())


def student_password(sid):
    first = name_map[sid].split()[0].lower()
    return f"{first}@{sid[-4:]}"


def authenticate(username, password):
    u = username.strip().lower()
    p = password.strip().lower()
    if u == FACULTY_USER and same(p, FACULTY_PASS):
        return {"role": "faculty", "id": "faculty", "name": "Faculty Advisor"}
    sid = u.upper()
    if sid in name_map and same(p, student_password(sid)):
        return {"role": "student", "id": sid, "name": name_map[sid]}
    return None


def sample_accounts():
    rows = [df[df["student_id"] == "S0001"].iloc[0]]
    for cond in (df["risk_level"] == "High", df["segment"] == "All-Rounders"):
        pick = df[cond]
        if len(pick):
            rows.append(pick.iloc[0])
    return pd.DataFrame({
        "Student ID": [r["student_id"] for r in rows],
        "Name": [r["name"] for r in rows],
        "Password": [student_password(r["student_id"]) for r in rows],
    }).set_index("Student ID")


# =====================================================
# TOP BAR (Scenario Switcher + Theme Toggle)
# =====================================================
def top_bar():
    c_scen, c_sp, c_role, c_theme = st.columns([3, 2, 2, 1])

    with c_scen:
        scenarios = [
            "Standard Cohort",
            "🚨 Pre-Exam Attendance Crisis",
            "💼 Placement Season Crunch",
            "🌟 Post-Intervention Success"
        ]
        curr = st.session_state.get("scenario", "Standard Cohort")
        idx = scenarios.index(curr) if curr in scenarios else 0
        chosen_scen = st.selectbox("⚡ Hackathon Live Demo Scenario:", scenarios, index=idx, key="scen_selector")
        if chosen_scen != curr:
            st.session_state["scenario"] = chosen_scen
            st.rerun()

    with c_role:
        if "user" in st.session_state:
            u = st.session_state["user"]
            st.caption(f"Signed in as **{u['name']}** ({u['role'].title()})")
        else:
            st.caption("Not signed in")

    with c_theme:
        st.button("☀️ Light" if DARK else "🌙 Dark", on_click=flip_theme, key="theme_btn", use_container_width=True)


# =====================================================
# AI COPILOT INTERACTIVE WIDGET
# =====================================================
def render_copilot(role, student_row=None, cohort_df=None):
    st.subheader("🤖 Smart Campus AI Advisor & Copilot")
    st.caption("Context-aware conversational intelligence for predictive student success and targeted faculty action.")

    chips = [
        "💼 How can I improve my placement readiness?",
        "🔍 What is pulling my Success Score down?",
        "📅 Suggest a 4-week study plan",
        "🏆 Which certifications should I take?"
    ] if role == "student" else [
        "📊 Summarize cohort risk factors",
        "⚠️ Show students with CGPA > 7.5 but weak coding",
        "🏛️ Which department has lowest LMS engagement?",
        "✉️ Draft mentor message for Ravi Kumar"
    ]

    st.markdown("**Quick Inquiries (1-Click Prompts):**")
    cols = st.columns(4)
    selected_chip = None
    for i, chip in enumerate(chips):
        if cols[i].button(chip, key=f"chip_{role}_{i}", use_container_width=True):
            selected_chip = chip

    chat_key = f"chat_{role}_{student_row['student_id'] if student_row is not None else 'faculty'}"
    if chat_key not in st.session_state:
        st.session_state[chat_key] = [
            {"role": "assistant", "content": f"Hello! I am your Smart Campus AI Advisor. {'Ask me anything about your academic trajectory, coding progress, or career fit.' if role == 'student' else 'Ask me about cohort risk factors, departmental breakdowns, or intervention drafting.'}"}
        ]

    if selected_chip:
        st.session_state[chat_key].append({"role": "user", "content": selected_chip})
        resp = generate_advisor_response(selected_chip, role, student_row, cohort_df)
        st.session_state[chat_key].append({"role": "assistant", "content": resp})

    # Display message history
    for msg in st.session_state[chat_key]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # User input
    user_query = st.chat_input("Ask the AI Advisor a question..." if role == "student" else "Query cohort intelligence or draft interventions...")
    if user_query:
        st.session_state[chat_key].append({"role": "user", "content": user_query})
        resp = generate_advisor_response(user_query, role, student_row, cohort_df)
        st.session_state[chat_key].append({"role": "assistant", "content": resp})
        st.rerun()


# =====================================================
# LOGIN SCREEN
# =====================================================
FEATURES = [
    ("🔗", "7-Source Unified Integration", "Academic, attendance, LMS, engagement, placement, skills and feedback in one SQLite pipeline."),
    ("🤖", "Dual Machine Learning Engines", "Random Forest models predicting placement readiness (90.7% CV Acc) and academic distress risk (99.7% CV Acc)."),
    ("🎯", "Holistic Radar & Career Pathways", "7-Dimension spider benchmarking and automatic role matching with skill gap identification."),
    ("📋", "Closed-Loop Intervention Hub", "Actionable case management with one-click faculty intervention assignments and tracking."),
]

STUDENT_TIPS = [
    ("📅", "Attend regularly", "Students below 65% attendance score noticeably lower. Aim for 75% or more."),
    ("💻", "Code a little every day", "30 minutes of daily practice beats one long weekend session."),
    ("📝", "Finish LMS assignments on time", "Log in at least 5 times a week and submit before the deadline."),
    ("🏆", "Join one club or hackathon", "Events and certifications build skills that marks cannot show."),
    ("🎤", "Do mock interviews early", "Start before placement season, not during it."),
    ("🙋", "Ask for help early", "Talk to your mentor the moment a subject starts slipping."),
]
FACULTY_TIPS = [
    ("❓", "Open with a 3-question quiz", "Quick recall questions at the start help students remember more."),
    ("📋", "Review the At-Risk list every Monday", "Talk to the 5 lowest-scoring students each week."),
    ("🤝", "Pair strong with struggling students", "Peer tutoring helps both sides learn."),
    ("🎬", "Try a flipped classroom", "Post short videos on the LMS and use class time for problems."),
    ("⚡", "Give feedback within 48 hours", "Fast feedback keeps students on track."),
    ("📣", "Collect 2-minute monthly feedback", "A short anonymous form shows what to fix early."),
]
COLLEGE_TIPS = [
    ("🚀", "Run coding bootcamps", "Target students with strong academics but weak placement readiness."),
    ("🔔", "Do an early-warning check in week 4", "Catch problems before the first internal exams."),
    ("📊", "Track LMS use by department", "Support the department with the lowest engagement."),
    ("👨‍👩‍👧", "Involve parents for disengaged students", "A call home early prevents drop-offs."),
    ("🏢", "Invite industry for talks and drives", "Real exposure improves aptitude and interview scores."),
    ("🧭", "Assign every student a mentor", "One person who knows their story makes a big difference."),
]
ALL_TIPS = STUDENT_TIPS + FACULTY_TIPS + COLLEGE_TIPS


def login_screen():
    hero("Smart Campus Analytics: Student Success Platform",
         "Transforming campus signals into early predictive action, career pathways, and closed-loop faculty intervention.",
         icon="🎓", pill=f"Active Scenario: {st.session_state.get('scenario', 'Standard Cohort')}")

    flagged = int(((df["risk_level"] == "High") | df["academic_risk"] | df["placement_risk"]).sum())
    st.markdown(stat_cards([
        ("🎓", len(df), "Students tracked"),
        ("🏛️", df["department"].nunique(), "Departments"),
        ("🤖", "90.7%", "ML Placement Accuracy"),
        ("⚠️", flagged, "Students flagged for support"),
    ]), unsafe_allow_html=True)

    left, right = st.columns([1, 1.2], gap="large")

    with left:
        section("🔐 Sign in")

        st.markdown("**⚡ 1-Click Fast Judge Demo Sign-In:**")
        qc1, qc2, qc3 = st.columns(3)
        if qc1.button("👨‍🏫 Faculty Advisor", use_container_width=True):
            st.session_state["user"] = {"role": "faculty", "id": "faculty", "name": "Faculty Advisor"}
            st.rerun()
        if qc2.button("🎓 Ravi (S0001)", use_container_width=True):
            st.session_state["user"] = {"role": "student", "id": "S0001", "name": name_map.get("S0001", "Ravi Kumar")}
            st.rerun()
        if qc3.button("🎓 Aditya (S0015)", use_container_width=True):
            st.session_state["user"] = {"role": "student", "id": "S0015", "name": name_map.get("S0015", "Aditya Verma")}
            st.rerun()

        st.markdown("---")
        with st.form("login"):
            username = st.text_input("Username (faculty, or your student ID)", placeholder="e.g. S0001")
            password = st.text_input("Password", type="password", placeholder="first name@last 4 digits of ID")
            submitted = st.form_submit_button("Sign in with Credentials", use_container_width=True)
        if submitted:
            user = authenticate(username, password)
            if user:
                st.session_state["user"] = user
                st.rerun()
            else:
                st.error("Wrong username or password.")

        with st.expander("🔑 Demo Credentials Reference", expanded=False):
            st.markdown(
                "**Students:** username is your student ID. Password is your "
                "**first name in lowercase + @ + the last 4 digits of your ID**. "
                "Example: `S0001` / `ravi@0001`.")
            st.table(sample_accounts())
            st.markdown("**Faculty:** `faculty` / `faculty@123`")

    with right:
        section("✨ Platform Highlights")
        st.markdown(tip_grid(FEATURES), unsafe_allow_html=True)

    c1, c2 = st.columns(2, gap="large")
    with c1:
        section("🎯 Try it: What-if Habit Simulator")
        cgpa = st.slider("CGPA", 4.0, 10.0, 7.0, 0.1)
        att = st.slider("Attendance %", 40, 100, 75)
        code = st.slider("Coding practice score", 0, 100, 50)
        assign = st.slider("LMS assignments completed %", 0, 100, 70)
        score, level, place_risk = estimate_score(cgpa, att, code, assign)
        note = "Placement readiness needs attention" if place_risk else "Placement readiness looks fine"
        st.markdown(
            f'<div class="score"><div class="score-v" style="color:{RISK_COLORS[level]}">'
            f'{score:.1f}</div><div class="score-l">Estimated Success Score · '
            f'{level} risk · {note}</div></div>', unsafe_allow_html=True)
        st.progress(min(int(score), 100))

    with c2:
        section("🧮 Success Score Weight Architecture")
        st.markdown(formula_bars(), unsafe_allow_html=True)

        section("💡 Smart Tip of the Day")
        if "tip" not in st.session_state:
            st.session_state["tip"] = random.choice(ALL_TIPS)
        if st.button("🎲 Show another tip"):
            st.session_state["tip"] = random.choice(ALL_TIPS)
        e, t, x = st.session_state["tip"]
        st.markdown(f'<div class="daytip">{e} <b>{t}</b><br>{x}</div>', unsafe_allow_html=True)


# =====================================================
# STUDENT VIEW (Personalized Student Portal)
# =====================================================
STATUS_INFO = {
    "Low": "Your score is above 65. You are doing well. Keep it up.",
    "Medium": "Your score is between 55 and 65. You are on track, with room to grow.",
    "High": "Your score is below 55. Extra support, such as mentoring, is available to help you.",
}
STATUS = {"Low": "🟢 Doing well",
          "Medium": "🟡 On track, with room to grow",
          "High": "🟠 Extra support is available"}


def student_view(sid):
    s = df[df["student_id"] == sid].iloc[0]
    score = float(s["success_score"])
    delta = score - float(df["success_score"].mean())
    careers = get_career_recommendations(s)

    hero(f"Welcome, {s['name']}", f"{sid} · Department of {s['department']} · Student Success Portal",
         icon="👋", pill=f"Placement Probability: {s.get('ml_placement_prob', 0)}%")

    st.markdown(info_cards([
        dict(label="Your Success Score", value=f"{score:.1f}",
             delta=f"{'▲' if delta >= 0 else '▼'} {delta:+.1f} vs class average",
             tone="up" if delta >= 0 else "down"),
        dict(label="Status", value=STATUS[str(s["risk_level"])],
             note=STATUS_INFO[str(s["risk_level"])]),
        dict(label="ML Placement Probability", value=f"{s.get('ml_placement_prob', 0):.1f}%",
             note="Trained Random Forest prediction"),
        dict(label="Assigned Segment", value=s["segment"],
             note=f"Action: {s['suggested_action']}"),
    ]), unsafe_allow_html=True)
    st.progress(min(int(score), 100))

    t_diag, t_careers, t_goals, t_mentorship, t_copilot = st.tabs([
        "📊 Holistic Diagnostic",
        "🎯 Career Pathways & Skill Fit",
        "🚀 Target Goal Simulator",
        "📋 Assigned Mentorship & Action",
        "🤖 AI Academic Copilot"
    ])

    # ---------- Tab 1: Holistic Diagnostic ----------
    with t_diag:
        c_rad, c_imp = st.columns([1.1, 1])

        with c_rad:
            section("🕸️ 7-Dimension Holistic Spider Radar")
            top10_sub = df[df["success_score"] >= df["success_score"].quantile(0.9)]
            radar_fig = make_radar_chart(s, df.mean(numeric_only=True), top10_sub.mean(numeric_only=True))
            show(radar_fig)
            st.caption("Layered radar comparing your 7 core signals against the cohort average and top 10% honor students.")

        with c_imp:
            section("⚖️ Points Impact (vs Class Average)")
            impacts = {c: float(s[f"impact_{c}"]) for c in WEIGHTS}
            chart = pd.DataFrame({"component": list(WEIGHTS.keys()), "impact": [impacts[c] for c in WEIGHTS]})
            chart["effect"] = chart["impact"].apply(lambda v: "Lifts your score" if v >= 0 else "Room to improve")
            fig = px.bar(chart, x="component", y="impact", color="effect", text="impact",
                         color_discrete_map={"Lifts your score": "#10B981", "Room to improve": "#F59E0B"},
                         title="Score Pull vs Push Points")
            show(fig)

            strengths = [c for c in sorted(impacts, key=impacts.get, reverse=True)[:2] if impacts[c] > 0]
            focus = [c for c in sorted(impacts, key=impacts.get)[:2] if impacts[c] < 0]
            st.success("**Your Top Strengths:** " + (", ".join(strengths) if strengths else "Consistent all-round progress"))
            st.warning("**Immediate Focus Areas:** " + (", ".join(focus) if focus else "No urgent deficits"))

        # Printable diagnostic card
        section("📄 Printable Student Performance Diagnostic Card")
        card_html = make_student_diagnostic_card(s, careers)
        st.markdown(card_html, unsafe_allow_html=True)
        st.download_button("⬇️ Download Diagnostic Card (HTML)", data=card_html,
                           file_name=f"{sid}_diagnostic_card.html", mime="text/html")

    # ---------- Tab 2: Career Pathways & Skill Fit ----------
    with t_careers:
        section("🎯 Personalized Industry Career Match")
        st.caption("Algorithm evaluates your 15 academic and skill indicators to find best-fit technology tracks.")

        top_match = careers[0]
        st.info(f"🏆 **Top Match**: You are an **{top_match['match_score']}% Match** for **{top_match['icon']} {top_match['title']}**!")

        st.markdown('<div class="career-grid">', unsafe_allow_html=True)
        for c in careers:
            badge_color = "badge-green" if c["match_score"] >= 75 else ("badge-amber" if c["match_score"] >= 55 else "badge-blue")
            st.markdown(f"""
            <div class="career-card">
                <div class="career-header">
                    <span class="career-title">{c['icon']} {c['title']}</span>
                    <span class="career-badge">{c['match_score']}% Match</span>
                </div>
                <p style="font-size:0.86rem; color:var(--muted); margin-bottom:8px;">{c['summary']}</p>
                <div style="font-size:0.83rem; margin-bottom:6px;"><b>✅ Strengths:</b> {', '.join(c['strengths'][:2])}</div>
                <div style="font-size:0.83rem; color:#EF4444; margin-bottom:8px;"><b>⚠️ Target Gaps:</b> {', '.join(c['gaps'][:2])}</div>
                <div style="font-size:0.82rem; background:var(--soft); padding:8px 10px; border-radius:8px;">
                    <b>🎓 Recommended Certifications:</b><br/>
                    • {c['certifications'][0]}<br/>
                    • {c['certifications'][1] if len(c['certifications']) > 1 else ''}
                </div>
            </div>
            """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # ---------- Tab 3: Target Goal Simulator ----------
    with t_goals:
        section("🚀 Target Success Score & Growth Roadmap")
        st.caption("Select your target score to receive an algorithmically optimized, achievable step-by-step milestone plan.")

        target = st.slider("Select your Target Success Score:",
                           min_value=float(round(score)),
                           max_value=98.0,
                           value=float(min(round(score + 10.0), 95.0)),
                           step=1.0)
        goal_plan = calculate_goal_plan(s, target)

        if goal_plan["achieved"]:
            st.success(goal_plan["message"])
        else:
            st.markdown(f"**Target Gap**: **+{goal_plan['gap']} points** required to reach **{target:.1f}**.")
            st.markdown(f"**Projected Score with plan**: **{goal_plan['projected']}** (Gain: +{goal_plan['total_gain']} pts)")

            for step in goal_plan["steps"]:
                st.markdown(f"""
                <div class="step-card">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <b>{step['icon']} {step['action']}</b>
                        <span class="badge badge-green">+{step['gain']} pts</span>
                    </div>
                    <div style="font-size:0.85rem; color:var(--muted); margin-top:4px;">{step['details']}</div>
                </div>
                """, unsafe_allow_html=True)

    # ---------- Tab 4: Assigned Mentorship & Action ----------
    with t_mentorship:
        section("📋 Faculty Mentorship & Active Interventions")
        interventions = get_interventions(sid)

        if interventions.empty:
            st.info("No active faculty interventions currently assigned. Keep up the good work!")
        else:
            st.write(f"You have **{len(interventions)} intervention records** on file.")
            for _, r in interventions.iterrows():
                badge_cls = "badge-green" if r["status"] == "Completed" else ("badge-amber" if r["status"] == "In Progress" else "badge-red")
                st.markdown(f"""
                <div class="mcard" style="margin-bottom:12px;">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <div>
                            <span class="badge {badge_cls}">{r['status']}</span>
                            <span style="font-weight:700; margin-left:8px; font-size:1.05rem;">{r['action_type']}</span>
                        </div>
                        <span style="font-size:0.85rem; color:var(--muted);">Due: {r['target_date']}</span>
                    </div>
                    <div style="font-size:0.88rem; margin-top:8px;"><b>Faculty Mentor:</b> {r['faculty_name']} · <b>Priority:</b> {r['priority']}</div>
                    <div style="font-size:0.88rem; color:var(--muted); margin-top:4px;"><b>Action Notes:</b> {r['notes']}</div>
                </div>
                """, unsafe_allow_html=True)

        # Smart Milestones & Badges
        section("🏆 Smart Achievements & Badges")
        b1, b2, b3, b4 = st.columns(4)
        if s["attendance_pct"] >= 80:
            b1.success("📅 **Attendance Star**\n\n80%+ consistent presence")
        else:
            b1.caption("📅 Attendance Star: Requires 80%+")

        if s["coding"] >= 50:
            b2.success("💻 **Code Sprinter**\n\n50+ problem solving score")
        else:
            b2.caption("💻 Code Sprinter: Requires 50+ coding")

        if s["assignment_completion_pct"] >= 80:
            b3.success("📝 **LMS Champion**\n\n80%+ coursework on time")
        else:
            b3.caption("📝 LMS Champion: Requires 80%+ LMS")

        if s["hackathons"] >= 1 or s["certifications"] >= 1:
            b4.success("🏆 **Innovation Builder**\n\nHackathon / Certification earner")
        else:
            b4.caption("🏆 Innovation Builder: 1+ hackathon/cert")

    # ---------- Tab 5: AI Academic Copilot ----------
    with t_copilot:
        render_copilot("student", student_row=s)


# =====================================================
# UPLOAD DATA PANEL (Sidebar)
# =====================================================
def upload_panel():
    with st.sidebar.expander("📤 Use your own data"):
        if st.session_state.get("custom_sources"):
            st.success("The dashboard is showing your uploaded data.")
            if st.button("Back to demo data"):
                del st.session_state["custom_sources"]
                st.rerun()
            return
        st.caption("Upload all 7 CSV files named academic.csv, attendance.csv, "
                   "lms.csv, engagement.csv, placement.csv, skills.csv and "
                   "feedback.csv, with the same columns as the demo data.")
        files = st.file_uploader("CSV files", type="csv", accept_multiple_files=True)
        if not files:
            return

        sources, problems = {}, []
        for f in files:
            stem = f.name.lower().rsplit(".", 1)[0]
            if stem in FILES:
                try:
                    sources[stem] = pd.read_csv(f)
                except Exception:
                    problems.append(f"{f.name} could not be read as a CSV file.")
        for name in FILES:
            if name not in sources:
                problems.append(f"Missing file: {name}.csv")
            else:
                absent = [c for c in REQUIRED[name] if c not in sources[name].columns]
                if absent:
                    problems.append(f"{name}.csv is missing columns: {', '.join(absent)}")
        for p in problems:
            st.warning(p)
        if problems or not st.button("Use this data"):
            return

        ok = False
        try:
            get_data(sources)
            ok = True
        except Exception as err:
            st.error(f"These files could not be processed: {err}")
        if ok:
            st.session_state["custom_sources"] = sources
            st.rerun()


# =====================================================
# STUDENT EDITOR (SQLite Data Management)
# =====================================================
def student_editor():
    st.subheader("Add or edit a student")
    if st.session_state.get("custom_sources"):
        st.info("You are viewing uploaded data. Click 'Back to demo data' in the "
                "sidebar to add or edit students in the database.")
        return

    saved = st.session_state.get("last_saved")
    if saved:
        st.success(saved)

    mode = st.radio("What do you want to do?",
                    ["Add a new student", "Edit an existing student"], horizontal=True)
    adding = mode == "Add a new student"
    depts = sorted(df["department"].unique())

    if adding:
        sid = next_student_id()
        base = None
        st.write(f"New student ID: **{sid}**")
    else:
        sid = st.selectbox("Student", df["student_id"].tolist(), key="editor_sid",
                           format_func=lambda x: f"{x} - {name_map[x]}")
        base = df[df["student_id"] == sid].iloc[0]

    def start(col):
        v = float(df[col].median()) if base is None else float(base[col])
        return v

    def num(label, col, lo, hi, whole=False):
        v = min(max(start(col), lo), hi)
        key = f"{sid}_{col}_{mode}"
        if whole:
            return st.number_input(label, int(lo), int(hi), int(round(v)), 1, key=key)
        return st.number_input(label, float(lo), float(hi), float(round(v, 1)), 0.1, key=key)

    with st.form(f"editor_{sid}_{mode}"):
        name = st.text_input("Full name", value="" if base is None else str(base["name"]),
                             key=f"{sid}_name_{mode}")
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**Academic and attendance**")
            dept = st.selectbox("Department", depts,
                                index=0 if base is None else depts.index(base["department"]),
                                key=f"{sid}_dept_{mode}")
            cgpa = num("CGPA", "cgpa", 0, 10)
            backlogs = num("Backlogs", "backlogs", 0, 20, True)
            attendance_pct = num("Attendance %", "attendance_pct", 0, 100)
        with c2:
            st.markdown("**LMS and engagement**")
            logins = num("Logins per week", "logins_per_week", 0, 50, True)
            assign = num("Assignments completed %", "assignment_completion_pct", 0, 100)
            events = num("Events", "events", 0, 50, True)
            clubs = num("Clubs", "clubs", 0, 10, True)
            hacks = num("Hackathons", "hackathons", 0, 30, True)
            certs = num("Certifications", "certifications", 0, 30, True)
        with c3:
            st.markdown("**Placement, skills and feedback**")
            apt = num("Aptitude score", "aptitude", 0, 100, True)
            code = num("Coding score", "coding", 0, 100, True)
            mock = num("Mock interview score", "mock_interview", 0, 100, True)
            tech = num("Technical skills", "technical", 0, 100, True)
            soft = num("Soft skills", "softskill", 0, 100, True)
            sat = num("Satisfaction (1 to 5)", "satisfaction", 1, 5, True)
        submitted = st.form_submit_button("💾 Save student to Database")

    if submitted:
        if not name.strip():
            st.error("Please enter the student's name.")
        else:
            values = dict(name=name.strip(), department=dept, cgpa=cgpa, backlogs=backlogs,
                          attendance_pct=attendance_pct, logins_per_week=logins,
                          assignment_completion_pct=assign, events=events, clubs=clubs,
                          hackathons=hacks, certifications=certs, aptitude=apt,
                          coding=code, mock_interview=mock, technical=tech,
                          softskill=soft, satisfaction=sat)
            ok = False
            try:
                save_student(sid, values)
                ok = True
            except Exception as err:
                st.error(f"Could not save: {err}")
            if ok:
                get_data.clear()
                st.session_state["flash"] = {"sid": sid, "action": "added" if adding else "updated"}
                st.rerun()

    st.markdown("---")
    if st.button("↩️ Reset database to the original CSV data"):
        reset_database()
        get_data.clear()
        st.session_state["flash"] = {"reset": True}
        st.rerun()


# =====================================================
# FACULTY VIEW (Complete Faculty & Admin Hub)
# =====================================================
def faculty_view():
    flash = st.session_state.pop("flash", None)
    if flash:
        if flash.get("reset"):
            msg = "Database reset to the original CSV data."
        else:
            r = df[df["student_id"] == flash["sid"]].iloc[0]
            msg = f"{r['name']} ({flash['sid']}) {flash['action']} successfully in SQLite database."
        st.session_state["last_saved"] = msg
        st.toast(msg, icon="✅")

    hero("Smart Campus Faculty Intelligence Hub",
         "Predictive Analytics · Dual Machine Learning Models · Closed-Loop Case Management",
         icon="👨‍🏫", pill=f"Scenario: {st.session_state.get('scenario', 'Standard Cohort')}")

    with st.expander("🧹 Multi-Source SQLite Data Integration & Cleaning Pipeline"):
        st.write(f"Merged 7 campus streams into unified view of **{len(df)} students**.")
        if clean_report:
            st.table(pd.DataFrame({"issue": [k for k in clean_report.keys() if k != "ml_metadata"],
                                   "count": [clean_report[k] for k in clean_report.keys() if k != "ml_metadata"]}
                                  ).set_index("issue"))
        st.code(MASTER_SQL, language="sql")

    # Sidebar Filters
    st.sidebar.header("Cohort Filters")
    all_depts = sorted(df["department"].unique())
    all_segs = sorted(df["segment"].unique())
    dept = st.sidebar.multiselect("Department", all_depts, default=all_depts)
    risk = st.sidebar.multiselect("Risk level", ["High", "Medium", "Low"], default=["High", "Medium", "Low"])
    segs = st.sidebar.multiselect("Segment", all_segs, default=all_segs)
    upload_panel()

    view = df[df["department"].isin(dept)
              & df["risk_level"].isin(risk)
              & df["segment"].isin(segs)]
    if view.empty:
        st.warning("No students match these filters. Please widen your selection.")
        st.stop()

    # Top KPI Metrics
    ml_hidden_count = int(view.get("ml_hidden_risk", pd.Series([False]*len(view))).sum())
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Students", len(view))
    k2.metric("Avg Success Score", f"{view['success_score'].mean():.1f}")
    k3.metric("High Risk", int((view["risk_level"] == "High").sum()))
    k4.metric("Placement Risk", int(view["placement_risk"].sum()))
    k5.metric("ML Hidden Risk", ml_hidden_count, help="Students passing simple CGPA filter but flagged by AI due to behavioral drift.")
    k6.metric("ML Place Ready", f"{(view.get('ml_placement_prob', pd.Series([0]*len(view))) >= 50).mean()*100:.0f}%")

    # Main Tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📊 Overview & ML Intelligence",
        "⚠️ At-Risk Early Warning & Action",
        "📋 Intervention Management Hub",
        "🧩 AI Segments & Clusters",
        "🔍 Student 360° Drill-down",
        "🤖 AI Campus Copilot",
        "✏️ Student Data Management"
    ])

    # ---------- Tab 1: Overview & ML Intelligence ----------
    with tab1:
        # ML Model Performance Cards
        st.subheader("🤖 Dual Machine Learning Predictive Engines")
        ml_meta = clean_report.get("ml_metadata", {})
        c_rf1, c_rf2 = st.columns(2)
        with c_rf1:
            p_acc = ml_meta.get("placement_model", {}).get("accuracy", 0.907)
            st.markdown(f"""
            <div class="mcard" style="border-left-color:#4F46E5;">
                <div class="ml">Machine Learning Classifier 1</div>
                <div class="mv" style="color:#4F46E5;">{p_acc*100:.1f}% CV Accuracy</div>
                <div style="font-weight:700; font-size:0.95rem; margin-top:4px;">Random Forest Placement Readiness Predictor</div>
                <div class="mn">Predicts multi-signal probability of cracking campus placements.<br/>
                <b>Top Indicators:</b> {', '.join(ml_meta.get('placement_model', {}).get('top_drivers', ['coding', 'aptitude', 'technical']))}</div>
            </div>
            """, unsafe_allow_html=True)
        with c_rf2:
            r_acc = ml_meta.get("risk_model", {}).get("accuracy", 0.997)
            st.markdown(f"""
            <div class="mcard" style="border-left-color:#EF4444;">
                <div class="ml">Machine Learning Classifier 2</div>
                <div class="mv" style="color:#EF4444;">{r_acc*100:.1f}% CV Accuracy</div>
                <div style="font-weight:700; font-size:0.95rem; margin-top:4px;">Random Forest Academic Failure Early Warning</div>
                <div class="mn">Detects early disengagement signals weeks before semester examinations.<br/>
                <b>Top Indicators:</b> {', '.join(ml_meta.get('risk_model', {}).get('top_drivers', ['cgpa', 'attendance_pct', 'assignment_completion_pct']))}</div>
            </div>
            """, unsafe_allow_html=True)

        # Feature Importance Plot
        if ml_meta:
            show(make_feature_importance_plot(ml_meta))

        # Strategic Quadrant
        show(make_quadrant_matrix(view))

        # Campus Policy Simulator for Deans/HODs
        section("🧪 Dean & HOD Interactive Policy Simulator (What-If)")
        st.caption("Simulate institutional interventions to test projected risk reduction and placement rate boosts before implementing.")
        sp1, sp2, sp3 = st.columns(3)
        sim_att = sp1.slider("Mandatory Attendance Push (% boost)", 0, 15, 5)
        sim_code = sp2.slider("Pre-Placement Coding Bootcamp (+ points)", 0, 25, 12)
        sim_assign = sp3.slider("LMS Assignment Submission Incentive (+ %)", 0, 20, 8)

        policy_res = simulate_campus_policy(view, att_delta=sim_att, coding_delta=sim_code, assign_delta=sim_assign)
        r_c1, r_c2, r_c3 = st.columns(3)
        r_c1.metric("Projected Avg Success Score", f"{policy_res['sim_avg']:.1f}", f"+{policy_res['avg_delta']:.1f} pts")
        r_c2.metric("High Risk Student Count", f"{policy_res['sim_high_risk']}", f"{policy_res['risk_delta']} students", delta_color="inverse")
        r_c3.metric("Placement-Ready Students", f"{policy_res['sim_place_ready']}", f"+{policy_res['place_delta']} students")

        # Distributions
        c_h1, c_h2 = st.columns(2)
        with c_h1:
            fig = px.histogram(view, x="success_score", color="risk_level", nbins=20,
                               color_discrete_map=RISK_COLORS,
                               category_orders={"risk_level": ["High", "Medium", "Low"]},
                               title="Success Score Distribution")
            show(fig)
        with c_h2:
            dept_avg = view.groupby("department")["success_score"].mean().round(1).reset_index()
            fig = px.bar(dept_avg, x="department", y="success_score", text="success_score",
                         title="Average Success Score by Department")
            show(fig)

    # ---------- Tab 2: At-Risk Early Warning & Action ----------
    with tab2:
        section("⚠️ Early Warning & Vulnerability Matrix")
        risk_filter = st.selectbox("Filter Risk Sub-Category:",
                                   ["All Flagged Students", "High Overall Risk (<55)", "Academic Risk",
                                    "Placement Risk", "ML Hidden Risk (AI Detected Drift)"])

        if risk_filter == "High Overall Risk (<55)":
            atrisk = view[view["risk_level"] == "High"]
        elif risk_filter == "Academic Risk":
            atrisk = view[view["academic_risk"]]
        elif risk_filter == "Placement Risk":
            atrisk = view[view["placement_risk"]]
        elif risk_filter == "ML Hidden Risk (AI Detected Drift)":
            atrisk = view[view.get("ml_hidden_risk", pd.Series([False]*len(view)))]
        else:
            atrisk = view[(view["risk_level"] == "High") | view["academic_risk"] | view["placement_risk"]]

        atrisk = atrisk.sort_values("success_score")
        st.subheader(f"{len(atrisk)} students flagged in this category")

        show_cols = ["student_id", "name", "department", "success_score", "risk_level",
                     "ml_placement_prob", "ml_risk_prob", "academic_risk", "placement_risk",
                     "risk_reasons", "suggested_action"]
        st.dataframe(atrisk[[c for c in show_cols if c in atrisk.columns]], hide_index=True)

        st.download_button("⬇️ Download Filtered At-Risk List (CSV)",
                           data=atrisk[[c for c in show_cols if c in atrisk.columns]].to_csv(index=False),
                           file_name="at_risk_students.csv", mime="text/csv")

        # 1-Click Fast Intervention Assigner
        section("⚡ 1-Click Assign Faculty Intervention")
        if not atrisk.empty:
            sel_student_id = st.selectbox("Select Flagged Student:", atrisk["student_id"].tolist(),
                                          format_func=lambda x: f"{x} - {name_map[x]} ({atrisk.loc[atrisk.student_id==x, 'department'].values[0]})")
            st_row = atrisk[atrisk["student_id"] == sel_student_id].iloc[0]

            with st.form("quick_intervention_form"):
                qi1, qi2, qi3 = st.columns(3)
                f_name = qi1.text_input("Faculty Mentor Name", value="Prof. Sharma")
                act_type = qi2.selectbox("Action Type", [
                    "Coding Bootcamp Enrollment",
                    "Remedial Peer Tutoring",
                    "Attendance Counseling",
                    "Placement Mock Interview Prep",
                    "Parent-Teacher Consultation",
                    "Well-being & Motivation Counseling"
                ], index=0 if st_row["placement_risk"] else 1)
                prio = qi3.selectbox("Priority", ["High", "Medium", "Low"], index=0 if st_row["risk_level"] == "High" else 1)

                notes = st.text_area("Action Plan / Mentor Notes",
                                     value=f"Assigned for {st_row['suggested_action']}. Score: {st_row['success_score']:.1f}, Reasons: {st_row['risk_reasons']}.")
                submitted_int = st.form_submit_button("⚡ Assign Intervention Now", use_container_width=True)

            if submitted_int:
                save_intervention(sel_student_id, st_row["name"], st_row["department"], f_name, act_type, prio, notes)
                st.toast(f"Intervention assigned to {st_row['name']} ({sel_student_id})!", icon="✅")
                st.rerun()

    # ---------- Tab 3: Intervention Management Hub ----------
    with tab3:
        section("📋 Closed-Loop Intervention & Case Management Hub")
        st.caption("Track, assign, and update student mentoring workflows with persistent SQLite storage.")

        all_interventions = get_interventions()
        total_int = len(all_interventions)
        pend_int = int((all_interventions["status"] == "Pending").sum()) if total_int > 0 else 0
        prog_int = int((all_interventions["status"] == "In Progress").sum()) if total_int > 0 else 0
        comp_int = int((all_interventions["status"] == "Completed").sum()) if total_int > 0 else 0

        st.markdown(stat_cards([
            ("📋", total_int, "Total Interventions"),
            ("⏳", pend_int, "Pending Action"),
            ("🔄", prog_int, "In Progress"),
            ("✅", comp_int, "Resolved / Completed"),
        ]), unsafe_allow_html=True)

        if not all_interventions.empty:
            st.dataframe(all_interventions, hide_index=True)

            # Update Intervention Status
            section("✏️ Update Intervention Status")
            u_c1, u_c2, u_c3 = st.columns([2, 2, 3])
            int_id = u_c1.selectbox("Select Record ID:", all_interventions["id"].tolist(),
                                    format_func=lambda x: f"ID #{x} - {all_interventions.loc[all_interventions.id==x, 'student_name'].values[0]} ({all_interventions.loc[all_interventions.id==x, 'action_type'].values[0]})")
            new_st = u_c2.selectbox("New Status:", ["In Progress", "Completed", "Pending"])
            st_notes = u_c3.text_input("Follow-up progress note (optional):", "")
            if st.button("Update Record"):
                update_intervention_status(int_id, new_st, st_notes if st_notes.strip() else None)
                st.toast(f"Record #{int_id} updated to {new_st}!", icon="✅")
                st.rerun()

            st.download_button("⬇️ Download Full Intervention Log (CSV)",
                               data=all_interventions.to_csv(index=False),
                               file_name="campus_interventions_log.csv", mime="text/csv")
        else:
            st.info("No active interventions logged yet.")

    # ---------- Tab 4: AI Segments & Clusters ----------
    with tab4:
        section("🧩 Multi-Signal Cohort Segmentation")
        seg_counts = view["segment"].value_counts().reset_index()
        seg_counts.columns = ["segment", "students"]
        fig = px.bar(seg_counts, x="students", y="segment", orientation="h",
                     text="students", title="Students per Rule-Based Segment")
        show(fig)

        st.subheader("Segment Profile & Action Playbook")
        profile = view.groupby("segment")[CLUSTER_FEATURES].mean().round(1)
        profile["suggested_action"] = view.groupby("segment")["suggested_action"].first()
        profile.insert(0, "students", view.groupby("segment").size())
        st.dataframe(profile)

        st.subheader("KMeans AI-Discovered Clusters")
        st.caption("Unsupervised clustering (K=4) grouping students by underlying multi-signal features.")
        cluster_profile = view.groupby("cluster")[CLUSTER_FEATURES].mean().round(1)
        cluster_profile.insert(0, "students", view.groupby("cluster").size())
        st.dataframe(cluster_profile)

    # ---------- Tab 5: Student 360° Drill-Down ----------
    with tab5:
        ids = view["student_id"].tolist()
        default_idx = ids.index("S0001") if "S0001" in ids else 0
        sid = st.selectbox("Select Student to Inspect:", ids, index=default_idx,
                           format_func=lambda x: f"{x} - {name_map[x]} ({view.loc[view.student_id==x, 'department'].values[0]})")
        s = df[df["student_id"] == sid].iloc[0]
        careers = get_career_recommendations(s)

        hero(f"{s['name']}", f"ID: {sid} · Department of {s['department']} · 360° Diagnostic Profile",
             icon="🔍", pill=f"Placement Probability: {s.get('ml_placement_prob', 0)}%")

        st.markdown(info_cards([
            dict(label="Success Score", value=f"{float(s['success_score']):.1f}"),
            dict(label="Risk Level", value=str(s["risk_level"])),
            dict(label="ML Placement Probability", value=f"{s.get('ml_placement_prob', 0):.1f}%"),
            dict(label="ML Risk Probability", value=f"{s.get('ml_risk_prob', 0):.1f}%"),
            dict(label="Segment", value=s["segment"]),
        ]), unsafe_allow_html=True)

        c_rad2, c_car2 = st.columns([1.1, 1])
        with c_rad2:
            section("🕸️ 7-Dimension Spider Radar vs Honors Benchmark")
            top10_sub = df[df["success_score"] >= df["success_score"].quantile(0.9)]
            show(make_radar_chart(s, df.mean(numeric_only=True), top10_sub.mean(numeric_only=True)))

        with c_car2:
            section("🎯 Top Recommended Career Track")
            top_c = careers[0]
            st.markdown(f"""
            <div class="mcard" style="border-left-color:#10B981;">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span style="font-size:1.15rem; font-weight:700;">{top_c['icon']} {top_c['title']}</span>
                    <span class="badge badge-green">{top_c['match_score']}% Match</span>
                </div>
                <p style="font-size:0.86rem; color:var(--muted); margin-top:6px;">{top_c['summary']}</p>
                <div style="font-size:0.85rem; margin-top:6px;"><b>Strengths:</b> {', '.join(top_c['strengths'][:2])}</div>
                <div style="font-size:0.85rem; color:#EF4444; margin-top:4px;"><b>Gaps:</b> {', '.join(top_c['gaps'][:2])}</div>
                <div style="font-size:0.84rem; background:var(--soft); padding:8px; border-radius:6px; margin-top:8px;">
                    <b>Certs:</b> {', '.join(top_c['certifications'][:2])}
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.write(f"**Reasons:** {s['risk_reasons']}")
            st.write(f"**Drivers:** {s['drivers']}")
            st.write(f"**Recommended Action:** {s['suggested_action']}")

        # Student's Interventions
        section(f"📋 Interventions Logged for {s['name']}")
        st_ints = get_interventions(sid)
        if st_ints.empty:
            st.info("No interventions currently logged for this student.")
        else:
            st.dataframe(st_ints[["id", "action_type", "faculty_name", "priority", "status", "created_at", "notes"]], hide_index=True)

        # Printable card
        card_html = make_student_diagnostic_card(s, careers)
        st.download_button(f"⬇️ Download Diagnostic Card for {s['name']} (HTML)",
                           data=card_html, file_name=f"{sid}_diagnostic_card.html", mime="text/html")

    # ---------- Tab 6: AI Campus Copilot ----------
    with tab6:
        render_copilot("faculty", cohort_df=view)

    # ---------- Tab 7: Student Data Management ----------
    with tab7:
        student_editor()


# =====================================================
# MAIN ROUTING
# =====================================================
inject_css()
top_bar()

if "user" not in st.session_state:
    login_screen()
    st.stop()

user = st.session_state["user"]
st.sidebar.markdown(f"Signed in as **{user['name']}** ({user['role'].title()})")
if st.sidebar.button("🚪 Log out"):
    del st.session_state["user"]
    st.rerun()

if user["role"] == "student":
    student_view(user["id"])
else:
    faculty_view()
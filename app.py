import hmac
import random

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from pipeline import (build, WEIGHTS, CLUSTER_FEATURES, compute_scores,
                      add_risk_flags, MASTER_SQL, REQUIRED, FILES)

# ---------- Page setup (must be the first Streamlit command) ----------
st.set_page_config(page_title="Smart Campus Analytics", page_icon="🎓", layout="wide")

# ---------- Demo accounts (synthetic data only) ----------
# Faculty:  faculty / faculty@123
# Student:  username = student ID (e.g. S0001)
#           password = first name in lowercase + @ + last 4 digits of the ID
#           example  = ravi@0001
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
animation:shift 10s ease infinite,fadeUp .6s ease both;padding:24px 30px;border-radius:18px;
margin-bottom:18px;box-shadow:0 10px 30px rgba(79,70,229,.25);}
.hero-t{color:#fff;font-size:2rem;font-weight:800;}
.hero-s{color:#fff;opacity:.92;margin-top:4px;}
.float{display:inline-block;animation:float 3s ease-in-out infinite;}

/* section titles */
.sec{font-size:1.25rem;font-weight:700;color:var(--text);margin:22px 0 10px;
padding-bottom:6px;border-bottom:2px solid var(--accent);display:inline-block;}

/* stat cards */
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:14px;margin:6px 0 10px;}
.stat{background:var(--card);border:1px solid var(--border);border-radius:14px;padding:14px 16px;
text-align:center;animation:pop .6s ease both;transition:transform .2s,box-shadow .2s;}
.stat:hover{transform:translateY(-4px);box-shadow:0 10px 24px rgba(79,70,229,.2);}
.stat-ic{font-size:1.5rem;}
.stat-v{font-size:1.7rem;font-weight:800;color:var(--accent);}
.stat-l{color:var(--muted);font-size:.85rem;}
.stats .stat:nth-child(2){animation-delay:.1s}
.stats .stat:nth-child(3){animation-delay:.2s}
.stats .stat:nth-child(4){animation-delay:.3s}

/* tip / feature cards */
.tips{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:12px;}
.tip{display:flex;gap:12px;background:var(--card);border:1px solid var(--border);
border-left:4px solid var(--accent);border-radius:12px;padding:12px 14px;
animation:fadeUp .5s ease both;transition:transform .2s,box-shadow .2s;}
.tip:hover{transform:translateY(-3px);box-shadow:0 8px 20px rgba(79,70,229,.18);}
.tip-ic{font-size:1.6rem;}
.tip-t{font-weight:700;color:var(--text);}
.tip-x{color:var(--muted);font-size:.9rem;margin-top:2px;}
.tips .tip:nth-child(2){animation-delay:.08s}
.tips .tip:nth-child(3){animation-delay:.16s}
.tips .tip:nth-child(4){animation-delay:.24s}
.tips .tip:nth-child(5){animation-delay:.32s}
.tips .tip:nth-child(6){animation-delay:.40s}

/* score formula bars */
.bar-row{display:flex;align-items:center;gap:10px;margin:9px 0;}
.bar-n{width:92px;color:var(--text);font-size:.9rem;}
.track{flex:1;background:var(--soft);border-radius:8px;height:12px;overflow:hidden;}
.fill{height:100%;background:linear-gradient(90deg,#4F46E5,#0EA5E9);border-radius:8px;
transform-origin:left;animation:grow 1s ease both;}
.bar-v{width:42px;text-align:right;color:var(--muted);font-weight:700;}

/* tip of the day + estimator result */
.daytip{background:var(--soft);border:1px dashed var(--accent);border-radius:14px;
padding:14px 16px;margin:10px 0;animation:pop .5s ease both;color:var(--text);}
.score{text-align:center;background:var(--card);border:1px solid var(--border);
border-radius:14px;padding:12px;margin:8px 0;animation:pop .4s ease both;}
.score-v{font-size:2.6rem;font-weight:800;}
.score-l{color:var(--muted);}

/* Streamlit widgets */
div[data-testid="stMetric"]{background:var(--card);border:1px solid var(--border);
border-left:5px solid var(--accent);border-radius:12px;padding:14px 16px;
box-shadow:0 2px 8px rgba(79,70,229,.08);animation:fadeUp .7s ease both;
transition:transform .2s,box-shadow .2s;}
div[data-testid="stMetric"]:hover{transform:translateY(-4px);box-shadow:0 8px 20px rgba(79,70,229,.2);}
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


def top_bar():
    _, right = st.columns([5, 1])
    right.button("☀️ Light mode" if DARK else "🌙 Dark mode",
                 on_click=flip_theme, key="theme_btn")


def show(fig):
    """Draw a Plotly chart that matches the current theme."""
    fig.update_layout(
        template="plotly_dark" if DARK else "plotly_white",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=T["text"]))
    st.plotly_chart(fig, theme=None)


# =====================================================
# HTML helpers
# =====================================================
def hero(title, subtitle, icon="🎓"):
    st.markdown(
        f'<div class="hero"><div class="hero-t"><span class="float">{icon}</span> '
        f'{title}</div><div class="hero-s">{subtitle}</div></div>',
        unsafe_allow_html=True)


def section(text):
    st.markdown(f'<div class="sec">{text}</div>', unsafe_allow_html=True)


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


# =====================================================
# Login-page content
# =====================================================
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
FEATURES = [
    ("🔗", "One unified student view", "Academic, attendance, LMS, engagement, placement, skills and feedback in one place."),
    ("🎯", "A transparent Success Score", "See exactly which indicators lift or pull down each student."),
    ("⚠️", "Early warnings", "Academic and placement risks flagged before results suffer."),
    ("🧩", "Smart segments", "Groups like 'High academic, low placement-ready' with a suggested action."),
]
ALL_TIPS = STUDENT_TIPS + FACULTY_TIPS + COLLEGE_TIPS

RAW_COLS = ["cgpa", "backlogs", "attendance_pct", "logins_per_week",
            "assignment_completion_pct", "events", "clubs", "hackathons",
            "certifications", "aptitude", "coding", "mock_interview",
            "technical", "softskill", "satisfaction"]



# =====================================================
# Student names (created here, so only this file needs updating)
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
    """Replace placeholder names like 'Student 12' with unique full names.
    If the data already has real names, it is left untouched."""
    placeholder = data["name"].astype(str).str.match(r"^Student \d+$").mean() > 0.5
    is_old_ravi = (data["student_id"] == "S0001").any()
    if not (placeholder or (is_old_ravi and data.loc[data["student_id"] == "S0001", "name"].iloc[0] == "Ravi")):
        return data
    ids = sorted(data["student_id"])
    rng = np.random.RandomState(7)
    combos = [f"{f} {l}" for f in FIRST_NAMES for l in LAST_NAMES]
    picked = rng.choice(len(combos), len(ids), replace=False)
    names = [combos[i] for i in picked]
    mapping = dict(zip(ids, names))
    if "S0001" in mapping:
        mapping["S0001"] = "Ravi Kumar"
    data = data.copy()
    data["name"] = data["student_id"].map(mapping)
    return data


@st.cache_data
def get_data(sources=None):
    data, report = build(with_report=True, sources=sources)
    return give_real_names(data), report


custom_sources = st.session_state.get("custom_sources")
df, clean_report = get_data(custom_sources)
name_map = dict(zip(df["student_id"], df["name"]))


def estimate_score(cgpa, attendance, coding, assignments):
    """What-if score: a typical student, with these four habits changed."""
    row = df[RAW_COLS].median().to_frame().T
    row["cgpa"] = cgpa
    row["attendance_pct"] = attendance
    row["coding"] = coding
    row["assignment_completion_pct"] = assignments
    r = add_risk_flags(compute_scores(row)).iloc[0]
    return float(r["success_score"]), str(r["risk_level"]), bool(r["placement_risk"])


# =====================================================
# LOGIN
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
        return {"role": "faculty", "id": "faculty", "name": "Faculty"}
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


def login_screen():
    hero("Student Success Platform",
         "Turn campus data into early action. Sign in to see your dashboard.")

    flagged = int(((df["risk_level"] == "High") | df["academic_risk"]
                   | df["placement_risk"]).sum())
    st.markdown(stat_cards([
        ("🎓", len(df), "Students tracked"),
        ("🏛️", df["department"].nunique(), "Departments"),
        ("🔗", 7, "Data sources merged"),
        ("⚠️", flagged, "Students flagged for support"),
    ]), unsafe_allow_html=True)

    left, right = st.columns([1, 1.2], gap="large")

    with left:
        section("🔐 Sign in")
        with st.form("login"):
            username = st.text_input("Username (faculty, or your student ID)",
                                     placeholder="e.g. S0001")
            password = st.text_input("Password", type="password",
                                     placeholder="first name@last 4 digits of ID")
            submitted = st.form_submit_button("Sign in")
        if submitted:
            user = authenticate(username, password)
            if user:
                st.session_state["user"] = user
                st.rerun()
            else:
                st.error("Wrong username or password.")

        with st.expander("🔑 How do I log in? (demo accounts)", expanded=True):
            st.markdown(
                "**Students:** username is your student ID. Password is your "
                "**first name in lowercase + @ + the last 4 digits of your ID**. "
                "Example: `S0001` / `ravi@0001`.")
            st.table(sample_accounts())
            st.markdown("**Faculty:** `faculty` / `faculty@123`")

    with right:
        section("✨ What you get")
        st.markdown(tip_grid(FEATURES), unsafe_allow_html=True)

    section("📚 How to improve classes: ideas that work")
    t1, t2, t3 = st.tabs(["🎓 For students", "👩‍🏫 For faculty", "🏫 For the college"])
    with t1:
        st.markdown(tip_grid(STUDENT_TIPS), unsafe_allow_html=True)
    with t2:
        st.markdown(tip_grid(FACULTY_TIPS), unsafe_allow_html=True)
    with t3:
        st.markdown(tip_grid(COLLEGE_TIPS), unsafe_allow_html=True)

    c1, c2 = st.columns(2, gap="large")
    with c1:
        section("🎯 Try it: how do habits change your score?")
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
        st.caption("A typical student is used for everything else. "
                   "Move the sliders to see the effect.")

    with c2:
        section("🧮 How the Success Score is built")
        st.markdown(formula_bars(), unsafe_allow_html=True)
        st.caption("Each bar shows how much that area counts toward the 100-point score.")

        section("💡 Tip of the day")
        if "tip" not in st.session_state:
            st.session_state["tip"] = random.choice(ALL_TIPS)
        if st.button("🎲 Show another tip"):
            st.session_state["tip"] = random.choice(ALL_TIPS)
        e, t, x = st.session_state["tip"]
        st.markdown(f'<div class="daytip">{e} <b>{t}</b><br>{x}</div>',
                    unsafe_allow_html=True)


# =====================================================
# STUDENT VIEW (only their own data)
# =====================================================
STATUS = {"Low": "🟢 Doing well",
          "Medium": "🟡 On track, with room to grow",
          "High": "🟠 Extra support is available"}


def student_view(sid):
    s = df[df["student_id"] == sid].iloc[0]
    score = float(s["success_score"])
    delta = score - float(df["success_score"].mean())

    hero(f"Welcome, {s['name']}", f"{sid} · {s['department']} department", icon="👋")

    c1, c2, c3 = st.columns(3)
    c1.metric("Your Success Score", f"{score:.1f}", f"{delta:+.1f} vs class average")
    c2.metric("Status", STATUS[str(s["risk_level"])])
    c3.metric("Group", s["segment"])
    st.progress(min(int(score), 100))

    impacts = {c: float(s[f"impact_{c}"]) for c in WEIGHTS}
    strengths = [c for c in sorted(impacts, key=impacts.get, reverse=True)[:2] if impacts[c] > 0]
    focus = [c for c in sorted(impacts, key=impacts.get)[:2] if impacts[c] < 0]

    a, b = st.columns(2)
    a.success("**Your strengths:** " + (", ".join(strengths) if strengths else "keep going!"))
    b.warning("**Focus areas:** " + (", ".join(focus) if focus else "nothing urgent"))

    st.info(f"**Suggested next step:** {s['suggested_action']}")
    if s["placement_risk"]:
        st.warning("Placement tip: your coding and interview scores have the most room "
                   "to grow. Regular practice before placement season will help.")
    if s["risk_reasons"] != "None":
        st.write(f"**Things to work on:** {s['risk_reasons']}")

    chart = pd.DataFrame({"component": list(WEIGHTS.keys()),
                          "impact": [impacts[c] for c in WEIGHTS]})
    chart["effect"] = chart["impact"].apply(
        lambda v: "Lifts your score" if v >= 0 else "Room to improve")
    fig = px.bar(chart, x="component", y="impact", color="effect", text="impact",
                 color_discrete_map={"Lifts your score": "#10B981",
                                     "Room to improve": "#F59E0B"},
                 title="What is lifting your score, and where can you grow? (points vs class average)")
    show(fig)

    st.subheader("Your scores vs the class average")
    compare = pd.DataFrame({
        "component": list(WEIGHTS.keys()),
        "you": [round(float(s[c]), 1) for c in WEIGHTS],
        "class average": [round(float(df[c].mean()), 1) for c in WEIGHTS],
    })
    st.dataframe(compare, hide_index=True)

    with st.expander("💡 Ideas to improve your classes"):
        st.markdown(tip_grid(STUDENT_TIPS), unsafe_allow_html=True)



# =====================================================
# Upload your own data (faculty sidebar)
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
            get_data(sources)          # test run, and cache the result
            ok = True
        except Exception as err:
            st.error(f"These files could not be processed: {err}")
        if ok:
            st.session_state["custom_sources"] = sources
            st.rerun()

# =====================================================
# FACULTY VIEW (all students)
# =====================================================
def faculty_view():
    hero("Student Success Platform",
         "Faculty dashboard: unified view of academic, attendance, LMS, "
         "engagement, placement, skills and feedback data")

    with st.expander("🧹 Data integration & cleaning report"):
        st.write(f"Merged 7 sources into one view of **{len(df)} students**.")
        if clean_report:
            st.table(pd.DataFrame({"issue": list(clean_report.keys()),
                                   "count": list(clean_report.values())}
                                  ).set_index("issue"))
        else:
            st.write("No data issues found.")
        st.caption("The 7 sources are loaded into a SQLite database. Duplicates are "
                   "removed in the *_clean views, then everything is joined with this SQL:")
        st.code(MASTER_SQL, language="sql")

    st.sidebar.header("Filters")
    all_depts = sorted(df["department"].unique())
    all_segs = sorted(df["segment"].unique())
    dept = st.sidebar.multiselect("Department", all_depts, default=all_depts)
    risk = st.sidebar.multiselect("Risk level", ["High", "Medium", "Low"],
                                  default=["High", "Medium", "Low"])
    segs = st.sidebar.multiselect("Segment", all_segs, default=all_segs)
    upload_panel()

    view = df[df["department"].isin(dept)
              & df["risk_level"].isin(risk)
              & df["segment"].isin(segs)]
    if view.empty:
        st.warning("No students match these filters. Please widen your selection.")
        st.stop()

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Students", len(view))
    k2.metric("Avg Success Score", f"{view['success_score'].mean():.1f}")
    k3.metric("High Risk", int((view["risk_level"] == "High").sum()))
    k4.metric("Academic Risk", int(view["academic_risk"].sum()))
    k5.metric("Placement Risk", int(view["placement_risk"].sum()))

    tab1, tab2, tab3, tab4 = st.tabs(
        ["📊 Overview", "⚠️ At-Risk Students", "🧩 Segments", "🔍 Student Drill-down"])

    # ---------- Tab 1: Overview ----------
    with tab1:
        left, right = st.columns(2)
        fig = px.histogram(view, x="success_score", color="risk_level", nbins=20,
                           color_discrete_map=RISK_COLORS,
                           category_orders={"risk_level": ["High", "Medium", "Low"]},
                           title="Success Score Distribution")
        with left:
            show(fig)

        dept_avg = view.groupby("department")["success_score"].mean().round(1).reset_index()
        fig = px.bar(dept_avg, x="department", y="success_score", text="success_score",
                     title="Average Success Score by Department")
        with right:
            show(fig)

        fig = px.scatter(view, x="academic", y="placement", color="segment",
                         hover_name="name", hover_data=["student_id", "success_score"],
                         title="Academic Strength vs Placement Readiness")
        fig.add_hline(y=50, line_dash="dash", line_color="grey")
        fig.add_vline(x=70, line_dash="dash", line_color="grey")
        show(fig)
        st.caption("Bottom-right corner = strong academics but weak placement readiness.")

        st.subheader("💡 Key insights")
        insights = []
        hidden = view[(view["risk_level"] == "Low") & view["placement_risk"]]
        insights.append(f"{len(hidden)} students look safe on the overall score "
                        f"but have placement risk.")
        strong = view[view["cgpa"] > 7.5]
        if len(strong) > 0:
            pct = strong["placement_risk"].mean() * 100
            insights.append(f"{pct:.0f}% of students with CGPA above 7.5 "
                            f"still have placement risk.")
        low_att = view[view["attendance_pct"] < 65]
        ok_att = view[view["attendance_pct"] >= 65]
        if len(low_att) > 0 and len(ok_att) > 0:
            gap = ok_att["success_score"].mean() - low_att["success_score"].mean()
            insights.append(f"Students with attendance below 65% score "
                            f"{gap:.1f} points lower on average.")
        lms_dept = view.groupby("department")["lms"].mean()
        insights.append(f"{lms_dept.idxmin()} has the lowest average LMS engagement "
                        f"({lms_dept.min():.0f}/100).")
        for text in insights:
            st.info(text)

        with st.expander("📚 Ideas to improve classes (for faculty)"):
            st.markdown(tip_grid(FACULTY_TIPS), unsafe_allow_html=True)

    # ---------- Tab 2: At-risk ----------
    with tab2:
        atrisk = view[(view["risk_level"] == "High") | view["academic_risk"]
                      | view["placement_risk"]].sort_values("success_score")
        st.subheader(f"{len(atrisk)} students need attention")
        show_cols = ["student_id", "name", "department", "success_score", "risk_level",
                     "academic_risk", "placement_risk", "risk_reasons",
                     "segment", "suggested_action"]
        st.dataframe(atrisk[show_cols], hide_index=True)
        st.download_button("⬇️ Download at-risk list (CSV)",
                           data=atrisk[show_cols].to_csv(index=False),
                           file_name="at_risk_students.csv", mime="text/csv")

    # ---------- Tab 3: Segments ----------
    with tab3:
        seg_counts = view["segment"].value_counts().reset_index()
        seg_counts.columns = ["segment", "students"]
        fig = px.bar(seg_counts, x="students", y="segment", orientation="h",
                     text="students", title="Students per Segment")
        show(fig)

        st.subheader("Segment profile and recommended action")
        profile = view.groupby("segment")[CLUSTER_FEATURES].mean().round(1)
        profile["suggested_action"] = view.groupby("segment")["suggested_action"].first()
        profile.insert(0, "students", view.groupby("segment").size())
        st.dataframe(profile)

        st.subheader("KMeans clusters (AI-discovered groups)")
        st.caption("Cluster numbers are just labels. Read the averages to see what each group represents.")
        cluster_profile = view.groupby("cluster")[CLUSTER_FEATURES].mean().round(1)
        cluster_profile.insert(0, "students", view.groupby("cluster").size())
        st.dataframe(cluster_profile)

    # ---------- Tab 4: Drill-down ----------
    with tab4:
        ids = view["student_id"].tolist()
        default_idx = ids.index("S0001") if "S0001" in ids else 0
        sid = st.selectbox("Select a student", ids, index=default_idx,
                           format_func=lambda x: f"{x} - {name_map[x]}")
        s = df[df["student_id"] == sid].iloc[0]

        st.subheader(f"{s['name']} ({sid}, {s['department']})")
        c1, c2, c3 = st.columns(3)
        c1.metric("Success Score", f"{float(s['success_score']):.1f}")
        c2.metric("Risk Level", str(s["risk_level"]))
        c3.metric("Segment", s["segment"])

        if s["academic_risk"]:
            st.error("Academic risk flag: YES")
        else:
            st.success("Academic risk flag: No")
        if s["placement_risk"]:
            st.error("Placement risk flag: YES")
        else:
            st.success("Placement risk flag: No")

        st.write(f"**Reasons:** {s['risk_reasons']}")
        st.write(f"**Drivers:** {s['drivers']}")
        st.write(f"**Suggested action:** {s['suggested_action']}")

        impact = pd.DataFrame({
            "component": list(WEIGHTS.keys()),
            "impact": [float(s[f"impact_{c}"]) for c in WEIGHTS]})
        impact["effect"] = impact["impact"].apply(
            lambda v: "Pushes score up" if v >= 0 else "Pulls score down")
        fig = px.bar(impact, x="component", y="impact", color="effect", text="impact",
                     color_discrete_map={"Pushes score up": "#10B981",
                                         "Pulls score down": "#EF4444"},
                     title="What is pulling this student's score up or down? (points vs class average)")
        show(fig)

        st.subheader("Component scores vs class average")
        compare = pd.DataFrame({
            "component": list(WEIGHTS.keys()),
            "student": [round(float(s[c]), 1) for c in WEIGHTS],
            "class average": [round(float(df[c].mean()), 1) for c in WEIGHTS]})
        st.dataframe(compare, hide_index=True)


# =====================================================
# MAIN: decide what to show
# =====================================================
inject_css()
top_bar()

if "user" not in st.session_state:
    login_screen()
    st.stop()

user = st.session_state["user"]
st.sidebar.markdown(f"Signed in as **{user['name']}** ({user['role']})")
if st.sidebar.button("Log out"):
    del st.session_state["user"]
    st.rerun()

if user["role"] == "student":
    student_view(user["id"])
else:
    faculty_view()
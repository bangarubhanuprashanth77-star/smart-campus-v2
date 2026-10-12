import streamlit as st
import plotly.express as px
import pandas as pd
from pipeline import build, WEIGHTS, CLUSTER_FEATURES

# ---------- Page setup (must be the first Streamlit command) ----------
st.set_page_config(page_title="Smart Campus Analytics", page_icon="🎓", layout="wide")

RISK_COLORS = {"High": "#e74c3c", "Medium": "#f39c12", "Low": "#2ecc71"}


@st.cache_data
def get_data():
    return build(with_report=True)


df, clean_report = get_data()
name_map = dict(zip(df["student_id"], df["name"]))

st.title("🎓 Student Success Platform")
st.caption("Unified view of academic, attendance, LMS, engagement, placement, skills and feedback data")

# ---------- Data quality (shows the cleaning done during integration) ----------
with st.expander("🧹 Data integration & cleaning report"):
    st.write(f"Merged 7 sources into one view of **{len(df)} students**.")
    if clean_report:
        st.table(pd.DataFrame({"issue": list(clean_report.keys()),
                               "count": list(clean_report.values())}))
    else:
        st.write("No data issues found.")

# ---------- Sidebar filters ----------
st.sidebar.header("Filters")
all_depts = sorted(df["department"].unique())
all_segs = sorted(df["segment"].unique())

dept = st.sidebar.multiselect("Department", all_depts, default=all_depts)
risk = st.sidebar.multiselect("Risk level", ["High", "Medium", "Low"],
                              default=["High", "Medium", "Low"])
segs = st.sidebar.multiselect("Segment", all_segs, default=all_segs)

view = df[
    df["department"].isin(dept)
    & df["risk_level"].isin(risk)
    & df["segment"].isin(segs)
]

if view.empty:
    st.warning("No students match these filters. Please widen your selection.")
    st.stop()

# ---------- KPI cards ----------
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Students", len(view))
k2.metric("Avg Success Score", f"{view['success_score'].mean():.1f}")
k3.metric("High Risk", int((view["risk_level"] == "High").sum()))
k4.metric("Academic Risk", int(view["academic_risk"].sum()))
k5.metric("Placement Risk", int(view["placement_risk"].sum()))

# ---------- Tabs ----------
tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Overview", "⚠️ At-Risk Students", "🧩 Segments", "🔍 Student Drill-down"]
)

# =====================================================
# TAB 1: Overview
# =====================================================
with tab1:
    left, right = st.columns(2)

    fig = px.histogram(view, x="success_score", color="risk_level", nbins=20,
                       color_discrete_map=RISK_COLORS,
                       category_orders={"risk_level": ["High", "Medium", "Low"]},
                       title="Success Score Distribution")
    left.plotly_chart(fig)

    dept_avg = view.groupby("department")["success_score"].mean().round(1).reset_index()
    fig = px.bar(dept_avg, x="department", y="success_score", text="success_score",
                 title="Average Success Score by Department")
    right.plotly_chart(fig)

    fig = px.scatter(view, x="academic", y="placement", color="segment",
                     hover_name="name",
                     hover_data=["student_id", "success_score"],
                     title="Academic Strength vs Placement Readiness")
    fig.add_hline(y=50, line_dash="dash", line_color="grey")
    fig.add_vline(x=70, line_dash="dash", line_color="grey")
    st.plotly_chart(fig)
    st.caption("Bottom-right corner = strong academics but weak placement readiness.")

    # Key insights, calculated from the filtered data
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

# =====================================================
# TAB 2: At-risk students
# =====================================================
with tab2:
    atrisk = view[
        (view["risk_level"] == "High") | view["academic_risk"] | view["placement_risk"]
    ].sort_values("success_score")

    st.subheader(f"{len(atrisk)} students need attention")

    show_cols = ["student_id", "name", "department", "success_score", "risk_level",
                 "academic_risk", "placement_risk", "risk_reasons",
                 "segment", "suggested_action"]
    st.dataframe(atrisk[show_cols], hide_index=True)

    st.download_button("⬇️ Download at-risk list (CSV)",
                       data=atrisk[show_cols].to_csv(index=False),
                       file_name="at_risk_students.csv", mime="text/csv")

# =====================================================
# TAB 3: Segments
# =====================================================
with tab3:
    seg_counts = view["segment"].value_counts().reset_index()
    seg_counts.columns = ["segment", "students"]
    fig = px.bar(seg_counts, x="students", y="segment", orientation="h",
                 text="students", title="Students per Segment")
    st.plotly_chart(fig)

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

# =====================================================
# TAB 4: Student drill-down (explainable score)
# =====================================================
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
        "impact": [float(s[f"impact_{c}"]) for c in WEIGHTS],
    })
    impact["effect"] = impact["impact"].apply(
        lambda v: "Pushes score up" if v >= 0 else "Pulls score down")

    fig = px.bar(impact, x="component", y="impact", color="effect", text="impact",
                 color_discrete_map={"Pushes score up": "#2ecc71",
                                     "Pulls score down": "#e74c3c"},
                 title="What is pulling this student's score up or down? (points vs class average)")
    st.plotly_chart(fig)

    st.subheader("Component scores vs class average")
    compare = pd.DataFrame({
        "component": list(WEIGHTS.keys()),
        "student": [round(float(s[c]), 1) for c in WEIGHTS],
        "class average": [round(float(df[c].mean()), 1) for c in WEIGHTS],
    })
    st.dataframe(compare, hide_index=True)
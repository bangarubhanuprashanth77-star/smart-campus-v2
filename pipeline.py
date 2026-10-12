import sqlite3
from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).parent
DATA_DIR = BASE / "data"
DB_PATH = BASE / "campus.db"

FILES = ["academic", "attendance", "lms", "engagement",
         "placement", "skills", "feedback"]

# Columns every source must have (used to check uploaded files)
REQUIRED = {
    "academic":   ["student_id", "name", "department", "cgpa", "backlogs"],
    "attendance": ["student_id", "attendance_pct"],
    "lms":        ["student_id", "logins_per_week", "assignment_completion_pct"],
    "engagement": ["student_id", "events", "clubs", "hackathons", "certifications"],
    "placement":  ["student_id", "aptitude", "coding", "mock_interview"],
    "skills":     ["student_id", "technical", "softskill"],
    "feedback":   ["student_id", "satisfaction"],
}

# The integration step, written in SQL. Each <source>_clean view keeps the
# first row per student_id (this removes duplicates), then everything is
# joined into one row per student.
MASTER_SQL = """SELECT a.student_id, a.name, a.department, a.cgpa, a.backlogs,
       t.attendance_pct,
       l.logins_per_week, l.assignment_completion_pct,
       e.events, e.clubs, e.hackathons, e.certifications,
       p.aptitude, p.coding, p.mock_interview,
       s.technical, s.softskill,
       f.satisfaction
FROM academic_clean a
LEFT JOIN attendance_clean t ON a.student_id = t.student_id
LEFT JOIN lms_clean        l ON a.student_id = l.student_id
LEFT JOIN engagement_clean e ON a.student_id = e.student_id
LEFT JOIN placement_clean  p ON a.student_id = p.student_id
LEFT JOIN skills_clean     s ON a.student_id = s.student_id
LEFT JOIN feedback_clean   f ON a.student_id = f.student_id
ORDER BY a.student_id"""


# =====================================================
# PART 2: Load the 7 sources into SQLite, clean and join with SQL
# =====================================================
def ensure_data():
    """If any CSV is missing (e.g. not pushed to GitHub), create the data."""
    if not all((DATA_DIR / f"{n}.csv").exists() for n in FILES):
        import generate_data
        generate_data.main()


def _open_db(use_file):
    """campus.db on disk if it can be written, otherwise an in-memory database."""
    if use_file:
        try:
            conn = sqlite3.connect(DB_PATH)
            conn.execute("CREATE TABLE IF NOT EXISTS _probe (x)")
            conn.execute("DROP TABLE _probe")
            return conn
        except sqlite3.Error:
            pass
    return sqlite3.connect(":memory:")


def _integrate(sources, use_file):
    """Load every source as a table, build the clean views, run the join."""
    conn = _open_db(use_file)
    report = {}
    try:
        for name in FILES:
            sources[name].to_sql(name, conn, if_exists="replace", index=False)
            conn.execute(f"DROP VIEW IF EXISTS {name}_clean")
            conn.execute(
                f"CREATE VIEW {name}_clean AS SELECT * FROM {name} "
                f"WHERE rowid IN (SELECT MIN(rowid) FROM {name} GROUP BY student_id)")
            removed = conn.execute(
                f"SELECT COUNT(*) - COUNT(DISTINCT student_id) FROM {name}"
            ).fetchone()[0]
            if removed > 0:
                report[f"duplicates removed ({name})"] = int(removed)
        conn.commit()
        master = pd.read_sql_query(MASTER_SQL, conn)
    finally:
        conn.close()
    return master, report


def load_and_merge(sources=None):
    """sources=None reads the CSV files in data/. Or pass a dict
    {name: DataFrame} (for example files uploaded in the dashboard)."""
    use_file = sources is None
    if sources is None:
        ensure_data()
        sources = {n: pd.read_csv(DATA_DIR / f"{n}.csv") for n in FILES}

    try:
        master, report = _integrate(sources, use_file)
    except sqlite3.Error:
        if not use_file:
            raise
        master, report = _integrate(sources, False)   # fall back to memory

    missing = master.isna().sum()
    missing = missing[missing > 0]
    for col, n in missing.items():
        report[f"missing values filled ({col})"] = int(n)

    num_cols = master.select_dtypes("number").columns
    for c in num_cols:
        master[c] = master.groupby("department")[c].transform(
            lambda s: s.fillna(s.median())
        )

    assert master.isna().sum().sum() == 0, "Some missing values are still left!"
    return master, report


# =====================================================
# PART 3: Student Success Score
# =====================================================
WEIGHTS = {
    "academic":   0.25,
    "attendance": 0.15,
    "lms":        0.10,
    "engagement": 0.10,
    "placement":  0.20,
    "skills":     0.15,
    "feedback":   0.05,
}
assert round(sum(WEIGHTS.values()), 6) == 1.0, "Weights must add up to 1"


def compute_scores(df):
    d = df.copy()

    backlog_score = (100 - d["backlogs"] * 20).clip(lower=0)
    d["academic"] = 0.8 * (d["cgpa"] * 10) + 0.2 * backlog_score

    d["attendance"] = d["attendance_pct"]

    login_score = (d["logins_per_week"] / 10 * 100).clip(upper=100)
    d["lms"] = 0.4 * login_score + 0.6 * d["assignment_completion_pct"]

    d["engagement"] = (
        d["events"] * 8 + d["clubs"] * 10
        + d["hackathons"] * 15 + d["certifications"] * 10
    ).clip(upper=100)

    d["placement"] = (
        0.4 * d["aptitude"] + 0.4 * d["coding"] + 0.2 * d["mock_interview"]
    )

    d["skills"] = (d["technical"] + d["softskill"]) / 2

    d["feedback"] = (d["satisfaction"] - 1) / 4 * 100

    d["success_score"] = sum(d[k] * w for k, w in WEIGHTS.items()).round(1)

    assert d["success_score"].between(0, 100).all(), "Score out of range!"
    return d


# =====================================================
# PART 4: Risk flags
# =====================================================
def add_risk_flags(df):
    d = df.copy()

    d["academic_risk"] = (
        (d["cgpa"] < 6.0) | (d["backlogs"] >= 2) | (d["attendance_pct"] < 65)
    )

    d["placement_risk"] = (d["placement"] < 50) | (d["coding"] < 40)

    d["risk_level"] = pd.cut(
        d["success_score"],
        bins=[0, 55, 65, 100],
        labels=["High", "Medium", "Low"],
        include_lowest=True,
    )

    def reasons(r):
        out = []
        if r["cgpa"] < 6.0:            out.append("Low CGPA")
        if r["backlogs"] >= 2:         out.append("2+ backlogs")
        if r["attendance_pct"] < 65:   out.append("Low attendance")
        if r["placement"] < 50:        out.append("Low placement score")
        if r["coding"] < 40:           out.append("Weak coding")
        return ", ".join(out) if out else "None"

    d["risk_reasons"] = d.apply(reasons, axis=1)
    return d


# =====================================================
# PART 5: Explainable score
# =====================================================
def explain(df):
    d = df.copy()
    comps = list(WEIGHTS.keys())
    class_avg = d[comps].mean()

    for c in comps:
        d[f"impact_{c}"] = ((d[c] - class_avg[c]) * WEIGHTS[c]).round(2)

    def top_drivers(row):
        impacts = {c: row[f"impact_{c}"] for c in comps}
        ordered = sorted(impacts, key=impacts.get)
        worst = [c for c in ordered[:2] if impacts[c] < 0]
        best = [c for c in ordered[::-1][:1] if impacts[c] > 0]
        down = ", ".join(worst) if worst else "none"
        up = ", ".join(best) if best else "none"
        return f"Pulling down: {down} | Pushing up: {up}"

    d["drivers"] = d.apply(top_drivers, axis=1)
    return d


# =====================================================
# PART 6: Segmentation
# =====================================================
ACTIONS = {
    "High Academic, Low Placement-Ready": "Enrol in coding bootcamp + mock interviews",
    "Disengaged & Struggling":            "Mentor meeting + parent communication",
    "All-Rounders":                       "Encourage leadership roles, advanced projects",
    "Active but Needs Academic Support":  "Remedial classes + peer tutoring",
    "Average / Monitor":                  "Routine monitoring",
}

CLUSTER_FEATURES = ["academic", "attendance", "lms", "engagement", "placement"]


def segment(df):
    d = df.copy()

    # Method 1: rule-based segments (the first matching rule wins)
    def rule(r):
        if r["academic"] >= 70 and r["placement"] < 60:
            return "High Academic, Low Placement-Ready"
        if r["academic"] < 55 and r["attendance"] < 65:
            return "Disengaged & Struggling"
        if r["academic"] >= 70 and r["placement"] >= 58:
            return "All-Rounders"
        if r["engagement"] >= 50 and r["academic"] < 65:
            return "Active but Needs Academic Support"
        return "Average / Monitor"

    d["segment"] = d.apply(rule, axis=1)
    d["suggested_action"] = d["segment"].map(ACTIONS)

    # Method 2: KMeans clustering (finds groups on its own)
    X = StandardScaler().fit_transform(d[CLUSTER_FEATURES])
    d["cluster"] = KMeans(n_clusters=4, n_init=10, random_state=42).fit_predict(X)
    return d


def build(with_report=False, sources=None):
    """Runs the whole pipeline. The dashboard calls this one function."""
    master, report = load_and_merge(sources)
    d = compute_scores(master)
    d = add_risk_flags(d)
    d = explain(d)
    d = segment(d)
    return (d, report) if with_report else d


# =====================================================
# Test block: runs only when you run pipeline.py directly
# =====================================================
if __name__ == "__main__":
    df = build()

    print("Ravi's segment:")
    print(df.loc[df.student_id == "S0001",
                 ["success_score", "segment", "suggested_action", "cluster"]].T)

    print("\nStudents per segment:")
    print(df["segment"].value_counts())

    print("\nKMeans cluster profiles (average component scores):")
    print(df.groupby("cluster")[CLUSTER_FEATURES].mean().round(1))
    print("\nStudents per cluster:")
    print(df["cluster"].value_counts().sort_index())
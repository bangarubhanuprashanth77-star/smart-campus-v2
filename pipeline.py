from pathlib import Path
from functools import reduce

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path(__file__).parent / "data"

FILES = ["academic", "attendance", "lms", "engagement",
         "placement", "skills", "feedback"]


# =====================================================
# PART 2: Load, clean and merge the 7 data sources
# =====================================================
def ensure_data():
    """If any CSV is missing (e.g. not pushed to GitHub), create the data."""
    if not all((DATA_DIR / f"{n}.csv").exists() for n in FILES):
        import generate_data
        generate_data.main()


def load_and_merge():
    ensure_data()
    report = {}

    tables = []
    for name in FILES:
        df = pd.read_csv(DATA_DIR / f"{name}.csv")
        before = len(df)
        df = df.drop_duplicates(subset="student_id")
        removed = before - len(df)
        if removed > 0:
            report[f"duplicates removed ({name})"] = removed
        tables.append(df)

    master = reduce(
        lambda left, right: left.merge(right, on="student_id", how="left"),
        tables
    )

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


def build(with_report=False):
    """Runs the whole pipeline. The dashboard calls this one function."""
    master, report = load_and_merge()
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
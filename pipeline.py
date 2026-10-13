import datetime
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
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


# Editable columns of each source (everything except student_id)
FIELDS = {n: [c for c in REQUIRED[n] if c != "student_id"] for n in FILES}


# =====================================================
# PART 2: SQLite database (load, clean, join, add and edit students)
# =====================================================
_LOCK = threading.RLock()
_MEMORY = {}     # shared in-memory database, used only if campus.db cannot be written


def ensure_data():
    """If any CSV is missing (e.g. not pushed to GitHub), create the data."""
    if not all((DATA_DIR / f"{n}.csv").exists() for n in FILES):
        import generate_data
        generate_data.main()


def _connect():
    """campus.db on disk if it can be written, otherwise one shared in-memory database."""
    try:
        conn = sqlite3.connect(DB_PATH, timeout=15)
        conn.execute("CREATE TABLE IF NOT EXISTS _probe (x)")
        conn.execute("DROP TABLE _probe")
        return conn, True
    except sqlite3.Error:
        if "conn" not in _MEMORY:
            _MEMORY["conn"] = sqlite3.connect(":memory:", check_same_thread=False)
        return _MEMORY["conn"], False


@contextmanager
def db():
    """Open the database, save on success, undo on error."""
    with _LOCK:
        conn, is_file = _connect()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            if is_file:
                conn.close()


def _make_views(conn):
    """<source>_clean keeps the first row per student_id (removes duplicates)."""
    for name in FILES:
        conn.execute(f"DROP VIEW IF EXISTS {name}_clean")
        conn.execute(
            f"CREATE VIEW {name}_clean AS SELECT * FROM {name} "
            f"WHERE rowid IN (SELECT MIN(rowid) FROM {name} GROUP BY student_id)")


def _load_tables(conn, sources):
    for name in FILES:
        sources[name].to_sql(name, conn, if_exists="replace", index=False)
    _make_views(conn)


def _duplicate_report(conn):
    report = {}
    for name in FILES:
        removed = conn.execute(
            f"SELECT COUNT(*) - COUNT(DISTINCT student_id) FROM {name}").fetchone()[0]
        if removed > 0:
            report[f"duplicates removed ({name})"] = int(removed)
    return report


def _tables_ready(conn):
    have = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    return all(n in have for n in FILES)


def _init_interventions_table(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS interventions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            student_id TEXT NOT NULL,
            student_name TEXT,
            department TEXT,
            faculty_name TEXT,
            action_type TEXT,
            priority TEXT,
            notes TEXT,
            status TEXT,
            created_at TEXT,
            target_date TEXT
        )
    """)
    count = conn.execute("SELECT COUNT(*) FROM interventions").fetchone()[0]
    if count == 0:
        sample_data = [
            ("S0001", "Ravi Kumar", "CSE", "Prof. Sharma", "Coding Bootcamp Enrollment", "High", "Strong academics (8.2 CGPA) but coding score is low (28). Assigned intensive 4-week LeetCode problem-solving track.", "In Progress", "2026-10-01", "2026-10-25"),
            ("S0015", "Aditya Verma", "ECE", "Dr. Reddy", "Remedial Peer Tutoring", "High", "Attendance dropped below 60%, 2 backlogs in Digital Electronics. Assigned peer tutor and weekly review.", "Pending", "2026-10-05", "2026-10-20"),
            ("S0042", "Sneha Rao", "MECH", "Prof. Kulkarni", "Attendance & Mentor Check-in", "Medium", "LMS assignment completion is 45%. Conducted counseling session to resolve hostel connectivity issue.", "Completed", "2026-09-28", "2026-10-10"),
            ("S0088", "Karthik Nair", "CSE", "Prof. Sharma", "Placement Mock Interview Prep", "Medium", "Aptitude score is 75 but mock interview score is 42. Enrolled in presentation and technical HR prep club.", "In Progress", "2026-10-08", "2026-10-30"),
            ("S0105", "Priya Sharma", "CIVIL", "Dr. Rao", "Academic Counseling", "Low", "Routine follow-up on project deliverables; student progressing well on certifications.", "Completed", "2026-09-15", "2026-09-30")
        ]
        conn.executemany("""
            INSERT INTO interventions 
            (student_id, student_name, department, faculty_name, action_type, priority, notes, status, created_at, target_date)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, sample_data)


def init_db(force=False):
    """Fill the database from the CSV files, but only if it is empty (or forced).
    After that, the database is the source of truth, so edits are kept."""
    with _LOCK:
        with db() as conn:
            if force or not _tables_ready(conn):
                ensure_data()
                sources = {n: pd.read_csv(DATA_DIR / f"{n}.csv") for n in FILES}
                _load_tables(conn, sources)
            else:
                _make_views(conn)
            _init_interventions_table(conn)


def reset_database():
    """Throw away all edits and reload the original CSV files."""
    with _LOCK:
        with db() as conn:
            conn.execute("DROP TABLE IF EXISTS interventions")
    init_db(force=True)


def get_interventions(student_id=None):
    init_db()
    with db() as conn:
        _init_interventions_table(conn)
        if student_id:
            df = pd.read_sql_query("SELECT * FROM interventions WHERE student_id = ? ORDER BY id DESC", conn, params=[student_id])
        else:
            df = pd.read_sql_query("SELECT * FROM interventions ORDER BY id DESC", conn)
    return df


def save_intervention(student_id, student_name, department, faculty_name, action_type, priority, notes, target_date=None, status="Pending"):
    init_db()
    today = datetime.date.today().strftime("%Y-%m-%d")
    if not target_date:
        target_date = (datetime.date.today() + datetime.timedelta(days=14)).strftime("%Y-%m-%d")
    with db() as conn:
        _init_interventions_table(conn)
        conn.execute("""
            INSERT INTO interventions 
            (student_id, student_name, department, faculty_name, action_type, priority, notes, status, created_at, target_date)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (student_id, student_name, department, faculty_name, action_type, priority, notes, status, today, target_date))


def update_intervention_status(intervention_id, new_status, notes=None):
    init_db()
    with db() as conn:
        _init_interventions_table(conn)
        if notes:
            conn.execute("UPDATE interventions SET status = ?, notes = notes || ' | Update: ' || ? WHERE id = ?",
                         (new_status, notes, intervention_id))
        else:
            conn.execute("UPDATE interventions SET status = ? WHERE id = ?", (new_status, intervention_id))


def delete_intervention(intervention_id):
    init_db()
    with db() as conn:
        _init_interventions_table(conn)
        conn.execute("DELETE FROM interventions WHERE id = ?", (intervention_id,))


def next_student_id():
    init_db()
    with db() as conn:
        last = conn.execute("SELECT MAX(student_id) FROM academic").fetchone()[0]
    try:
        return f"S{int(str(last)[1:]) + 1:04d}"
    except (TypeError, ValueError):
        return "S0001"


def _py(v):
    return v.item() if hasattr(v, "item") else v


def save_student(student_id, values):
    """Insert a new student, or update an existing one, in all 7 tables.
    values = {column: value} for every column in FIELDS."""
    absent = [c for n in FILES for c in FIELDS[n] if c not in values]
    if absent:
        raise ValueError(f"Missing values for: {', '.join(absent)}")
    init_db()
    with db() as conn:
        for name in FILES:
            cols = FIELDS[name]
            vals = [_py(values[c]) for c in cols]
            sets = ", ".join(f"{c} = ?" for c in cols)
            cur = conn.execute(f"UPDATE {name} SET {sets} WHERE student_id = ?",
                               vals + [student_id])
            if cur.rowcount == 0:
                marks = ", ".join("?" * (len(cols) + 1))
                conn.execute(
                    f"INSERT INTO {name} (student_id, {', '.join(cols)}) VALUES ({marks})",
                    [student_id] + vals)


def load_and_merge(sources=None):
    """sources=None reads the SQLite database (filled from data/*.csv the first
    time). Or pass {name: DataFrame}, e.g. files uploaded in the dashboard; those
    are joined in a temporary in-memory database and nothing is saved."""
    if sources is None:
        init_db()
        with db() as conn:
            master = pd.read_sql_query(MASTER_SQL, conn)
            report = _duplicate_report(conn)
    else:
        conn = sqlite3.connect(":memory:")
        try:
            _load_tables(conn, sources)
            master = pd.read_sql_query(MASTER_SQL, conn)
            report = _duplicate_report(conn)
        finally:
            conn.close()

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


# =====================================================
# PART 7: Machine Learning Models (Random Forest)
# =====================================================
ML_FEATURE_COLS = [
    "cgpa", "backlogs", "attendance_pct", "logins_per_week",
    "assignment_completion_pct", "events", "clubs", "hackathons",
    "certifications", "aptitude", "coding", "mock_interview",
    "technical", "softskill", "satisfaction"
]


def train_ml_models(df):
    """Trains Random Forest models for placement success and academic failure risk."""
    d = df.copy()
    X = d[ML_FEATURE_COLS]

    # Target 1: Placement Readiness
    y_place = ((d["placement"] >= 58) & (d["coding"] >= 45)).astype(int)
    rf_place = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    rf_place.fit(X, y_place)

    # Target 2: Early Academic Risk (Distress / Dropout warning)
    y_risk = ((d["cgpa"] < 6.2) | (d["backlogs"] >= 2) | (d["attendance_pct"] < 68)).astype(int)
    rf_risk = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    rf_risk.fit(X, y_risk)

    d["ml_placement_prob"] = (rf_place.predict_proba(X)[:, 1] * 100).round(1)
    d["ml_risk_prob"] = (rf_risk.predict_proba(X)[:, 1] * 100).round(1)

    # Hidden Risk: passes heuristic (Low risk), but ML detects multi-signal drift
    d["ml_hidden_risk"] = (d["risk_level"] == "Low") & (d["ml_risk_prob"] >= 35.0)

    place_imp = dict(zip(ML_FEATURE_COLS, rf_place.feature_importances_.round(4)))
    risk_imp = dict(zip(ML_FEATURE_COLS, rf_risk.feature_importances_.round(4)))

    cv_place = float(cross_val_score(rf_place, X, y_place, cv=5).mean().round(3))
    cv_risk = float(cross_val_score(rf_risk, X, y_risk, cv=5).mean().round(3))

    ml_metadata = {
        "placement_model": {
            "name": "RandomForest Placement Predictor",
            "accuracy": cv_place,
            "feature_importances": dict(sorted(place_imp.items(), key=lambda kv: -kv[1])),
            "top_drivers": sorted(place_imp, key=place_imp.get, reverse=True)[:4]
        },
        "risk_model": {
            "name": "RandomForest Academic Distress Early Warning",
            "accuracy": cv_risk,
            "feature_importances": dict(sorted(risk_imp.items(), key=lambda kv: -kv[1])),
            "top_drivers": sorted(risk_imp, key=risk_imp.get, reverse=True)[:4]
        }
    }
    return d, ml_metadata


# =====================================================
# PART 8: Career Pathway Recommendation Engine
# =====================================================
CAREER_TRACKS = [
    {
        "id": "swe",
        "title": "Software Engineering & Full-Stack",
        "icon": "💻",
        "weights": {"coding": 0.35, "technical": 0.25, "aptitude": 0.20, "assignment_completion_pct": 0.10, "hackathons": 0.10},
        "required_keys": [("coding", "Coding & DSA", 60), ("technical", "Tech Fundamentals", 65), ("aptitude", "Problem Solving", 60)],
        "certifications": ["AWS Certified Developer Associate", "LeetCode 75 Mastery Badge", "Meta Full-Stack Engineer"],
        "recommended_electives": ["Distributed Systems & Microservices", "Cloud Native Computing", "Advanced Algorithms & DSA"],
        "summary": "Building scalable web platforms, backend systems, and modern software applications."
    },
    {
        "id": "ds_ai",
        "title": "AI & Data Science Specialist",
        "icon": "🤖",
        "weights": {"cgpa": 0.25, "technical": 0.25, "coding": 0.20, "aptitude": 0.15, "hackathons": 0.15},
        "required_keys": [("cgpa", "Academic / Math Foundation", 7.0), ("technical", "ML & Data Skills", 65), ("coding", "Python Algorithms", 55)],
        "certifications": ["DeepLearning.AI Machine Learning", "Google Professional Data Engineer", "TensorFlow Developer"],
        "recommended_electives": ["Deep Learning & Neural Networks", "Natural Language Processing", "Big Data Analytics & Spark"],
        "summary": "Designing neural models, predictive analytics systems, and large-scale data pipelines."
    },
    {
        "id": "devops_cloud",
        "title": "Cloud, SRE & DevOps Engineer",
        "icon": "☁️",
        "weights": {"technical": 0.30, "certifications": 0.25, "assignment_completion_pct": 0.20, "coding": 0.15, "aptitude": 0.10},
        "required_keys": [("technical", "System Fundamentals", 60), ("certifications", "Cloud Certs", 2), ("assignment_completion_pct", "Lab Hands-on", 75)],
        "certifications": ["AWS Solutions Architect Associate", "Certified Kubernetes Administrator (CKA)", "HashiCorp Terraform Associate"],
        "recommended_electives": ["Cloud Architecture & Infra as Code", "Linux & Shell Automation", "Network & Cyber Security"],
        "summary": "Managing cloud reliability, Kubernetes clusters, and automated CI/CD pipelines."
    },
    {
        "id": "pm_consulting",
        "title": "Product Management & Tech Consulting",
        "icon": "📊",
        "weights": {"softskill": 0.35, "mock_interview": 0.25, "aptitude": 0.20, "events": 0.10, "clubs": 0.10},
        "required_keys": [("softskill", "Communication & Leadership", 65), ("mock_interview", "Executive Presence", 60), ("aptitude", "Analytical Reasoning", 60)],
        "certifications": ["Certified Scrum Product Owner (CSPO)", "Google Project Management", "McKinsey/BCG Case Prep"],
        "recommended_electives": ["Software Product Management", "Technology Strategy & Entrepreneurship", "Business Intelligence & KPIs"],
        "summary": "Aligning engineering execution, stakeholder roadmaps, and high-impact business outcomes."
    },
    {
        "id": "core_systems",
        "title": "Core Engineering & Embedded Systems",
        "icon": "⚙️",
        "weights": {"cgpa": 0.35, "technical": 0.30, "mock_interview": 0.15, "attendance_pct": 0.10, "certifications": 0.10},
        "required_keys": [("cgpa", "Core Theory Marks", 7.5), ("technical", "Hardware / Domain Knowledge", 60), ("attendance_pct", "Lab Attendance", 75)],
        "certifications": ["ARM Accredited Engineer", "MATLAB & Simulink Specialist", "IoT Systems & Firmware"],
        "recommended_electives": ["Embedded Real-Time Systems", "Robotics & Microcontrollers", "VLSI Design & Architecture"],
        "summary": "Developing hardware-firmware integration, IoT devices, and specialized domain engineering systems."
    }
]


def get_career_recommendations(row):
    """Evaluates student fit for 5 career paths, calculates match %, strengths, gaps, and certs."""
    results = []

    def norm(col):
        v = float(row[col])
        if col == "cgpa": return min(max(v * 10.0, 0), 100)
        if col in ["hackathons", "certifications", "clubs"]: return min(v / 3.0 * 100, 100)
        if col == "events": return min(v / 5.0 * 100, 100)
        return min(max(v, 0), 100)

    for track in CAREER_TRACKS:
        score = sum(norm(k) * w for k, w in track["weights"].items())
        score = round(min(max(score, 5.0), 99.0), 1)

        strengths = []
        gaps = []
        for col, label, thresh in track["required_keys"]:
            val = float(row[col])
            if val >= thresh:
                strengths.append(f"{label} ({val:g})")
            else:
                gaps.append(f"{label} (Current: {val:g}, Target: {thresh:g})")

        results.append({
            "id": track["id"],
            "title": track["title"],
            "icon": track["icon"],
            "match_score": score,
            "summary": track["summary"],
            "strengths": strengths if strengths else ["Foundational prerequisites met"],
            "gaps": gaps if gaps else ["All target readiness thresholds met!"],
            "certifications": track["certifications"],
            "electives": track["recommended_electives"]
        })

    results.sort(key=lambda x: x["match_score"], reverse=True)
    return results


# =====================================================
# PART 9: Student Goal Planner & Roadmap Calculator
# =====================================================
def calculate_goal_plan(row, target_score):
    """Calculates practical steps for a student to reach target_score."""
    current = float(row["success_score"])
    gap = target_score - current
    if gap <= 0:
        return {
            "achieved": True,
            "current": current,
            "target": target_score,
            "message": f"You've already surpassed this goal! Current score: {current:.1f} vs Target: {target_score:.1f}.",
            "steps": []
        }

    steps = []
    # Lever 1: Attendance
    att = float(row["attendance_pct"])
    if att < 90:
        new_att = min(att + 10, 95)
        gain = round((new_att - att) * 0.15, 1)
        steps.append({
            "area": "Attendance",
            "icon": "📅",
            "action": f"Raise attendance from {att:.0f}% to {new_att:.0f}%",
            "gain": gain,
            "details": "Attend all scheduled lectures and laboratory sessions over the next 4 weeks."
        })

    # Lever 2: Coding Practice
    code = float(row["coding"])
    if code < 85:
        target_code = min(code + 25, 90)
        gain = round((target_code - code) * 0.4 * 0.20, 1)
        steps.append({
            "area": "Coding Practice",
            "icon": "💻",
            "action": f"Boost coding score from {code:.0f} to {target_code:.0f}",
            "gain": gain,
            "details": "Solve 3 medium DSA problems weekly on LeetCode/HackerRank and attend coding labs."
        })

    # Lever 3: LMS Assignments
    assign = float(row["assignment_completion_pct"])
    if assign < 95:
        target_assign = min(assign + 20, 98)
        gain = round((target_assign - assign) * 0.6 * 0.10, 1)
        steps.append({
            "area": "LMS Assignments",
            "icon": "📝",
            "action": f"Improve assignment completion from {assign:.0f}% to {target_assign:.0f}%",
            "gain": gain,
            "details": "Submit all outstanding coursework 48 hours before the deadline."
        })

    # Lever 4: Mock Interviews
    mock = float(row["mock_interview"])
    if mock < 80:
        target_mock = min(mock + 20, 85)
        gain = round((target_mock - mock) * 0.2 * 0.20, 1)
        steps.append({
            "area": "Mock Interviews",
            "icon": "🎤",
            "action": f"Increase mock interview score from {mock:.0f} to {target_mock:.0f}",
            "gain": gain,
            "details": "Book 2 peer mock interviews and review resume pitch with campus career cell."
        })

    # Lever 5: Certifications & Hackathons
    certs = int(row["certifications"])
    if certs < 3:
        steps.append({
            "area": "Certifications & Events",
            "icon": "🏆",
            "action": f"Earn 1 new industry certification (Current: {certs})",
            "gain": 1.0,
            "details": "Enroll in a free cloud or technical certification (AWS/Google/NPTEL)."
        })

    total_gain = sum(s["gain"] for s in steps)
    projected = round(min(current + total_gain, 100.0), 1)

    return {
        "achieved": False,
        "current": current,
        "target": target_score,
        "gap": round(gap, 1),
        "total_gain": round(total_gain, 1),
        "projected": projected,
        "steps": steps
    }


# =====================================================
# PART 10: Campus Policy & What-If Simulator
# =====================================================
def simulate_campus_policy(df, att_delta=0, coding_delta=0, assign_delta=0, hack_delta=0):
    """Simulates campus-wide intervention policies and projects before/after impact."""
    sim = df.copy()

    if att_delta != 0:
        sim["attendance_pct"] = (sim["attendance_pct"] + att_delta).clip(0, 100)
    if coding_delta != 0:
        sim["coding"] = (sim["coding"] + coding_delta).clip(0, 100)
    if assign_delta != 0:
        sim["assignment_completion_pct"] = (sim["assignment_completion_pct"] + assign_delta).clip(0, 100)
    if hack_delta != 0:
        sim["hackathons"] = (sim["hackathons"] + hack_delta).clip(0, 10)

    sim = compute_scores(sim)
    sim = add_risk_flags(sim)
    sim = explain(sim)
    sim = segment(sim)
    sim, _ = train_ml_models(sim)

    base_avg = float(df["success_score"].mean())
    sim_avg = float(sim["success_score"].mean())

    base_high_risk = int((df["risk_level"] == "High").sum())
    sim_high_risk = int((sim["risk_level"] == "High").sum())

    base_place_ready = int(((df["placement"] >= 50) & (df["coding"] >= 40)).sum())
    sim_place_ready = int(((sim["placement"] >= 50) & (sim["coding"] >= 40)).sum())

    base_acad_risk = int(df["academic_risk"].sum())
    sim_acad_risk = int(sim["academic_risk"].sum())

    return {
        "sim_df": sim,
        "base_avg": round(base_avg, 1),
        "sim_avg": round(sim_avg, 1),
        "avg_delta": round(sim_avg - base_avg, 1),
        "base_high_risk": base_high_risk,
        "sim_high_risk": sim_high_risk,
        "risk_delta": sim_high_risk - base_high_risk,
        "base_place_ready": base_place_ready,
        "sim_place_ready": sim_place_ready,
        "place_delta": sim_place_ready - base_place_ready,
        "base_acad_risk": base_acad_risk,
        "sim_acad_risk": sim_acad_risk,
        "acad_delta": sim_acad_risk - base_acad_risk,
    }


# =====================================================
# PART 11: Demo Scenarios Generator
# =====================================================
def get_scenario_data(scenario_type, base_df):
    """Generates perturbed scenario datasets for instant hackathon demonstrations."""
    if scenario_type == "Standard Cohort":
        return base_df.copy()

    s = base_df.copy()
    if scenario_type == "🚨 Pre-Exam Attendance Crisis":
        mask = s["attendance_pct"] < s["attendance_pct"].quantile(0.4)
        s.loc[mask, "attendance_pct"] = (s.loc[mask, "attendance_pct"] - 16).clip(30, 100)
        s.loc[mask, "assignment_completion_pct"] = (s.loc[mask, "assignment_completion_pct"] - 22).clip(15, 100)
        s.loc[mask, "logins_per_week"] = (s.loc[mask, "logins_per_week"] - 3).clip(1, 20)
    elif scenario_type == "💼 Placement Season Crunch":
        mask = s["department"].isin(["ECE", "MECH", "CIVIL"])
        s.loc[mask, "coding"] = (s.loc[mask, "coding"] * 0.7).clip(15, 100).round()
        s.loc[mask, "mock_interview"] = (s.loc[mask, "mock_interview"] - 12).clip(20, 100).round()
    elif scenario_type == "🌟 Post-Intervention Success":
        mask = (s["risk_level"] == "High") | (s["placement_risk"])
        s.loc[mask, "attendance_pct"] = (s.loc[mask, "attendance_pct"] + 14).clip(0, 95)
        s.loc[mask, "coding"] = (s.loc[mask, "coding"] + 24).clip(0, 85)
        s.loc[mask, "assignment_completion_pct"] = (s.loc[mask, "assignment_completion_pct"] + 20).clip(0, 95)
        s.loc[mask, "mock_interview"] = (s.loc[mask, "mock_interview"] + 15).clip(0, 85)

    s = compute_scores(s)
    s = add_risk_flags(s)
    s = explain(s)
    s = segment(s)
    s, _ = train_ml_models(s)
    return s


# =====================================================
# PART 12: AI Campus Advisor Copilot Query Engine
# =====================================================
def generate_advisor_response(query, role, student_row=None, cohort_df=None):
    """Synthesizes instant, contextual advisory intelligence for students and faculty."""
    q = query.strip().lower()

    if role == "student" and student_row is not None:
        s = student_row
        name = s.get("name", "Student")
        score = float(s.get("success_score", 0))
        cgpa = float(s.get("cgpa", 0))
        att = float(s.get("attendance_pct", 0))
        code = float(s.get("coding", 0))
        mock = float(s.get("mock_interview", 0))
        place_prob = float(s.get("ml_placement_prob", 0))
        risk_level = str(s.get("risk_level", "Medium"))
        segment = str(s.get("segment", "Average / Monitor"))

        if any(w in q for w in ["placement", "job", "career", "interview", "coding"]):
            return (
                f"### 💼 Placement Advisory for {name}\n\n"
                f"- **Current Placement Probability**: **{place_prob:.1f}%** (Placement Score: {float(s.get('placement', 0)):.1f}/100)\n"
                f"- **Coding Benchmark**: **{code:.0f}/100** ({'⚠️ Critical gap: below campus cutoff 40' if code < 40 else '✅ Solid standing'})\n"
                f"- **Mock Interview**: **{mock:.0f}/100**\n\n"
                f"**🎯 3-Step Recommendation Plan:**\n"
                f"1. **Daily Coding Sprint**: Spend 45 minutes daily on LeetCode Top 75 / HackerRank Easy & Medium.\n"
                f"2. **Aptitude Mastery**: Practice quantitative and logical reasoning modules on the campus LMS.\n"
                f"3. **Mock Interviews**: Schedule a peer mock interview with your department placement cell before week 6."
            )
        elif any(w in q for w in ["why", "pull", "down", "reason", "weak"]):
            drivers = str(s.get("drivers", ""))
            return (
                f"### 🔍 Diagnostic Breakdown for {name}\n\n"
                f"- **Current Success Score**: **{score:.1f}/100** ({risk_level} Risk)\n"
                f"- **Primary Score Drivers**: `{drivers}`\n\n"
                f"**Detailed Breakdown:**\n"
                f"- **CGPA**: {cgpa:.2f} (Counts 20% toward score)\n"
                f"- **Attendance**: {att:.1f}% (Counts 15% toward score — {'⚠️ Needs immediate boost to 75%+' if att < 75 else '✅ Solid'})\n"
                f"- **LMS Completion**: {float(s.get('assignment_completion_pct', 0)):.1f}%\n\n"
                f"**Recommended Focus**: Focus on improving `{s.get('risk_reasons', 'General consistency')}` to see the quickest point increase."
            )
        elif any(w in q for w in ["plan", "roadmap", "study", "schedule", "improve"]):
            return (
                f"### 📅 4-Week Personalized Improvement Roadmap for {name}\n\n"
                f"| Week | Focus Area | Weekly Milestone |\n"
                f"|---|---|---|\n"
                f"| **Week 1** | Attendance & LMS | Attend all lectures (Target 85%+), submit 2 pending LMS assignments. |\n"
                f"| **Week 2** | Coding & DSA | Solve 10 problems on arrays, strings, and hash maps. Target coding +10. |\n"
                f"| **Week 3** | Technical Skills | Complete 1 module of cloud/web tech certification. |\n"
                f"| **Week 4** | Mock Interview | Schedule formal faculty/peer mock interview & resume review. |\n\n"
                f"**Target Outcome**: Projected score increase from **{score:.1f}** to **{min(score + 8.5, 98.0):.1f}**."
            )
        elif any(w in q for w in ["cert", "course", "elective", "skill"]):
            return (
                f"### 🏆 Recommended Certifications & Electives\n\n"
                f"Based on your profile in `{s.get('department')}`:\n"
                f"1. **Cloud & Infrastructure**: *AWS Certified Cloud Practitioner* or *Google Cloud Associate*\n"
                f"2. **Core Programming**: *Python for Data Structures & Algorithms* (Coursera / LeetCode)\n"
                f"3. **Hackathon Opportunity**: Register for upcoming Smart Campus Intra-College Hackathon to gain +15 engagement points!"
            )
        else:
            return (
                f"### 🎓 Academic Advisor Summary for {name}\n\n"
                f"- **Success Score**: **{score:.1f}/100**\n"
                f"- **Segment**: **{segment}**\n"
                f"- **Immediate Action**: {s.get('suggested_action', 'Continue current academic pace.')}\n\n"
                f"You can ask me about **placement readiness**, **why your score is pulling down**, **a 4-week study plan**, or **recommended certifications**!"
            )

    elif role == "faculty" and cohort_df is not None:
        view = cohort_df
        total = len(view)
        high_risk = int((view["risk_level"] == "High").sum())
        acad_risk = int(view["academic_risk"].sum())
        place_risk = int(view["placement_risk"].sum())
        avg_score = float(view["success_score"].mean())
        hidden = int(view.get("ml_hidden_risk", pd.Series([False]*total)).sum())

        if any(w in q for w in ["summary", "overview", "cohort", "risk"]):
            return (
                f"### 📊 Faculty Executive Briefing\n\n"
                f"- **Cohort Size**: **{total} Students** across {view['department'].nunique()} departments\n"
                f"- **Average Success Score**: **{avg_score:.1f}/100**\n"
                f"- **High Risk Flagged**: **{high_risk} students** ({high_risk/total*100:.1f}%)\n"
                f"- **Placement Risk**: **{place_risk} students** ({place_risk/total*100:.1f}%)\n"
                f"- **Academic Risk**: **{acad_risk} students** ({acad_risk/total*100:.1f}%)\n"
                f"- **ML Hidden Risk Detected**: **{hidden} students** (students who pass static filters but show multi-signal decline)\n\n"
                f"**⚡ Priority Recommendation**: Trigger immediate remedial coding clinic for students in the *High Academic, Low Placement-Ready* segment."
            )
        elif any(w in q for w in ["coding", "placement", "cgpa", "high academic"]):
            sub = view[(view["cgpa"] >= 7.5) & (view["coding"] < 45)]
            return (
                f"### ⚠️ Students with High CGPA (≥7.5) but Weak Coding (<45)\n\n"
                f"Found **{len(sub)} students** in this high-vulnerability bracket.\n\n"
                f"**Top Candidates for Placement Bootcamp:**\n" +
                "\n".join([f"- **{r['name']}** ({r['student_id']}, {r['department']}): CGPA {r['cgpa']:.2f}, Coding {r['coding']:.0f}/100, Success Score {r['success_score']:.1f}" for _, r in sub.head(5).iterrows()]) +
                f"\n\n*Action*: Assign to 3-week competitive programming clinic."
            )
        elif any(w in q for w in ["lms", "department", "lowest", "engagement"]):
            lms_dept = view.groupby("department")["lms"].mean().round(1)
            lowest_d = lms_dept.idxmin()
            return (
                f"### 🏛️ Departmental LMS Engagement Analysis\n\n" +
                "\n".join([f"- **{d}**: Average LMS Score **{score:.1f}/100**" for d, score in lms_dept.items()]) +
                f"\n\n**⚠️ Lowest Engagement**: **{lowest_d}** department ({lms_dept.min():.1f}/100).\n"
                f"**Suggested Action**: Notify department HOD to review assignment submission deadlines and course module uploads."
            )
        elif any(w in q for w in ["draft", "mentor", "note", "message", "ravi"]):
            return (
                f"### ✉️ Drafted Mentorship Message for Student S0001 (Ravi Kumar)\n\n"
                f"> **Subject**: Placement Readiness & Coding Clinic Support\n>\n"
                f"> *Dear Ravi,*\n>\n"
                f"> *Your academic record in CSE is commendable with an 8.2 CGPA and consistent attendance. However, our analytics indicate that your coding assessment score (28/100) and mock interview score (40/100) require targeted preparation prior to the upcoming campus recruitment drives.*\n>\n"
                f"> *We have reserved a seat for you in the Department Coding Bootcamp starting this Thursday. Please connect with your faculty mentor Prof. Sharma for your personalized problem-solving schedule.*\n>\n"
                f"> *Best regards,*\n"
                f"> *Academic Success Cell*"
            )
        else:
            return (
                f"### 🎓 Faculty AI Advisor Assistant\n\n"
                f"- Currently tracking **{total} students**.\n"
                f"- **{high_risk} High-Risk** and **{place_risk} Placement-Risk** candidates require faculty attention.\n\n"
                f"You can ask me to:\n"
                f"- *'Summarize cohort risk factors'*\n"
                f"- *'Show students with CGPA > 8.0 but low coding'*\n"
                f"- *'Which department has lowest LMS engagement?'*\n"
                f"- *'Draft mentor message for Ravi'*."
            )

    return "I am your Smart Campus AI Advisor. How can I assist you today?"


def build(with_report=False, sources=None):
    """Runs the whole pipeline. The dashboard calls this one function."""
    master, report = load_and_merge(sources)
    d = compute_scores(master)
    d = add_risk_flags(d)
    d = explain(d)
    d = segment(d)
    d, ml_meta = train_ml_models(d)
    if with_report:
        report["ml_metadata"] = ml_meta
        return d, report
    return d


# =====================================================
# Test block: runs only when you run pipeline.py directly
# =====================================================
if __name__ == "__main__":
    df, report = build(with_report=True)

    print("Ravi's segment & ML predictions:")
    print(df.loc[df.student_id == "S0001",
                 ["success_score", "segment", "suggested_action", "cluster",
                  "ml_placement_prob", "ml_risk_prob", "ml_hidden_risk"]].T)

    print("\nStudents per segment:")
    print(df["segment"].value_counts())

    print("\nML Placement Model CV Accuracy:", report["ml_metadata"]["placement_model"]["accuracy"])
    print("ML Risk Model CV Accuracy:", report["ml_metadata"]["risk_model"]["accuracy"])
    print("\nInterventions count:", len(get_interventions()))
"""Creates synthetic student data as 7 CSV files inside the data/ folder."""
from pathlib import Path
import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"


def main():
    np.random.seed(42)
    N = 300
    DATA_DIR.mkdir(exist_ok=True)

    ids = [f"S{i:04d}" for i in range(1, N + 1)]
    clip = lambda x, lo, hi: np.clip(x, lo, hi)
    effort = np.random.normal(0, 1, N)  # hidden "hard-working level"

    academic = pd.DataFrame({
        "student_id": ids,
        "name": [f"Student {i}" for i in range(1, N + 1)],
        "department": np.random.choice(["CSE", "ECE", "MECH", "CIVIL"], N),
        "cgpa": clip(7.0 + 0.8 * effort + np.random.normal(0, 0.6, N), 3, 10).round(2),
        "backlogs": np.random.poisson(np.clip(0.8 - 0.4 * effort, 0.05, None)),
    })

    attendance = pd.DataFrame({
        "student_id": ids,
        "attendance_pct": clip(78 + 8 * effort + np.random.normal(0, 7, N), 30, 100).round(1),
    })

    lms = pd.DataFrame({
        "student_id": ids,
        "logins_per_week": np.random.poisson(np.clip(6 + 2 * effort, 1, None)),
        "assignment_completion_pct": clip(75 + 12 * effort + np.random.normal(0, 10, N), 0, 100).round(1),
    })

    engagement = pd.DataFrame({
        "student_id": ids,
        "events": np.random.poisson(3, N),
        "clubs": np.random.randint(0, 4, N),
        "hackathons": np.random.poisson(1, N),
        "certifications": np.random.poisson(np.clip(1.5 + 0.5 * effort, 0.1, None)),
    })

    placement = pd.DataFrame({
        "student_id": ids,
        "aptitude": clip(60 + 6 * effort + np.random.normal(0, 12, N), 0, 100).round(0),
        "coding": clip(55 + 8 * effort + np.random.normal(0, 16, N), 0, 100).round(0),
        "mock_interview": clip(55 + 6 * effort + np.random.normal(0, 14, N), 0, 100).round(0),
    })

    skills = pd.DataFrame({
        "student_id": ids,
        "technical": clip(60 + 6 * effort + np.random.normal(0, 12, N), 0, 100).round(0),
        "softskill": clip(62 + 4 * effort + np.random.normal(0, 12, N), 0, 100).round(0),
    })

    feedback = pd.DataFrame({
        "student_id": ids,
        "satisfaction": np.random.randint(1, 6, N),
    })

    # Demo student "Ravi": good marks but weak coding (high academic, low placement)
    academic.loc[0, "name"] = "Ravi"
    academic.loc[0, "department"] = "CSE"
    academic.loc[0, "cgpa"] = 8.2
    academic.loc[0, "backlogs"] = 0
    attendance.loc[0, "attendance_pct"] = 82.0
    lms.loc[0, "logins_per_week"] = 5
    lms.loc[0, "assignment_completion_pct"] = 70.0
    for col, val in {"events": 2, "clubs": 1, "hackathons": 0, "certifications": 1}.items():
        engagement.loc[0, col] = val
    for col, val in {"aptitude": 55, "coding": 28, "mock_interview": 40}.items():
        placement.loc[0, col] = val
    skills.loc[0, "technical"] = 50
    skills.loc[0, "softskill"] = 60
    feedback.loc[0, "satisfaction"] = 4

    # Make the data messy like real life: missing values and duplicate rows
    attendance.loc[np.random.choice(range(1, N), 12, replace=False), "attendance_pct"] = np.nan
    placement.loc[np.random.choice(range(1, N), 10, replace=False), "coding"] = np.nan
    lms = pd.concat([lms, lms.sample(5, random_state=1)])

    tables = {"academic": academic, "attendance": attendance, "lms": lms,
              "engagement": engagement, "placement": placement,
              "skills": skills, "feedback": feedback}
    for name, df in tables.items():
        df.to_csv(DATA_DIR / f"{name}.csv", index=False)
        print(f"Saved data/{name}.csv ({len(df)} rows)")


if __name__ == "__main__":
    main()

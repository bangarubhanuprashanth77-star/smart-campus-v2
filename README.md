# Smart Campus Analytics: Student Success Platform

An analytics dashboard that integrates student data from 7 sources, calculates a
Student Success Score, flags at-risk students, and suggests actions for faculty
and administrators.

**Live demo:** https://YOUR-APP-NAME.streamlit.app

## What it does
- Integrates academic, attendance, LMS, engagement, placement, skills and feedback data
- Cleans the data (duplicates removed, missing values filled with department medians)
- Calculates a weighted Student Success Score (0-100)
- Flags academic risk and placement risk separately
- Explains each score (which indicators pull it up or down)
- Segments students (rule-based + KMeans) and recommends an action for each group

## Student Success Score (0-100)
Weighted average of 7 component scores, each on a 0-100 scale:

| Component | Weight | Built from |
|---|---|---|
| Academic | 25% | 80% CGPA + 20% backlog score |
| Placement | 20% | 40% aptitude + 40% coding + 20% mock interview |
| Attendance | 15% | Overall attendance % |
| Skills | 15% | Average of technical and soft skills |
| LMS | 10% | 40% login frequency + 60% assignment completion |
| Engagement | 10% | Events, clubs, hackathons, certifications |
| Feedback | 5% | Student satisfaction (1-5 scaled to 0-100) |

**Risk level:** High below 55, Medium 55-65, Low above 65.
**Academic risk:** CGPA < 6, 2+ backlogs, or attendance < 65%.
**Placement risk:** placement score < 50 or coding < 40.

## Tech stack
Python, pandas, NumPy, scikit-learn (KMeans), Streamlit, Plotly

## Project structure
- `generate_data.py`: creates synthetic student data (7 CSV files in `data/`)
- `pipeline.py`: cleaning, merging, scoring, risk flags, explainability, segmentation
- `app.py`: the Streamlit dashboard

## Run locally
    pip install -r requirements.txt
    python generate_data.py
    streamlit run app.py

## Note
All data is synthetic, created to protect student privacy. The pipeline works
unchanged on real exports with the same columns. If the CSV files are missing,
the app regenerates them automatically.

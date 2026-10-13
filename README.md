# Smart Campus Analytics: AI Student Success Platform 🎓

> **A Next-Generation Predictive Analytics & Closed-Loop Intervention System for Higher Education**  
> Unifies 7 fragmented campus data streams, deploys dual Machine Learning predictive models, calculates holistic multi-signal diagnostics, and enables proactive, empathetic faculty interventions.

---

## 🏆 Hackathon Innovation Highlights

Traditional campus systems only flag students after exams or placement test rejections—when it is already too late. **Smart Campus Analytics 2.0** turns reactive monitoring into **predictive early warning, career alignment, and closed-loop faculty mentoring**.

### 🌟 Key Hackathon Features

1. **🔗 7-Source SQLite Data Fusion Pipeline**
   - Ingests academic records, RFID attendance, LMS Moodle logs, campus engagement, placement mock tests, technical/soft skills, and student sentiment surveys into a high-performance SQLite database.
   - Cleans duplicate records using SQL views and imputes missing signals using department-level medians.

2. **🤖 Dual Supervised Machine Learning Engines (Scikit-Learn)**
   - **Placement Readiness Predictor (`RandomForestClassifier`)**: **90.7% 5-Fold Cross-Validation Accuracy**. Predicts probability (0–100%) of clearing competitive campus recruitment.
   - **Academic Distress Early Warning Detector (`RandomForestClassifier`)**: **99.7% Cross-Validation Accuracy**. Identifies subtle multi-signal disengagement weeks before exams.
   - **Explainable AI (XAI)**: Visualizes exact feature importances driving predictions (e.g. coding problem-solving vs assignment consistency).
   - **ML Hidden Risk Detection**: Uncovers students who pass simple heuristic filters (e.g., normal CGPA) but display behavioral drift.

3. **🕸️ 7-Dimension Holistic Spider / Radar Chart**
   - Polar benchmark comparing each student across Academic, Attendance, LMS, Engagement, Placement, Skills, and Feedback.
   - Layers the student's polygon against the **Cohort Average** and the **Top 10% Honors Benchmark**.

4. **🎯 Personalized Industry Career Pathway & Skill Gap Engine**
   - Evaluates students against 5 high-demand technology tracks:
     - 💻 *Software Engineering & Full-Stack*
     - 🤖 *AI & Data Science Specialist*
     - ☁️ *Cloud, SRE & DevOps Engineer*
     - 📊 *Product Management & Tech Consulting*
     - ⚙️ *Core Engineering & Embedded Systems*
   - Computes match percentage, pinpoints exact skill gaps, and recommends targeted certifications and campus electives.

5. **📋 Closed-Loop Faculty Intervention Management Hub**
   - Moves beyond static charts: Faculty can assign concrete mentoring actions (*Coding Bootcamp, Remedial Tutoring, Attendance Counseling, Mock Interview Prep, Parent Consultation*).
   - Full **Persistent SQLite Case Management** (`interventions` table) with status workflow (*Pending → In Progress → Completed*), priority badges, and CSV export.

6. **🧪 Dean & HOD Interactive Policy Simulator (What-If Analysis)**
   - Real-time simulation lab for institutional leadership:
   - Move sliders to simulate campus-wide policies (*e.g., +5% Attendance enforcement, +12 Coding bootcamp points, +8% LMS push*).
   - Instantly calculates projected campus-wide risk reduction and placement rate jumps!

7. **🤖 Interactive AI Campus Advisor Copilot**
   - Context-aware intelligent assistant for both **Students** and **Faculty**.
   - Includes 1-click query chips (*e.g. "How do I improve placement readiness?", "Summarize cohort risk factors", "Draft mentor note for Ravi"*).

8. **⚡ 1-Click Judge Demo Accounts & Live Scenario Switcher**
   - **Instant Judge Login**: One-click buttons to sign in as Faculty, Student Ravi (High marks, low coding), or Student Aditya (At-risk).
   - **Scenario Switcher**: Toggle between *Standard Cohort*, *🚨 Pre-Exam Attendance Crisis*, *💼 Placement Season Crunch*, and *🌟 Post-Intervention Success*.

9. **🚀 Student Target Goal Simulator & Growth Roadmap**
   - Students set a target Success Score (e.g., 63 → 75) and receive an algorithmically generated, milestone-based 4-week roadmap with point gains.

10. **📄 Printable Student Diagnostic Card (HTML Export)**
    - One-click export of an executive, confidential diagnostic performance report.

---

## 🧮 Success Score Architecture (0–100 Scale)

Weighted multi-signal formula combining 7 core dimensions:

| Dimension | Weight | Measured Signals |
|---|:---:|---|
| **Academic** | **25%** | 80% CGPA (scaled 0–100) + 20% Backlog mitigation score |
| **Placement** | **20%** | 40% Aptitude + 40% Coding DSA + 20% Mock Interview |
| **Attendance** | **15%** | Class & lab attendance percentage |
| **Skills** | **15%** | 50% Technical domain proficiency + 50% Soft skills |
| **LMS** | **10%** | 40% Login consistency + 60% Assignment completion rate |
| **Engagement** | **10%** | Clubs, college events, hackathons, and certifications |
| **Feedback** | **5%** | Student satisfaction index (1–5 scaled to 0–100) |

- **High Risk:** Success Score < 55
- **Medium Risk:** Success Score 55–65
- **Low Risk:** Success Score > 65

---

## 🛠️ Tech Stack

- **Frontend / UI:** Streamlit, Custom Responsive CSS (Glassmorphism, Dark/Light Themes)
- **Visualizations:** Plotly Express & Plotly Graph Objects (Spider Radar, 2x2 Quadrant, Feature Importance, Histograms)
- **Data Engineering:** Python, pandas, NumPy, SQLite3
- **Machine Learning:** Scikit-Learn (`RandomForestClassifier`, `KMeans`, `StandardScaler`, 5-Fold Cross Validation)

---

## 📂 Project Architecture

```
smart-campus/
├── app.py              # Main Streamlit application (Faculty & Student portals, Copilot, Interventions)
├── pipeline.py         # Data cleaning, scoring, ML models, career matcher, simulator, SQLite CRUD
├── generate_data.py    # Synthetic multi-source campus dataset generator
├── campus.db           # SQLite database storing 7 source tables + interventions table
├── requirements.txt    # Python dependencies
└── data/               # 7 Raw CSV streams (academic, attendance, lms, engagement, placement, skills, feedback)
```

---

## 🚀 Quickstart & Local Setup

```bash
# 1. Activate virtual environment (if using the local venv):
.\venv\Scripts\activate   # Windows

# 2. Install dependencies:
pip install -r requirements.txt

# 3. Launch the platform:
streamlit run app.py
```

### 🔑 Demo Logins
- **1-Click Buttons:** Available directly on the login screen!
- **Faculty:** `faculty` / `faculty@123`
- **Student (Ravi Kumar):** `S0001` / `ravi@0001`
- **Student (Aditya Verma):** `S0015` / `aditya@0015`

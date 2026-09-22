
# AI-Based Sprint Task Assignment System

An AI-based system that recommends optimal sprint task assignments by analyzing employee skills, experience, workload, velocity, and task requirements.

The system uses Machine Learning to predict task success probability and expected completion time, and then uses the Hungarian Algorithm to optimize task assignments while considering employee capacity.

## 🚀 Project Overview

Assigning sprint tasks manually can be difficult when multiple employees have different skills, experience levels, workloads, and capacities.

This project provides an AI-assisted approach to task assignment.

The system:

* Analyzes employee profiles and skills
* Extracts required skills from task descriptions
* Calculates weighted skill matching
* Predicts task success probability
* Predicts expected completion time
* Considers employee workload and capacity
* Uses the Hungarian Algorithm for optimized assignment
* Provides recommended assignments through a Streamlit dashboard
* Provides REST APIs using FastAPI

## 🎯 Objectives

* Improve sprint task assignment
* Match tasks with suitable employee skills
* Consider employee workload and capacity
* Reduce manual assignment effort
* Provide data-driven assignment recommendations
* Support future improvement through actual sprint outcomes

## 🏗️ System Architecture

```text
Employee Data
      +
Task Data
      +
Historical Data
      ↓
Feature Engineering
      ↓
Skill Matching
      ↓
Machine Learning Models
      ↓
┌──────────────────────────────┐
│ Random Forest Classifier     │
│ → Success Probability        │
│                              │
│ Random Forest Regressor      │
│ → Expected Completion Days   │
└──────────────────────────────┘
      ↓
Hungarian Algorithm
      ↓
Capacity-Aware Assignment
      ↓
FastAPI Backend
      ↓
Streamlit Dashboard
      ↓
Recommended Sprint Assignments
```

## 🧠 Machine Learning

The project uses two Random Forest models.

### 1. Random Forest Classifier

Used to predict the probability that an employee will successfully complete a task.

**Output:**

```text
Success Probability
```

Example:

```text
Employee: Arun
Task: API Development

Predicted Success Probability: 82%
```

### 2. Random Forest Regressor

Used to predict the expected number of days required to complete a task.

**Output:**

```text
Expected Completion Days
```

Example:

```text
Predicted Completion Time: 3.4 days
```

## 📊 Features Used

The ML models use features such as:

* Task type
* Skill match
* Experience years
* Average velocity
* Current workload
* Story points

## 🔍 Skill Matching

The system uses a weighted skill matching approach.

Employee skills are represented using proficiency values between `0` and `1`.

Example:

```text
Python: 0.9
SQL: 0.8
React: 0.6
```

The system compares employee skills with the required skills of a task and generates a skill-match score between `0` and `1`.

## ⚙️ Optimization

Machine Learning predicts employee-task suitability, while the Hungarian Algorithm is used for assignment optimization.

The optimizer considers:

* Predicted fit score
* Employee capacity
* Task story points
* Remaining workload

The system first finds preferred employee-task matches and then checks whether the employee has enough remaining capacity.

If the preferred employee does not have enough capacity, the task can be rerouted to another suitable employee.

## 🛠️ Technologies Used

### Programming

* Python

### Machine Learning

* Scikit-learn
* Random Forest Classifier
* Random Forest Regressor
* Feature Engineering

### Backend

* FastAPI
* Pydantic
* REST API
* Uvicorn

### Optimization

* Hungarian Algorithm
* SciPy

### Data Processing

* Pandas
* NumPy

### Frontend

* Streamlit
* Plotly

### Database

* SQLite
* SQLAlchemy

### Development Tools

* Git
* GitHub
* Joblib

### Integration

* Jira Cloud REST API

## 📁 Project Structure

```text
AI-Sprint-Task-Assignment-System/
│
├── backend/
│   ├── app/
│   ├── api/
│   ├── db/
│   ├── integrations/
│   ├── ml/
│   ├── models/
│   ├── data/
│   ├── requirements.txt
│   └── ...
│
├── frontend/
│   ├── dashboard.py
│   └── ...
│
├── README.md
└── .gitignore
```

## 📂 Dataset

The project currently uses synthetic data for development and model training.

The synthetic dataset contains:

* 25 employees
* 20 backlog tasks
* 3000 historical records

The main data files are:

```text
employees.csv
tasks.csv
history.csv
```

### employees.csv

Contains employee information such as:

* Employee ID
* Name
* Role
* Experience
* Skills
* Velocity
* Capacity
* Current workload

### tasks.csv

Contains sprint backlog information such as:

* Task ID
* Sprint ID
* Task title
* Description
* Story points
* Task type
* Priority
* Required skills

### history.csv

Contains historical assignment records used to train the Machine Learning models.

## 🔄 Project Workflow

### Step 1 – Data Ingestion

Employee, task, and historical data are loaded into the system.

### Step 2 – Skill Extraction

Required skills are identified from task descriptions using keyword and synonym matching.

### Step 3 – Feature Engineering

Features such as skill match, experience, velocity, workload, task type, and story points are generated.

### Step 4 – ML Prediction

The Random Forest models predict:

* Success probability
* Expected completion days

### Step 5 – Optimization

The Hungarian Algorithm creates globally preferred task assignments while considering employee capacity.

### Step 6 – Dashboard

The Streamlit dashboard displays:

* Employee data
* Task data
* Recommended assignments
* Fit scores
* Predicted completion time
* Capacity utilization
* Priority-based analysis

### Step 7 – Outcome Recording

Actual sprint outcomes can be recorded for future model improvement.

## 🌐 API Endpoints

### Employees

```text
GET /api/employees
```

Returns employee information.

### Tasks

```text
GET /api/tasks
```

Returns sprint task information.

### Sprint Planning

```text
POST /api/sprint/plan
```

Generates recommended sprint assignments.

### Employee Profile

```text
POST /api/employees/profile
```

Creates or updates employee profile information.

### Employee Profiles

```text
GET /api/employees/profiles
```

Returns stored employee profiles.

### Sprint Outcomes

```text
POST /api/sprint/outcomes
```

Records actual sprint outcomes.

```text
GET /api/sprint/outcomes
```

Retrieves recorded sprint outcomes.

## 🖥️ Running the Project

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/AI-Sprint-Task-Assignment-System.git
```

```bash
cd AI-Sprint-Task-Assignment-System
```

### 2. Create Virtual Environment

```bash
python -m venv venv
```

### 3. Activate Virtual Environment

Windows:

```bash
venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install -r backend/requirements.txt
```

### 5. Run Backend

Go to the backend folder:

```bash
cd backend
```

Run:

```bash
uvicorn app.main:app --reload
```

Backend will be available at:

```text
http://localhost:8000
```

FastAPI documentation:

```text
http://localhost:8000/docs
```

### 6. Run Frontend

Open another terminal and go to the frontend folder:

```bash
cd frontend
```

Run:

```bash
streamlit run dashboard.py
```

The dashboard will be available at:

```text
http://localhost:8501
```

## 🔗 Application Flow

```text
Streamlit Dashboard
        ↓
HTTP Request
        ↓
FastAPI Backend
        ↓
Data Processing
        ↓
ML Prediction
        ↓
Hungarian Optimization
        ↓
Recommended Assignment
        ↓
JSON Response
        ↓
Streamlit Dashboard
```

## 📈 Example Assignment Output

```text
Task: Build REST API

Assigned Employee: Employee 07

Fit Score: 0.82
Predicted Success Probability: 82%
Predicted Completion Time: 3.4 days

Reason:
Predicted 82% success probability, ~3.4 days to complete.
```

## 🔮 Future Improvements

* Train the model using real Jira sprint history
* Add authentication and authorization
* Improve NLP-based skill extraction
* Integrate real-time Jira data
* Add more employee performance features
* Improve model evaluation using real-world outcomes
* Deploy the application to the cloud
* Restrict CORS configuration for production
* Use PostgreSQL for production deployment

## ⚠️ Current Limitations

* The default training dataset is synthetic.
* Jira-imported employees require profile information such as skills, experience, and capacity.
* Authentication is not currently implemented.
* SQLite is used by default for local development.
* CORS is configured broadly for development.

## 👩‍💻 Developer

**Sharoline Lurdu Mariya V**

B.Tech Artificial Intelligence and Data Science
Mailam Engineering College

## ⭐ Project Highlights

* Machine Learning based task assignment
* Random Forest classification and regression
* Weighted skill matching
* Capacity-aware assignment
* Hungarian Algorithm optimization
* FastAPI REST backend
* Streamlit interactive dashboard
* Jira Cloud integration
* Feedback-based future retraining

---

If you find this project useful, feel free to ⭐ the repository.


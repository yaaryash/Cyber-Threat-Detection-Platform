# 🛡️ Cyber Threat Detection Platform

> An end-to-end MLOps pipeline that detects phishing websites
> using machine learning — built with production-grade engineering practices.

---

## 🔍 What This Project Does

Takes a CSV of website features (SSL state, URL length, domain age,
redirect behaviour etc.) and predicts whether each website is a
**phishing attempt or legitimate**.

Built as a complete MLOps system — not just a model, but a full
pipeline from raw data to deployed API.

---

## 🏗️ How It Works

```
MongoDB Atlas (raw data storage)
       ↓
Data Ingestion → Data Validation → Data Transformation
       ↓
Model Training (comparing multiple ML models)
       ↓
Best model saved → Served via FastAPI REST API
       ↓
CI/CD via GitHub Actions → Docker → AWS ECR → EC2
```

---

## 🧰 Tech Stack

| Layer | Tools Used |
|---|---|
| Data Storage | MongoDB Atlas |
| ML Models | Scikit-learn, XGBoost |
| Experiment Tracking | MLflow + DagsHub |
| API | FastAPI |
| Containerization | Docker |
| Cloud Infrastructure | AWS S3, ECR, EC2 |
| CI/CD | GitHub Actions |

---

## 📊 Model Performance
<!-- Will be updated after training runs -->

| Model | F1 Score | Precision | Recall |
|---|---|---|---|
| XGBoost | -- | -- | -- |
| Random Forest | -- | -- | -- |
| Gradient Boosting | -- | -- | -- |
| Decision Tree | -- | -- | -- |
| Logistic Regression | -- | -- | -- |

---

## ⚙️ Setup

```bash
# Clone
git clone https://github.com/yaaryash/Cyber-Threat-Detection-Platform.git
cd cyber-threat-detection-platform

# Create environment
conda create -p venv python==3.10 -y
conda activate venv/

# Install
pip install -r requirements.txt

# Environment variables
cp .env.example .env
# Fill in your credentials in .env
```

---

## 🚀 Run

```bash
# Step 1: Push data to MongoDB
python push_data.py

# Step 2: Train the pipeline
python main.py

# Step 3: Start API
python app.py
# Open http://localhost:8000/docs
```

---

## 👤 Author

**Yash** — ML Engineer
[LinkedIn](https://www.linkedin.com/in/yaaryash) · [GitHub](https://github.com/yaaryash)
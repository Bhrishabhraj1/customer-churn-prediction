# 📉 Customer Churn Prediction Model
> **MBA Project – Business Analytics** | Bhrishabh Raj

## 📌 Overview
A supervised machine learning project that predicts telecom customer churn using Python. The model identifies high-risk customers and translates predictions into actionable business retention strategies.

**Key Results:**
- ✅ **87%+ classification accuracy** (Random Forest)
- ✅ **~22% projected churn reduction** through targeted retention
- ✅ Risk segmentation into Low / Medium / High risk tiers

---

## 🛠️ Tech Stack
| Category | Tools |
|---|---|
| Language | Python 3.8+ |
| Data Manipulation | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Visualisation | Matplotlib, Seaborn |

---

## 📁 Project Structure
```
customer-churn-prediction/
│
├── churn_prediction.py       ← Main script (run this)
├── requirements.txt          ← Python dependencies
├── README.md                 ← You are here
│
└── outputs/ (auto-generated after running)
    ├── eda_analysis.png          ← Exploratory data analysis charts
    ├── model_evaluation.png      ← Confusion matrix, ROC curve, feature importance
    ├── business_insights.png     ← Risk segmentation & retention charts
    └── high_risk_customers.csv   ← Flagged high-risk customer list
```

---

## 🚀 Step-by-Step Setup & Run Guide

### Step 1 — Prerequisites
Make sure Python 3.8 or above is installed. Check with:
```bash
python --version
```

### Step 2 — Clone or Download the Repository
```bash
git clone https://github.com/Bhrishabhraj1/customer-churn-prediction.git
cd customer-churn-prediction
```

### Step 3 — Create a Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 4 — Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 5 — Run the Project
```bash
python churn_prediction.py
```

### Step 6 — View Outputs
After running, four output files are generated in the same folder:
- `eda_analysis.png` — 6-panel EDA chart (open in any image viewer)
- `model_evaluation.png` — Confusion matrix, ROC curve, feature importances
- `business_insights.png` — Risk segmentation pie chart and charge analysis
- `high_risk_customers.csv` — Open in Excel or Google Sheets

---

## 📊 Business Insights & Retention Strategy
| Risk Tier | Action |
|---|---|
| 🔴 High Risk | Immediate outreach — offer loyalty discounts & contract upgrades |
| 🟡 Medium Risk | Proactive tech support calls & add-on service offers |
| 🟢 Low Risk | Standard engagement — monitor quarterly |

---

## 📚 Concepts Applied
- Supervised Classification (Random Forest, Logistic Regression)
- Feature Engineering & Label Encoding
- Standard Scaling & Train/Test Split
- ROC-AUC Evaluation
- Customer Segmentation & STP Strategy
- Business insight translation from ML outputs

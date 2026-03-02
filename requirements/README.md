# 🌍 AI-Based Carbon Emission Prediction & Explainability System

An end-to-end **Machine Learning + Explainable AI (XAI)** project that predicts carbon-emission-related outcomes and explains model decisions using **SHAP**, **LIME**, and **Permutation Importance**.  
The project also includes an **interactive Streamlit web application** for easy visualization and interpretation.

---

## 📌 Project Motivation
Most AI models act as **black boxes**, especially in environmental and policy-driven domains where trust and transparency are critical.

This project focuses on:
- Accurate prediction using machine learning
- Clear interpretation of model behavior
- Making AI **explainable, ethical, and reliable**

---

## 🎯 Objectives
- Build a complete ML pipeline from raw data to predictions
- Perform data preprocessing and feature handling
- Train and evaluate a machine learning model
- Apply Explainable AI techniques
- Visualize explanations through a Streamlit web app

---

## 🧠 Key Features
✔ End-to-end machine learning workflow  
✔ Interactive Streamlit dashboard  
✔ SHAP global & local explanations  
✔ LIME instance-level explanation  
✔ Permutation feature importance  
✔ Modular and scalable code structure  
✔ Ready-to-use trained model  

---

## 📂 Project Structure
AI_Project/
│
├── data/
│ ├── raw/
│ │ └── carbon_data.csv
│ └── processed/
│ ├── train.csv
│ └── test.csv
│
├── models/
│ └── model.joblib
│
├── notebooks/
│ └── EDA.ipynb
│
├── reports/
│ ├── lime_local.txt
│ ├── permutation_importance.csv
│ ├── shap_local.html
│ └── shap_summary.png
│
├── requirements/
│ ├── requirements.txt
│ └── README.md
│
├── src/
│ ├── app.py # Streamlit application
│ ├── data_prep.py # Data preprocessing
│ ├── train_model.py # Model training
│ ├── explain.py # XAI explanations
│ ├── pdp_plot.py # Partial dependence plots
│ └── init.py
│
└── README.md

---

## 🛠️ Technologies Used
- **Python**
- **Pandas & NumPy** – Data processing
- **Scikit-learn** – Machine learning
- **SHAP** – Model explainability
- **LIME** – Local explanations
- **Matplotlib / Seaborn** – Visualization
- **Streamlit** – Web application
- **Joblib** – Model saving/loading
- **Jupyter Notebook** – EDA

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone <your-repository-url>
cd AI_Project

✅ Install Dependencies
pip install -r requirements/requirements.txt

✅ Step 1: Data Preprocessing
python src/data_prep.py

✅ Step 2: Train the Model
python src/train_model.py

✅ Step 3: Generate Explainability Reports
python src/explain.py

✅ Step 4: Run Streamlit Application (Windows)
python -m streamlit run src/app.py
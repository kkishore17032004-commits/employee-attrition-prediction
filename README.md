# Employee Attrition Prediction System

Production-quality **Streamlit** dashboard for HR analytics and employee attrition prediction.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)
![License](https://img.shields.io/badge/License-MIT-green)

## Overview

Enterprise HR analytics platform that predicts employee attrition risk using machine learning. The dashboard provides interactive visualizations, model comparison, feature importance analysis, and HR retention recommendations.

## Project Structure

```text
Employee_Attrition_Prediction/
│
├── app.py                                      # Main Streamlit application
├── attrition_model.pkl                         # Trained model bundle (joblib)
├── DATASET.csv                                 # IBM HR dataset
├── train_model.py                              # Model training script
├── requirements.txt                            # Python dependencies
├── README.md                                   # Project documentation
│
├── assets/
│     company_logo.png                          # Company logo
│
├── images/                                     # Additional image assets
└── reports/                                    # Generated prediction reports
```

## Features

- **Dashboard** – KPI cards and 12+ interactive Plotly charts
- **Employee Prediction** – Full feature form with probability gauge and risk classification
- **Analytics** – Filterable workforce analytics by department, gender, role, and more
- **Feature Importance** – Top 10 Random Forest features with interactive bar chart
- **Model Comparison** – Accuracy, precision, recall, F1, confusion matrix, and ROC curves
- **HR Recommendations** – Risk-based retention action engine
- **CSV Download** – Export prediction results

## Quick Start

### 1. Install Dependencies

```bash
cd Employee_Attrition_Prediction
python -m pip install -r requirements.txt
```

### 2. Train Model (if `attrition_model.pkl` is missing)

```bash
python train_model.py
```

### 3. Run the Application

```bash
python -m streamlit run app.py
```

> **Note:** If `streamlit` is not recognized, use `python -m streamlit run app.py` instead. This works even when Python's Scripts folder is not on your PATH.

The app opens at `http://localhost:8501`.

## Model Details

| Model               | Accuracy |
|---------------------|----------|
| Random Forest       | 79.93%   |
| Decision Tree       | 75.17%   |
| Logistic Regression | 74.49%   |

- **Best Model:** Random Forest (200 estimators)
- **Preprocessing:** Label encoding, SMOTE oversampling
- **Features:** 30 input attributes after dropping constant columns

## Technologies

- Python, Streamlit, Pandas, NumPy
- Scikit-learn, imbalanced-learn, Joblib
- Plotly, Matplotlib

## Dataset

IBM HR Analytics Employee Attrition dataset with 1,470 employee records and 35 attributes covering demographics, job satisfaction, compensation, and tenure.

## License

MIT License – Free for educational and enterprise use.

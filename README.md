# Healthcare ETL Pipeline with Diabetes Prediction

> Academic project - Demonstrates an end-to-end ETL pipeline and a reproducible ML workflow on clinical data.

---

## Project Overview

An end-to-end ETL pipeline for medical data combined with a machine learning model
predicting Type 2 Diabetes. Focus is on data quality, reproducibility, and interpretable ML.

**Goals:**
- Clean, validate, and transform clinical data (ETL pipeline)
- Engineer auditable features for ML
- Train a reproducible ML pipeline (Logistic Regression + GridSearchCV)
- Evaluate using stratified test set, classification metrics, and ROC-AUC
- Save all artefacts and metadata for reproducibility

---

## Dataset

- **Source:** [Kaggle - Pima Indians Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
- **Target variable:** `Outcome` (0 = no diabetes, 1 = diabetes)
- **Features:** Numeric (Glucose, BMI, BloodPressure, etc.) + engineered features

---

## Technologies

| Area | Tools |
|---|---|
| Language | Python 3.9+ |
| ETL / Data | pandas, numpy |
| ML | scikit-learn (pipelines, GridSearchCV, metrics) |
| Visualizations | matplotlib, seaborn |
| Serialization | joblib, JSON |
| Experimentation | Jupyter Notebook |

---

## Machine Learning

- **Model:** Logistic Regression (baseline) + GridSearchCV hyperparameter tuning
- **Preprocessing:** StandardScaler (numeric), OneHotEncoder (categorical)
- **Split:** Stratified train/test split
- **Evaluation:** Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix

---

## Results

| Metric | Score |
|---|---|
| Accuracy | 75.3% |
| Precision (class 1) | 62.1% |
| Recall (class 1) | 75.9% |
| F1-Score (class 1) | 68.3% |
| ROC-AUC | 0.8256 |

ROC-AUC of 0.83 indicates strong discriminative ability despite class imbalance.

---

## Project Structure
```
├── data/
│ ├── raw/ # original Kaggle dataset
│ ├── processed/ # cleaned & feature-engineered data
│ └── cleaned/ # model-ready parquet + metadata
├── notebooks/
│ ├── 01_extract_eda.ipynb 
│ ├── 02_data_cleaning.ipynb
│ ├── 03_feature_engineering.ipynb
│ ├── 05_model_training.ipynb 
│ └── 06_evaluation.ipynb 
├── src/
│ └── etl_pipeline.py # executable ETL pipeline
├── results/
│ ├── metrics/ # CSV reports
│ ├── models/ # serialized models + training metadata
│ └── visualizations/ # confusion_matrix.png, roc_curve.png
├── requirements.txt
└── README.md
```

---

## How to Run

```bash
git clone https://github.com/damian-balinski/healthcare-diabetes-etl
cd healthcare-diabetes-etl
pip install -r requirements.txt

# Run ETL pipeline
python -m src.etl_pipeline

# Launch notebooks
jupyter notebook
```

---

## Author

Damian Baliński 
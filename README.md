# Healthcare ETL Pipeline with Diabetes Prediction (AI)

## Project Overview
This project implements an end-to-end ETL pipeline for medical data and a machine learning model predicting Type 2 Diabetes.
Focus is on data quality, reproducibility, and interpretable ML, designed as both an academic and portfolio project for a junior data engineer in healthcare.

## Project Goals
- Clean, validate, and transform clinical data (ETL pipeline)
- Engineer auditable features for ML
- Train a reproducible ML pipeline (Logistic Regression)
- Evaluate using stratified test set, classification metrics, and ROC-AUC
- Save all artefacts and metadata for reproducibility

## Dataset
- **Source:** Kaggle – Pima Indians Diabetes Dataset
- **Target variable:** Outcome (0 = no diabetes, 1 = diabetes)
- **Features:** Numeric (Glucose, BMI, BloodPressure, etc.) + engineered features

---

## Technologies
- Python 3.9+ (core language)
- pandas / numpy (ETL and data processing)
- scikit-learn (ML pipelines, preprocessing, training, evaluation)
- matplotlib / seaborn (visualizations)
- joblib / JSON (model & metadata serialization)
- Jupyter Notebook (experimentation and documentation)

---

## Machine Learning
- Logistic Regression baseline, stratified train/test split
- Preprocessing: StandardScaler (numeric), OneHotEncoder (categorical)
- Hyperparameter tuning: GridSearchCV
- Evaluation metrics: Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix

---

## Project Structure 

- project-root/  
data/  
raw/ # original dataset  
processed/ # cleaned + feature-engineered data  
cleaned/ # model-ready dataset  
  
- notebooks/  
01_exploration.ipynb  
02_cleaning.ipynb  
03_feature_engineering.ipynb  
05_model_training.ipynb  
06_evaluation.ipynb  
  
- src/  
etl_pipeline.py # executable ETL pipeline  
  
- results/  
metrics/ # CSV metrics & reports  
visualizations/ # confusion matrix, ROC curve  
models/ # serialized ML models & metadata
  
README.md  
requirements.txt

---

## How to Run

```bash
git clone https://github.com/damian-balinski/healthcare-diabetes-etl

cd healthcare-diabetes-etl

pip install -r requirements.txt 

python -m src.etl_pipeline

jupyter notebook 
```
----

## Author 
Damian Baliński - 123702  gr.3 
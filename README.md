# HeartAttackPredictionML

# 🫀 GP2: Heart Attack Prediction

This repository contains the solution to the **GP2: Heart Attack Prediction** challenge hosted on [Kaggle](https://kaggle.com/competitions/gp-2-heart-attack-prediction). The goal is to build a classification model that can predict the risk of heart disease based on a patient’s demographic and medical features.

---

## 📌 Project Overview

**Objective**  
Given a set of features describing a patient's physical health and demographics, predict whether the patient is likely to experience heart disease.

**Evaluation Metric**  
The competition is evaluated using the **mean F1-score**, focusing on the balance between precision and recall in the predictions.

---

## 🧠 Features

The dataset includes 11 input features:

| Feature        | Description |
|----------------|-------------|
| Age            | Age of the patient (years) |
| Sex            | Male (`M`) or Female (`F`) |
| ChestPainType  | Chest pain type: `TA`, `ATA`, `NAP`, `ASY` |
| RestingBP      | Resting blood pressure (mm Hg) |
| Cholesterol    | Serum cholesterol (mg/dl) |
| FastingBS      | Fasting blood sugar > 120 mg/dl (`1` if true, `0` otherwise) |
| RestingECG     | ECG results: `Normal`, `ST`, or `LVH` |
| MaxHR          | Maximum heart rate achieved |
| ExerciseAngina | Exercise-induced angina: `Y` or `N` |
| Oldpeak        | Depression of ST segment induced by exercise |
| ST_Slope       | Slope of the peak exercise ST segment: `Up`, `Flat`, `Down` |

---

## ⚙️ Model Pipeline

The following steps were used to train and optimize the model:

1. **Preprocessing**
   - Standard scaling for numerical features
   - One-hot encoding for categorical features

2. **Modeling**
   - Classifier: `XGBoost (XGBClassifier)`
   - Hyperparameter tuning via `GridSearchCV` with 5-fold `StratifiedKFold`
   - Scoring metric: `macro F1-score`

3. **Post-processing**
   - Calibrated probabilities using `CalibratedClassifierCV`
   - Optimal decision threshold selected via cross-validated predictions to maximize F1 score

4. **Submission**
   - Final predictions generated on test set using the optimized and calibrated model
   - Output format: `PatientID`, `HeartDisease` (0 or 1)

---

## 📁 Files

| File Name             | Description |
|-----------------------|-------------|
| `train.csv`           | Training dataset with labels |
| `test_X.csv`          | Test dataset without labels |
| `y_predict.csv`       | Final submission file |
| `model_pipeline.py`   | Full modeling pipeline, including preprocessing, tuning, calibration, and prediction |
| `README.md`           | Project overview and documentation |

---



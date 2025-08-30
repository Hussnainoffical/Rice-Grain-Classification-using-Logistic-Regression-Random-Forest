# 🌾 Rice Grain Classification using ML (Logistic Regression vs Random Forest)

This project is part of my learning journey after completing the **Google Machine Learning Crash Course: Classification Badge** 🎉.  
It focuses on applying **supervised machine learning** techniques to classify rice varieties (Cammeo vs Osmancik) based on physical features.

---

## 🚀 Project Overview
- Dataset: [Rice Cammeo vs Osmancik](https://download.mlcc.google.com/mledu-datasets/Rice_Cammeo_Osmancik.csv)  
- Task: **Binary Classification** of rice grains  
- Models used:
  - **Logistic Regression** (with scaling)
  - **Random Forest Classifier**
- Metrics: Accuracy, Precision, Recall, F1 Score, ROC AUC
- Visualizations: ROC Curve, Precision-Recall Curve
- Validation: 5-Fold Cross Validation

---

## ⚙️ Workflow
1. **Data Preprocessing**
   - Load dataset from Google’s MLCC repo
   - Automatic detection of target column (`Class`)
   - Label Encoding for categorical target
   - Train-test split (80/20)

2. **Model Building**
   - Logistic Regression (with `StandardScaler`)
   - Random Forest Classifier (200 trees)

3. **Evaluation**
   - Metrics: Accuracy, Precision, Recall, F1, ROC AUC
   - Confusion Matrix
   - ROC Curve & Precision-Recall Curve
   - Cross-validation mean AUC

---

## 📊 Results
- Both models performed strongly on classification.
- Random Forest generally achieved higher **AUC** compared to Logistic Regression.
- Visual comparisons highlight differences in precision-recall tradeoffs.

---


## 🛠️ Tech Stack
- **Python**
- **NumPy**, **Pandas**
- **Matplotlib**
- **Scikit-learn**

---

## 🎯 Key Learnings
- Hands-on practice with **Logistic Regression vs Random Forest**.
- Understanding the role of **scaling** in linear models.
- Evaluation with **ROC/PR curves** and **cross-validation**.
- Building reproducible ML pipelines using `sklearn.Pipeline`.

---

## 📜 Acknowledgements
- Google’s **[Machine Learning Crash Course](https://developers.google.com/machine-learning/crash-course)**  
- Scikit-learn documentation  

---

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
       accuracy_score, roc_auc_score, recall_score, precision_score,
       confusion_matrix, f1_score, roc_curve, precision_recall_curve
)


URL = "https://download.mlcc.google.com/mledu-datasets/Rice_Cammeo_Osmancik.csv"
r = requests.get(URL)
with open("File1.csv", 'w', encoding='utf-8') as f:
    f.write(r.text)
df = pd.read_csv("File1.csv")

print(df.head())
print(df.describe())

all_columns = df.columns
candidates = []

for c in all_columns:

       col_lower = c.lower()
       if ("class" in col_lower or
               "species" in col_lower or
               "label" in col_lower or
               "target" in col_lower or
               "type" in col_lower):
              candidates.append(c)

if len(candidates) > 0:
       target_col = candidates[0]

else:
       target_col = all_columns[-1]

print("Using target column:", target_col)

y_raw = df[target_col]
le = LabelEncoder()
y = le.fit_transform(y_raw)
print("Label mapping:", dict(zip(le.classes_, le.transform(le.classes_))))

X = df.drop(columns=[target_col])

X_train, X_test, Y_train, Y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

print("Train size", X_train.shape, "Test size", X_test.shape)


pipeline_log = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(max_iter=1000, solver='liblinear'))
])

pipeline_rf = Pipeline([
    ('rf', RandomForestClassifier(n_estimators=200, random_state=42))
])


pipeline_log.fit(X_train, Y_train)
pipeline_rf.fit(X_train, Y_train)

y_log_proba = pipeline_log.predict_proba(X_test)[:, 1]
y_rf_proba = pipeline_rf.predict_proba(X_test)[:, 1]

y_log_predict = (y_log_proba > 0.5).astype(int)
y_rf_predict = (y_rf_proba > 0.5).astype(int)

def show_results(model_name, y_true, y_pred, y_proba):
    print("\nModel:", model_name)
    print("Accuracy :", accuracy_score(y_true, y_pred))
    print("Precision:", precision_score(y_true, y_pred))
    print("Recall   :", recall_score(y_true, y_pred))
    print("F1 Score :", f1_score(y_true, y_pred))
    print("ROC AUC  :", roc_auc_score(y_true, y_proba))
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))

show_results("Logistic Regression", Y_test, y_log_predict, y_log_proba)
show_results("Random Forest", Y_test, y_rf_predict, y_rf_proba)


fpr_log, tpr_log, thr_log = roc_curve(Y_test, y_log_proba)
fpr_rf, tpr_rf, thr_rf   = roc_curve(Y_test, y_rf_proba)

auc_log = roc_auc_score(Y_test, y_log_proba)
auc_rf  = roc_auc_score(Y_test, y_rf_proba)

plt.figure(figsize=(8,6))
plt.plot(fpr_log, tpr_log, label=f'Logistic (AUC = {auc_log:.3f})')
plt.plot(fpr_rf,  tpr_rf,  label=f'RandomForest (AUC = {auc_rf:.3f})')
plt.plot([0,1],[0,1],'k--', label='chance')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate (Recall)')
plt.title('ROC Curve Comparison')
plt.legend()
plt.grid(True)
plt.show()


prec_log, rec_log, _ = precision_recall_curve(Y_test, y_log_proba)
prec_rf, rec_rf, _   = precision_recall_curve(Y_test, y_rf_proba)

plt.figure(figsize=(8,6))
plt.plot(rec_log, prec_log, label="Logistic")
plt.plot(rec_rf,  prec_rf,  label="RandomForest")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.grid(True)
plt.show()

cv_auc_log = cross_val_score(pipeline_log, X, y, scoring='roc_auc', cv=5).mean()
cv_auc_rf  = cross_val_score(pipeline_rf,  X, y, scoring='roc_auc', cv=5).mean()
print("CV AUC Logistic:", cv_auc_log)
print("CV AUC RandomForest:", cv_auc_rf)



import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

train_df = pd.read_csv("UNSW_NB15_training-set.csv")
test_df = pd.read_csv("UNSW_NB15_testing-set.csv")
data = pd.concat([train_df, test_df], ignore_index=True)

features = ["dur", "proto", "service", "state", "spkts", "dpkts", "sbytes", "dbytes"]
X = data[features]
y = data["label"]  # 0=норм трафик, 1=атака

label_encoders = {}
for col in ["proto", "service", "state"]:
    le = LabelEncoder()
    X.loc[:, col] = le.fit_transform(X[col])
    label_encoders[col] = le
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred) * 100
print(f"Accuracy: {accuracy:.2f}%")
report = classification_report(y_test, y_pred, output_dict=True)
print("Classification Report:")
for label, metrics in report.items():
    if isinstance(metrics, dict):
        print(f"Class {label}: Precision: {metrics['precision'] * 100:.2f}%, Recall: {metrics['recall'] * 100:.2f}%, F1-score: {metrics['f1-score'] * 100:.2f}%")
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)
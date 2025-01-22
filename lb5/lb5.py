import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.metrics import r2_score, mean_absolute_error
from tensorflow.keras.utils import Sequence

try:
    train_data = pd.read_csv("train.csv", sep=',')
    test_data = pd.read_csv("test.csv", sep=';')
except Exception as e:
    print(f"Error reading CSV files: {e}")
    exit()

def preprocess_data(data, is_train=True, label_encoders=None, scaler=None):
    data = data.copy()
    data.fillna(data.median(numeric_only=True), inplace=True)
    if is_train:
        label_encoders = {}
        for column in ['POSTED_BY', 'BHK_OR_RK', 'ADDRESS']:
            if column in data.columns:
                le = LabelEncoder()
                data[column] = le.fit_transform(data[column].astype(str))
                label_encoders[column] = le
    else:
        for column, le in label_encoders.items():
            if column in data.columns:
                known_classes = set(le.classes_)
                data[column] = data[column].astype(str).apply(lambda x: x if x in known_classes else 'unknown')
                le.classes_ = np.append(le.classes_, 'unknown')
                data[column] = le.transform(data[column])
    numeric_columns = ['SQUARE_FT', 'LONGITUDE', 'LATITUDE']
    for col in numeric_columns:
        if col not in data.columns:
            print(f"Warning: Missing column '{col}' in data. Filling with zeros.")
            data[col] = 0
    if is_train:
        scaler = StandardScaler()
        data[numeric_columns] = scaler.fit_transform(data[numeric_columns])
    else:
        if scaler:
            data[numeric_columns] = scaler.transform(data[numeric_columns])
    if is_train:
        X = data.drop(['TARGET(PRICE_IN_LACS)'], axis=1)
        y = data['TARGET(PRICE_IN_LACS)']
        return X, y, label_encoders, scaler
    else:
        return data, None, None

X_train, y_train, train_encoders, scaler = preprocess_data(train_data)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
val_predictions = model.predict(X_val)
r2 = r2_score(y_val, val_predictions)
mae = mean_absolute_error(y_val, val_predictions)
print(f"Validation R2 Score: {r2:.4f}")
print(f"Validation MAE: {mae:.4f}")
average_price = np.mean(y_val)
accuracy = 100 - (mae / average_price * 100)
print(f"ИИИИТАААААК точность модели: Model Accuracy = {accuracy:.2f}%")

test_data_processed, _, _ = preprocess_data(test_data, is_train=False, label_encoders=train_encoders, scaler=scaler)
test_predictions = model.predict(test_data_processed.drop(['TARGET(PRICE_IN_LACS)'], axis=1, errors='ignore'))
test_data['TARGET(PRICE_IN_LACS)'] = test_predictions
test_data.to_csv("test_predictions.csv", index=False)
print("Гадание по ладошке лежит в 'test_predictions.csv'")

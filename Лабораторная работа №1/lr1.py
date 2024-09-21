import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

# Загрузка данных
df = pd.read_csv('fixed_values_ds.csv')

# Если есть категориальные переменные, закодируем их
labelencoder = LabelEncoder()

# Предположим, что 'ssl_state' и 'domain_registered' — это категориальные переменные
df['ssl_state'] = labelencoder.fit_transform(df['ssl_state'])
df['domain_registered'] = labelencoder.fit_transform(df['domain_registered'])

# Разделение данных на признаки и целевую переменную
X = df.drop('result', axis=1)  # Все столбцы кроме 'result' — признаки
y = df['result']  # Целевая переменная

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Обучение модели Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Прогнозирование
y_pred = model.predict(X_test)

# Оценка модели
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
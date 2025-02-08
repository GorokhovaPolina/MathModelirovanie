import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

data = pd.read_csv("Fitness_trackers.csv")
data.dropna(inplace=True)
features = ["Brand Name", "Device Type", "Selling Price", "Original Price", "Display", "Rating (Out of 5)", "Strap Material", "Average Battery Life (in days)"]
df = data[features].copy()
df.loc[:, "Selling Price"] = df["Selling Price"].str.replace(",", "").astype(float)
df.loc[:, "Original Price"] = df["Original Price"].str.replace(",", "").astype(float)
num_features = ["Selling Price", "Original Price", "Rating (Out of 5)", "Average Battery Life (in days)"]
cat_features = ["Brand Name", "Device Type", "Display", "Strap Material"]
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
])

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
pipeline = make_pipeline(preprocessor, kmeans)
pipeline.fit(df)
labels = pipeline.named_steps["kmeans"].labels_
data["Cluster"] = labels
inertia = kmeans.inertia_
print(f"Инерция модели: {inertia}")
accuracy = (1 - inertia / np.max(df[num_features].var())) * 100
print(f"Точность модели: {accuracy:.2f}%")

# первые строки с кластерами
print(data[["Brand Name", "Model Name", "Selling Price", "Rating (Out of 5)", "Cluster"]].head())

output_file = 'data_with_clusters.xlsx'
data[["Brand Name", "Model Name", "Selling Price", "Rating (Out of 5)", "Cluster"]].to_excel(output_file, index=False)
print(f"Таблица с кластерами сохранена в файл {output_file}")

plt.figure(figsize=(8, 6))
plt.scatter(data["Selling Price"], data["Rating (Out of 5)"], c=labels, cmap="viridis", alpha=0.7)
plt.xlabel("Selling Price", fontfamily = "Times New Roman", fontsize=14)
plt.ylabel("Rating (Out of 5)", fontfamily = "Times New Roman", fontsize=14)
plt.title("Кластеры фитнес-браслетов", fontfamily = "Times New Roman")
plt.colorbar(label="Cluster")
plt.xticks(fontfamily = "Times New Roman", fontsize=7, rotation=90)
plt.yticks(fontfamily = "Times New Roman", fontsize=7)
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()
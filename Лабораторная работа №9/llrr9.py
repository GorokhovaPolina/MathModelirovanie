import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FactorAnalysis
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from minisom import MiniSom  

file_path_1 = 'дискуссия.xlsx'
file_path_2 = 'дискуссия2.xlsx'
data1 = pd.read_excel(file_path_1, sheet_name='Выборка для анализа')
data2 = pd.read_excel(file_path_2, sheet_name='Статистика')
features1 = data1[['а', 'е', 'у', 'э', 'о', 'и', '!', '?', 'Кол-во', 'Длина']]
features2 = data2[['а', 'е', 'и', 'о', 'у', 'э', '!', '?']]
features1 = features1.dropna()
features2 = features2.dropna()
features1_grouped = features1.groupby(data1['Автор']).mean()
features2_grouped = features2.groupby(data2['Автор']).mean()
all_columns = list(set(features1_grouped.columns) | set(features2_grouped.columns))
features1_grouped = features1_grouped.reindex(columns=all_columns, fill_value=0)
features2_grouped = features2_grouped.reindex(columns=all_columns, fill_value=0)
merged_data = pd.concat([features1_grouped, features2_grouped], axis=0)
scaler = StandardScaler()
scaled_features = scaler.fit_transform(merged_data)

fa = FactorAnalysis(n_components=12, random_state=42)
factor_scores = fa.fit_transform(scaled_features)
print(factor_scores.shape)

num_factors = factor_scores.shape[1]
final_data = pd.DataFrame(factor_scores, columns=[f'Factor_{i+1}' for i in range(num_factors)])
final_data['Author'] = merged_data.index

kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(final_data.drop(columns='Author'))
final_data['Cluster'] = kmeans.labels_

test_data = pd.read_excel(file_path_2, sheet_name='Стат по тест выб без сообщений')
test_data.columns = test_data.columns.str.strip()
print(test_data.columns)
test_features = test_data[['а', 'е', 'у', 'э', 'о', 'и', '!', '?', 'Количество_постов', 'Средняя_длина_сообщения']]
test_features = test_features.reindex(columns=features1_grouped.columns, fill_value=0)
scaled_test_features = scaler.transform(test_features)

test_factor_scores = fa.transform(scaled_test_features)
print(test_factor_scores.shape)
num_factors = test_factor_scores.shape[1]
test_final_data = pd.DataFrame(test_factor_scores, columns=[f'Factor_{i+1}' for i in range(num_factors)])
test_final_data['Author'] = test_data['Код']
test_final_data = pd.DataFrame(test_factor_scores, columns=[f'Factor_{i+1}' for i in range(10)])
test_final_data['Author'] = test_data['Код']

test_final_data['Cluster'] = kmeans.predict(test_final_data.drop(columns='Author'))

output_file = 'test_data_with_clusters.xlsx'
test_final_data[['Author', 'Cluster']].to_excel(output_file, index=False)
print(f"Таблица с кластерами сохранена в файл {output_file}")

som_size = 30
som = MiniSom(som_size, som_size, final_data.shape[1] - 1, sigma=1.0, learning_rate=0.5)
som.train(final_data.drop(columns='Author').values, 100, verbose=True)
plt.figure(figsize=(14, 12))

plt.imshow(som.distance_map().T, cmap='Spectral', origin='lower', interpolation='bilinear')

for x, t in zip(final_data.drop(columns='Author').values, final_data['Cluster']):
    w = som.winner(x) 
    plt.text(w[0] + 0.5, w[1] + 0.5, str(t), color=plt.cm.viridis(t / 2), 
             fontweight='bold', ha='center', va='center', fontsize=10)

plt.title('Карта Кохонена')
plt.xlabel('X-координата')
plt.ylabel('Y-координата')
plt.colorbar()
plt.show()


# silhouette_scores = []
# for k in range(2, 25):  # меняю число кластеров
#     kmeans = KMeans(n_clusters=k, random_state=42)
#     kmeans.fit(final_data.drop(columns='Author'))
#     score = silhouette_score(final_data.drop(columns='Author'), kmeans.labels_)
#     silhouette_scores.append(score)

# plt.plot(range(2, 25), silhouette_scores)
# plt.xlabel('Количество кластеров')
# plt.ylabel('Сила силуэта')
# plt.title('Метод силуэта')
# plt.show()
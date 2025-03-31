import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Загружаем данные
iris = load_iris()
X = iris.data

# Русские названия признаков
feature_names_rus = [
    "Длина чашелистика",
    "Ширина чашелистика",
    "Длина лепестка",
    "Ширина лепестка"
]

# Масштабируем данные
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Находим оптимальное количество кластеров методом локтя
inertia = []
k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Определяем оптимальное k по методу локтя (вторая производная)
optimal_k = k_range[np.argmin(np.diff(inertia, 2))]

# Визуализация метода локтя
plt.figure(figsize=(8, 6))
plt.plot(k_range, inertia, marker='o', label="Инерция")
plt.axvline(optimal_k, linestyle="--", color="r", label=f"Оптимальное k = {optimal_k}")
plt.title("Метод локтя")
plt.xlabel("Количество кластеров")
plt.ylabel("Сумма квадратов расстояний")
plt.legend()
plt.show()

# Запускаем K-Means с найденным optimal_k
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
labels = kmeans.fit_predict(X_scaled)
centroids = kmeans.cluster_centers_

# Визуализация всевозможных пар признаков
pairs = list(combinations(range(X.shape[1]), 2))  # Всевозможные комбинации признаков

plt.figure(figsize=(12, 8))

for i, (x_idx, y_idx) in enumerate(pairs, 1):
    plt.subplot(2, 3, i)
    for cluster in range(optimal_k):
        plt.scatter(X_scaled[labels == cluster, x_idx], X_scaled[labels == cluster, y_idx],
                    label=f'Кластер {cluster + 1}')

    # Рисуем центроиды
    plt.scatter(centroids[:, x_idx], centroids[:, y_idx], c='black', marker='x', s=100, label='Центроиды')

    plt.xlabel(feature_names_rus[x_idx])
    plt.ylabel(feature_names_rus[y_idx])
    plt.legend()
    plt.title(f'Проекция: {feature_names_rus[x_idx]} vs {feature_names_rus[y_idx]}')

plt.tight_layout()
plt.show()

print(f"Оптимальное число кластеров: {optimal_k}")
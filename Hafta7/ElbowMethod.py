import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler

# 1. Veri setini yükleme
data = load_wine()
X = data.data
y = data.target

# 2. Ölçekleme (Standardization)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Elbow Method: inertia değerlerini hesapla
inertias = []
k_values = range(1, 11)

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)

# 4. Elbow grafiğini çiz
plt.figure(figsize=(8, 4))
plt.plot(k_values, inertias, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.grid(True)
plt.show()

# 5. En uygun k ile KMeans modelini yeniden eğit
optimal_k = 3  # Elbow grafiğine göre belirlenen k değeri
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
labels = kmeans.fit_predict(X_scaled)

# 6. Kümeleme sonuçlarını görselleştir (ilk iki özellik üzerinden)
plt.figure(figsize=(8, 5))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap='viridis')
plt.title(f'KMeans Clustering (k={optimal_k})')
plt.xlabel('Feature 1 (Scaled)')
plt.ylabel('Feature 2 (Scaled)')
plt.grid(True)
plt.show()

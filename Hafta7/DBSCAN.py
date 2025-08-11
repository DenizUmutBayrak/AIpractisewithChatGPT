import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

# Veriyi yükle
data = load_wine()
X = data.data
y = data.target

# Veriyi ölçekle
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Boyut indirgeme (sadece görselleştirme için)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# DBSCAN modeli
dbscan = DBSCAN(eps=0.1, min_samples=20)
clusters = dbscan.fit_predict(X_scaled)

# Silhouette skoru (eğer -1'den fazla cluster varsa)
n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
if n_clusters > 1:
    score = silhouette_score(X_scaled, clusters)
    print(f"Silhouette Score: {score:.2f}")
else:
    print("Yeterli küme bulunamadı.")

# Görselleştirme
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', s=50)
plt.title("DBSCAN ile Kümeleme (PCA ile 2D)")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.colorbar(label='Cluster')
plt.show()

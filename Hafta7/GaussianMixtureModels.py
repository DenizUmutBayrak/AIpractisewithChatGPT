import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.metrics import silhouette_score

# Veri setini yükleyelim (Wine dataset'i)
data = load_wine()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.DataFrame(data.target, columns=["class"])

# Veriyi normalize et
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# GMM modelini eğitelim
gmm = GaussianMixture(n_components=3, random_state=42)  # 3 küme olduğunu varsayıyoruz
gmm.fit(X_scaled)

# Verinin her bir noktasını kümelemek
clusters = gmm.predict(X_scaled)

# Kümeleme sonucunu görselleştirme (ilk 2 özellik için)
plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_scaled[:, 0], y=X_scaled[:, 1], hue=clusters, palette='Set2', s=100, edgecolor='k')
plt.title("Gaussian Mixture Model Clustering")
plt.xlabel("Alcohol")
plt.ylabel("Malic Acid")
plt.show()

# Silhouette Score hesaplama
score = silhouette_score(X_scaled, clusters)
print(f"Silhouette Score: {score:.2f}")

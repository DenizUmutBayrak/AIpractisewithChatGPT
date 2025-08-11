import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

# Veri setini yükleyelim (Wine dataset'i)
from sklearn.datasets import load_wine
data = load_wine()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.DataFrame(data.target, columns=["class"])

# Veriyi normalize et
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Linkage yöntemi ile hiyerarşik kümeleme
linked = linkage(X_scaled, method='ward')

# Dendrogram görselleştirme
plt.figure(figsize=(10, 7))
dendrogram(linked, labels=y['class'].values)
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Index")
plt.ylabel("Distance")
plt.show()

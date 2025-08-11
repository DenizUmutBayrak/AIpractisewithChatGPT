from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Veri setini yükleyelim
iris = load_iris()
X = iris.data
y = iris.target

# KMeans modelini oluşturalım
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X)

# Tahmin edilen etiketler
predicted_labels = kmeans.labels_

# Sonuçları görselleştirme (ilk iki özellik ile)
df = pd.DataFrame(X, columns=iris.feature_names)
df['Cluster'] = predicted_labels

sns.scatterplot(data=df, x=iris.feature_names[0], y=iris.feature_names[1], hue='Cluster', palette='deep')
plt.title("K-Means Clustering (İlk İki Özellik)")
plt.show()

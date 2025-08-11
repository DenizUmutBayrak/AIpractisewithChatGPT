import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

# --- 1. Veri Oluşturma (Kümelenmiş) ---
class0 = np.random.randn(50, 2) + [2, 3]
class1 = np.random.randn(50, 2) + [7, 8]
X = np.vstack((class0, class1))
y = np.array([0]*50 + [1]*50)

# --- 2. DataFrame oluştur ---
df = pd.DataFrame(X, columns=['X1', 'X2'])
df['Label'] = y

# --- 3. Feature Engineering ---
df['Sum'] = df['X1'] + df['X2']
df['Diff'] = df['X1'] - df['X2']
df['Prod'] = df['X1'] * df['X2']
df['Ratio'] = df['X1'] / (df['X2'] + 1e-5)

# --- 4. Model için X ve y ---
X_new = df[['X1', 'X2', 'Sum', 'Diff', 'Prod', 'Ratio']].values
y = df['Label'].values

# --- 5. Eğitim-Test Bölünmesi ---
X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.3, random_state=42)

# --- 6. KNN Model Eğitimi ---
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# --- 7. Tahmin ---
y_pred = knn.predict(X_test)

# --- 8. Doğruluk ---
acc = accuracy_score(y_test, y_pred)
print(f"Feature Engineering sonrası KNN doğruluğu: {acc:.2f}")

# --- 9. Görselleştirme ---

# Yöntem 1: Sadece X1 ve X2 ile gerçek ve tahmin
X_test_2d = X_test[:, :2]

plt.figure(figsize=(8,6))
plt.scatter(X_test_2d[:, 0], X_test_2d[:, 1], c=y_test, cmap='coolwarm', edgecolor='k', s=100, label='Gerçek')
plt.scatter(X_test_2d[:, 0], X_test_2d[:, 1], c=y_pred, cmap='coolwarm', marker='x', s=100, label='Tahmin')
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('KNN Tahminleri ve Gerçek Sınıflar (X1 ve X2)')
plt.legend()
plt.show()

# Yöntem 2: PCA ile 2 boyuta indirgeme ve görselleştirme
pca = PCA(n_components=2)
X_test_pca = pca.fit_transform(X_test)

plt.figure(figsize=(8,6))
plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_test, cmap='coolwarm', edgecolor='k', s=100, label='Gerçek')
plt.scatter(X_test_pca[:, 0], X_test_pca[:, 1], c=y_pred, cmap='coolwarm', marker='x', s=100, label='Tahmin')
plt.xlabel('PCA1')
plt.ylabel('PCA2')
plt.title('KNN Tahminleri ve Gerçek Sınıflar (PCA 2D)')
plt.legend()
plt.show()

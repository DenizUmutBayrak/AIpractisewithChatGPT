# Gerekli kütüphaneleri içe aktar
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Veri kümesini yükle
iris = load_iris()
X = iris.data
y = iris.target

# Veriyi eğitim ve test olarak ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Veriyi ölçeklendir (KNN için önemli!)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# KNN modelini oluştur (k=3)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Tahmin yap
y_pred = knn.predict(X_test)

# Değerlendirme
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

import numpy as np

# Gürültü ekleyelim (rastgele küçük değişiklikler)
np.random.seed(42)
noise = np.random.normal(0, 0.5, X.shape)  # Ortalama 0, std 0.5 olan gürültü
X_noisy = X + noise

# Aynı şekilde train/test ayır ve modeli yeniden eğit
X_train_n, X_test_n, y_train_n, y_test_n = train_test_split(X_noisy, y, test_size=0.3, random_state=42)
X_train_n = scaler.fit_transform(X_train_n)
X_test_n = scaler.transform(X_test_n)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_n, y_train_n)
y_pred_n = knn.predict(X_test_n)

# Yeni sonuçlar
print("Gürültülü Veri ile Accuracy:", accuracy_score(y_test_n, y_pred_n))
print("\nConfusion Matrix:\n", confusion_matrix(y_test_n, y_pred_n))
print("\nClassification Report:\n", classification_report(y_test_n, y_pred_n))


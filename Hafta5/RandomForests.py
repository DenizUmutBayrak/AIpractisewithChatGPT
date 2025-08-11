# Gerekli kütüphaneler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Veri setini yükle
iris = load_iris()
X = iris.data
y = iris.target

print("Özellikler:", iris.feature_names)
print("Sınıflar:", iris.target_names)

# Veriyi eğitim ve test olarak ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🔵 1. Model: Normal Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("\n🔵 Normal Model")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=iris.target_names))

# Özellik önemlerini görselleştir
feature_importances = model.feature_importances_
features = iris.feature_names

plt.figure(figsize=(8, 4))
plt.barh(features, feature_importances, color='forestgreen')
plt.xlabel("Öneme Skoru")
plt.title("Özellik Önemleri (Normal Model)")
plt.show()

# 🔴 2. Model: Derinliği sınırlı Random Forest (Overfitting gözlemi için)
model_limited = RandomForestClassifier(
    n_estimators=100,
    max_depth=2,                # Derinlik sınırı
    min_samples_leaf=4,         # Minimum örnek sayısı
    random_state=42
)

model_limited.fit(X_train, y_train)
y_pred_limited = model_limited.predict(X_test)

print("\n🔴 Sınırlı Model")
print("Limited Model Accuracy:", accuracy_score(y_test, y_pred_limited))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred_limited))
print("\nClassification Report:\n", classification_report(y_test, y_pred_limited, target_names=iris.target_names))

# Sınırlı modelin özellik önemlerini görselleştir
feature_importances_limited = model_limited.feature_importances_

plt.figure(figsize=(8, 4))
plt.barh(features, feature_importances_limited, color='darkorange')
plt.xlabel("Öneme Skoru")
plt.title("Özellik Önemleri (Sınırlı Model)")
plt.show()

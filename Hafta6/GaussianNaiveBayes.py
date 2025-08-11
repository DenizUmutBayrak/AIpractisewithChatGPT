import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Veri setini yükle
data = load_wine()
X = data.data
y = data.target
feature_names = data.feature_names
class_names = data.target_names

print("Özellikler:", feature_names)
print("Sınıflar:", class_names)

# Veriyi train-test olarak ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Gaussian Naive Bayes modeli eğit (orijinal veri)
model = GaussianNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("\nOrijinal Veri ile Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Gürültü ekle (train verisine küçük rastgele gürültü)
noise_factor = 0.9
X_train_noisy = X_train + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=X_train.shape)

# Gürültülü veriye model eğit
noisy_model = GaussianNB()
noisy_model.fit(X_train_noisy, y_train)
y_noisy_pred = noisy_model.predict(X_test)

print("\nGürültülü Veri ile Accuracy:", accuracy_score(y_test, y_noisy_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_noisy_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_noisy_pred))

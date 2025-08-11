import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Veri oluşturma (X - Bağımsız değişken, y - Bağımlı değişken)
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)
y = np.array([1, 4, 9, 16, 25, 36, 49, 64, 81, 100])

# Polinomal özellikleri ekleme (2. dereceden)
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

# Doğrusal regresyon modelini oluşturma
model = LinearRegression()
model.fit(X_poly, y)

# Modelin tahminlerini yapma
y_pred = model.predict(X_poly)

# Sonuçları görselleştirme
plt.scatter(X, y, color='blue', label='Gerçek Veri')
plt.plot(X, y_pred, color='red', label='Polinomal Regresyon')
plt.legend()
plt.show()

# Modelin katsayıları ve sabiti
print("Modelin Katsayıları:", model.coef_)
print("Modelin Sabiti:", model.intercept_)


import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Veri oluşturma
X = np.array([[1], [2], [3], [4], [5]])  # Bağımsız değişken (input)
y = np.array([1, 2, 2.9, 4.1, 5])       # Bağımlı değişken (output)

# Model oluşturma
model = LinearRegression()

# Modeli eğitme
model.fit(X, y)

# Tahmin yapma
y_pred = model.predict(X)

# Sonuçları görselleştirme
plt.scatter(X, y, color='blue', label='Gerçek Veri')
plt.plot(X, y_pred, color='red', label='Lineer Regresyon')
plt.legend()
plt.show()

# Modelin performansını değerlendirme
print("Modelin Katsayıları:", model.coef_)
print("Modelin Sabiti:", model.intercept_)
print("Ortalama Kare Hata:", mean_squared_error(y, y_pred))
print("R2 Skoru:", r2_score(y, y_pred))

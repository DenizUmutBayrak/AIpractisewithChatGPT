import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import math

# Veri: Evin metrekare büyüklüğü, oda sayısı, yaşı ve fiyatı
X = np.array([[50, 2, 10], [60, 3, 15], [80, 3, 20], [100, 4, 25], [120, 4, 30]])
y = np.array([200, 250, 300, 350, 400])

# Veriyi eğitim ve test olarak ayırma (test_size=0.5 daha fazla test verisi kullanma)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Modeli oluşturma
model = LinearRegression()

# Modeli eğitme
model.fit(X_train, y_train)

# Test seti üzerinde tahmin yapma
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Modelin katsayıları ve sabiti
print("Modelin Katsayıları:", model.coef_)
print("Modelin Sabiti:", model.intercept_)

# Hata metriklerini hesaplama
mse_train = mean_squared_error(y_train, y_pred_train)
mse_test = mean_squared_error(y_test, y_pred_test)

# RMSE hesaplama (karekökünü almak)
rmse_train = math.sqrt(mse_train)
rmse_test = math.sqrt(mse_test)

# MAE hesaplama (ortalama mutlak hata)
mae_train = mean_absolute_error(y_train, y_pred_train)
mae_test = mean_absolute_error(y_test, y_pred_test)

# R2 skoru
r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)

# Sonuçları yazdırma
print(f"Training MSE: {mse_train}")
print(f"Testing MSE: {mse_test}")
print(f"Training RMSE: {rmse_train}")
print(f"Testing RMSE: {rmse_test}")
print(f"Training MAE: {mae_train}")
print(f"Testing MAE: {mae_test}")
print(f"Training R2: {r2_train}")
print(f"Testing R2: {r2_test}")

# Eğitim verisi ve tahminini görselleştir
plt.scatter(X_train[:, 0], y_train, color='blue', label='Eğitim Verisi')
plt.plot(X_train[:, 0], y_pred_train, color='red', label='Eğitim Eğrisi')

# Test verisi ve tahminini görselleştir
plt.scatter(X_test[:, 0], y_test, color='green', label='Test Verisi')
plt.plot(X_test[:, 0], y_pred_test, color='orange', label='Test Eğrisi')

plt.xlabel('X Değeri (Metrekare)')
plt.ylabel('Y Değeri (Fiyat)')
plt.title('Linear Regression Modeli')
plt.legend()
plt.show()

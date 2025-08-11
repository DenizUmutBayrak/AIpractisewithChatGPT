import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression


# X: öğrencinin çalıştığı saatler
# y: sınavı geçme durumu (1: geçti, 0: kaldı)

X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

model = LogisticRegression()
model.fit(X, y)

# 0 ile 10 arasında daha fazla noktada tahmin yapalım
X_test = np.linspace(0, 11, 300).reshape(-1, 1)
y_prob = model.predict_proba(X_test)[:, 1]  # Sınavı geçme olasılığı (1 olan sınıf)

plt.figure(figsize=(8, 5))
plt.scatter(X, y, color='red', label='Gerçek Veri (0: Kaldı, 1: Geçti)')
plt.plot(X_test, y_prob, color='blue', label='Lojistik Regresyon (Sigmoid Eğrisi)')
plt.xlabel('Çalışma Süresi (saat)')
plt.ylabel('Sınavı Geçme Olasılığı')
plt.title('Lojistik Regresyon Modeli')
plt.legend()
plt.grid(True)
plt.show()

import os
# Uyarı seviyesini düşür
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# oneDNN optimizasyonunu devre dışı bırak (isteğe bağlı)
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from sklearn.model_selection import train_test_split

# Veri üretimi
length_A = np.random.normal(30, 5, 50)
color_A = np.random.normal(40, 10, 50)
length_B = np.random.normal(50, 7, 50)
color_B = np.random.normal(70, 8, 50)

X = np.column_stack([
    np.concatenate([length_A, length_B]),
    np.concatenate([color_A, color_B])
])
y = np.array([0]*50 + [1]*50)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Model tanımı
model = Sequential([
    Input(shape=(2,)),
    Dense(4, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Eğitim
model.fit(X_train, y_train, epochs=100, verbose=0)

# Değerlendirme
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Doğruluğu: {acc:.2f}")

# Karar sınırı çizimi
x_min, x_max = X[:, 0].min() - 5, X[:, 0].max() + 5
y_min, y_max = X[:, 1].min() - 5, X[:, 1].max() + 5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                     np.linspace(y_min, y_max, 200))
grid = np.c_[xx.ravel(), yy.ravel()]
Z = model.predict(grid, verbose=0).reshape(xx.shape)

plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, levels=50, cmap="coolwarm", alpha=0.6)
plt.scatter(X[y==0][:, 0], X[y==0][:, 1], color='green', label='Tür A')
plt.scatter(X[y==1][:, 0], X[y==1][:, 1], color='orange', label='Tür B')
plt.title("Keras ile Yapay Sinir Ağı - Karar Sınırı")
plt.xlabel("Bitki Uzunluğu (cm)")
plt.ylabel("Renk Tonu Skoru")
plt.legend()
plt.show()

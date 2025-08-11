import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Veriler (Features: metrekare ve oda sayısı)
X = np.array([
    [50, 2],
    [60, 3],
    [70, 3],
    [80, 4],
    [90, 4],
    [100, 5]
])

X_new = np.array([[120, 5]])

# Hedef değerler (Prices: ev fiyatları)
y = np.array([150, 200, 250, 300, 350, 400])

model = LinearRegression()

model.fit(X, y)

y_pred = model.predict(X)
y_new_pred = model.predict(X_new)

print("Predicted Price:", y_new_pred[0])
print("Modelin Katsayıları (Coefficients):", model.coef_)
print("Modelin Sabiti (Intercept):", model.intercept_)
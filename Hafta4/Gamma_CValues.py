import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

# Veri seti oluşturma
X, y = make_moons(n_samples=100, noise=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Parametre kombinasyonları
C_values = [0.1, 1, 100]
gamma_values = [0.01, 1, 10]

# Çizim
fig, axes = plt.subplots(len(C_values), len(gamma_values), figsize=(12, 9))
for i, C in enumerate(C_values):
    for j, gamma in enumerate(gamma_values):
        model = SVC(kernel='rbf', C=C, gamma=gamma)
        model.fit(X_train, y_train)

        ax = axes[i, j]
        ax.set_title(f"C={C}, gamma={gamma}")
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap='autumn')

        # Karar sınırı

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        xx = np.linspace(xlim[0], xlim[1], 30)
        yy = np.linspace(ylim[0], ylim[1], 30)
        YY, XX = np.meshgrid(yy, xx)
        xy = np.vstack([XX.ravel(), YY.ravel()]).T
        Z = model.decision_function(xy).reshape(XX.shape)

        ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1], linestyles=['--', '-', '--'])

plt.tight_layout()
plt.show()

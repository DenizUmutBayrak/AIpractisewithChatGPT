import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.svm import SVC

# Veri kümesini oluştur
X, y = make_circles(n_samples=200, factor=0.3, noise=0.1, random_state=42)

# RBF kernel kullanan SVM modelini oluştur ve eğit
model = SVC(kernel='rbf', C=1)
model.fit(X, y)

# Sınırları çizmek için kullanılan fonksiyon
def plot_svm(model, X, y):
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='autumn')

    ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30)
    YY, XX = np.meshgrid(yy, xx)
    xy = np.vstack([XX.ravel(), YY.ravel()]).T
    Z = model.decision_function(xy).reshape(XX.shape)

    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1],
               linestyles=['--', '-', '--'])

    ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1],
               s=100, linewidth=1, facecolors='none', edgecolors='k')
    plt.title("SVM with RBF Kernel")
    plt.show()

# Görselleştir
plot_svm(model, X, y)

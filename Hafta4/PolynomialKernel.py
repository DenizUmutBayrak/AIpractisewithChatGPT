from sklearn.datasets import make_circles
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np

# Veri setini oluştur
X, y = make_circles(noise=0.1, factor=0.3, random_state=5)

# Polinomal kernel ile model
model = SVC(kernel='poly', degree=3, C=1, coef0=1)
model.fit(X, y)

# Görselleştirme fonksiyonu
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
    plt.title("SVM with Polynomial Kernel")
    plt.show()

# Modeli çiz
plot_svm(model, X, y)

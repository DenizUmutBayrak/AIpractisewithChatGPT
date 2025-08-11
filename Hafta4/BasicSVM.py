import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC

# Veri seti: 2 sınıflı basit örnek veri
X, y = datasets.make_blobs(n_samples=100, centers=2, random_state=6)

# SVM modelini tanımla ve eğit
model = SVC(kernel='linear', C=1.0)
model.fit(X, y)


# Veriyi ve karar sınırını çiz
def plot_svm(model, X, y):    #grafiğin en son çizimi için
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='autumn')

    ax = plt.gca()
    xlim = ax.get_xlim()  #düzlemin alt ve üst sınırlarını belirliyor
    ylim = ax.get_ylim()

    # Grid oluştur
    xx = np.linspace(xlim[0], xlim[1], 30)
    yy = np.linspace(ylim[0], ylim[1], 30) #karar yapısını çizebilmek için grid oluşturur
    YY, XX = np.meshgrid(yy, xx)

    xy = np.vstack([XX.ravel(), YY.ravel()]).T    #tüm noktaları (x,y) formatında yazar
    Z = model.decision_function(xy).reshape(XX.shape) #SVM değerlerini atar
    # 0 = karar sınırı üzeri    1 = bir sınıfın marjini  -1 = diğer sınıfın marjini

    # Karar sınırı ve marjin çizgileri
    ax.contour(XX, YY, Z, colors='k', levels=[-1, 0, 1],
               linestyles=['--', '-', '--'])
    # Destek vektörleri
    ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1],
               s=100, linewidth=1, facecolors='none', edgecolors='k')
    plt.title("SVM Decision Boundary with Support Vectors")
    plt.show()


plot_svm(model, X, y)

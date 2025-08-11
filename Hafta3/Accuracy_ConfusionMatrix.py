import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report

# Veri: Öğrencilerin sınav notları ve geçip geçmedikleri (0 = kaldı, 1 = geçti)
X_binary = np.array([30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90]).reshape(-1, 1)
y_binary = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1])

# Modeli oluştur ve eğit
model = LogisticRegression()
model.fit(X_binary, y_binary)

# Test için tahminler
y_pred = model.predict(X_binary)

# Accuracy (Doğruluk) hesaplama
accuracy = accuracy_score(y_binary, y_pred)
print("Model Doğruluğu (Accuracy):", accuracy)

# Confusion Matrix (Karışıklık Matrisi) hesaplama
conf_matrix = confusion_matrix(y_binary, y_pred)

# Karışıklık Matrisini görselleştirme
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

# Sınıflandırma raporunu (precision, recall, f1-score) yazdırma
class_report = classification_report(y_binary, y_pred)
print("Sınıflandırma Raporu:\n", class_report)

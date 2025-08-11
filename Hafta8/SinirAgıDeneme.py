import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix

# 1. Veri Yükleme
data = load_wine()
X = data.data
y = data.target

# 2. One-hot encoding (output katmanı için gerekli)
y_categorical = to_categorical(y)

# 3. Veriyi eğitim/test olarak ayır
X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.3, random_state=42)

# 4. Verileri ölçekle
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Modeli oluştur
model = Sequential()
model.add(Dense(16, input_shape=(X.shape[1],), activation='relu'))  # 13 giriş -> 16 nöron
model.add(Dense(12, activation='relu'))  # 12 nöronlu gizli katman
model.add(Dense(3, activation='softmax'))  # 3 sınıf -> softmax çıkış

# 6. Modeli derle
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 7. Modeli eğit
model.fit(X_train_scaled, y_train, epochs=100, batch_size=8, verbose=0)

# 8. Tahmin yap
y_pred_probs = model.predict(X_test_scaled)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)

# 9. Sonuçları değerlendir
print("Confusion Matrix:")
print(confusion_matrix(y_true, y_pred))
print("\nClassification Report:")
print(classification_report(y_true, y_pred))

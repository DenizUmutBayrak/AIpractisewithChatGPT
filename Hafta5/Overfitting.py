from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# 1. Veri seti
iris = load_iris()
X = iris.data
y = iris.target

# 2. Veri setini ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Aşırı büyümüş model (kontrolsüz)
overfit_model = DecisionTreeClassifier(random_state=42)
overfit_model.fit(X_train, y_train)

# 4. Kontrollü model (sınırlı derinlik)
pruned_model = DecisionTreeClassifier(max_depth=3, min_samples_leaf=4, random_state=42)
pruned_model.fit(X_train, y_train)

# 5. Test sonuçları
print("=== Overfit Model ===")
print(classification_report(y_test, overfit_model.predict(X_test)))

print("=== Pruned Model ===")
print(classification_report(y_test, pruned_model.predict(X_test)))

# 6. Ağaçları çiz
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plot_tree(overfit_model, filled=True, feature_names=iris.feature_names, class_names=iris.target_names)
plt.title("Overfit Tree")

plt.subplot(1, 2, 2)
plot_tree(pruned_model, filled=True, feature_names=iris.feature_names, class_names=iris.target_names)
plt.title("Pruned Tree")

plt.tight_layout()
plt.show()

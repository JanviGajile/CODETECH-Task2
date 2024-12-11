# CODETECH-Task2

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# Generating a sample dataset
np.random.seed(42)
X = np.random.rand(200, 2)  # Two feature columns
y = (X[:, 0] + X[:, 1] > 1).astype(int)  # Label: 1 if sum of features > 1, else 0

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Function to evaluate model performance
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"Accuracy: {acc:.2f}")
    print(f"Precision: {prec:.2f}")
    print(f"Recall: {rec:.2f}")
    print(f"F1-Score: {f1:.2f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

# Logistic Regression
print("Logistic Regression:")
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
evaluate_model(log_reg, X_test, y_test)

# Decision Tree
print("\nDecision Tree:")
dec_tree = DecisionTreeClassifier(random_state=42)
dec_tree.fit(X_train, y_train)
evaluate_model(dec_tree, X_test, y_test)

# Random Forest
print("\nRandom Forest:")
rand_forest = RandomForestClassifier(random_state=42)
rand_forest.fit(X_train, y_train)
evaluate_model(rand_forest, X_test, y_test)

# Support Vector Machine
print("\nSupport Vector Machine:")
svm = SVC()
svm.fit(X_train, y_train)
evaluate_model(svm, X_test, y_test)

# Visualizing decision boundaries for each model
def plot_decision_boundary(model, X, y, title):
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.coolwarm)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.coolwarm)
    plt.title(title)
    plt.show()

print("\nVisualizing Decision Boundaries:")
plot_decision_boundary(log_reg, X, y, "Logistic Regression Decision Boundary")
plot_decision_boundary(dec_tree, X, y, "Decision Tree Decision Boundary")
plot_decision_boundary(rand_forest, X, y, "Random Forest Decision Boundary")
plot_decision_boundary(svm, X, y, "Support Vector Machine Decision Boundary")

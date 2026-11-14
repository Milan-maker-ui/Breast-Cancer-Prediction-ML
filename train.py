

import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

# 1. Load Dataset

print("\nLoading Breast Cancer Dataset...")

data = load_breast_cancer()

X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="target")

print("Dataset loaded successfully!")
print("Dataset shape:", X.shape)

print("\nTarget Classes:")
print(data.target_names)

# 2. Dataset Information

print("\nDataset Information:")

print("Number of samples:", X.shape[0])
print("Number of features:", X.shape[1])

print("\nMissing Values:", X.isnull().sum().sum())

print("\nClass Distribution:")
print(y.value_counts().rename(index={0: "Malignant", 1: "Benign"}))

# 3. Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

# 4. Train Decision Tree Model

print("\nTraining Decision Tree Model...")

model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=4,
    random_state=42
)

model.fit(X_train, y_train)

print("Model trained successfully!")

# 5. Make Predictions

y_pred = model.predict(X_test)

# 6. Model Evaluation

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test, y_pred, zero_division=0
)

recall = recall_score(
    y_test, y_pred, zero_division=0
)

f1 = f1_score(
    y_test, y_pred, zero_division=0
)

print("\n========== MODEL PERFORMANCE ==========")

print(f"Accuracy  : {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")

# 7. Classification Report

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=data.target_names,
        zero_division=0
    )
)



# 8. Confusion Matrix


cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

output_dir = Path(__file__).resolve().parents[1] / "outputs"
output_dir.mkdir(exist_ok=True)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=data.target_names
)

display.plot(
    cmap="Blues",
    values_format="d"
)

plt.title("Breast Cancer Prediction - Confusion Matrix")
plt.tight_layout()

plt.savefig(
    output_dir / "confusion_matrix.png",
    dpi=150
)

plt.show()
plt.close()



# 9. Feature Importance


feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 10 Important Features:")

print(feature_importance.head(10).to_string(index=False))

plt.figure(figsize=(10, 7))

plt.barh(
    feature_importance.head(10)["Feature"][::-1],
    feature_importance.head(10)["Importance"][::-1]
)

plt.xlabel("Importance")
plt.ylabel("Features")
plt.title("Top 10 Important Features")

plt.tight_layout()

plt.savefig(
    output_dir / "feature_importance.png",
    dpi=150
)

plt.show()
plt.close()



# 10. Sample Prediction


sample = X_test.iloc[[0]]

prediction = model.predict(sample)[0]

print("\n========== SAMPLE PREDICTION ==========")

print("Predicted Diagnosis:", data.target_names[prediction])
print("Actual Diagnosis:", data.target_names[y_test.iloc[0]])



# 11. Final Summary


print("\n========== PROJECT SUMMARY ==========")

print("Project: Breast Cancer Prediction")
print("Algorithm: Decision Tree Classifier")
print("Dataset: Wisconsin Diagnostic Dataset")
print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nProject execution completed successfully!")

print("\nDisclaimer: Educational use only. Not for medical diagnosis.")
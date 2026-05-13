**Iris Classification**

📌 Objective

Build a machine learning model to classify iris flowers into species using sepal and petal features.

---

⚙️ Models Used

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Tree

---

📊 Best Model

Logistic Regression (Accuracy: ~97–100%)

---

🧠 Features

- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

---

💾 Saved Files

- "iris_model.pkl" → trained model
- "scaler.pkl" → feature scaling
- "label_encoder.pkl" → target encoding

---

🔮 Inference Example

import joblib

# Load files
model = joblib.load("iris_model.pkl")
scaler = joblib.load("scaler.pkl")
le = joblib.load("label_encoder.pkl")

# Sample input
sample = [[5.1, 3.5, 1.4, 0.2]]

# Apply scaling
sample = scaler.transform(sample)

# Predict
prediction = model.predict(sample)

# Convert back to label
print("Predicted Flower:", le.inverse_transform(prediction))

---

Output Example

Predicted Flower: Iris-setosa

---

📌 How to Run

1. Install dependencies:

pip install scikit-learn pandas joblib

2. Run the inference code above

---

📈 Conclusion

The model performs very well due to clear separability of iris species.
---
Author:
Shivi Sanjay


import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import messagebox
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# ===============================
# 1. Load Titanic Dataset
# ===============================
print("Program Started")
data = pd.read_csv("train.csv")

# Select useful columns
data = data[["Survived", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]]

# Fill missing Age values
data["Age"].fillna(data["Age"].median(), inplace=True)

# Convert Sex to numeric
data["Sex"] = data["Sex"].map({"male": 0, "female": 1})

# ===============================
# 2. Prepare Data
# ===============================
X = data.drop("Survived", axis=1)
y = data["Survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ===============================
# 3. Train SVM Model
# ===============================
model = SVC(kernel="rbf", probability=True)
model.fit(X_train, y_train)

# Model Accuracy
accuracy = accuracy_score(y_test, model.predict(X_test))

# ===============================
# 4. Create GUI
# ===============================
root = tk.Tk()
root.title("Titanic Survival Prediction (SVM)")
root.geometry("400x550")

title = tk.Label(root, text="Titanic Survival Prediction", font=("Arial", 16))
title.pack(pady=10)

accuracy_label = tk.Label(root, text=f"Model Accuracy: {accuracy:.2f}", fg="green")
accuracy_label.pack(pady=5)

# Labels and Entry fields
labels = ["Pclass (1-3)", "Sex (male/female)", "Age",
          "SibSp", "Parch", "Fare"]

entries = []

for label in labels:
    tk.Label(root, text=label).pack()
    entry = tk.Entry(root)
    entry.pack()
    entries.append(entry)

result_label = tk.Label(root, text="", font=("Arial", 14))
result_label.pack(pady=15)

# ===============================
# 5. Prediction Function
# ===============================
def predict():
    try:
        pclass = int(entries[0].get())
        sex = entries[1].get().lower()
        age = float(entries[2].get())
        sibsp = int(entries[3].get())
        parch = int(entries[4].get())
        fare = float(entries[5].get())

        # Convert sex to numeric
        if sex == "male":
            sex = 0
        elif sex == "female":
            sex = 1
        else:
            messagebox.showerror("Error", "Sex must be 'male' or 'female'")
            return

        input_data = np.array([[pclass, sex, age, sibsp, parch, fare]])
        input_scaled = scaler.transform(input_data)

        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][prediction]

        if prediction == 1:
            result = "Survived"
        else:
            result = "Did Not Survive"

        result_label.config(
            text=f"Prediction: {result}\nConfidence: {probability:.2f}"
        )

    except:
        messagebox.showerror("Error", "Please enter valid numeric values")

# Predict Button
predict_button = tk.Button(root, text="Predict", command=predict)
predict_button.pack(pady=10)

root.mainloop()

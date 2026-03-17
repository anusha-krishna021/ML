import pandas as pd
import tkinter as tk
from tkinter import messagebox
import urllib.request

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

print("Program started...")

# -----------------------------
# Download dataset
# -----------------------------

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
urllib.request.urlretrieve(url, "heart.csv")

print("Dataset downloaded")

# Dataset column names
columns = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "target"
]

# Load dataset
data = pd.read_csv("heart.csv", names=columns)

# Convert target values
data['target'] = data['target'].apply(lambda x: 1 if x > 0 else 0)

# Select features
data = data[['age', 'sex', 'cp', 'chol', 'thalach', 'target']]

print("Dataset loaded successfully")

# -----------------------------
# Create Bayesian Network
# -----------------------------

model = DiscreteBayesianNetwork([
    ('age', 'target'),
    ('sex', 'target'),
    ('cp', 'target'),
    ('chol', 'target'),
    ('thalach', 'target')
])

print("Training model...")

model.fit(data, estimator=MaximumLikelihoodEstimator)

inference = VariableElimination(model)

print("Model training completed")

# -----------------------------
# Prediction function
# -----------------------------


def predict():

    try:

        age = int(age_entry.get())
        sex = int(sex_entry.get())
        cp = int(cp_entry.get())
        chol = int(chol_entry.get())
        thalach = int(thalach_entry.get())

        result = inference.query(
            variables=['target'],
            evidence={
                'age': age,
                'sex': sex,
                'cp': cp,
                'chol': chol,
                'thalach': thalach
            })

        prob = result.values[1]

        messagebox.showinfo(
            "Prediction Result",
            f"Heart Disease Probability: {prob:.2f}"
        )

    except:
        messagebox.showerror("Error", "Please enter valid numbers")

# -----------------------------
# UI Design
# -----------------------------


root = tk.Tk()
root.title("Heart Disease Diagnosis")
root.geometry("400x420")

tk.Label(root, text="Heart Disease Prediction",
         font=("Arial", 14)).pack(pady=10)

tk.Label(root, text="Age").pack()
age_entry = tk.Entry(root)
age_entry.pack()

tk.Label(root, text="Sex (0 = Female, 1 = Male)").pack()
sex_entry = tk.Entry(root)
sex_entry.pack()

tk.Label(root, text="Chest Pain Type (0-3)").pack()
cp_entry = tk.Entry(root)
cp_entry.pack()

tk.Label(root, text="Cholesterol").pack()
chol_entry = tk.Entry(root)
chol_entry.pack()

tk.Label(root, text="Max Heart Rate").pack()
thalach_entry = tk.Entry(root)
thalach_entry.pack()

tk.Button(root, text="Predict Heart Disease", command=predict).pack(pady=20)

print("Opening UI...")

root.mainloop()

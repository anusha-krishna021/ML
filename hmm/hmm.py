import numpy as np
from hmmlearn import hmm
import tkinter as tk
from tkinter import messagebox

# -----------------------------
# Train HMM Model (Simple Example)
# -----------------------------
model = hmm.MultinomialHMM(n_components=2, n_iter=100)

# Example dataset (3 sequences)
X = np.array([
    [0], [1], [2],
    [1], [0], [0],
    [2], [1], [0]
])

lengths = [3, 3, 3]

model.fit(X, lengths)

# -----------------------------
# Function to Predict States
# -----------------------------


def predict_states():
    try:
        user_input = entry.get()

        # Convert input string to list
        obs = list(map(int, user_input.strip().split()))

        obs_array = np.array(obs).reshape(-1, 1)

        # Predict hidden states
        states = model.predict(obs_array)

        result_label.config(text=f"Hidden States: {states}")

    except:
        messagebox.showerror("Error", "Enter valid numbers (e.g., 0 1 2)")


# -----------------------------
# UI Design
# -----------------------------
root = tk.Tk()
root.title("Hidden Markov Model Predictor")
root.geometry("400x250")

title = tk.Label(root, text="HMM State Predictor", font=("Arial", 16))
title.pack(pady=10)

label = tk.Label(root, text="Enter Observation Sequence (e.g., 0 1 2):")
label.pack()

entry = tk.Entry(root, width=30)
entry.pack(pady=5)

btn = tk.Button(root, text="Predict Hidden States", command=predict_states)
btn.pack(pady=10)

result_label = tk.Label(root, text="", fg="blue", font=("Arial", 12))
result_label.pack(pady=10)

root.mainloop()

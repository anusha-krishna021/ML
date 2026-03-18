import tkinter as tk
import numpy as np
from PIL import ImageGrab
from tensorflow.keras.models import load_model
import cv2

# Load trained model
model = load_model("digit_model.h5")

# Draw function (thin white line)
def draw(event):
    x = event.x
    y = event.y
    canvas.create_line(x, y, x+1, y+1, fill="white", width=3)

# Clear canvas
def clear_canvas():
    canvas.delete("all")
    result_label.config(text="Draw a digit")

# Predict digit
def predict_digit():

    # Get canvas position
    x = root.winfo_rootx() + canvas.winfo_x()
    y = root.winfo_rooty() + canvas.winfo_y()

    x1 = x + canvas.winfo_width()
    y1 = y + canvas.winfo_height()

    # Capture image
    img = ImageGrab.grab().crop((x, y, x1, y1))

    # Convert to grayscale
    img = np.array(img.convert('L'))

    # Resize
    img = cv2.resize(img, (28,28))

    # Normalize
    img = img / 255.0

    # Reshape for CNN
    img = img.reshape(1,28,28,1)

    # Predict
    prediction = model.predict(img)

    digit = np.argmax(prediction)

    result_label.config(text="Predicted Digit: " + str(digit))


# Window
root = tk.Tk()
root.title("Digit Recognition")

# Canvas
canvas = tk.Canvas(root, width=200, height=200, bg="black")
canvas.pack()

canvas.bind("<B1-Motion>", draw)

# Buttons
predict_button = tk.Button(root, text="Predict", command=predict_digit)
predict_button.pack()

clear_button = tk.Button(root, text="Clear", command=clear_canvas)
clear_button.pack()

# Result
result_label = tk.Label(root, text="Draw a digit")
result_label.pack()

root.mainloop()

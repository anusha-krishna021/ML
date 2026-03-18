import os
import cv2
from tensorflow.keras.datasets import mnist

print("Loading MNIST dataset...")

(x_train, y_train), (x_test, y_test) = mnist.load_data()

dataset_path = "dataset/train"

print("Creating folders...")

for i in range(10):
    os.makedirs(os.path.join(dataset_path, str(i)), exist_ok=True)

print("Saving images...")

count = 0

for i in range(len(x_train)):
    
    img = x_train[i]
    label = y_train[i]

    folder = os.path.join(dataset_path, str(label))
    filename = os.path.join(folder, f"{i}.png")

    cv2.imwrite(filename, img)
    count += 1

print("Total images saved:", count)
print("Dataset creation completed!")

import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

data = []
labels = []

dataset_path = "dataset/train"

print("Loading dataset...")

for digit in os.listdir(dataset_path):

    digit_path = os.path.join(dataset_path, digit)

    for img_name in os.listdir(digit_path):

        img_path = os.path.join(digit_path, img_name)

        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            continue

        img = cv2.resize(img,(28,28))

        data.append(img)
        labels.append(int(digit))

data = np.array(data)
labels = np.array(labels)

data = data / 255.0
data = data.reshape(-1,28,28,1)

labels = to_categorical(labels,10)

print("Dataset loaded:",len(data))

X_train,X_test,y_train,y_test = train_test_split(
    data,labels,test_size=0.2,random_state=42
)

model = Sequential()

model.add(Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1)))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D((2,2)))

model.add(Flatten())

model.add(Dense(128,activation='relu'))
model.add(Dense(10,activation='softmax'))

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("Training started...")

model.fit(X_train,y_train,epochs=5,validation_data=(X_test,y_test))

loss,accuracy = model.evaluate(X_test,y_test)

print("Test Accuracy:",accuracy)

model.save("digit_model.h5")

print("Model saved successfully!")

import cv2
import os

person_name = input("Enter person name: ")
dataset_path = f"dataset/{person_name}"

os.makedirs(dataset_path, exist_ok=True)

cap = cv2.VideoCapture(0)
count = 0

while True:
    ret, frame = cap.read()
    cv2.imshow("Capturing Faces", frame)

    cv2.imwrite(f"{dataset_path}/{count}.jpg", frame)
    count += 1

    if count == 200:   # Capture 200 images
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

import cv2
import numpy as np
import math
import csv
import os
import pyttsx3
from datetime import datetime
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier


cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier(r"C:\Users\vdhar\Downloads\Sign-Language-detection\Sign-Language-detection\Model-new\keras_model.h5", r"C:\Users\vdhar\Downloads\Sign-Language-detection\Sign-Language-detection\Model-new\labels.txt")
engine = pyttsx3.init()
offset = 20
imgSize = 300
labels = ["Yes","Thank You", "Please", "Perfect", "No","I love you", "Hello", "Bye"]


log_file = "predictions_log.csv"
if not os.path.exists(log_file):
    with open(log_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Timestamp", "Prediction"])

prev_label = ""

while True:
    success, img = cap.read()
    imgOutput = img.copy()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap: wCal + wGap] = imgResize
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap: hCal + hGap, :] = imgResize

        prediction, index = classifier.getPrediction(imgWhite, draw=False)
        label = labels[index]

        # Only speak if new label
        if label != prev_label:
            engine.say(label)
            engine.runAndWait()
            prev_label = label

            # Log to file
            with open(log_file, mode="a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), label])

        # Drawing
        cv2.rectangle(imgOutput, (x - offset, y - offset - 70), (x - offset + 300, y - offset - 10), (0, 255, 0), cv2.FILLED)
        cv2.putText(imgOutput, label, (x, y - 30), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 0), 2)
        cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (0, 255, 0), 4)

        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Image", imgOutput)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

#-- coding: utf8 --
#!/usr/bin/env python3

import cv2
import os
import numpy as np
import time

from colorFile import *

os.system('clear')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')

faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml");
eyeCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
smileCascade = cv2.CascadeClassifier(cv2.data.haarcascades +
"haarcascade_smile.xml")

# Тип шрифта
font = cv2.FONT_HERSHEY_SIMPLEX
color = GenRGB()
id = 0 # начальный счетчик идентификаторов

# начальный счетчик идентификаторов
names = ['None', 'Egor', 'Larisa', 'Zahar', 'Z', 'W']
detected_faces = []


font = cv2.FONT_HERSHEY_SIMPLEX

# Initialize and start realtime video capture
cam = cv2.VideoCapture(0)
cam.set(3, 640) # set video widht
cam.set(4, 480) # set video height

# Define min window size to be recognized as a face
minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)

print('\n', INFOcolored('[INFO]'), ('Подготовка камеры...'))
time.sleep(2.0)
print('\n', INFOcolored('[INFO]'), ('Воспроизведение видео...'))

def draw_detections(img, parts, border=2):
    for (x, y, w, h) in parts:
        pad_w, pad_h = int(0.15*w), int(0.05*h)
        cv2.rectangle(roi_color, (x+pad_w, y+pad_h), (x+w-pad_w, y+h-pad_h), (color), border)

while True:
    ret, img =cam.read()
    img = cv2.flip(img, 1) # Flip vertically
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = (int(minW), int(minH)),
       )

    for(x,y,w,h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])

        # Распознание глаз
        eyes = eyeCascade.detectMultiScale(
            roi_gray,
            scaleFactor= 1.5,
            minNeighbors=5,
            minSize=(5, 5),
            )

        draw_detections(img, eyes)

        # Распознание улыбки
        smile = smileCascade.detectMultiScale(
            roi_gray,
            scaleFactor= 1.5,
            minNeighbors=15,
            minSize=(25, 25),
            )

        draw_detections(img, smile)

        # Проверяем что лицо распознано
        if (confidence < 100):
            id = names[id]
            confidence = f"{round(100 - confidence)}"

            if str(id) not in detected_faces and str(id) in names:
                detected_faces.append(str(id))
                print(('Обнаружен'), SimpleTextColor(str(id)))
        else:
            id = "Who are u?"
            confidence = f"{round(100 - confidence)}"

        cv2.putText(img, str(id), (x+5,y-5), font, 1, (255,255,255), 2)
        cv2.putText(img, str(confidence), (x+5,y+h-5), font, 1, (255,255,0), 1)

    cv2.imshow('camera',img)

    k = cv2.waitKey(10) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        break


cam.release()
cv2.destroyAllWindows()

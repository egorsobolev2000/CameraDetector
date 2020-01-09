import cv2
import os

from colorFile import INFOcolored

cam = cv2.VideoCapture(0)
cam.set(3, 640) # Ширина
cam.set(3, 640) # Высота

os.system('clear')
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Для каждого человека ввести один числовой идентификатор лица
face_id = input('\n введите идентификатор пользователя и нажмите Enter ==> ')
print('\n', INFOcolored('[INFO]'), ('Инициализация захвата лица. Посмотри камеру и подожди ...'))

# Инициализация индивидуального отсчета лица
count = 0
dataset_count = []
while True:
    ret, img = cam.read()
    img = cv2.flip(img, 1)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Вызов классификаторной функции
    faces = face_detector.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x,y), (x+w, y+h), (255, 0, 0), 2)
        count += 1

        # Сохраняем захваченное изображение в папку наборов данных
        cv2.imwrite(f'dataset/User.{str(face_id)}.{str(count)}.jpg', gray[y:y + h, x:x + w])
        cv2.imshow('image', img)


    # Если нажать ESC программа закроется
    k = cv2.waitKey(100) & 0xff
    if k == 27:
        break
    # Захватуем 300 образцов лица и завершаем программу
    elif count <= 100:
        if count not in dataset_count:
            dataset_count.append(count)
            print(count)
    elif count >= 100:
        break

# Небольшая уборка
print('\n [INFO] Выход из программы')
cam.release()
cv2.destroyAllWindows()

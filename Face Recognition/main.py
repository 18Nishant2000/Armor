import cv2
import tensorflow as tf
import numpy as np
from face_recognition_model import encoder

def start():
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    model = tf.keras.models.load_model('face_recognizer')
    size = (100, 100)
    cap = cv2.VideoCapture(0)

    while True:

        _, img = cap.read()
        img = cv2.flip(img, 1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 1)
        for (x, y, w, h) in faces:
            x -=20
            y -=30
            w +=x+20
            h +=y+30
            cv2.rectangle(img, (x,y), (w,h), (255,0,0), 2)
            img2 = img[y:h,x:w]
            img2 = cv2.resize(img2, size)
            img2 = img2.reshape(1, 100,100,3)
            pred = model.predict(img2)
            result = encoder.inverse_transform(pred)
            print(result)
            

        cv2.imshow('Scanning', img)
            
        if cv2.waitKey(1) == 13:
            cv2.destroyAllWindows()
            break

start()
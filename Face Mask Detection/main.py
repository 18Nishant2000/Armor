import cv2
import numpy as np
import smtplib
import tensorflow as tf
config = tf.compat.v1.ConfigProto(
    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.9)
)
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)
import credentials as c

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
SUB='Alert'
TEXT='You violated Face Mask Policy. A penalty of 5000 is levied on you.'

face_recognizer = tf.keras.models.load_model('D:\PROJECTS\Armor\Face Recognition\\face_recognizer')

result = {
    0 : ['Mask', (0,255,0)],
    1 : ['No Mask', (0,0,255)]
}

cap = cv2.VideoCapture(0)
model = tf.keras.models.load_model('mask_detector')
size = (100, 100)


def prediction(img):
    pred = model.predict(img)
    label = np.argmax(pred, axis=1)[0]
    # print(label)
    return label


while True:
    
    _ , img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 1)
    
    for (x, y, w, h) in faces:
        gray_face = gray[y:y+w,x:x+w]
        gray_face = cv2.resize(gray_face, size)
        gray_face = gray_face/255.0
        
        img2 = img/255.0
        img2 = cv2.resize(img2, size)
        gray_face = np.reshape(img2, (1,100,100,3))
        label = prediction(gray_face)

        cv2.rectangle(img, (x, y), (x+w, y+h), result[label][1], 2)
        cv2.rectangle(img, (x, y-40), (x+w, y), result[label][1], -1)
        cv2.putText(img, result[label][0], (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)

        if label == 1:
            print(face_recognizer.predict(img2))

            # mess = f'Subject: {SUB}\n\n{TEXT}'
            # mail = smtplib.SMTP('smtp.gmail.com', 587)
            # mail.ehlo()
            # mail.starttls()
            # mail.login(c.email, c.password)
            # mail.sendmail(c.from_email, c.to_email, mess)
            # mail.close()


    cv2.imshow('Face Mask Detection', img)


    if cv2.waitKey(1) == 13:
        cv2.destroyAllWindows()
        break

    
cap.release()
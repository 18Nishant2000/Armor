import cv2
import numpy as np
import tensorflow as tf
config = tf.compat.v1.ConfigProto(
    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.9)
)
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

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

    cv2.imshow('Face Mask Detection', img)


    if cv2.waitKey(1) == 13:
        cv2.destroyAllWindows()
        break

    
cap.release()
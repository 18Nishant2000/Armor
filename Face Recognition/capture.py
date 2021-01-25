import cv2
import os

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

k = 0
dir_name = input('Enter your name: ')

if not os.path.isdir(f'people/{dir_name}'):
    os.mkdir(f'people/{dir_name}')

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
        if k < 1000:
            cv2.imwrite(f'people/{dir_name}/image{k}.jpg', img2)
            k+=1
        else:
            print('done')
            break
    
    cv2.imshow('Window', img)
    
    if cv2.waitKey(1) == 13:
        cv2.destroyAllWindows()
        break

cap.release()
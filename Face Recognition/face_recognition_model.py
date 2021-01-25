import os
import numpy as np
import cv2
from sklearn.preprocessing import OneHotEncoder
import random
import tensorflow as tf
config = tf.compat.v1.ConfigProto(
    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.7)
)
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)
import matplotlib.pyplot as plt



label = [i for i in os.listdir('people')]
label = np.array(label)
label = np.reshape(label, (-1,1))
encoder = OneHotEncoder()
l = encoder.fit_transform(label).toarray()
print(l)
size = (100, 100)

X_train = []
Y_train = []
X_test = []
Y_test = []

dirs = os.listdir('people/')
print(dirs)
for dir_num in range(len(dirs)):
    k = 0
    images = os.listdir(f'people/{dirs[dir_num]}/')
    random.shuffle(images)
    for i in images:
        img = cv2.imread(f'people/{dirs[dir_num]}/{i}')
        img = cv2.resize(img, size)
        if k < 950:
            X_train.append(img)
            Y_train.append(l[dir_num])
        else:
            X_test.append(img)
            Y_test.append(l[dir_num])
        k+=1

X_train_len = len(X_train)
X_test_len = len(X_test)
X_train = np.array(X_train)
X_test = np.array(X_test)


X_train = X_train.reshape(X_train_len, 100, 100, 3)
X_test = X_test.reshape(X_test_len, 100, 100, 3)
Y_train = np.array(Y_train)
Y_test = np.array(Y_test)

# print(X_train_len)
# print(X_test_len)
# print(Y_train)
# print(Y_test)


model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(64, 3, activation='relu', input_shape=(100, 100, 3)))
model.add(tf.keras.layers.MaxPool2D(2, 2))
model.add(tf.keras.layers.Conv2D(128, 3, activation='relu'))
model.add(tf.keras.layers.MaxPool2D(2, 2))
model.add(tf.keras.layers.Conv2D(200, 3, activation='relu'))
model.add(tf.keras.layers.MaxPool2D(2, 2))
model.add(tf.keras.layers.Conv2D(100, 3, activation='relu'))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(len(label), activation='softmax'))
model.summary()


# model = Sequential([
#     Conv2D(128, 3, activation='relu', padding='same', input_shape=(300, 300, 1)),
#     # MaxPool2D(2, 2, padding='valid'),
#     Conv2D(128, 3, activation='relu', padding='same'),
#     MaxPool2D(2, 2, padding='valid'),
#     Conv2D(64, 5, activation='relu', padding='same'),
#     # MaxPool2D(2, 2, padding='valid'),
#     Conv2D(64, 5, activation='relu', padding='same'),
#     MaxPool2D(2, 2, padding='valid'),
#     Conv2D(32, 3, activation='relu', padding='same'),
#     # MaxPool2D(2, 2, padding='valid'),
#     Conv2D(8, 3, activation='relu', padding='same'),
#     Flatten(),
#     Dense(len(label), activation='softmax')
# ])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
# history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=5, batch_size=1)
# model.summary()
# model.save('face_recognizer')

# loss_curve = history.history['loss']
# val_loss_curve = history.history['val_loss']
# plt.plot(loss_curve, 'b', label='Training Loss')
# plt.plot(val_loss_curve, 'r', label='Validation Loss')
# plt.title('Loss Curve')
# plt.legend(loc='upper right')
# plt.xlabel('No. of epochs')
# plt.ylabel('Loss')
# plt.show()

# accuracy_curve = history.history['accuracy']
# val_accuracy_curve = history.history['val_accuracy']
# plt.plot(accuracy_curve, 'b', label='Training Accuracy')
# plt.plot(val_accuracy_curve, 'g', label='Validation Accuracy')
# plt.title('Accuracy Curve')
# plt.legend(loc='upper left')
# plt.xlabel('No. of epochs')
# plt.ylabel('Accuracy')
# plt.show()

print(model.evaluate(X_test, Y_test))
pred = model.predict(X_test)
print(pred)
for i in pred:
    print(encoder.inverse_transform([i]))

import os
import cv2
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer

config = tf.compat.v1.ConfigProto(
    gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.7)
)
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)

categories = [
    'with mask',
    'without mask'
]
X_train = []
Y_train = []
X_test = []
Y_test = []
size = (100, 100)
data = []


for i in categories:
    images = os.listdir(f'dataset/{i}')
    for j in range(len(images)):
        # mat = cv2.imread(f'dataset/{i}/{images[j]}', cv2.IMREAD_GRAYSCALE)
        mat = cv2.imread(f'dataset/{i}/{images[j]}')
        mat = cv2.resize(mat, size)
        data.append((mat, i))


print(data[:2])
print(len(data))
train_size = int(len(data)*.95)
for i in range(10):
    random.shuffle(data)

images = []
label = []

for i in range(len(data)):
    images.append(data[i][0])
    label.append(data[i][1])

images = np.array(images)/255.0
images = np.reshape(images, (images.shape[0], 100, 100, 3))
lb = LabelBinarizer()
label = lb.fit_transform(label)
label = to_categorical(label)
label = np.array(label)

X_train = images[:train_size+1]
Y_train = label[:train_size+1]
X_test = images[train_size+1:]
Y_test = label[train_size+1:]

print(len(X_train))
print(len(Y_train))
print(len(X_test))
print(len(Y_test))
print(X_train)
print(Y_train)
print(X_test)
print(Y_test)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(64, 3, activation='relu', input_shape=(100, 100, 3)))
model.add(tf.keras.layers.MaxPool2D(2, 2))
model.add(tf.keras.layers.Conv2D(128, 3, activation='relu'))
model.add(tf.keras.layers.MaxPool2D(2, 2))
model.add(tf.keras.layers.Conv2D(200, 3, activation='relu'))
model.add(tf.keras.layers.MaxPool2D(2, 2))
model.add(tf.keras.layers.Conv2D(100, 3, activation='relu'))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(2, activation='softmax'))
model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=5, batch_size=50)
model.save('mask_detector_colored')
# tf.keras.utils.plot_model(model, to_file='model_architecture.png')

loss_curve = history.history['loss']
val_loss_curve = history.history['val_loss']
plt.plot(loss_curve, 'b', label='Training Loss')
plt.plot(val_loss_curve, 'r', label='Validation Loss')
plt.title('Loss Curve')
plt.legend(loc='upper right')
plt.xlabel('No. of epochs')
plt.ylabel('Loss')
plt.show()

accuracy_curve = history.history['accuracy']
val_accuracy_curve = history.history['val_accuracy']
plt.plot(accuracy_curve, 'b', label='Training Accuracy')
plt.plot(val_accuracy_curve, 'g', label='Validation Accuracy')
plt.title('Accuracy Curve')
plt.legend(loc='upper left')
plt.xlabel('No. of epochs')
plt.ylabel('Accuracy')
plt.show()

print(model.evaluate(X_test, Y_test))
pred = model.predict(X_test)
print(pred)
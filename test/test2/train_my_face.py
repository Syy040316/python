
import cv2
import tensorflow as tf
import numpy as np
import os
import random
from tensorflow.keras import models, layers

import sys
from sklearn.model_selection import train_test_split

my_faces_path = './my_faces'
other_faces_path = 'input_img'
size = 64
input_shape = (64,64,3)

imgs = []
labs = []

def getPaddingSize(img):
    h, w, _ = img.shape
    top, bottom, left, right = (0,0,0,0)
    longest = max(h, w)

    if w < longest:
        tmp = longest - w
        # //表示整除符号
        left = tmp // 2
        right = tmp - left
    elif h < longest:
        tmp = longest - h
        top = tmp // 2
        bottom = tmp - top
    else:
        pass
    return top, bottom, left, right

def readData(path , h=size, w=size):
    for filename in os.listdir(path):
        if filename.endswith('.jpg'):
            filename = path + '/' + filename

            img = cv2.imread(filename)

            top,bottom,left,right = getPaddingSize(img)
            # 将图片放大， 扩充图片边缘部分
            img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0,0,0])
            img = cv2.resize(img, (h, w))

            imgs.append(img)
            labs.append(path)

readData(my_faces_path)
readData(other_faces_path)

imgs = np.array(imgs)
labs = np.array([[1,] if lab == my_faces_path else [0,] for lab in labs])

X_train, x_test, Y_train, y_test = train_test_split(imgs, labs,
                test_size= 0.1,random_state= random.randint(0,100))

X_train = X_train.reshape(X_train.shape[0], size, size ,3)
x_test = x_test.reshape(x_test.shape[0], size, size, 3)

x_test = x_test.astype('float32') / 255.0
X_train = X_train.astype('float32') / 255.0

batch_size = 128


def build():
    model = models.Sequential()

    model.add(
        layers.Conv2D(32, (3,3), padding= 'same',activation= 'relu', input_shape= input_shape)
    )
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(32,(3,3), padding= 'same', activation= 'relu' ))
    model.add(layers.MaxPool2D(pool_size= (2,2)))
    model.add(layers.Dropout(0.2))


    model.add(layers.Conv2D(64, (3,3), padding= 'same', activation= 'relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(64, (3,3), padding='same', activation= 'relu'))
    model.add(layers.MaxPool2D(pool_size= (2,2)))
    model.add(layers.Dropout(0.2))

    model.add(layers.Conv2D(128, (3,3), padding= 'same', activation= 'relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(128, (3,3), padding='same', activation= 'relu'))
    model.add(layers.MaxPool2D(pool_size= (2,2)))
    model.add(layers.Dropout(0.25))

    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation= 'relu', ))

    model.add(layers.Dropout(0.2))
    model.add(layers.Dense(128, activation= 'relu'))

    model.add(layers.Dropout(0.35))
    model.add(layers.Dense(32, activation= 'relu'))
    model.add(layers.Dropout(0.5))

    model.add(layers.Dense(1, activation= 'sigmoid'))
    return model
model = build()
model.summary()

model.compile(
    optimizer= 'adam',
    loss= 'binary_crossentropy',
    metrics= ['accuracy']
)

model.fit(X_train,Y_train, batch_size= batch_size,
        verbose= 1,epochs= 20, validation_data=(x_test, y_test)
        )
model.save('model',)

score = model.evaluate(x_test, y_test, batch_size= batch_size)

test_loss = score[0]
test_acc = score[1]
print('\ntest_loss: ', test_loss)
print(' test_acc: ', test_acc)





from tensorflow import train
from tensorflow.keras import models, layers
import cv2
import numpy as np
import os
import random
import sys
from sklearn.model_selection import train_test_split
import dlib
import tensorflow as tf
my_faces_path = './my_faces'
other_faces_path = 'input_img'
size = 64
imgs = []
labs = []
INPUT_SHAPE = (64, 64, 3)
NB_CLASSES = 1
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
            img = cv2.copyMakeBorder(img, top, bottom,
                                     left, right, cv2.BORDER_CONSTANT, value=[0,0,0])
            img = cv2.resize(img, (h, w))

            imgs.append(img)
            labs.append(path)
def img_change(img,height= size, width = size ):
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
    top,bottom,left,right = getPaddingSize(img)
    # 将图片放大， 扩充图片边缘部分
    img = cv2.copyMakeBorder(img, top, bottom,
                             left, right, cv2.BORDER_CONSTANT, value=[0,0,0])
    img = cv2.resize(img, (height, width))
    return img

readData(my_faces_path)
readData(other_faces_path)
# 将图片数据与标签转换成数组
imgs = np.array(imgs)
labs = np.array([[0,1] if lab == my_faces_path else [1,0] for lab in labs])

# 随机划分测试集与训练集
train_x,test_x,train_y,test_y = train_test_split(imgs, labs,
                                                 test_size=0.05, random_state=random.randint(0,100))
# 参数：图片数据的总数，图片的高、宽、通道
train_x = train_x.reshape(train_x.shape[0], size, size, 3)
test_x = test_x.reshape(test_x.shape[0], size, size, 3)
# 将数据转换成小于1的数
train_x = train_x.astype('float32')/255.0
test_x = test_x.astype('float32')/255.0


batch_size = 100
num_batch = len(train_x) // batch_size


def build():
    model = models.Sequential()
    model.add(
        layers.Conv2D(32,(3,3), activation= 'relu', input_shape= INPUT_SHAPE, padding= 'same')
    )
    model.add(
        layers.MaxPool2D(pool_size= (2,2), strides= (2,2))
    )
    model.add(layers.Dropout(0.2))
    model.add(
        layers.Conv2D(64, (3,3), activation= 'relu')
    )
    model.add(layers.MaxPool2D((2,2), strides= (2,2)))
    model.add(layers.Dropout(0.5))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))

    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation='sigmoid'))
    return model

9

model = models.load_model('model')
detector = dlib.get_frontal_face_detector()
cap = cv2.VideoCapture(0)
while True:
    ret, img = cap.read()
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    glur = cv2.GaussianBlur(img,(55,55), 10)
    dets = detector(gray_img, 1)
    if not len(dets):
        cv2.putText(glur, 'Can`t get face.', (50,60), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0,0,255), 2)
        cv2.imshow('video', glur)
        key = cv2.waitKey(10) & 0xff
        if key == ord('q'):
            sys.exit(0)
    for i, d in enumerate(dets):
        x1 = d.top() if d.top() > 0 else 0
        y1 = d.bottom() if d.bottom() > 0 else 0
        x2 = d.left() if d.left() > 0 else 0
        y2 = d.right() if d.right() > 0 else 0
        face = img[x1:y1, x2:y2]

        face = img_change(face)

        face = tf.reshape(face, (1, 64, 64, 3))
        face = tf.cast(face, tf.float32) / 255.0

        cv2.putText(glur,'is this my face?',(50,60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2 )

        pred = model.predict(face)
        print(pred, pred[0])

        if pred[0] > 0.96:

            cv2.putText(glur,'Yes', (50,100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
            cv2.rectangle(glur,(x2, x1), (y2, y1),(0,255,255), 2)
        else:
            cv2.putText(glur,'No', (50,100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,0), 2)
            cv2.rectangle(glur,(x2, x1), (y2, y1),(0,255,255), 1)
        cv2.imshow('video', glur)
        key = cv2.waitKey(10) & 0xff
        if key == ord('q'):
            sys.exit(0)






        
        






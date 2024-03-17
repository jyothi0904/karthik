import numpy as np
import pandas as pd 
â€‹
â€‹
â€‹
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
â€‹



import keras 
from keras.models import Sequential
from keras.layers import Conv2D,Flatten,Dense,MaxPooling2D,Dropout
from sklearn.metrics import accuracy_score


import ipywidgets as widgets
import io
from PIL import Image
import tqdm
from sklearn.model_selection import train_test_split
import cv2
from sklearn.utils import shuffle 
import tensorflow as tf


X_train = []
Y_train = []
image_size = 150
labels = ['glioma_tumor','meningioma_tumor','no_tumor','pituitary_tumor']
for i in labels:
    folderpath = os.path.join('/kaggle/input/brain-tumor-classification-mri/Training',i)
    for j in os.listdir(folderpath):
        img = cv2.imread(os.path.join(folderpath,j))
        img = cv2.resize(img,(image_size,image_size))
        X_train.append(img)
        Y_train.append(i)
for i in labels:
    folderpath = os.path.join('/kaggle/input/brain-tumor-classification-mri/Testing',i)
    for j in os.listdir(folderpath):
        img = cv2.imread(os.path.join(folderpath,j))
        img = cv2.resize(img,(image_size,image_size))
        X_train.append(img)
        Y_train.append(i)
X_train = np.array(X_train)
Y_train = np.array(Y_train)
        

X_train,Y_train  = shuffle(X_train,Y_train,random_state=101)
X_train.shape


X_train,X_test,y_train,y_test= train_test_split(X_train,Y_train,test_size=0.1,random_state=101)


y_train_new=[]
for i in y_train:
    y_train_new.append(labels.index(i))
y_train=y_train_new
Y_train=tf.keras.utils.to_categorical(y_train)

y_test_new=[]
for i in y_test:
    y_test_new.append(labels.index(i))
y_test=y_test_new
Y_test=tf.keras.utils.to_categorical(y_test)


model = Sequential()
model.add(Conv2D(32,(3,3),activation = 'relu',input_shape=(150,150,3)))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.3))
model.add(Conv2D(64,(3,3),activation = 'relu'))
model.add(Conv2D(64,(3,3),activation = 'relu'))
model.add(Dropout(0.3))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.3))
model.add(Conv2D(128,(3,3),activation = 'relu'))
model.add(Conv2D(128,(3,3),activation = 'relu'))
model.add(Conv2D(128,(3,3),activation = 'relu'))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.3))
model.add(Conv2D(128,(3,3),activation = 'relu'))
model.add(Conv2D(256,(3,3),activation = 'relu'))
model.add(MaxPooling2D(2,2))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(512,activation = 'relu'))
model.add(Dense(512,activation = 'relu'))
model.add(Dropout(0.3))
model.add(Dense(4,activation = 'softmax'))




model.summary()


model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])

import matplotlib.pyplot as plt
import seaborn as sns


history = model.fit(X_train,Y_train,epochs=20,validation_split=0.1)



model.save ('braintumor.h5')

acc=history.history['accuracy']
val_acc = history.history['val_accuracy']
epochs = range(len(acc))
fig = plt.figure(figsize=(14,7))
plt.plot(epochs,acc,'r',label="Training Accuracy")
plt.plot(epochs,val_acc,'r',label="Validation Accuracy")
plt.legend(loc='upper left')
plt.show()


loss =history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(loss))
fig = plt.figure(figsize=(14,7))
plt.plot(epochs,acc,'r',label="Training Loss")
plt.plot(epochs,val_acc,'r',label="Validation Loss")
plt.legend(loc='upper left')
plt.show()


img = cv2.imread('/kaggle/input/brain-tumor-classification-mri/Training/glioma_tumor/gg (10).jpg')
img = cv2.resize(img,(150,150))
img_array=np.array(img)
img_array.shape


img_array=img_array.reshape(1,150,150,3)
img_array.shape

from tensorflow.keras.preprocessing import image
img=image.load_img('/kaggle/input/brain-tumor-classification-mri/Training/glioma_tumor/gg (10).jpg')
plt.imshow(img,interpolation='nearest')
plt.show()

a=model.predict(img_array)
indices=a.argmax()
indices

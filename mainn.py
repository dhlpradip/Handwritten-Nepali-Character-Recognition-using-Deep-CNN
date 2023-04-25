import numpy as np
import os
import cv2
import random
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Activation,Flatten,Conv2D,MaxPooling2D,Dropout
from keras.utils import to_categorical
from keras.regularizers import l1

DATADIR='./dhcd/'
CATEGORIES=["0","1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20","21","22","23","24","25","26","27","28","29","30","31","32","33","34","35","36","37","38","39","40","41","42","43","44","45"]
training_data=[]
def create_training_data():
    for category in CATEGORIES:
        path=os.path.join(DATADIR,category)
        class_num=CATEGORIES.index(category)
        for img in tqdm(os.listdir(path)):
            img_array=cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
            training_data.append([img_array,class_num])


create_training_data()
a=len(training_data)
print("Training data:", a)
random.shuffle(training_data)
for sample in training_data[:25]:
    print(sample[1])

x=[]
y=[]

for features,label in training_data:
    x.append(features)
    y.append(label)

x=np.array(x).reshape(-1,32,32,1)
y=y=to_categorical(y) 

x=x/250.0

model=Sequential()

model.add(Conv2D(64,(3,3),input_shape=x.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64,(3,3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Dropout(0.3))

model.add(Flatten())#this converts our 3D feature to 1D features vector

model.add(Dense(46))
model.add(Activation("softmax"))


model.compile(loss="categorical_crossentropy",optimizer="nadam",metrics=['accuracy'])
model.fit(x,y,batch_size=512,epochs=7,validation_split=0.3)
model.save('train_modelRADAM.hdf5')
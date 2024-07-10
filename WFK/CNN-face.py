########人脸数据集##########
###########保存模型############
########1层隐含层（全连接层）##########
#60000条训练数据和10000条测试数据，28x28像素的灰度图像
#隐含层激活函数：ReLU函数
#输出层激活函数：softmax函数（实现多分类）
#损失函数：稀疏交叉熵损失函数
#输入层有784个节点，隐含层有128个神经元，输出层有10个节点
import os
import glob
import h5py
import keras
import numpy as np
from tkinter import *
from tkinter import ttk
from PIL import Image,ImageTk
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.imagenet_utils import preprocess_input
from keras.models import Model
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image  
from sklearn.model_selection import train_test_split

import time
print('--------------')
nowtime = time.strftime('%Y-%m-%d %H:%M:%S')
print(nowtime)

#指定GPU
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
plt.rcParams['font.sans-serif'] = ['SimHei']

image_dir = "orl"

def Preprocessing(image_dir):  
    images = []    
    for root, dirs, files in os.walk(image_dir):  
        for file in files:  
            # 检查文件扩展名  
            if file.lower().endswith((".png", ".pgm", ".tif", ".bmp")):  
                # 构建图像的完整路径  
                image_path = os.path.join(root, file)  
                # 打开图像并转换为灰度图像  
                with Image.open(image_path) as image:  
                    image = image.convert('L')  # 转换为灰度图  
                    # 将图像转换为NumPy数组  
                    image_array = np.array(image)  
                    # 将图像数据展平为一个列向量  
                    # images.append(image_array.flatten())
                    images.append(image_array)
    
    # 将列表转换为NumPy数组  
    images_matrix = np.array(images)

    return images_matrix  

X = Preprocessing(image_dir)

label = []
x = 0
for j in range(X.shape[0]):
    label = label + [x]
    if (j+1) % 10 == 0:
        x += 1

y = np.array(label)

#加载数据
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42)

print('\n train_x:%s, train_y:%s, test_x:%s, test_y:%s'%(train_x.shape,train_y.shape,test_x.shape,test_y.shape)) 

#数据预处理
#X_train = train_x.reshape((60000,28*28))
#Y_train = train_y.reshape((60000,28*28))       #后面采用tf.keras.layers.Flatten()改变数组形状
X_train,X_test = tf.cast(train_x/255.0,tf.float32),tf.cast(test_x/255.0,tf.float32)     #归一化
y_train,y_test = tf.cast(train_y,tf.int16),tf.cast(test_y,tf.int16)

model = tf.keras.Sequential()  # 建立模型
# 第一层
model.add(tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=(112,92,1)))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
# 第二层
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Dropout(0.25))
# 第三层
model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Dropout(0.25))
# 全连接层
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
# 全连接层
model.add(tf.keras.layers.Dense(40, activation='softmax'))
print('\n',model.summary())     #查看网络结构和参数信息

#配置模型训练方法
#adam算法参数采用keras默认的公开参数，损失函数采用稀疏交叉熵损失函数，准确率采用稀疏分类准确率函数
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.05),loss='sparse_categorical_crossentropy',metrics=['sparse_categorical_accuracy'])   

#训练模型
#批量训练大小为64，迭代5次，测试集比例0.2
print('--------------')
nowtime = time.strftime('%Y-%m-%d %H:%M:%S')
print('Start Training'+str(nowtime))

history = model.fit(X_train,y_train,batch_size=10,epochs=200,validation_split=0.2)

print('--------------')
nowtime = time.strftime('%Y-%m-%d %H:%M:%S')
print('Stop Training'+str(nowtime))
#评估模型
model.evaluate(X_test,y_test,verbose=2) 

#保存整个模型
model.save('face.h5')


#结果可视化
# print(history.history)
loss = history.history['loss']          #训练集损失
val_loss = history.history['val_loss']  #测试集损失
acc = history.history['sparse_categorical_accuracy']            #训练集准确率
val_acc = history.history['val_sparse_categorical_accuracy']    #测试集准确率

plt.figure(figsize=(10,3))

plt.subplot(121)
plt.plot(loss,color='b',label='train')
plt.plot(val_loss,color='r',label='test')
plt.ylabel('loss')
plt.legend()

plt.subplot(122)
plt.plot(acc,color='b',label='train')
plt.plot(val_acc,color='r',label='test')
plt.ylabel('Accuracy')
plt.legend()

#使用模型
plt.figure()
for i in range(10):
    num = np.random.randint(1,80)

    plt.subplot(2,5,i+1)
    plt.axis('off')
    plt.imshow(test_x[num],cmap='gray')
    demo = tf.reshape(X_test[num],(1,112,92))
    y_pred = np.argmax(model.predict(demo))
    plt.title('标签值：'+str(test_y[num])+'\n预测值：'+str(y_pred))
#y_pred = np.argmax(model.predict(X_test[0:5]),axis=1)
#print('X_test[0:5]: %s'%(X_test[0:5].shape))
#print('y_pred: %s'%(y_pred))

#plt.ion()     
plt.show()
#plt.pause(5)
#plt.close()

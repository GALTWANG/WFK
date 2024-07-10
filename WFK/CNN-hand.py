########手写数字数据集##########
###########保存模型############
########1层隐含层（全连接层）##########
#60000条训练数据和10000条测试数据，28x28像素的灰度图像
#隐含层激活函数：ReLU函数
#输出层激活函数：softmax函数（实现多分类）
#损失函数：稀疏交叉熵损失函数
#输入层有784个节点，隐含层有128个神经元，输出层有10个节点
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

import time
print('--------------')
nowtime = time.strftime('%Y-%m-%d %H:%M:%S')
print(nowtime)

#指定GPU
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
plt.rcParams['font.sans-serif'] = ['SimHei']

#加载数据
mnist = tf.keras.datasets.mnist
(train_x,train_y),(test_x,test_y) = mnist.load_data()
print('\n train_x:%s, train_y:%s, test_x:%s, test_y:%s'%(train_x.shape,train_y.shape,test_x.shape,test_y.shape)) 

#数据预处理
#X_train = train_x.reshape((60000,28*28))
#Y_train = train_y.reshape((60000,28*28))       #后面采用tf.keras.layers.Flatten()改变数组形状
X_train,X_test = tf.cast(train_x/255.0,tf.float32),tf.cast(test_x/255.0,tf.float32)     #归一化
y_train,y_test = tf.cast(train_y,tf.int16),tf.cast(test_y,tf.int16)

# X_train = X_train[None, :, :, None]

num_classes = 10

#建立模型
model = tf.keras.Sequential()
model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(28,28,1)))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
model.add(tf.keras.layers.Dropout(0.25))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
print('\n',model.summary())     #查看网络结构和参数信息

#配置模型训练方法
#adam算法参数采用keras默认的公开参数，损失函数采用稀疏交叉熵损失函数，准确率采用稀疏分类准确率函数
model.compile(optimizer='SGD',loss='sparse_categorical_crossentropy',metrics=['sparse_categorical_accuracy'])   

#训练模型
#批量训练大小为64，迭代5次，测试集比例0.2（48000条训练集数据，12000条测试集数据）
print('--------------')
nowtime = time.strftime('%Y-%m-%d %H:%M:%S')
print('Start Training'+str(nowtime))

history = model.fit(X_train,y_train,batch_size=64,epochs=5,validation_split=0.2)

print('--------------')
nowtime = time.strftime('%Y-%m-%d %H:%M:%S')
print('Stop Training'+str(nowtime))
#评估模型
model.evaluate(X_test,y_test,verbose=2) 

#保存整个模型
model.save('mnist.h5')


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
    num = np.random.randint(1,10000)

    plt.subplot(2,5,i+1)
    plt.axis('off')
    plt.imshow(test_x[num],cmap='gray')
    demo = tf.reshape(X_test[num],(1,28,28))
    y_pred = np.argmax(model.predict(demo))
    plt.title('标签值：'+str(test_y[num])+'\n预测值：'+str(y_pred))
#y_pred = np.argmax(model.predict(X_test[0:5]),axis=1)
#print('X_test[0:5]: %s'%(X_test[0:5].shape))
#print('y_pred: %s'%(y_pred))

#plt.ion() 
plt.show()
#plt.pause(5)
#plt.close()

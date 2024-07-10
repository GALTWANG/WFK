import tensorflow as tf
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image  
from sklearn.model_selection import train_test_split

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
    #if (j+1) % 11 == 0:
    if (j+1) % 10 == 0:
    #if (j+1) % 26 == 0:
        x += 1

y = np.array(label)

#加载数据
train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42)

X_train,X_test = tf.cast(train_x/255.0,tf.float32),tf.cast(test_x/255.0,tf.float32)     #归一化
y_train,y_test = tf.cast(train_y,tf.int16),tf.cast(test_y,tf.int16)

# 假设你的模型文件名为'my_model.h5'，这是一个HDF5格式文件
model_path = 'face_3.h5'

# 使用load_model函数加载模型
model = tf.keras.models.load_model(model_path)

model.evaluate(X_test,y_test,verbose=2) 
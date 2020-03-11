#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
tf.contrib.eager.enable_eager_execution()


# In[9]:


fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()


# In[3]:


train_images.shape,train_labels.shape,test_images.shape,test_labels.shape


# In[4]:


# train_images[0]


# In[5]:


#labels对应的物品名称
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


# In[6]:


plt.figure(figsize=(1,1))
plt.imshow(train_images[0],cmap = 'binary')
# plt.colorbar()
# plt.grid(False)
plt.show()


# In[7]:


train_images = train_images / 255.0
test_images = test_images / 255.0


# In[8]:


plt.figure(figsize=(6,6))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()


# In[16]:


model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),  #对（batch_size，28,28）数据进行扁平化 —> （batch_size,784）
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(#optimizer= tf.train.GradientDescentOptimizer(learning_rate=0.01), #tf1.9，eager模式下，要用tf.train的优化器
              optimizer= tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# In[ ]:


model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),  #对（batch_size，28,28）数据进行扁平化 —> （batch_size,784）
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])


# In[12]:


model.fit(train_images, train_labels, epochs=10)


# In[10]:


test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print('\nTest accuracy:', test_acc)


# ## 模型保存为H5类型

# In[12]:


model.save('fashion_mnist.h5')


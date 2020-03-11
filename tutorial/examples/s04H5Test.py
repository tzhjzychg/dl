#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
tf.contrib.eager.enable_eager_execution()


# ## 加载模型

# In[3]:


model = tf.keras.models.load_model('../../output/models/fashion_mnist.h5')


# ## 模型测试

# In[4]:


#图像数据加载
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()


# In[8]:


class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


# In[9]:


x1 = train_images[0]/255
x = np.reshape(x1,(1,28,28))


# In[10]:


y = model.predict(x = x)


# In[11]:


plt.imshow(train_images[0],cmap = 'binary')
plt.xlabel(class_names[np.argmax(y)])
plt.show()


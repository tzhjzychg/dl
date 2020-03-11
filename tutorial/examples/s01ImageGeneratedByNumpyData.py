#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()


# In[3]:


#labels对应的物品名称
class_names = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


# In[4]:


plt.figure(figsize=(1,1))
plt.imshow(train_images[0],cmap = 'binary')
# plt.colorbar()
# plt.grid(False)
plt.show()


# ### 图片保存

# In[11]:


from PIL import Image
#Python Imaging Library
im = Image.fromarray(train_images[0])
im.save("your_file.jpg")


# In[26]:


for i in range(60000):
    im = Image.fromarray(train_images[i])
    im.save("data/train/{}.png".format(i))


# In[27]:


for i in range(10000):
    im = Image.fromarray(test_images[i])
    im.save("data/test/{}.jpg".format(i))


# ### labels 保存

# In[41]:


train_labels.tofile('data/train.m')


# In[31]:


test_labels.tofile('data/test.m')


# ### 测试

# In[32]:


import numpy as np


# In[44]:


a = np.fromfile('data/train.m' ,dtype=np.int8)


# In[46]:


a[0:10]


# In[47]:


train_labels[0:10]


#!/usr/bin/env python
# coding: utf-8

# In[96]:


import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# tf.contrib.eager.enable_eager_execution()

#labels对应的物品名称
class_names = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


# ## 加载H5模型

# In[97]:


def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

# 重置图
tf.reset_default_graph()
# 新建会话
tf.keras.backend.set_session(get_session())
model = tf.keras.models.load_model('../../output/models/fashion_mnist.h5')


# ## 模型信息查看

# In[146]:


#查看模型层信息
model.summary()


# In[147]:


#查看某层信息
model.layers[0].name


# In[6]:


# 查看输入节点
input_node_names = [node.name for node in model.inputs]
print('Input nodes names are: %s'%str(input_node_names))


# In[7]:


# 查看输出节点,重命名
orig_output_node_names = [node.name for node in model.outputs]
print('output nodes names are: %s'%str(orig_output_node_names))


# ## 图像预处理
#     添加reshape，归一化的ops

# In[98]:


# 添加输入图像
image = tf.placeholder(tf.uint8,shape=[None,None,None,3],name='input_image')
# 图像reshape
img_reshape = tf.image.resize_images(image,(28,28),method=3)
# 图像灰度化
img_gray = tf.image.rgb_to_grayscale(images=img_reshape,name='gray_image')
# 图像归一化
img_reshape2 = tf.reshape(img_gray,(-1,28,28))
img_prepared = (255 - img_reshape2)/255


# In[136]:


# # 图像预处理检测
# sess = tf.Session()
# a = Image.open('test01.jpg')
# b = np.asarray(a)
# b = b.reshape(1,460, 500, 3)
# c = sess.run(img_process,feed_dict={image:b})
# c = c*255
# d  = d.astype(np.uint8)
# Image.fromarray(d)


# In[99]:


# 添加到网络中
outputs_rate = model(img_prepared)
outputs_catg = tf.argmax(outputs_rate,1,name='output_category')


# In[100]:


# 获取当前model的后台框架实例化的seeion
sess = tf.keras.backend.get_session()


# ## test

# In[101]:


a = Image.open('test06.jpg')
b = np.asarray(a)
print(b.shape)


# In[102]:


b = b.reshape(1,500, 500, 3)
x = sess.run(outputs_catg,feed_dict={image:b})
class_names[x[0]]


# In[103]:


x = sess.run(img_prepared,feed_dict={image:b})
plt.figure(figsize=(1,1))
plt.imshow(x[0],cmap = 'binary')
# plt.colorbar()
# plt.grid(False)
plt.show()


# ## 保存pb

# In[104]:


# 模型保存时，eager模式关闭下,只对tf.keras的model保存
# tf.saved_model.simple_save(sess,'./data/1',inputs={'input':model.input},outputs={'output':model.output})
# 通过sess保存,outputs自定义信号如下，为符合平台在线测试用的
tf.saved_model.simple_save(sess,export_dir='./data/5',
                           inputs={"input_image": image},outputs={"labels": outputs_catg,
                                                                  "logits": outputs_rate})


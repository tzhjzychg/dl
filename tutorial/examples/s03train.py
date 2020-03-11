#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
tf.contrib.eager.enable_eager_execution()


# ## 类别-index

# In[2]:


class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


# ## tf.data读取tfrecords文件
#     tf.data.TFRecordDataset

# In[3]:


filenames = ['../../input/datasets/train.tfrecords']
raw_dataset = tf.data.TFRecordDataset(filenames)


# ## 数据解析
#     tf.parse_single_example

# In[4]:


# Create a description of the features.
feature_description = {
    'img': tf.FixedLenFeature([], tf.string, default_value=''), #矩阵数据需要tf.string格式
    'label': tf.FixedLenFeature([], tf.int64, default_value=0)
#     'feature1': tf.io.FixedLenFeature([], tf.int64, default_value=0),
#     'feature2': tf.io.FixedLenFeature([], tf.string, default_value=''),
#     'feature3': tf.io.FixedLenFeature([], tf.float32, default_value=0.0),
}

def _parse_function(example_proto):
  # Parse the input `tf.Example` proto using the dictionary above.
    parsed_features = tf.parse_single_example(example_proto, feature_description)
    
    # return parsed_features
    
    #图像数据：由tf.tsring转为matrix，再归一化即可（该过程放在网络里边）
    img = tf.decode_raw(parsed_features['img'], tf.uint8)/255
    img = tf.reshape(img, (28, 28))
    
    #标签数据：无变化
    #tf.decode_raw(img_encode.values, tf.uint8)
    label = tf.cast(parsed_features['label'],tf.float32)
    label = tf.reshape(label,(1,))
    
    return (img,label) 


# In[5]:


parsed_dataset = raw_dataset.map(_parse_function)
parsed_dataset


# ### 查看结果

# In[6]:


for parsed_record in parsed_dataset.take(1):
    print(repr(parsed_record[1]))


# ## 创建datasets迭代器
#     
#     shuffle
#     batch
#     make_one_shot_iterator

# In[7]:


#batch 设置
parsed_dataset = parsed_dataset.shuffle(buffer_size=1000).batch(32).repeat(10)


# In[8]:


for parsed_dataset_iter in parsed_dataset.take(1):
    print(parsed_dataset_iter)


# In[20]:


# 数据查看，batchsize = 32 
imgs = parsed_dataset_iter[0].numpy().reshape((32,28,28))
labels = parsed_dataset_iter[1].numpy()

plt.figure(figsize=(7,7))
for i in range(32):
    plt.subplot(4,8,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    
    plt.imshow(imgs[i],cmap = 'binary')
    plt.xlabel(class_names[int(labels[i])])
plt.show()


# In[19]:


parsed_dataset_iter[0].numpy()


# ### 暂时不用

# In[10]:


#创建了数据迭代器
a = parsed_dataset.make_one_shot_iterator()
#查看下结果
a.get_next()[0]


# ## X-Y指定?

# 只要保证parsed_datasets为两个属性，且X在前，Y在后即可

# ## 建模

# In[11]:


model = tf.keras.Sequential([
    #添加归一化，rehsape层
#     tf.keras.layers.Lambda(lambda x:x), #函数处理，train报错，未解决，考虑是sequential的局限？
    #tf.keras.layers.Reshape((28,28)), #重置形状
    tf.keras.layers.Flatten(input_shape=(28, 28)),  #对（batch_size，28,28）数据进行扁平化 —> （batch_size,784）
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer = tf.train.AdamOptimizer(),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# In[12]:


model.fit(parsed_dataset, epochs=10,steps_per_epoch= int(60000/32))
# model.fit(parsed_dataset, epochs=1,steps_per_epoch= 5)


# In[13]:


model.summary()


# ## 推理

# In[14]:


import numpy as np
y = model.predict(x=np.reshape(imgs[23]/255,(1,28*28)))


# In[15]:


plt.imshow(imgs[23],cmap = 'binary')
plt.xlabel(class_names[np.argmax(y)])
plt.show()


# ## 模型保存

# In[16]:


model.save('../../output/models/fashion_mnist.h5')


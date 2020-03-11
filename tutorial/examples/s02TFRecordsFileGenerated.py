#!/usr/bin/env python
# coding: utf-8

# In[3]:


import tensorflow as  tf
import os
from PIL import Image
import numpy as np


# In[14]:


def get_tfrecords_file(figure_path,label_path_name,target_name,target_path):
    #读取labels
    labels = np.fromfile(label_path_name,dtype=np.int8)
    print('label file is read !')
    #读图片，写文件
    writer = tf.python_io.TFRecordWriter(target_path+'/'+"{}.tfrecords".format(target_name))
    for img_name in os.listdir(figure_path):
        if 'png' not in img_name:
            pass
        else:
            img_path = figure_path + '/' + img_name
            Img = Image.open(img_path)
            # 将图片转化为固定大小
            #img = Img.resize((128, 128))
            # 将图片转化为二进制格式
            img_raw = Img.tobytes()
            # 创建样本
            example = tf.train.Example(
                features = tf.train.Features(
                    feature = {
                    # 设定标签和图像数据的属性
                        "label": tf.train.Feature(int64_list = tf.train.Int64List(value = [labels[int(img_name.split('.')[0])]])),
                        "img": tf.train.Feature(bytes_list = tf.train.BytesList(value = [img_raw]))
                    }))
            writer.write(example.SerializeToString())
    writer.close()
    print('TFRecords File Generation Finished!')
    return 0


# In[12]:


#生成test
get_tfrecords_file('data/test','data/test.m','test','.')


# In[15]:


#生成train
get_tfrecords_file('data/train','data/train.m','train','.')


# ### 测试
# 

# In[ ]:





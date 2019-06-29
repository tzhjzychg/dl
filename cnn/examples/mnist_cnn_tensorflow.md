### mnist cnn model
    <simple cnn and look at the 'mnist_cnn_tensorflow.png'>
input:

      [6000,784] #6000条数据，每条为28*28的图片平展后的向量

output:

      [6000,10]  #6000条数据，每条为1个数字(即0-9)的one-hot-encoding编码

cnn process_convolution:

      W = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1))     #初始化权值（卷积核）
      
      b_conv_1 = tf.Variable(tf.constant(0.1, shape=shape))               #初始化偏差

      conv2d = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')   #卷积过程
      
      conv_h1 = tf.nn.relu(conv2d + b_conv1)                              #relu处理得到临时logit结果

cnn process_maxpooling:

     有多种，如logistic、cross entropy等
     
     cross entropy = sum(-y_*log(y)),即预测概率与实际概率越贴近，该值越小
     
     


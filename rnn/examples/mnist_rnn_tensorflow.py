
# coding: utf-8

# In[1]:


# -*- coding: UTF-8 -*-
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow.contrib as contrib
tf.set_random_seed(1)


# In[3]:


##导入数据
mnist = input_data.read_data_sets('../Datasets/',one_hot=True)


# In[5]:


# 超参数
lr = 0.001    # learning rate
training_iters = 100000  # train step 上限
batch_size = 128
# 输入按照时间序列展开是：每个字符数据的输入 = 单次28个值（列），需要输入28次（行）
n_inputs = 28       # MNIST data input (img shape: 28*28)
n_steps = 28      # time steps

n_hidden_units = 200   # neurons in hidden layer
n_classes = 10   # MNIST classes (0-9 digits)


# In[6]:


# x,y placeholder
x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_classes])

# 对 weights biases 的初始值的定义
weights = {
    # shape (28, 128)
    "in": tf.Variable(tf.truncated_normal([n_inputs, n_hidden_units])),
    # shape (128, 10)
    'out': tf.Variable(tf.truncated_normal([n_hidden_units, n_classes]))
}
biases = {
    # shape (128, )
    'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ])),
    # shape (10, )
    'out': tf.Variable(tf.constant(0.1, shape=[n_classes, ]))
}


# In[7]:


# 定义 RNN 的主体结构（input_layer, cell, output_layer）
def RNN(X,weights,biases):
    #--------before RNN cell--------#
    # 原始的 X 是 3 维数据, 我们需要把它变成 2 维数据才能使用 weights 的矩阵乘法
    # X ==> (128 batches * 28 steps, 28 inputs)
    X = tf.reshape(X,[-1,n_inputs])
    # X_in = W*X + b
    X_in = tf.matmul(X,weights['in'])+biases['in']
    # X_in ==> (128 batches, 28 steps, 128 hidden) 换回3维
    X_in = tf.reshape(X_in,[-1,n_steps,n_hidden_units])

    #--------RNN cell--------#
    # 使用 basic RNN Cell.
    rnn_cell = contrib.rnn.BasicRNNCell(num_units=n_hidden_units)
    init_state = rnn_cell.zero_state(batch_size, dtype=tf.float32)
    
    ## 如果 inputs 为 (batches, steps, inputs) ==> time_major=False(默认); 如果 inputs 为 (steps, batches, inputs) ==> time_major=True;
    ## 返回两个结果：
    ### outputs：表示rnn循环隐含层的直接结果输出，shape为（128,28,200）(可能有其他形状，如上解释)；
    ### final_state:state： 最终的状态.
    # 一般情况下state的形状为 [batch_size(128), cell.output_size(200)],实际上就是output中最后一步的结果：
    #                                    即output[:,-1,:]为[128,200]
    # 如果cell是LSTMCells,则state将是包含每个单元格的LSTMStateTuple的元组，state的形状为[2，batch_size, cell.output_size ]
    outputs, final_state = tf.nn.dynamic_rnn(cell=rnn_cell,inputs=X_in,initial_state=init_state)

    #--------after RNN cell--------#
    results = tf.matmul(final_state,weights['out'])+biases['out']

    return results #final_state[1],outputs,final_state[0] 


# In[8]:


# 定义好了 RNN 主体结构后, 我们就可以来计算 cost 和 train_op
pred = RNN(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y))
train_op = tf.train.AdamOptimizer(lr).minimize(cost)
# train_op = tf.train.AdadeltaOptimizer(lr).minimize(cost)  ## 这出错了啦！！！！！！

correct_pred = tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))


# In[9]:


init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)
    step = 0
    while step * batch_size < training_iters:
        batch_xs,batch_ys = mnist.train.next_batch(batch_size)
        batch_xs = batch_xs.reshape([batch_size,n_steps,n_inputs])
        sess.run([train_op],feed_dict={x:batch_xs,y:batch_ys})

        if step % 20 ==0:
            print(sess.run(accuracy,feed_dict={x:batch_xs,y:batch_ys}))
        step += 1
    
#     print(sess.run(pred.shape))
    
#     print(sess.run(biases['out']))


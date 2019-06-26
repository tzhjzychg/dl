import tensorflow as tf
import numpy as np
from matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from keras.datasets import mnist

#加载MNIST数据
(X_train_image,Y_train_label),(X_test_image,Y_test_label) = mnist.load_data()

#数据查看
X_train_image.shape,Y_train_label.shape
#((60000, 28, 28), (60000,))

#数据矩阵打印
def plot(image):
    fig = plt.gcf()
    plt.imshow(image,cmap = 'binary')
    plt.show()
    
plot(X_train_image[0])

#建立模型，单隐层
#输入输出
x = tf.placeholder('float',[None,784])
y_ = tf.placeholder('float',[None,10])
#权值阈值
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))
#推到过程
y = tf.nn.softmax(tf.matmul(x,W)+b)
#损失函数（交叉熵）
cross_entropy = -tf.reduce_sum(y_*tf.log(y)) 
#训练引擎（梯度下降法，学习速率0.01，最小化损失值）
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

#模型执行
#创建回话
sess = tf.InteractiveSession()
#初始化变量
sess.run(tf.global_variables_initializer())
#建立训练过程（确定循环次数600、样本输入）
for ii in range(600):
    #随机获取100个数据的索引
    selected_columns = np.random.choice(60000,100) 
    #获取输入的X
    selected_X_return = X_train_image[selected_columns,:,:].reshape((100,784))/255.0  #获取100个样本数据，并且拉伸（降维）为100*784矩阵
    #获取输入的Y
    selected_Y = Y_train_label[selected_columns]
    selected_Y_onehort = np.zeros([100,10])
    for i,j in enumerate(selected_Y):
        selected_Y_onehort[i,j] = 1
    #执行训练
    sess.run(train_step,feed_dict={x: selected_X_return, y_: selected_Y_onehort})
    
    #过程精度打印
    if ii% 20 == 0:
        print('完成了20步，10000条测试样本数据的准确率为：')
        correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1)) #判断y与y_的差异
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float")) #类型转换float，求均值
        test_X = X_test_image.reshape((10000,784))/255.0
        test_Y = np.zeros([10000,10])
        for i,j in enumerate(Y_test_label):
            test_Y[i,j] = 1.0
        print(sess.run(accuracy,feed_dict={x: test_X, y_: test_Y}))



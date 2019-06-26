### mnist nn model
    <simple nn with single layer>
input:

      [6000,784] #6000条数据，每条为28*28的图片平展后的向量

output:

      [6000,10]  #6000条数据，每条为1个数字(即0-9)的one-hot-encoding编码

logit function:

      y = Softmax(xW+b)

      Softmax:e^x1/(e^x1 + e^x2 + e^x3 + …),为了得到对分类结果进行概率统计

loss function:

     有多重，如logistic、cross entropy等
     
     cross entropy = sum(-y_*log(y)),即预测概率与实际概率越贴近，该值越小
     
     

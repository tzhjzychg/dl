## RNN原理

### RNN结构分解
  
  【模型理解】 
  
       不要被时间维度上展开图误导（比较流行的那个，我第一次看时顿生迷茫），实际上是RNN可以理解成一个全连接网络结构如下：
![Image](https://github.com/tzhjzychg/dl/blob/master/material/RNN%E7%BB%8F%E5%85%B8%E7%BB%93%E6%9E%84.png)
  
  【字符说明】 
  
        RNN结构如图所示，其中网络的各个组件如下：
        Xt ：输入X的在t时刻的值，值的维度不确定（N批次M维）；
        W ：权值，包括in/out；
        B  ：偏差，包括in/out；
        Ht：隐含层处理过程，需要加载上次的Ht-1，得到本次的Ht以及隐含层输出；
        Wt：前后迭代（循环）的共享权值。

  
  【公式说明】 
  
        输入：input = Xt*Win+Bin
        隐含层：middle = tanh(input + Ht-1*W)
        输出：output = softmax(middle*Wout + Bout)

### MNIST数据集的RNN结构分解

[*诸图引用请添加来源*](https://github.com/tzhjzychg/dl/blob/master/material/)

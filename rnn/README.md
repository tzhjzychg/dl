## RNN原理

### RNN结构分解
  
  【模型理解】 
  
       不要被时间维度上展开图误导（比较流行的那个，我第一次看时顿生迷茫），实际上是RNN可以理解成一个全连接网络结构如下：
<p align="center">
	<img src="https://github.com/tzhjzychg/dl/blob/master/material/RNN%E7%BB%8F%E5%85%B8%E7%BB%93%E6%9E%84.png" alt="Sample">
	<p align="center">
		<em>RNN典型结构图</em>
	</p>
</p>
  
  【字符说明】 
  
        RNN结构如图所示，其中网络的各个组件如下：
        Xt ：输入X的在t时刻的值，值的维度不确定（N批次M维）；
        Win/Wout ：输入输出权值；
        Bin/Bout ：输入输出偏差；
        Ht：隐含层处理过程，需要加载上次的Ht-1，得到本次的Ht以及隐含层输出；
        Wt：前后迭代（循环）的共享权值。

  
  【公式说明】 
  
        输入：input = Xt*Win+Bin
        隐含层：middle = tanh(input + Ht-1*W)
        输出：output = softmax(middle*Wout + Bout)

### MNIST数据集的RNN结构分解
    
    输入是MNIST手写数据集，batch为128，单个样本大小为（28,28），即单个批次数据维度为（128,28,28）；
  
  【RNN循环训练的理解】
  
       图片按照28个循环训练，每次训练图像的1行，即第一次网络的输入的数据为（128,1,28），第二次为（128,2,28）……，第28次为（128,28,28）。
       其中（128,1,28）表示128个批次，每个图型矩阵的第1行的28个列数据。流程拆分如下：
<p align="center">
	<img src="https://github.com/tzhjzychg/dl/blob/master/material/mnist%E6%A0%B7%E6%9C%AC%E5%9B%BE%E8%A7%A31.png" alt="Sample">
	<p align="center">
		<em>RNN循环过程输入数据截取模式</em>
	</p>
</p>
    
   【数据维度变化】
       
       单次循环中模型输入：假设当前为第1个时刻，输入取源数据集[:,0,:]，即输入数据维度为（128,28）；
       输入数据转化:input为(128,200);
       隐含层转化：Ht为(128,200);
       输出层转化：(128,10)。
       
<p align="center">
	<img src="https://github.com/tzhjzychg/dl/blob/master/material/mnist%E6%A0%B7%E6%9C%AC%E5%9B%BE%E8%A7%A32.png" alt="Sample">
	<p align="center">
		<em>RNN过程数据维度变化</em>
	</p>
</p>

   【补充：隐含层RNN Cell的输出】
       
       上述是循环中的某次过程，若是全部循环的，需要在中间增加一个维度，即H all维度为（:,step_n,:）。
       RNN Cell输出看应用情况，若是一般分类任务，则cell 的 output（H all）在最后的t时刻的输出，即[:,-1,:]，或者Hn，或者status。
       解释如下图：

<p align="center">
	<img src="https://github.com/tzhjzychg/dl/blob/master/material/%E7%AE%80%E5%8D%95RNN%E5%9B%BE%E8%A7%A3.png" alt="Sample">
	<p align="center">
		<em>简单RNN分类任务全过程</em>
	</p>
</p>

[*诸图引用请添加来源*](https://github.com/tzhjzychg/dl/blob/master/material/)

[*参考资料*](https://www.jianshu.com/p/f89c7f540f6e)

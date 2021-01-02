### Batch Normalization 

作用防止梯度消失，原因是能将神经元输出值拉回梯度变化明显的区域。针对sigmoid和tanh这种有梯度饱和区域的激活函数。<br/>
所有的归一化都能起到平滑损失平面的作用，加速收敛速度？

#### BN & MLP
在MLP中，针对N,H的输入，对H上的每个神经元，在N（batch size）这个维度上做均值和方差的归一化

![](https://latex.codecogs.com/svg.latex?\hat{x}^{(k)}=\alpha^{(k)}\frac{x^{(k)}-E[x^{(k)}]}{\sqrt{Var[x^{(k)}]+\epsilon}}+\beta^{(k)})

加了两个可以学习的变量![](https://latex.codecogs.com/svg.latex?\alpha)和![](https://latex.codecogs.com/svg.latex?\beta)用于控制网络能够表达直接映射，也就是能够还原BN之前学习到的特征。 <br/>

#### BN & CNN
在CNN中，针对N,C,H,W的输入，对每个C，在N,H,W三个维度上做均值和方差的归一化，N即是batch size。

#### BN的缺点：
* batch size小的时候效果不好，稳定性差 
* 需要额外开辟内存缓存变量，空间消耗大
* 不适合RNN这种动态长度的模型，batch中的样本长度不一致，靠后的输入不能算。虽然实际操作中可以把长度差不多的并入一个batch中。
* BN不适合NLP，[引用](https://github.com/DA-southampton/NLP_ability/blob/master/%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E8%87%AA%E7%84%B6%E8%AF%AD%E8%A8%80%E5%A4%84%E7%90%86/Transformer/NLP%E4%BB%BB%E5%8A%A1%E4%B8%AD-layer-norm%E6%AF%94BatchNorm%E5%A5%BD%E5%9C%A8%E5%93%AA%E9%87%8C.md)分析，一个batch中同一位置的不同单词，表示不同特征，在此维度归一化不合理。

### Layer Normalization

#### LN & MLP

![](https://latex.codecogs.com/svg.latex?\mu^l=\frac{1}{H}\sum_{i=1}^Ha_i^l)

![](https://latex.codecogs.com/svg.latex?\sigma^l=\sqrt{\frac{1}{H}\sum_{i=1}^H({a_i^l-\mu^l})^2})

其中![](https://latex.codecogs.com/svg.latex?H)是隐藏层中节点数量，![](https://latex.codecogs.com/svg.latex?l)是MLP的层数。<br/>

![](https://latex.codecogs.com/svg.latex?\mathbf{h}={f(\frac{\mathbf{g}}{\sqrt{\sigma^2+\epsilon}}\odot(\mathbf{a}-\mu)+\mathbf{b})}) 
<br/>

和BN一样，用增益![](https://latex.codecogs.com/svg.latex?\mathbf{g})和偏置![](https://latex.codecogs.com/svg.latex?\mathbf{b})以及激活函数![](https://latex.codecogs.com/svg.latex?f)来保证归一化操作不破坏之前的信息。

#### LN & RNN
在RNN中，对每个时间步的输出![](https://latex.codecogs.com/svg.latex?\mathbf{h_t})作上述同样的归一化。

#### LN的优点
* LN有正则化的作用，得到的模型不容易过拟合。

### Thansformer中使用Layer Normalization
[引用](https://www.zhihu.com/question/395811291/answer/1260290120)这里的说法
* layer normalization有助于得到一个球体空间中符合0均值1方差高斯分布的 embedding， batch normalization不具备这个功能。
* layer normalization可以对transformer学习过程中由于多词条embedding累加可能带来的“尺度”问题施加约束，相当于对表达每个词一词多义的空间施加了约束，有效降低模型方差。batch normalization也不具备这个功能。

参考资料：
* [transformer 为什么使用 layer normalization，而不是其他的归一化方法](https://www.zhihu.com/question/395811291/answer/1260290120)
* [模型优化之Batch Normalization](https://zhuanlan.zhihu.com/p/54171297)
* [模型优化之Layer Normalization](https://zhuanlan.zhihu.com/p/54530247)
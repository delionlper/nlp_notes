### Scaled Dot-Product Attention
![](https://latex.codecogs.com/svg.latex?Attention(Q,K,V)=softmax(\frac{QK^T}{\sqrt{d_k}})V) <br/>

#### softmax归一化的特性

对于输入向量![](https://latex.codecogs.com/svg.latex?x\in\mathbb{R}^d)，softmax函数将其归一化到一个概率分布![](https://latex.codecogs.com/svg.latex?\hat{y}\in\mathbb{R}^d)。如果输入的数量级很大（每个元素都很大），那么具有最大值的位置![](https://latex.codecogs.com/svg.latex?\hat{y}_k)会非常接近于1。
<br/>
假设![](https://latex.codecogs.com/svg.latex?x=[a,a,2a]^\top)
* a=1时，![](https://latex.codecogs.com/svg.latex?\hat{y}_3=0.57611)
* a=10时，![](https://latex.codecogs.com/svg.latex?\hat{y}_3=0.99991)
* a=100时，![](https://latex.codecogs.com/svg.latex?\hat{y}_3\approx1.0)
<br/>

[大佬](https://www.zhihu.com/question/339723385/answer/782509914)给出了此种情况下的分布图，表明在数量级较大时，softmax将几乎全部的概率分布都分配给了最大值对应的标签。

另外根据[softmax梯度的推导]()，可得：<br/>
![](https://latex.codecogs.com/svg.latex?\frac{\partial{\hat{y}}}{\partial{x}}=diag{(\hat{y})}-\hat{y}{\hat{y}}^\top\in{\mathbb{R}}^{d\times{d}})

<br/>

![](https://latex.codecogs.com/svg.latex?\frac{\partial{\hat{y}}}{\partial{x}}=\begin{bmatrix}\hat{y}_1&0&\cdots&0\\0&\hat{y}_2&\cdots&0\\\vdots&\vdots&\ddots&\vdots\\0&0&\cdots&\hat{y}_d\end{bmatrix}-\begin{bmatrix}{\hat{y}_1}^2&\hat{y}_1\hat{y}_2&\cdots&\hat{y}_1\hat{y}_d\\\hat{y}_2\hat{y}_1&{\hat{y}_2}^2&\cdots&\hat{y}_2\hat{y}_d\\\vdots&\vdots&\ddots&\vdots\\\hat{y}_d\hat{y}_1&\hat{y}_d\hat{y}_2&\cdots&{\hat{y}_d}^2\end{bmatrix})

当输入很大时，![](https://latex.codecogs.com/svg.latex?\hat{y})会成为one-hot向量，梯度会消失为0。


#### 维度的根号来缩放
![](why_scaled.png)
<br/>

假设向量![](https://latex.codecogs.com/svg.latex?q)和![](https://latex.codecogs.com/svg.latex?k)的各个分量是互相独立的随机变量，均值是0，方差是1，那么点积![](https://latex.codecogs.com/svg.latex?q\cdot{k})的均值是0，方差是![](https://latex.codecogs.com/svg.latex?d_k)。

根据[大佬](https://www.zhihu.com/question/339723385/answer/782509914)的推导，两个均值为0方差为1的随机变量的乘积的均值为0，方差也为1。

<br/>

由于![](https://latex.codecogs.com/svg.latex?QK^\top)中每个元素是![](https://latex.codecogs.com/svg.latex?d_k)个随机变量pair的乘积和，所以变换之后的每个元素服从均值为0，方差为![](https://latex.codecogs.com/svg.latex?d_k)的分布。

<br/>

为了将方差稳定到1，所以每个元素要除以![](https://latex.codecogs.com/svg.latex?\sqrt{d_k})。

参考资料：
* [transformer中的attention为什么scaled?](https://www.zhihu.com/question/339723385/answer/782509914)
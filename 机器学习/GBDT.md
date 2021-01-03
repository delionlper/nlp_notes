### CART回归树

输入：训练数据集D <br/>
输出：回归树![](https://latex.codecogs.com/svg.latex?f(x)) <br/>
在训练数据集所在的输入空间中，递归地将每个区域划分为两个子区域并决定每个子区域上的输出值，构建二叉决策树。

1. 选择最有切分变量![](https://latex.codecogs.com/svg.latex?j)和切分点![](https://latex.codecogs.com/svg.latex?s)，求解<br/>
![](https://latex.codecogs.com/svg.latex?\mathop{min}_{j,s}\left[\mathop{min}_{c_1}\sum_{x_i\in{R_1(j,s)}}{(y_i-c_1)}^2+\mathop{min}_{c_2}\sum_{x_i\in{R_2(j,s)}}{(y_i-c_2)}^2\right])<br/>
遍历变量![](https://latex.codecogs.com/svg.latex?j)，对固定的切分变量![](https://latex.codecogs.com/svg.latex?j)扫描切分点![](https://latex.codecogs.com/svg.latex?s)，选择使上式达到最小值的对![](https://latex.codecogs.com/svg.latex?(j,s))。
2. 用选定的对![](https://latex.codecogs.com/svg.latex?(j,s))划分区域并决定相应的输出值：<br/>
![](https://latex.codecogs.com/svg.latex?R_1(j,s)=\left\\{x|x^{(j)}\leq{s}\right\\},R_2(j,s)=\left\\{x|x^{(j)}>{s}\right\\}) <br/>
![](https://latex.codecogs.com/svg.latex?\hat{c}_m=\frac{1}{N_m}\sum_{x_i\in{R_m(j,s)}}y_i,\quad{x\in{R_m}},\quad{m=1,2})
3. 继续对两个子空间调用步骤1，2，直至满足停止条件。
4. 将输入空间划分为M个区域![](https://latex.codecogs.com/svg.latex?R_1,R_2,...,R_m)，生成决策树：<br/>
![](https://latex.codecogs.com/svg.latex?f(x)=\sum_{m=1}^M\hat{c}_mI{(x\in{R_m})})

### GBDT
输入：训练数据集![](https://latex.codecogs.com/svg.latex?T=\\{{(x_1,y_1)},{(x_2,y_2)},...,{(x_N,y_N)}\\})，其中![](https://latex.codecogs.com/svg.latex?x_i\in\mathcal{X}\subseteq\mathbb{R}^n)，![](https://latex.codecogs.com/svg.latex?y_i\in{\mathcal{Y}}\subseteq\mathbb{R})，损失函数![](https://latex.codecogs.com/svg.latex?L(y,f(x)))。<br/>
输出：回归树![](https://latex.codecogs.com/svg.latex?\hat{f}(x))

1. 初始化<br/>
![](https://latex.codecogs.com/svg.latex?f_0{(x)}=arg\mathop{min}_{c}\sum_{i=1}^{N}{L(y_i,c)})
2. 对![](https://latex.codecogs.com/svg.latex?m=1,2,...,M) <br/>
    + 对![](https://latex.codecogs.com/svg.latex?i=1,2,...,N)，计算<br/>
![](https://latex.codecogs.com/svg.latex?r_{mi}=-{\[\frac{\partial{L(y_i,f(x_i))}}{\partial{f(x_i)}}\]}_{f(x)=f_{m-1}(x)})
    + 对![](https://latex.codecogs.com/svg.latex?r_{mi})拟合一个回归树，得到第m颗树的叶节点区域![](https://latex.codecogs.com/svg.latex?R_{mj},j=1,2,..,J)
    + 对![](https://latex.codecogs.com/svg.latex?j=1,2,..,J)，计算<br/>
    ![](https://latex.codecogs.com/svg.latex?c_{mj}=arg\mathop{min}_{c}\sum_{x_i\in{R_{mj}}}L(y_i,f_{m-1}(x_i)+c))
    + 更新 <br/>
    ![](https://latex.codecogs.com/svg.latex?f_m(x)=f_{m-1}(x)+\sum_{j=1}^Jc_{mj}I{(x\in{R_{mj}})})
3. 得到回归树 <br/>
![](https://latex.codecogs.com/svg.latex?\hat{f}(x)=f_M(x)=\sum_{m=1}^M\sum_{j=1}^{J}c_{mj}I{(x\in{R_{mj}})})

算法第1步初始化，估计使损失函数极小化的常数值，它是只有一个根节点的树。<br/>
第2.1步计算损失函数的负梯度在当前模型的值，将它作为残差的估计，对于平方损失函数，它就是通常所说的残差。<br/>
第2.2步估计回归树叶节点区域，以拟合残差近似值。<br/>
第2.3步利用线性搜索估计叶节点区域的值，使损失函数最小化。<br/>
第2.4步更新回归树。<br/>
第3步得到输出的最终模型。

### GBDT用于分类
当GBDT用于k分类问题时，每个类别依据上述的步骤训练一个GBDT回归模型，k个GBDT子模型的目标是一个k维one-hot向量。最终在所有子模型的输出上加softmax做分类判断。

参考资料
* [统计学习方法 李航]()
* [深入理解GBDT多分类算法](https://zhuanlan.zhihu.com/p/91652813?utm_source=wechat_session)

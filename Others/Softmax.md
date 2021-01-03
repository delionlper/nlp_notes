### softmax梯度推导

<br/>

![](https://latex.codecogs.com/svg.latex?\hat{y}_i=\frac{e^{x_i}}{\sum{e^{x_j}}})

<br/>

![](https://latex.codecogs.com/svg.latex?\frac{\partial{\hat{y}_i}}{\partial{x_j}}=\begin{cases}\frac{e^{x_i}\sum-e^{x_i}e^{x_j}}{\sum^2}=\hat{y}_i(1-\hat{y}_j),&i=j\\\\-\frac{e^{x_i}e^{x_j}}{\sum^2}=-\hat{y}_i\hat{y}_j,&i\neq{j}\\\\\end{cases})

<br/>
化简成矩阵形式：<br/>

![](https://latex.codecogs.com/svg.latex?\frac{\partial{\hat{y}}}{\partial{x}}=diag{(\hat{y})}-\hat{y}{\hat{y}}^\top\in{\mathbb{R}}^{d\times{d}})

<br/>

![](https://latex.codecogs.com/svg.latex?\frac{\partial{\hat{y}}}{\partial{x}}=\begin{bmatrix}\hat{y}_1&0&\cdots&0\\\\0&\hat{y}_2&\cdots&0\\\\\vdots&\vdots&\ddots&\vdots\\\\0&0&\cdots&\hat{y}_d\end{bmatrix}-\begin{bmatrix}{\hat{y}_1}^2&\hat{y}_1\hat{y}_2&\cdots&\hat{y}_1\hat{y}_d\\\\\\hat{y}_2\hat{y}_1&{\hat{y}_2}^2&\cdots&\hat{y}_2\hat{y}_d\\\\\\vdots&\vdots&\ddots&\vdots\\\\\\hat{y}_d\hat{y}_1&\hat{y}_d\hat{y}_2&\cdots&{\hat{y}_d}^2\end{bmatrix})

-------------

### CrossEntropy的梯度推导

![](https://latex.codecogs.com/svg.latex?L=-\sum_{j=1}^Ty_jlog\hat{y}_j)

![](https://latex.codecogs.com/svg.latex?\begin{aligned}\frac{\partial%20L}{\partial{x_i}}&=-\sum_k{y_k}\frac{\partial{log\hat{y}_k}}{\partial{x_i}}\\\\&=-\sum_k{y_k}\frac{1}{\hat{y}_k}\frac{\partial{\hat{y}_k}}{\partial{x_i}}\\\\&=-\sum_k{y_k}\frac{1}{\hat{y}_k}\begin{cases}\hat{y}_k(1-\hat{y}_i)&i=k\\\\-\hat{y}_k\hat{y}_i&i\neq{k}\\\\\end{cases}\\\\&=-y_i(1-\hat{y}_i)+\sum_{k\neq{i}}y_k\hat{y}_i\\\\&=\hat{y}_i-y_i,\quad{when}\sum_{y_k}=1\end{aligned})

<br/>



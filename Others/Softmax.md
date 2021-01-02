### softmax梯度推导

![](https://latex.codecogs.com/svg.latex?\hat{y}_i=\frac{e^{x_i}}{\sum{e^{x_j}}})

<br/>

![](https://latex.codecogs.com/svg.latex?\frac{\partial{\hat{y}_i}}{\partial{x_j}}=\begin{cases}\frac{e^{x_i}\sum-e^{x_i}e^{x_j}}{\sum^2}=\hat{y}_i(1-\hat{y}_j),&i=j\\\\-\frac{e^{x_i}e^{x_j}}{\sum^2}=-\hat{y}_i\hat{y}_j,&i\neq{j}\\\\\end{cases})

<br/>
化简成矩阵形式：<br/>

![](https://latex.codecogs.com/svg.latex?\frac{\partial{\hat{y}}}{\partial{x}}=diag{(\hat{y})}-\hat{y}{\hat{y}}^\top\in{\mathbb{R}}^{d\times{d}})

<br/>

![](https://latex.codecogs.com/svg.latex?\frac{\partial{\hat{y}}}{\partial{x}}=\begin{bmatrix}\hat{y}_1&0&\cdots&0\\\\0&\hat{y}_2&\cdots&0\\\\\vdots&\vdots&\ddots&\vdots\\\\0&0&\cdots&\hat{y}_d\end{bmatrix}-\begin{bmatrix}{\hat{y}_1}^2&\hat{y}_1\hat{y}_2&\cdots&\hat{y}_1\hat{y}_d\\\\\\hat{y}_2\hat{y}_1&{\hat{y}_2}^2&\cdots&\hat{y}_2\hat{y}_d\\\vdots&\vdots&\ddots&\vdots\\\\\\hat{y}_d\hat{y}_1&\hat{y}_d\hat{y}_2&\cdots&{\hat{y}_d}^2\end{bmatrix})

-------------

### CrossEntropy的梯度推导

![](softmax_crossentropy.jpg)

<br/>



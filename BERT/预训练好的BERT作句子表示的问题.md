
刚接触BERT，很自然会想到取BERT最后一层的CLS的output作为句子表示，用来无监督的做诸如相近句子挖掘或者语义匹配等任务，不过一般这样效果并不好。
</br>

最近ByteDance和CMU在EMNLP 2020上的一篇论文[On the Sentence Embeddings from Pre-trained Language Models](https://arxiv.org/pdf/2011.05864.pdf)关于这个问题做了一些详尽的分析和解决方案。
</br>

语言模型的统一定义，给定context(c)预测得到token(x)的概率分布，即: 
![](https://latex.codecogs.com/svg.latex?p(x|c)=\frac{exp(h_c^Tw_x)}{\sum_{{x}'}exp(h_c^Tw_{{x}'})})

这里![](https://latex.codecogs.com/svg.latex?h_c)是context embedding，![](https://latex.codecogs.com/svg.latex?w_x)表示word embedding。

### 各向异性嵌入空间
语言模型中最大似然目标的训练会产生各向异性的词向量空间，即向量各个方向分布并不均匀，并且在向量空间中占据了一个狭窄的圆锥体。

* 词频率影响词向量空间的分布 </br>
高频的词更接近原点。低频词远离原点。
* 词频影响词向量空间稀疏性 </br>
高频词分布更集中，相似度更近；而低频词分布则偏向稀疏。如果存在稀疏性，那么在词频低的词汇周围，就会存在很多意义不明的地方。

### 如何改进
这篇论文提出了将BERT embedding空间映射到一个标准高斯隐空间的方法。高斯分布满足各向同性；分布区域没有“洞”。
具体如何做的后续再分析。

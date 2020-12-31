
刚接触BERT,很自然会想到取BERT最后一层的CLS的output作为句子表示，用来无监督的做诸如相近句子挖掘或者语义匹配等任务，不过一般这样效果并不好。
</br>

最近ByteDance和CMU在EMNLP 2020上的一篇论文[On the Sentence Embeddings from Pre-trained Language Models](https://arxiv.org/pdf/2011.05864.pdf)关于这个问题做了一些详尽的分析和解决方案。
</br>

语言模型的统一定义，给定context(c)预测得到token(x)的概率分布，即:
![](https://latex.codecogs.com/gif.latex?\\p(x|c)=\frac{exp(h_c^Tw_x)}{\sum_{{x}%27}exp(h_c^Tw_{{x}%27})})

</br>

RoBERTa来自论文[RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/pdf/1907.11692.pdf)，相对于BERT主要做了如下改进。

* 使用了更大量的训练数据
* 使用了dynamic mask <br/>
[这里](https://github.com/delionlper/nlp_notes/blob/main/BERT/%E6%A6%82%E8%BF%B0.md)介绍过BERT是在数据预处理阶段把data复制了10份，15%的tokens中，80%做mask，10%做随机替换，还有10%保持不变。总共40个epoch那么每4个epoch看到的数据是一样的。<br/>
而RoBERTa做了动态掩码，不是在预处理阶段做mask，而是在数据送入模型之前才做mask，这样同一个句子，在不同epoch中，每次mask都不同。
* 去掉NSP，更改数据输入格式 <br/>
    - Segment+NSP：bert style
    - Sentence pair+NSP：使用两个连续的句子+NSP。用更大的batch size
    - Full-sentences：如果输入的最大长度为512，那么就是尽量选择512长度的连续句子。如果跨document了，就在中间加上一个特殊分隔符。无NSP。实验使用了这个，因为能够固定batch size的大小。
    - Doc-sentences：和full-sentences一样，但是不跨document。无NSP。最优。

* byte level BPE <br/>
BERT原型使用的是 character-level BPE vocabulary of size 30K。RoBERTa使用了GPT2的BPE实现，使用的是byte而不是unicode characters作为subword的单位。

* 更大的batch size
* 模型细节 <br/>
RoBERTa所有的训练样本几乎都是全长512的序列，这与BERT先通过小的序列长度进行训练不同。

参考资料：
* [RoBERTa: 捍卫BERT的尊严](https://zhuanlan.zhihu.com/p/149249619)
* [RoBERTa 详解](https://zhuanlan.zhihu.com/p/103205929)

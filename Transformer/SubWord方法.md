传统的tokenization技术的问题：
* char level embedding粒度太细，学到的语义过于抽象
* word level embedding词表很大，会有OOV问题，而且不利于学习词缀关系
* SubWord方法介于char level和word level之间

### Byte Pair Encoding（BPE）
来自于ACL 2016的这篇工作：[Neural machine translation of rare words with subword units](https://www.aclweb.org/anthology/P16-1162.pdf)

BPE(字节对)编码或二元编码是一种简单的数据压缩形式，其中最常见的一对连续字节数据被替换为该数据中不存在的字节。后期使用时需要一个替换表来重建原始数据。OpenAI GPT-2 与Facebook RoBERTa均采用此方法构建subword vector。

优点：可以有效地平衡词汇表大小和步数(编码句子所需的token数量)。 <br/>
缺点：基于贪婪和确定的符号替换，不能提供带概率的多个分片结果。ULM可以提供。


#### 算法过程
1. 准备足够大的训练语料
2. 确定期望的subword词表大小
3. 将单词拆分为字符序列并在末尾添加后缀“ </ w>”，统计单词频率。 本阶段的subword的粒度是字符。 例如，“ low”的频率为5，那么我们将其改写为“ l o w </ w>”：5
4. 统计每一个连续字节对的出现频率，选择最高频者合并成新的subword
5. 重复第4步直到达到第2步设定的subword词表大小或下一个最高频的字节对出现频率为1

随着合并的次数增加，词表大小通常先增加后减小。

#### 编码和解码
* 编码 <br/>
我们从最长的token迭代到最短的token，尝试将每个单词中的子字符串替换为token。 最终，我们将迭代所有tokens，并将所有子字符串替换为tokens。 如果仍然有子字符串没被替换但所有token都已迭代完毕，则将剩余的子词替换为特殊token，如\<unk\>。<br/>
编码的计算量很大。 在实践中，我们可以pre-tokenize所有单词，并在词典中保存单词tokenize的结果。 如果我们看到字典中不存在的未知单词。 我们应用上述编码方法对单词进行tokenize，然后将新单词的tokenization添加到字典中备用。

* 解码 <br/>
拼接所有tokens。

### WordPiece
WordPiece基于概率生成新的subword而不是下一最高频字节对。

#### 算法
1. 准备足够大的训练语料
2. 确定期望的subword词表大小
3. 将单词拆分成字符序列
4. 基于第3步数据训练语言模型
5. 从所有可能的subword单元中选择加入语言模型后能最大程度地增加训练数据概率的单元作为新的单元
6. 重复第5步直到达到第2步设定的subword词表大小或概率增量低于某一阈值

句子![](https://latex.codecogs.com/svg.latex?{S=(t_1,t_2,...,t_n)})由![](https://latex.codecogs.com/svg.latex?n)个子词组成，![](https://latex.codecogs.com/svg.latex?t_i)表示子词，且假设各个子词之间是独立存在的，则句子![](https://latex.codecogs.com/svg.latex?S)的语言模型似然值等价于所有子词概率的乘积：<br/>

![](https://latex.codecogs.com/svg.latex?{logP(S)=\sum_{i=1}^nP(t_i)})

假设把相邻位置的x和y两个子词进行合并，合并后产生的子词记为z，此时句子![](https://latex.codecogs.com/svg.latex?S)似然值的变化可表示为： <br/>
![](https://latex.codecogs.com/svg.latex?{logP(t_z)-(logP(t_x)+logP(t_y))=log(\frac{logP(t_z)}{logP(t_x)logP(t_y)})}) <br/>
似然值的变化就是两个子词之间的互信息。简而言之，WordPiece每次选择合并的两个子词，他们具有最大的互信息值，也就是两子词在语言模型上具有较强的关联性，它们经常在语料中以相邻方式同时出现。

### Unigram Language Model(ULM)
来自于ACL 2018的工作[Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates](https://www.aclweb.org/anthology/P18-1007.pdf)

初始化一个大词表，再根据评估准则不断丢弃词表，直到满足限定条件为止，词表由大变小。<br/>
ULM是另外一种subword分隔算法，它能够输出带概率的多个子词分段。假设：所有subword的出现都是独立的，并且subword序列由subword出现概率的乘积产生。

#### 算法
1. 准备足够大的训练语料
2. 确定期望的subword词表大小
3. 给定词序列优化下一个词出现的概率
4. 计算每个subword的损失
5. 基于损失对subword排序并保留前X%。为了避免OOV，建议保留字符级的单元
6. 重复第3至第5步直到达到第2步设定的subword词表大小或第5步的结果不再变化

### BERT中的tokenization
* 首先使用BasicTokenizer做一些预处理：
    - 转成unicode
    - 去除无意义字符
    - 把whitespace（空格，tab，换行，回车）变成空格
    - 在中文字符前后加上空格
    - codepoint归一化
    - 用空格切分
    - 用标点切分
* 使用WordpieceTokenizer再把词切分成更细粒度的wordpiece。<b>可以看到对于中文来说，已经是最细粒度的单字了，所以wordpiece对于中文字符没有做任何合并</b>

不同于ReBERTa采用的是Byte Pair，BERT采用的是基于unicode char level。

HuggingFace WordPieceTokenizer编码过程，直接使用了贪心算法，##代表非开头
``` python
class WordpieceTokenizer(object):
    """Runs WordPiece tokenization."""

    def __init__(self, vocab, unk_token, max_input_chars_per_word=100):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self, text):
        """
        Tokenizes a piece of text into its word pieces. This uses a greedy longest-match-first algorithm to perform
        tokenization using the given vocabulary.

        For example, :obj:`input = "unaffable"` wil return as output :obj:`["un", "##aff", "##able"]`.

        Args:
          text: A single token or whitespace separated tokens. This should have
            already been passed through `BasicTokenizer`.

        Returns:
          A list of wordpiece tokens.
        """

        output_tokens = []
        for token in whitespace_tokenize(text):
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue

            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end

            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
        return output_tokens
```


参考资料：
* [Neural machine translation of rare words with subword units](https://www.aclweb.org/anthology/P16-1162.pdf)
* [Google’s neural machine translation system: Bridging the gap between
human and machine translation](https://arxiv.org/pdf/1609.08144.pdf)
* [Japanese and Korean voice search](https://storage.googleapis.com/pub-tools-public-publication-data/pdf/37842.pdf)
* [Subword Regularization: Improving Neural Network Translation Models with Multiple Subword Candidates](https://www.aclweb.org/anthology/P18-1007.pdf)
* [Byte pair encoding](https://en.wikipedia.org/wiki/Byte_pair_encoding)
* [深入理解NLP Subword算法：BPE、WordPiece、ULM](https://zhuanlan.zhihu.com/p/86965595)
* [NLP Subword三大算法原理：BPE、WordPiece、ULM](https://mp.weixin.qq.com/s/dCImNYDmIk6tWJFCp5OE-w)
* [NLP三大Subword模型详解：BPE、WordPiece、ULM](https://zhuanlan.zhihu.com/p/191648421?utm_source=wechat_session)
* [BERT代码阅读](http://fancyerii.github.io/2019/03/09/bert-codes/#%E5%88%86%E8%AF%8D)
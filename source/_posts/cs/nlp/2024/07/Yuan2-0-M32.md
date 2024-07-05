---
title: 从Yuan2.0到Yuan2.0-M32
date: '2024-07-03 20:14:06 - NLP - LLM - transformer - 技术报告 - MoE'
categories:
  - CS
  - NLP
  - LLM
abbrlink: 3df0cd42
---

【本文已在同名 微信公众号 / 知乎 / [个人博客linsight.cn](http://www.linsight.cn/) 上线】  

***  

整理下Yuan2.0和Yuan2.0-M32技术报告的一些内容。  

# Yuan2.0  

Yuan2.0发布于23年11月，系列模型有3个规模：  

{% asset_img yuan2_intro.png Yuan2.0 %}  

## 模型  

通常的self-attention会计算token两两之间的关联性，但是没有显式加入“更近的token有更高重要性”这样的local dependency机制，按原文说法，有“short of neighbouring local associations of tokens”的问题。  

于是Yuan2.0把attention修改成Localized Filtering-based Attention（LFA），通过增加两个一维的convolution来增强相邻token之间的关联性，结构如下图  

{% asset_img lfa.png LFA %}  

为了防止训练时卷积泄露未来的信息，这里的convolution是单向的，只能往前看：  

{% asset_img lfa_conv.png LFA Conv %}  

和经典的Attention以及Attention with EMA对比，LFA在效果上更好，在模型参数的增加和计算性能上的损耗也相比EMA更小。具体的对比数据如下  

{% asset_img lfa_result.png LFA result %}  

EMA是《Mega: moving average equipped gated attention》所使用的方法，目前使用EMA的模型似乎不多。  

## 数据  

Yuan2.0的预训练数据分布如下  

{% asset_img yuan2_pretrain_data.png pretrain data %}  

主要是书籍、百科、专业知识、代码和数学相关的内容。  

一些数据的细节：  
- Baike和BOOK数据移除了小说数据  
- Code Instruct data：用生成的4M instruction获取大模型生成的 Python solution  
- StarCoder中的header如\<reponame\>, \<filename\>, \<gh_stars\>都移除了，一些code里的特殊token加到了tokenizer里  

微调数据集包括：  
- Code Instruction Dataset：专注在python上，其他语言去掉了  
- Math Instruction Dataset  
- Chat Instruction Dataset：数据分布如下表  

{% asset_img yuan2_chat_data.png chat数据 %}  

从数据上看，Yuan2.0主要是往代码和数学能力方向进行了提升。  

SFT的训练超参如下  

{% asset_img yuan2_sft_hp.png sft超参 %}  

## Tokenizer  

Yuan2.0使用SentencePiece，训练基于Unigram的tokenizer。  

由于训练数据量比较大，所以这里使用了paralle的训练方法：1.6T的中文数据切分为135个文件，每个文件各自训练一个vocab size为30000的tokenizer。  

获得135个tokenizer之后，每个tokenizer在各自训练数据上统计vocab中每个token占训练数据的byte size的比例。  

之后把各个tokenizer统计的token占比合并起来，只保留占比最高的50000个token。  

合并的过程中还会删掉包括数字、字母、特殊符号和长度>7个字的中文词。  

在这个基础上，再加入人工挑选的9000个低频中文字和30000个低频中文词，和前面的50000个token合并去重后得到了73417个token。  

最后，再把arxiv（上训练的） tokenizer、StarCoder（上训练的） tokenizer 和 LLaMA tokenizer和获得的词表进行合并，最终得到了词表大小为134953的tokenizer。  

## 训练  

Yuan2.0预训练的loss曲线走势如下  

{% asset_img yuan2_train_curve.png 训练 %}  

# Yuan2.0-M32  

Yuan2.0-M32是基于Yuan2.0-2B结构扩展的MoE模型（包括LFA），每层激活32个专家的其中2个，总参数量为40B，激活参数量为3.7B。  

{% asset_img m32_intro.png 模型 %}  

## 模型  

Yuan2.0-M32在结构上的主要改进是在router上使用了注意力机制。  

一般来说，router就是给每个专家赋一个可学习的向量，每次通过这个可学习的向量和输入token的向量的内积来决定这个token分配给哪个专家，如下图a。  

这种做法一个问题是没有考虑到分配的多个专家之间的关联性，而简单地把它们看作是独立的。  

考虑路由到的专家之间的关联性应该是对提升效果有帮助的。  

基于此Yuan2.0-M32提出attention router，如下图b。  

{% asset_img router.png attention router %}  

对于输入token向量I（维度=d，在Yuan2.0-M32里d=2048），以及N个候选专家，计算如下：  

$$Q=WI,\quad W\in\mathbb{R}^{N\times d}$$  

$$K=W^{\prime}I,\quad W^{\prime}\in\mathbb{R}^{N\times d}$$  

$$V=W^{^{\prime\prime}}I,\quad W^{^{\prime\prime}}\in\mathbb{R}^{N\times d}$$  

$$P=\mathrm{Softmax}(QK^T)\mathrm{V},\quad P\in R^N$$  

然后从P中选出top M个专家。  

不同router在相同的30B数据上进行训练，然后在另外10B数据进行评测，效果对比如下  

{% asset_img router_eval.png attention router %}  

其中attention router和classical router都是8个专家，而shared expert router总共16个专家，其中2个共享专家，再从另外14个里选择两个路由专家激活。  

另外还通过增加总的专家数来测试这个模型结构的scalability。在50B数据上训练，在另外10B数据上评测，总专家数为8/16/32时效果如下  

{% asset_img scalability.png scalability %}  

Yuan2.0-M32使用了和Yuan2.0一样的tokenizer。  

## 训练  

预训练和微调的超参如下  

{% asset_img train_hp.png train hp %}  

Yuan2.0-M32总共在2T token的数据上训练，loss变化如下，最终的loss下降到了1.22。  

{% asset_img pretrain.png pretrain %}  

预训练的时候窗口长度为4k，微调的时候为16k。参考CodeLLama的做法，这里要增大RoPE的base。这里不是简单地把10000扩展到500k或者1M，而是根据NTK-aware的公式计算：  

$$b^{\prime}=b\cdot s^{\frac{|D|}{|D|-2}}$$  

这里的D是head size，Yuan2.0-M32中head size为128。把4k扩展到16k，则s=4，计算得到新的base=40890。  

这里还拿base=40890和其他base（40000, 80000, 160000, 320000, 640000, 1280000, 2560000, 5120000, 10240000）进行效果对比，确认确实是40890的效果最好。  

## 评测  

Yuan2.0-M32在code generation、math、MMLU、AI2 Reasoning Challenge (ARC) benchmark上的评测效果如下。  

{% asset_img eval1.png 评测 %}  

{% asset_img eval2.png 评测 %}  

{% asset_img eval3.png 评测 %}  

{% asset_img eval4.png 评测 %}  

# 小结  

- Yuan2.0、Yuan2.0-M32使用了一些人造数据，在数学的代码上看起来有一定收益。  
- 结构上的改进感觉需要更多的实验来验证，另外这些改进在推理加速缺乏支持可能也是个问题。  

***  

读到这了，来一发点赞收藏关注吧~

博客：[http://www.linsight.cn/](http://www.linsight.cn/)  
知乎：[Linsight](https://www.zhihu.com/people/us4ever)  
微信公众号：Linsight  
![](/images/qrcode.jpg)  

***  

【往期文章】  
[MoE模型的前世今生](http://www.linsight.cn/44e38c1b.html)  
[昆仑万维-SkyworkMoE](https://www.linsight.cn/1d5bcd45.html)  
[成本10w刀的JetMoE](https://www.linsight.cn/f3acf042.html)  
[从loss视角理解大模型涌现能力](https://www.linsight.cn/f5fb75e4.html)  
[LLM长上下文的问题](http://www.linsight.cn/c4da56c0.html)  
[解锁大模型长上下文能力](http://www.linsight.cn/cc852861.html)  
[大模型推理窗口-从有限到无限大](http://www.linsight.cn/45ee1a6d.html)  
[理解Attention:从起源到MHA,MQA和GQA](http://www.linsight.cn/3dc22f96.html)  
[大模型推理加速-投机解码](http://www.linsight.cn/f5c015c.html)  
[大模型推理加速-MEDUSA](https://www.linsight.cn/7bbe2df6.html)  
[LLM的重复生成和ICL](https://www.linsight.cn/7381cae3.html)  
[大模型偏好对齐-DPO](http://www.linsight.cn/473f2b43.html)  
[大模型偏好对齐-ODPO](http://www.linsight.cn/da871ebe.html)  
[大模型偏好对齐-simPO](http://www.linsight.cn/280fa97a.html)  
[大模型偏好对齐-IPO](http://www.linsight.cn/4fe7b810.html)  
[Yi技术报告-划重点看细节](http://www.linsight.cn/41b6a819.html)  
[MiniCPM](https://www.linsight.cn/376db710.html)  
[GLM4报告的一些技术点](https://www.linsight.cn/a5206abd.html)  
[Gemma2](https://www.linsight.cn/cf3f1f81.html)  
[苹果的OpenELM](https://www.linsight.cn/f845f3e4.html)  
[transformer中normalization的二三事](http://www.linsight.cn/6a40bfa5.html)  
[从代码实现看normalization-到底做了什么](http://www.linsight.cn/b70b4a2d.html)  
[稀疏注意力计算:sliding window attention](http://www.linsight.cn/c61d17e3.html)  
[理解LLM位置编码:RoPE](http://www.linsight.cn/a051710f.html)  
[RoPE的远距离衰减](https://www.linsight.cn/f0902f1a.html)  
[大模型算法题(1)](http://www.linsight.cn/3345028a.html)  
[大模型算法题(2)](http://www.linsight.cn/ad0bba9d.html)  
[大模型算法题(3)](http://www.linsight.cn/1736008.html)  
[大模型算法题(4)](http://www.linsight.cn/1736008.html)  
[大模型算法题(5)](http://www.linsight.cn/336f2f3e.html)  
[大模型算法题(6)](http://www.linsight.cn/7c04944d.html)  
[大模型算法题(7)](https://www.linsight.cn/dd614e12.html)  

***  

# Reference  

【1】Yuan 2.0-M32: Mixture of Experts with Attention Router https://arxiv.org/abs/2405.17976  
【2】YUAN 2.0: A Large Language Model with Localized Filtering-based Attention https://arxiv.org/ftp/arxiv/papers/2311/2311.15786.pdf  


---
title: 对MoE模型的一些观察
tags:
  - NLP
  - LLM
  - transformer
  - MoE
categories:
  - CS
  - NLP
  - LLM
abbrlink: 5e1d14b3
date: 2024-07-16 20:14:40
---

【本文已在同名 微信公众号 / 知乎 / [个人博客linsight.cn](http://www.linsight.cn/) 上线】  

***  

包括清华和港科大的五所高校对几个MoE模型进行一些研究，并给出一些相应的模型设计建议。  

# MoE  

当前主流的Sparse Mixture-of-Experts模型在N个专家中激活k个，k < N，具体建模如下  

$$\mathbf{y}=\sum_{n\in N}g_n(\mathbf{x};\mathbf{G},\mathbf{k})E_n(\mathbf{x})$$  

$$\mathrm{Expert}(x)=W_\text{down}(W_\text{up}x\odot\mathrm{Act}(W_\text{gate}x))$$  

$$W_{\mathrm{up}},W_{\mathrm{gate}}\in\mathbb{R}^{d_{\mathrm{mid}}\times d_{\mathrm{hid}}}$$  

$$W_{\mathrm{down}}\in\mathbb{R}^{d_{\mathrm{hid}}\times d_{\mathrm{mid}}}$$  

这里把 $W_{\mathrm{up}}[i,:]$ 和 $W_{\mathrm{gate}}[i,:]$ 这两个行向量以及 $W_{\mathrm{down}}[:,i]$ 这个列向量定义为一个neuron，这样每个专家就包含d_mid个专家，这些neuron后面会有分析。  

# 研究对象  

文章中选择了Mixtral 8x7B，DeepSeekMoE 和 Grok-1三个MoE模型作为研究对象，另外还加上了Mistral 7B这个dense模型作为对比。  

各个模型设置对比如下  

{% asset_img models.png 模型 %}  

后续研究使用的相似度如无说明都是指cosine similarity。  

# Analysis of Static Parameters  

对这些模型的静态参数研究主要是（1）MoE experts和（2）gating，这两个也是MoE最重要的部分。  

## MoE experts  

参照《Transformer feed-forward layers are keyvalue memories》和《Empirical study on updating key-value memories in transformer feed-forward layers》的说法，expert的projection matrices可以看做是keys和values：  
- W_down的列向量表示possible outputs  
- W_up的行向量决定各个possible outputs的权重  
- W_gate决定是否激活对应的neuron  

对experts的研究又分为matrix level和neuron level。  

1、matrix level  

各个模型不同层下，所有专家三个投影矩阵对应的相似度如下图（DeepSeekMoE忽略了shared expert）。计算相似度的时候把矩阵进行了一维展开，之后通过PCA把维度转换到2维。  

{% asset_img matrix_level.png matrix level %}  

一些发现：  
- DeepSeekMoE和Grok-1的专家相似度比Mixtral低，而DeepSeekMoE和Grok-1是从零训练的，这表明Mixtral可能不是从零初始化的。  
- Mixtral中有一些专家和其他所有专家的相似度都极低，表明这些专家可能学到了一些特殊的内容。  
- 深层的专家相似度相比浅层更低，这说明深层专家可能有更高的专业化程度。  

2、neuron level  

matrix level的计算没有考虑到这样的情况：两个专家有相似的neuron，但是这些neuron的位置不同，这样也会导致相似度不高。因此这里通过取平均和重排序的方式来研究neuron level的相关性。重排序使用了Jonker-Volgenant算法。  

重排序后的相似度增长和Kendall’s coefficient如下表所示。Kendall’s coefficient是一种用于衡量多个评分者或多个方法对同一组对象进行评分或排名的一致性的统计量：1表示完全正相关，即两个变量的排名完全一致，-1表示完全负相关，即一个变量的排名与另一个变量的排名完全相反，0表示没有相关性，即两个变量的排名之间没有一致的模式。  

{% asset_img t2.png neuron level %}  

可以看到Mixtral的相关性依然是显著高于其他模型，说明其各个专家之间的初始化可能有关联。  

## Gate Embedding  

对于gating，研究人员首先计算了gate embedding向量之间的相似度，发现gate embedding的相似度和matrices之间的相似度结果相似。  

而后又研究了gate embedding（X）和W_up、W_gate、W_down（Y）的相似关系，并做了linear regression。下表是各个模型所有层平均之后的square of Pearson correlation coefficients。  

{% asset_img gating_1.png gating %}  

具体各层的数据  

{% asset_img gating_2.png gating %}  

一些发现：  
- gate embedding和W_gate的相似度最高。  
- Mixtral和DeepSeekMoE的（X，Y_gate）保持正相关，而Grok-1在>25层后出现了负相关。  
- gate embedding和W_gate的功能有些类似：前者决定专家的选择，后者则决定要激活的neuron，这两个部分有可能学习到了相近的知识。  

## Summary  

- 深层的专家间的相似度更低，可能是专门化的程度更高。  
- 专家的W_up、W_gate、W_down的相似度关系相近。  

# Analysis of Dynamic Behaviours  

前面研究的是模型的静态参数，这里通过使用一个6个token的短文本，和一个1100token的长文本对各个模型的动态特性进行探索。（emmm只用一两条数据是不是有点少）  

## Outputs of Experts  

对于MoE模型，一个自然的问题是，选中专家和未选中专家的输出之间有哪些相似性和差异性。  

短文本和长文本的各个专家（包含没有被选中的专家）的输出的相似度如下。  

{% asset_img dynamic.png 相似度 %}  

这里长文本使用的是angular similarity：  

$$\text{angular sim}=1-\frac{\arccos{(\text{cosine sim})}}{\pi}$$  

Mixtral：被选中的专家间的相似度更大，这可能是因为它们的norm更大。随着深度增加，在比较深的层中，整体的相似度较低，但是最后一两层却又突然变得特别相似。  

DeepSeek：和Mixtral相似，在最后一层也出现了相似度增大的情况。  

Grok：可能是因为Grok的expert size比较大，导致各个专家都能学到比较全面的内容，因此所有专家之间的输出相似度显著高于其他两个模型。  

## Norms of Expert Outputs and Gate Scores  

在上面这个实验发现被选中的专家的相似度会比较高，为了探索可能的原因，这里对experts的L2 norm和gating decision的关系进行了研究。  

使用了短文本作为输入，gate score和对应专家的norm如下  

{% asset_img norm.png norm %}  

Mixtral：发现被门控网络选中的两个expert通常都是feature vector norm最高的那两个。这个发现和《Competesmoe–effective training of sparse mixture of experts via competition》一致。另外层数越深，norm的值也越大，这和《Improved transformer pretraining with extra normalization》中的增长相似。  

DeepSeek：和Mixtral不同，DeepSeek的gating选择对norm的依赖看上去相对较低，但是top-1专家的得分更加突出，并且但是同样有随着层数增长，norm增大的特性。  

Grok：Grok的gating和norm没有明显的相关关系。可能的原因之一是GeLU相对较低的激活比率导致gating对norm的依赖性较弱。此外，与Mixtral和DeepSeek不同，Grok专家的norm在模型不同深度内几乎不变，而且其中一些norm值可能小于1，这是其他两个模型没有的现象。  

## Summary  

- 在Mixtral和DeepSeek中，各个专家的输出相似度随着深度加深而变弱，而在最后一两层相似度又会突然提高。  
- expert output的heat map和neuron-level相似度的heat map相似，这说明这两个测量某种程度上可以等价。  
- 对于Mixtral和DeepSeek，具有large norm output的expert更容易被选中。  

# Suggestions  

基于上面的一些观察，文章提出了一些建议：  
- Neuron-level experts：直观上gate embedding决定了专家的选择，而W_gate负责激活特定的neuron。而gate embedding和W_gate之间的相似性又存在关联。这意味着neuron可能是更细粒度的专家。应该进一步在微观层面上研究对专家的操作。  
- Model architecture：由于专家之间的相似性在深层/最后一层倾向于相对较低/高，可以考虑在深层增加专家的数量，而在最后一层减少专家数量。此外，gating机制通常选择输出norm较大的专家，因此使用norm作为路由机制是合理的。  
- Correlation measurement：在分析专家之间的相关性时，测量它们的weight matrices之间的相似性与测量它们在token上的输出特征向量之间的相似性在某种程度上是等价的。因此，测量weight matrices可以获取overview。  
- Training scheme：从零训练的DeepSeek和Grok比（可能）从dense模型初始化的Mixtra的专家相关性更低，说明从零训练能促进专家多样性。  

# 小结  

- 文章基于几个比较知名的MoE模型做了一些研究，但是研究样本量感觉应该再增大一些。  
- 基于观察结果的一些建议和一些其他工作结论不同，最好能一起对比下。  
- 说明MoE确实还有很多内容没有搞清楚。  

***  

读到这了，来一发点赞收藏关注吧~

博客：[http://www.linsight.cn/](http://www.linsight.cn/)  
知乎：[Linsight](https://www.zhihu.com/people/us4ever)  
微信公众号：Linsight  
![](/images/qrcode.jpg)  

***  

【往期文章】  
[MoE模型的前世今生](http://www.linsight.cn/44e38c1b.html)  
[DeepSeek-V2和MLA](https://www.linsight.cn/83c49df0.html)  
[昆仑万维-SkyworkMoE](https://www.linsight.cn/1d5bcd45.html)  
[成本10w刀的JetMoE](https://www.linsight.cn/f3acf042.html)  
[MoE的top-p routing](https://www.linsight.cn/224c42da.html)  
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
[从Yuan2.0到Yuan2.0-M32](https://www.linsight.cn/3df0cd42.html)  
[bilibili的index-1.9B](https://www.linsight.cn/770b63e1.html)  
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

# Reference  

【1】A Closer Look into Mixture-of-Experts in Large Language Models https://arxiv.org/abs/2406.18219  

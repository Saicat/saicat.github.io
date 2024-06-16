---
title: 从loss视角理解大模型涌现能力
tags:
  - NLP
  - LLM
  - transformer
  - 涌现能力
categories:
  - CS
  - NLP
  - LLM
abbrlink: f5fb75e4
date: 2024-06-15 16:13:55
---

【本文已在同名 微信公众号 / 知乎 / [个人博客linsight.cn](http://www.linsight.cn/) 上线】  

***  

智谱在《Understanding Emergent Abilities of Language Models from the Loss Perspective》中提出一个观察大模型涌现能力的视角 -- 预训练loss，主要内容是通过一系列实验结果来解释一些关于涌现能力的观察。可以作为一个理解大模型的参考角度，也可以用于指导预训练模型的开发和优化。  

# 背景  

《Emergent abilities of large language models》把emergent ability定义为在大规模模型中有，而在参数量较小的模型没有的能力。  

这个看法现在受到一些挑战：  
- 目前很多在更大规模数据集训练出来的小模型，展现出比之前大规模模型更强的能力，比如LLaMA3在大部分评测指标上就比GPT-3强，很多以前千亿模型才能做到的任务，现在百亿甚至十亿的模型也能做好。  
- 《Are emergent abilities of large language models a mirage?》认为产生涌现能力现象的因为是数据评测指标的非线性和不连续性带来的，如果使用更细粒度的连续指标，就能观察到指标的平滑提升。  

而《Training compute-optimal large language models》指出，相同的计算量下，不同的模型规模和数据量的组合会产生不同的效果。这说明单纯的模型规模或者数据规模并不是一个好的下游任务能力的indicator，预训练loss才是更合适的指标。

但是训练loss和下游任务表现具体是什么关系却还没有确定的说法，智谱针对这个问题做了一些预训练实验，并从预训练loss角度定义了emergent ability。  

# pretraining loss和下游任务表现的关系  

## 设置  

后续所有预训练实验使用相同的模型结构和同一份预训练数据（但是训练数据量可能有区别），一些通用设置如下：  
- 分词用BPE  
- 模型结构在LLaMA基础上，全部使用GQA，而RoPE只在一半的Q/K上应用  
- 使用AdamW优化器，$\beta_1=0.9$，$\beta_2=0.95$  
- 训练窗口长度为2048  
- 所有模型都在中英文比例为1:4的预训练数据集上训练，英文数据集分布如下  

{% asset_img eng_data.png 英文数据 %}  

所有模型都是从零开始预训练。  

评测模型的下游任务共有6类12个数据集，具体信息如下  

{% asset_img downstream_dataset.png 下游任务 %}  

{% asset_img downstream_dataset_num.png 下游任务 %}  

## 实验一：pretraining loss vs. performance  

第一个实验训练了3个规模的模型：1.5B、6B、32B，训练数据量分别为3T、3T、2.5T。具体设置如下  

{% asset_img exp1_param.png 实验设置 %}  

大约每训练43B token就会保存一次checkpoint。把3个模型所有checkpoint下，对应的预训练loss和下游任务评测结果画出来，如下所示  

{% asset_img exp1_plot.png loss vs. performance %}  

从上图可以观察到3个现象：  
- 无论模型规模如何，下游任务评测结果都随着预训练loss的降低而提升。从提升的具体情况可以分成两类，这个后面部分再分析。  
- 各个规模的模型所画出的点都落在了同一条曲线上，这说明下游任务的评测结果和预训练loss高度相关，而和模型规模没有直接关系。这点很重要。  
- 预训练loss对下游任务指标的表征能力同时适用于中英文，这说明中英文token在多语言预训练中具有相似的learning dynamics。  

而把计算量和下游任务指标的关系画出来，则有如下结果  

{% asset_img exp1_compute.png 下游任务效果和预训练计算量的关系 %}  

可以看到各个规模的模型所画出的点并没有落在同一条曲线上，这说明计算量并不是表征下游任务效果的好指标。  

## 实验二：training token count vs. performance  

第二个实验使用了不同的数据量训练了28个小一些的模型，具体设置如下  

{% asset_img exp2_param.png 实验设置 %}  

第一个实验中，每个规模的模型设置了一个固定的训练token数，然后取中间checkpoint进行评测。第二个实验是对每个规模的模型设置了多个不同的总训练token数。二者的区别在于，预训练的最后阶段会逐渐把学习率decay到最小值，而这样的学习率退火策略对效果有很大的影响。

取28个模型的最终checkpoint，画出对应的预训练loss和下游任务评测结果如下  

{% asset_img exp2_plot.png token count vs. performance %}  

结果和实验一类似，各个模型的点都落在了同一条曲线上。说明无论模型规模和训练量如何，只要loss相同，在下游任务上就有相同的表现。  

由于这28个模型相比实验一的较小，在图中最后一排的任务上效果都接近于随机。这个现象后续分析。  

## LLaMA’s loss vs. performance  

实验一和二是在从零开始训练的模型上评测的，这里用LLaMA来验证前面得到的结论。  

由于LLaMA没有放出中间checkpoint，这里直接从LLaMA的报告里抽出相应的数据点，在6个下游任务上的结果如下图

{% asset_img exp3_plot.png loss vs. performance %}  

可以看到基本上各个模型的点也是落在同一条曲线上。LLaMA和实验一实验二的训练框架、模型结构、训练数据都有所不同，但是也有相同的结论，说明这样的结论是具有普遍性的。  

> pre-training loss is a good indicator of LMs’ performance on downstream tasks, independent of model sizes, training tokens, languages, and pretraining frameworks  

# 进一步分析  

## 不同任务的趋势  

12个下游任务可以分为2类：  
- 第一类：TriviaQA, HellaSwag, RACE, WinoGrande, NLPCC-KBQA, ClozeT, CLUEWSC, C3。这些任务的效果随着预训练loss的下降，平滑上升。  
- 第二类：MMLU, C-Eval, GSM8K, GSM8K-Chinese。这些任务上，只有当预训练loss低于一定阈值，评测结果才开始提升。可以观察到，在实验一实验二的配置下，大概在预训练loss小于2.2这个阈值之后，下游任务表现开始提升。整体来说，第二类任务难度是大于第一类的。所以虽然第一类中有些任务的prompt或者形式与第二类中的任务有些相似，但是依然有不同的表现。  

第二类任务这个现象和《Grokking: Generalization beyond overfitting on small algorithmic datasets》提出的grokking有关联。  

grokking描述了下游任务的效果从随机水平（乱猜）提高到perfect generalization的improvement。这种improvement只有在过拟合到一定程度才会发生。在预训练中，模型整体上通常是欠拟合的。不过由于预训练语料库是不同文档的混合，因此模型可能在某些能力上过拟合（比如数值计算的能力，情感分析的能力），而在整体上依然欠拟合。  

当然第二类任务这个现象也和emergent ability有关联。按scaling law的说法，在训练token数固定的情况下，预训练loss与模型规模呈幂律关系。也就是说，模型大小和预训练损失之间存在单调关系。对于第二类任务，存在一个与预训练loss中的临界点相对应的模型规模阈值。当模型大小超过这个阈值时，模型就可以展现出超过随机猜测的能力。  

## 评测指标的影响  

前面提到，emergent ability这个现象有可能是因为评测指标的非线性和不连续性带来的。比如 MMLU这样的多项选择题，打分结果只能是0分或者满分。  

现在把这个评测指标换成两个连续的指标：  
- 一个是probability of the correct answer (CorrectChoiceProb)  
- 第二个是《Are emergent abilities of large language models a mirage?》中提出的Brier Score：  

$$\text{BrierScore}=\frac1N\sum_{i=1}^N\sum_{j=1}^C(y_{ij}-\hat{y}_{ij})^2$$  

N是样本数，C的类别数。  

把MMLU和C-Eval在这两个新指标上的评测结果画出来，如下所示  

{% asset_img metrics.png 指标 %}  

可以发现涌现能力的现象依然存在。  

值得注意的是，Brier Score的下降并不总是表示下游任务效果的提升。  

比如对于有A/B/C/D四个选项的多项选择题任务，假设正确答案的分布是均匀的。现在有两个模型，一个总是预测A，即（1，0，0，0），另一个总是给出平均分布的预测，即（0.25，0.25，0.25，0.25，0.25）。  

那么前者的Brier Score是1.5，而后者是0.75，但这并不能说明后者就更好。对于这个任务，实际上高于0.75的Brier Score都说明比随机猜测差。而低于随机猜测的指标变化并不能当做真正的提升，比如Brier Score从1.5提升到1.0并不能算作提升。  

另外《Training trajectories of language models across scales》提出用perplexity of correct options来作为评测，可以看到平滑的提升。但perplexity of correct options其实不能作为一个合适的指标。  

比如对于多项选择题，区分各个答案的能力才是我们想要的。而随着预训练进行，正确答案和错误答案的perplexity都在下降，只有当训练到二者的perplexity差异开始变大的时候，才能算是有提升。因此单纯的正确答案perplexity下降也能作为能力提升的指标，因为错误答案的perplexity可能下降更多。  

# 从loss角度定义emergent abilities  

基于前面的实验和分析，现在从预训练loss角度重新定义emergent ability：  

> Definition. An ability is emergent if it is not present in models with higher pre-training loss but is present in models with lower pre-training loss.  

一个emergent ability的normalized performance（比如多项选择题随机猜测的得分是0.25分，那这个任务原始的0.25分在normalized performance下就是0分）是预训练loss $L$ 的函数  

$$\begin{cases}f(L)&\mathrm{if~}L<\eta\\0&\mathrm{otherwise}&\end{cases}$$  

其中f是一个单调递减函数，$\eta$ 是阈值。  

《Scaling laws for autoregressive generative modeling》中提出，在固定的训练token数 $D$ 下，模型规模 $N$ 和预训练损失的关系是  

$$L(N)=L_\infty+\left(\frac{N_0}N\right)^{\alpha_N}$$  

其中 $L_{\infty}$ 是irreducible loss，$\alpha_{N}$ 是固定的系数。  

把上面两个式子结合起来，就有  

$$\begin{cases}f\left(L_\infty+\left(\frac{N_0}N\right)^{\alpha_N}\right)&\text{if }N\geq N_0\cdot\left(\eta-L_\infty\right)^{-\frac1{\alpha_N}}\\0&\text{otherwise}&\end{cases}$$  

当模型规模小于 $N_0\cdot(\eta-L_\infty)^{-1/\alpha_N}$ 这个阈值时，normalized performance为0；当模型规模超过这个阈值时，模型规模的增长带来了预训练loss的下降，从而带来了normalized performance的提升。  

# 小结  

通过预训练loss来预测下游任务的提升，这点用在预训练模型的分析和优化上还是有些帮助的。比如在loss较高的时候，在下游任务上的效果的变化可能更多是随机波动而不是真正的提升。  

不过文中只对一个model family做了实验，而loss和模型结构，词表等都有关系，因此还需要进一步的探索。  

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
[LLM长上下文的问题](http://www.linsight.cn/c4da56c0.html)  
[解锁大模型长上下文能力](http://www.linsight.cn/cc852861.html)  
[大模型推理窗口-从有限到无限大](http://www.linsight.cn/45ee1a6d.html)  
[理解Attention:从起源到MHA,MQA和GQA](http://www.linsight.cn/3dc22f96.html)  
[大模型推理加速-投机解码](http://www.linsight.cn/f5c015c.html)  
[大模型推理加速-MEDUSA](https://www.linsight.cn/7bbe2df6.html)  
[大模型偏好对齐-DPO](http://www.linsight.cn/473f2b43.html)  
[大模型偏好对齐-ODPO](http://www.linsight.cn/da871ebe.html)  
[大模型偏好对齐-simPO](http://www.linsight.cn/280fa97a.html)  
[大模型偏好对齐-IPO](http://www.linsight.cn/4fe7b810.html)  
[Yi技术报告-划重点看细节](http://www.linsight.cn/41b6a819.html)  
[transformer中normalization的二三事](http://www.linsight.cn/6a40bfa5.html)  
[从代码实现看normalization-到底做了什么](http://www.linsight.cn/b70b4a2d.html)  
[稀疏注意力计算:sliding window attention](http://www.linsight.cn/c61d17e3.html)  
[理解LLM位置编码:RoPE](http://www.linsight.cn/a051710f.html)  
[大模型算法题(1)](http://www.linsight.cn/3345028a.html)  
[大模型算法题(2)](http://www.linsight.cn/ad0bba9d.html)  
[大模型算法题(3)](http://www.linsight.cn/1736008.html)  
[大模型算法题(4)](http://www.linsight.cn/1736008.html)  
[大模型算法题(5)](http://www.linsight.cn/336f2f3e.html)  
[大模型算法题(6)](http://www.linsight.cn/7c04944d.html)  
[大模型算法题(7)](https://www.linsight.cn/dd614e12.html)  

***  

# Reference  

【1】Understanding Emergent Abilities of Language Models from the Loss Perspective https://arxiv.org/abs/2403.15796  
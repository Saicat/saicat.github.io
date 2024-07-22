---
title: 从dense到MoE -- sparse upcycling
tags:
  - NLP
  - LLM
  - transformer
  - MoE
  - 预训练
categories:
  - CS
  - NLP
  - LLM
abbrlink: a0824e29
date: 2024-07-19 21:03:12
---

【本文已在同名 微信公众号 / 知乎 / [个人博客linsight.cn](http://www.linsight.cn/) 上线】  

***  

目前已经有很多优秀的dense大模型，那么要通过MoE获得更强的模型，用已有的dense模型进行初始化是一个自然的想法。Google的sparse upcycling对此做了一些实验，由于实验是在2022年做的，模型用的是T5系列语言模型和Vision Transformer系列视觉模型。  

文中给出两个适合使用sparse upcycling的场景：  
- 已有dense模型，想在有限的计算资源下提升模型效果。  
- 要训一个模型，不知道dense模型和MoE哪个会有更好的效果（虽然通常来说MoE更好，但是训练难度和结果不确定也更大），那么就可以先训练一个dense模型保底，然后再在dense模型的基础上扩展成MoE结构继续优化。  

下面具体看下一些实验细节。  

# 设置  

对于transformer模型，sparse upcycling的操作如下图  

{% asset_img intro.png upcycling %}  

除了原模型的MLP层替换成MoE层外，其他组件包括layernorm、attention都直接从原dense模型copy到MoE模型。  

实验上，一些具体的基础设置如下：  
- 在原模型基础上，每2层替换一个MoE层，从第二层开始替换  
- MoE模型的总层数的dense模型层数相同  
- 每个MoE层专家数为32个；虽然使用更多的专家不会明显增大训练的FLOPS，但是更多的专家会带来larger initial quality drop relative to baseline dense model，而需要更多的计算资源来恢复这个quality drop；后续会有实验探索expert数量的影响  
- 每个expert都用原模型的MLP层参数初始化  
- router使用standard deviation=0.02的zero-mean normal distribution随机初始化  
- 在encoder使用expert choice routing，基础的设置是capacity factor C = 2，后面也做了关于capacity factor的消融实验  
- 在decoder使用token choice routing（top-k routing），k=2，并加上auxiliary loss帮助负载均衡，权重为0.01；在decoder使用top-k routing的原因是"to avoid train time (full batch teacher forcing) versus inference time (single token auto-regressive decoding) discrepancies"（和expert choice routing的设计相关）  

MoE模型训练时使用和dense模型一致的batch size、learning rate schedule和weight decay等。  

其中learning rate schedule用的是inverse square root learning rate schedule，因此MoE的训练可以接着dense模型的schedule接着进行。  

实验中所用的一些模型参数如下表  

{% asset_img models.png 模型 %}  

# 实验  

## CORE RESULTS  

1、dense模型继续训练 vs upcycling  

随着训练量的增加，upcycling相比dense模型继续预训练的优势逐渐扩大，如下图所示  

{% asset_img 1.png 实验 %}  

2、下游任务模型效果  

上面对比的是预训练模型。这些预训练模型经过微调后的效果对比如下。  

{% asset_img 2.png 实验 %}  

相比预训练模型，微调模型表现出相对更大的震荡，不过大致趋势还是可以看出MoE模型更有优势。  

3、MoE from scratch vs upcycling  

从零开始训练的MoE和upcycling方法的对比如下  

{% asset_img 3.png 实验 %}  

- 从零开始预训练的MoE模型效果提升得更快，这可能得益于多样化的专家初始化和更大的lr。  
- 只要给的计算资源足够多，从零开始训练的模型最终会赶上甚至超过upcycling的模型。  
- 在有限的训练资源下，upcycling的训练效率更高，从零开始训练的模型大约需要相当于原dense模型1.2倍的训练资源才能达到upcycling模型的效果。如果现在的训练资源<=训练dense模型的资源，那么sparse upcycling是更划算的。  

4、sparse upcycling vs dense upcycling  

对比《Scaling language models: Methods, analysis & insights from training gopher》中的depth tiling（dense upcycling） 和 sparse upcycling的预训练效果，结果当然是sparse upcycling效率更高点，如下图所示  

{% asset_img 4.png 实验 %}  

（不过这里没有提及depth tiling之后的模型规模）  

## 消融实验  

1、Amount of dense pretraining  

upcycling的效果可能受用于初始化的dense模型的收敛情况影响，因此取了不同step的dense模型checkpoint作为upcycling的初始化，并且都继续训练了200k个step，结果如下图  

{% asset_img a1.png 实验 %}  

结论是基本上无论从哪个checkpoint初始化MoE模型，收益都比较稳定。  

2、Router type  

使用不同的router（expert choice和token choice）对比结果如下  

{% asset_img a2.png 实验 %}  

结论是，在相同的step下，expert choice和token choice的效果基本一样，但是如果从时间上来看，使用expert choice routing的模型训练更快。  

3、Expert capacity factor  

每个专家处理的token越多，计算量就越大，理论上效果也越好。

使用不同的capacity factor，模型效果对比如下  

{% asset_img a3.png 实验 %}  

结论是，虽然理论上增加专家容量可以提升效果，但时间上，通常C = 2的效率比较好，即一定的时间内提升的效果最多（注意计算资源是有限的）。  


4、Number of MoE layers  

在视觉模型上对MoE层数的效果进行了式样。  

如下图右边两个小图，是使用不同的MoE层的效果，比如1表示只把最后一层MLP层替换为MoE层，以此类推  

{% asset_img a4.png 实验 %}  

结论是，更多的MoE层并不总是更好，大概是把5~6层替换成MoE层的时候效果最好（40%~50%的层数）。  

5、Initialization of experts  

对比了使用dense模型的MLP层初始化专家，和随机初始化专家，结果如下  

{% asset_img a5.png 实验 %}  

结果上看，使用dense模型的参数初始化专家效果更好。  

6、Number of experts  

如前面提到的，增加专家数并不会增大计算量，下图实验了2~128个专家下的效果  

{% asset_img a6.png 实验 %}  

结果上来看，效果是随着专家的增加而提升的，虽然最后表现出了收益递减的情况。  

## 其他  

1、optimizer  

在vision模型上，还尝试了使用dense模型的optimizer状态来训练MoE模型，但是并没有带来任何收益。  

2、router normalization  

另外，为了减少从dense到MoE初始化的performace drop，尝试了对router的输出进行normalization，以保持每个token得到的weight总和是1。

这个做法直觉上应该是有益的，不过会有一个小问题，那就是对于只被一个expert选中的token，会有vanishing routing gradients。  

实践上，router normalization在视觉模型上基本和不进行normalization的效果差不多，但是在语言模型上，会使得MoE模型效果变差。这二者的表现差异可能是因为语言模型上部分router使用了token choice routing。  

实际上目前大部分最新的MoE模型都没有开router normalization，但这里的原因感觉还有待深入验证。  

# 小结  

- 如果资源无限，那么从零初始化MoE模型更好，这点和天工MoE的实验是相同的；  
- 从结果上看，MoE层和专家数不是越多越好，但是多少是好，感觉至今其实没有很明确；  
- 目前大部分的MoE都使用token choice routing了，这里可能需要重新实验一下效果。  

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
[对MoE模型的一些观察](https://www.linsight.cn/5e1d14b3.html)  
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
[Qwen2技术报告](https://www.linsight.cn/a8f8b641.html)  
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
[大模型算法题(8)](https://www.linsight.cn/e287b9c3.html)  


# Reference  

【1】SPARSE UPCYCLING: TRAINING MIXTURE-OF-EXPERTS FROM DENSE CHECKPOINTS https://arxiv.org/abs/2212.05055  

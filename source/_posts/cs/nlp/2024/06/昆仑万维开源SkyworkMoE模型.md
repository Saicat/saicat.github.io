---
title: 昆仑万维-SkyworkMoE
abbrlink: 1d5bcd45
date: 2024-06-04 20:51:02
tags:
  - NLP
  - LLM
  - transformer
  - MoE
categories:
  - CS
  - NLP
  - LLM
---

【本文已在同名 微信公众号 / 知乎 / [个人博客linsight.cn](http://www.linsight.cn/) 上线】  

***  

之前我们对比较热门的十个MoE工作进行了整理：[MoE模型的前世今生](http://www.linsight.cn/44e38c1b.html)。  

最近昆仑万维开源了Skywork-MoE，一个总参数量为146B，激活参数量为22B的MoE模型。  

Skywork-MoE技术报告中针对几个实操会遇到的问题做了一些实验，还是挺有借鉴意义的。  

# Skywork-MoE模型  

分析之前，先看下Skywork-MoE的模型设计：  
- Llama-like architecture  
- RoPE  
- RMSNorm  
- SwiGLU activation function  

其他参数如下表  

{% asset_img structure.png 模型结构 %}  

Skywork-MoE共有146B参数，16个专家，激活参数量为22B。  

训练集群用到了192个NVIDIAHGX-A800节点，共1536个A800-80G显卡。  

训练框架是基于Megatron搭建的，data parallelism开了ZeRO-1优化，训练速度能达到690token/GPU/second，GPU利用率是38%。  

# 训练路线选择  

在当前的情况下，要训练一个MoE模型有两条路线可以选择：  
- upcycling：用一个dense模型做MoE模型的初始化，进行一定的继续预训练。这样的好处是MoE模型能在一个比较好的初始化点开始训练，直觉上这样的模型应该收敛得相对比较快，成本也比较低。存在的问题是dense模型的选择可能存在一些权衡取舍，且从dense进行初始化可能对最终效果存在负面影响。  
- from scratch：直接随机初始化一个MoE模型，从零开始训练。这样成本相比upcycling就比较高，但是效果可能比upcycling更好。  

当然还有一种方法是，先从零训一个dense模型，再从这个dense模型训练一个MoE模型。但是后面的实验告诉我们，如果这个dense模型纯粹是为最终的MoE模型服务的话，那这种方法是费力不讨好的。  

要决定是upcycling还是from scratch，需要看现有的dense模型的水平，以及MoE模型的训练预算。首先如果预算根本支持不了MoE模型这个规模的训练，那我们当然只能选择upcycling。只有当预算充足，我们才有机会选择from scratch这条路。而如果没有可用的dense模型，那就只能选择from scratch。  

前面我们从直觉上认为from scratch效果会更好，下面就从实验上来验证这个想法。  

首先，在300B token的数据上训练一个0.3B的dense模型，并分别取100B和300B时的checkpoint作为后续实验的起始点。这两个checkpoint起个名字叫"checkpoint-100B"和"checkpoint-300B"。  

然后在相同结构下，把dense模型扩成有8个专家的MoE模型，并使用3种不同的初始化策略：from-scratch / checkpoint-100B / checkpoint-300B。  

假设我们现在有两种MoE模型的训练预算，100B和300B（token）。  

对于100B训练预算，对比以下几个模型  

{% asset_img 100B.png 100B %}  

同样地，对于300B预算的情况，训练了init_scratch-decay_300b和init_100b-decay_300b。另外还训练了一个init_300b-3xLR，相比init_300b-const提升了3倍的学习率，用于验证学习率的影响。  

各个模型的训练结果如下图所示  

{% asset_img exp_1.png 实验 %}  

左图：在100B的训练预算下，from scratch已经可以和从dense初始化的MoE模型loss持平，甚至比init_300b-const好。报告认为init_300b-const效果不好有一部分原因是学习率太小了。  

中图：在300B的训练预算下，from scratch模型已经超越所有其他模型。另外学习率最小的模型表现最差。  

右图：把中图几个模型的expert similarity画出来，发现expert similarity越低的模型，表现越好，并且对于upcycling的模型，expert similarity在训练过程中越来越低，对应着模型效果越来越好。而from scratch的模型的expert similarity基本上一直保持为0，这也说明从dense模型初始化会使得专家多样性比较弱，从而使得模型收敛到suboptimal的点。  

据此，报告给出路线选择的经验法则。假设 $C_{\mathrm{dense}}$ 是dense模型的训练成本，$C_{\mathrm{MoE}}$ 是MoE模型的训练预算，那么：  
- 如果 $C_{\mathrm{MoE}}\ll C_{\mathrm{dense}}$，选择upcycling，upcycling能更好利用上dense模型已投入的成本。  
- 如果 $C_{\mathrm{MoE}}\geq2C_{\mathrm{dense}}$，选择from scratch，能获得更好的效果。  

另外，学习率的影响很大，这个要仔细设置。  

# 模型设计  

模型设计上，Skywork-MoE提出了两个主要的改进：gating logit normalization和adaptive auxiliary loss coefficients。  

## gating logit normalization  

研究人员在训练过程中发现有一个现象，那就是有时gating layer会输出熵很高的分布，也就是分配给各个专家的概率接近平均分布。这样的结果就是MoE层的输出基本上相当于是各个专家的平均值，而不是一个weighted average。  

而出现这种现象说明gating layer没有很好地区分各个专家，无法把相应的输入分配给最合适的专家。  

针对这个问题，Skywork-MoE给出的方法就是在gating layer的softmax之前引入一个normalization step，如下式  

$$\begin{aligned}&z=Wx+b\\&\tilde{z}=\lambda\cdot\frac{z-\mu}{\sigma}\\&g=\operatorname{softmax}(\tilde{z})\end{aligned}$$  

其中 $\lambda$ 是一个超参。  

这样归一化之后我们就得到一个均值为0，而方差受 $\lambda$ 控制的向量。大的 $\lambda$ 值会使得softmax之后的分布更显著，更不均匀。这就相当于给softmax加上一个放大器，把原本不显著的差异进行放大。  

为了验证这个设计的有效性，Skywork-MoE在2.5B参数16个专家的MoE模型上，分别使用和不使用gating logit normalization进行了训练。  

两个模型的gating分布差异如下图所示，normalization确实可以增大各个专家分配到的概率的差异。  

{% asset_img gate_dist.png gating distribution %}  

使用了normalization的模型在training loss和token drop rate上都有更好的表现，如下图所示。  

{% asset_img normaization.png gating logit normalization %}  

而统计gating layer输出的分布中的Max1/Max2和Max2/Max3比值也同样说明了各个expert被更有效地区分开了。  

在千亿Skywork-MoE模型的训练中，使用了 $\lambda=1$。  

## adaptive auxiliary loss coefficients  

一般来说，MoE模型在训练中都会加入一个auxiliary loss，帮助平衡专家的选择分布，提升训练效率，也增强专家的多样性。对于有M个MoE层的模型，最终loss如下式所示。  

$$\mathcal{L}_{\mathrm{total}}=\mathcal{L}_{\mathrm{ce}}+\sum_{l=1}^M\alpha\mathcal{L}_{\mathrm{aux}}^{(l)}$$  

每个MoE层都有对应的auxiliary loss。  

Skywork-MoE认为每层的auxiliary loss的系数 $\alpha$ 不一定要相同，并且随着训练进行，在gating的平衡已经比较好的时候，可以放宽auxiliary loss的限制强度，避免影响模型的最终效果。  

基于这两个想法，Skywork-MoE提出adaptive auxiliary loss coefficients。  

每个MoE层的auxiliary loss有自己的系数，而这个系数和当前这个MoE层的token drop rate联系了起来。大的token drop rate表示gating的分配不平衡，因此要加强auxiliary loss的约束，反之则可以减小约束。  

对于第l个MoE层，在第i个step的时候，auxiliary loss的系数计算如下  

$$\begin{array}{rcl}\hat\alpha_{i+1}^{(l)}&=&f(d_i^{(l)})\\\alpha_{i+1}^{(l)}&=&\beta\alpha_i^{(l)}+(1-\beta)\hat\alpha_{i+1}^{(l)}\end{array}$$  

其中d表示token drop rate，f是一个单调递增函数。$\alpha$ 会随着训练，通过moving average更新。$\beta$ 是moving average的权重，是一个超参。  

实际实现中，f设计成：  

$$f(d)=\left\{\begin{array}{ll}\xi d&\text{if }d\leq\alpha_{\text{max}}/\xi\\\alpha_{\text{max}}&\text{if }d>\alpha_{\text{max}}/\xi\end{array}\right.$$  

$\xi$ 表示auxiliary loss coefficient对token drop rate的敏感程度。  

最终训练中，各个超参的设置为：  
- $\xi=1/5$  
- $\alpha_{\max}=0.01$  
- $\beta=0.99$  

# 其他尝试  

报告中还给出了训练中一些其他尝试，虽然没有直接效果，但是也有参考意义。  

## 学习率  

MoE模型由于路由策略的存在，每个专家平均接受到的输入token数比global batch size要小。  

假设共有n个专家，激活专家数为k，那么平均每个专家接受到的输入只有模型输入的k/n。  

而有效batch size的减小意味着更容易引入noise，对此一般的应对方案就是减小learning rate，可以进行linear scaling（$k/n$），或者square root scaling（$\sqrt{k/n}$）。  

那么减小learning rate是否能提升效果呢？Skywork-MoE用一个1.8B参数，共32个专家，激活专家数为2的模型，按square root scaling，进行了以下3个实验  

{% asset_img lr_exp.png lr实验 %}  

所有模型在训了300B数据之后，lr会降到peak lr的10%，然后会再继续训10B，在这个过程里lr逐渐降为0。  

训练的loss如下图  

{% asset_img lr_result.png lr实验 %}  

虽然在300B的训练量下，减小lr有一点收益，但是随着最后10B的训练，三个模型都收敛到同样的loss。这说明前面的loss差异并不是不可弥补的，更可能只是因为在300B时三个模型的lr decay到不同的绝对值而已。  

这也说明根据专家数量减少MoE模型的训练学习率并没有太大必要。  

## 多样化初始化  

前面提到，用一个dense模型进行初始化，会导致各个专家相似度过高，从而损害MoE模型的效果。那么我们自然想到用多样化的几个dense模型进行MoE的初始化，效果是不是会更好。  

Skywork-MoE对此进行了实验。把原始dense模型分别用不同的100B数据进行训练，从而获得多个dense模型，并用这些多样化的dense模型初始化MoE模型。  

具体来说，基于原始dense模型 $M_{\mathrm{base}}$，用了中文、英文、代码三个不同的100B数据集进行训练，获得 $M_{\mathrm{cn}},M_{\mathrm{en}},M_{\mathrm{code}}$ 三个dense模型。之后把 $M_{\mathrm{cn}}$ 复制3份，$M_{\mathrm{en}}$ 复制3份，$M_{\mathrm{code}}$ 复制1份，$M_{\mathrm{base}}$ 复制1份，共同初始化一个有8个专家的MoE模型。  

多样化和无多样化的初始化方法，训练loss对比如下  

{% asset_img diff_dense.png 初始化实验 %}  

可以看到多样化的初始化方法确实有一点收益，不过随着训练进行，差异在逐渐减小。  

经过90B数据的训练之后，二者的loss只有不到0.01的差距。相较于dense模型的多次继续预训练成本，这个收益并不明显，因此Skywork-MoE最终没有采用多样化的初始化方法。  

# 效果  

146B参数的Skywork-MoE是从Skywork-13B初始化而来的。  

训练数据使用了SkyPile中的一部分数据，再加上一批合成数据。  

中文、英文、代码数据的比例为7:2:1。  

Skywork-MoE在和一些主流模型，在一些benchmark上的对比如下  

{% asset_img perf.png 效果 %}  

基本上达到了同归模型比较好的效果。  

# 小结  

Skywork-MoE开源了一个效果不错的MoE模型，同时对于初始化策略的探索也颇有借鉴意义。  

***  

读到这了，来一发点赞收藏关注吧~

博客：[http://www.linsight.cn/](http://www.linsight.cn/)  
知乎：[Linsight](https://www.zhihu.com/people/us4ever)  
微信公众号：Linsight  
![](/images/qrcode.jpg)  

***  

【往期文章】  

[MoE模型的前世今生](http://www.linsight.cn/44e38c1b.html)  
[LLM长上下文的问题](http://www.linsight.cn/c4da56c0.html)  
[解锁大模型长上下文能力](http://www.linsight.cn/cc852861.html)  
[大模型推理窗口-从有限到无限大](http://www.linsight.cn/45ee1a6d.html)  
[理解Attention:从起源到MHA,MQA和GQA](http://www.linsight.cn/3dc22f96.html)  
[大模型推理加速-投机解码](http://www.linsight.cn/f5c015c.html)  
[大模型偏好对齐-DPO](http://www.linsight.cn/473f2b43.html)  
[大模型偏好对齐-ODPO](http://www.linsight.cn/da871ebe.html)  
[大模型偏好对齐-simPO](http://www.linsight.cn/280fa97a.html)  
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

***  

# Reference  

【1】Skywork-MoE: A Deep Dive into Training Techniques for
Mixture-of-Experts Language Models https://github.com/SkyworkAI/Skywork-MoE/blob/main/skywork-moe-tech-report.pdf  

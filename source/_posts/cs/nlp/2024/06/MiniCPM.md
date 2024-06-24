---
title: MiniCPM
tags:
  - NLP
  - LLM
  - transformer
  - 技术报告
  - 学习率
  - 预训练
categories:
  - CS
  - NLP
  - LLM
abbrlink: 376db710
date: 2024-06-18 21:51:22
---

【本文已在同名 微信公众号 / 知乎 / [个人博客linsight.cn](http://www.linsight.cn/) 上线】  

***  

MiniCPM是面壁智能和清华开源的模型，MiniCPM开源系列包括非embedding参数为1.2B和2.4B两个规模的模型，以及对应的MiniCPM-DPO，MiniCPM-MoE和MiniCPM-128K模型。  

简单梳理一下MiniCPM提到的一些内容。  

# 背景  

大模型的训练成本很高，而且很多机制还没搞清楚，训出来的大规模模型在很多设备上也跑不起来，因此现在有不少机构对小一点的模型，即SLM，进行更全面的探索，比如Phi系列、TinyLlama、MobileLLM和Gemma等。  

MiniCPM也是对SLM的一次探索，从中得到的经验也可以推广到更大的模型上。  

# 风洞实验  

为了找到好的模型参数和训练参数，MiniCPM做了很多“风洞实验”（Model Wind Tunnel Experiments）。  

这些风洞实验主要包括三个部分：（1）搜索模型结构的超参（2）探索batch size的scaling（3）寻找最佳的learning rate。

后续风洞实验所用的模型具体参数如下  

{% asset_img exp_model.png 模型结构 %}  

## 模型超参  

预训练资源消耗很大，即使是SLM也不可能把所有参数的排列组合都搜索一遍。  

这里参考Tensor Program的做法（《Tensor programs v: Tuning large neural networks via zero-shot hyperparameter transfer》和《Tensor programs vi: Feature learning in infinite-depth neural networks》），分别对模型的宽度和深度进行了搜索。  

搜索所用的操作如下表所示，这里没有应用attention softmax的scaling技术。  

{% asset_img param_search_2.png 超参搜索 %}  

关于超参搜索的一些细节：  
- 用Maximal Update Parametrization的方法进行了调参。  
- 在一系列预定义的参数空间上进行贝叶斯搜索，所用模型参数为在N=0.009B。发现使用规模为10N和20N大小的数据集进行超参数优化时，超参的有效性表现出了一致性。因此调参的过程就使用D=10N=0.09B个token来进行实验了。  
- 应用了QK-Norm（《Querykey normalization for transformers》）和independent weight decay（《Decoupled weight decay regularization》）之后，发现模型对learning rate的敏感性显著降低。不过在找到最佳learning rate之后，后面的训练就不用再调整参数了，因此后面的实验就没有继续使用QK-Norm和independent weight decay。  

最终从下图展示的参数搜索，确定了最佳的hyper-parameters为：  
- scale depth = 1.4  
- scale emb = 12  
- init std = 0.1  
- lr = 0.01  

{% asset_img param_search.png 超参搜索 %}  

## Optimal Batch Size  

batch size决定了模型收敛速度与计算资源消耗之间的平衡。  

如果batch size太大，会导致很大的数据和计算成本（才能跑到足够的update次数，让模型收敛）。如果batch size太小，将会有大量的update step，并且相比大一些的batch size，loss的减小很有限，训练效率太低。  

这里参考OpenAI的《Scaling laws for neural language models》的方法来寻找最佳的batch size，并做了一些改动。  

《Scaling laws for neural language models》研究的是loss function和token数之间的关系。他们假设了更多的step=更多的训练时间。在这个假设下，OpenAI定义了一个critical batch size，在不消耗过多的step或者token的情况下，能达到一定的loss水平。  

这在无限GPU资源下是合理的。由于GPU资源是无限的，增加batch size不会增加单个step的耗时，但会减少总step数，因而提高了效率。但是实际上我们并没有无限GPU资源，因此将batch size增大相当于增加的每个step的时间，所以实际上通过增加batch size来减小step数，对总训练时间的影响并不大。  

因此MiniCPM放弃了“not consuming too many steps”的目标，转而追求“minimizing the token quantity to achieve the lowest loss”。  

关于optimal batch size与loss之间关系的估计，类似于“先有鸡还是先有蛋”的悖论，因为暂时没法完全搞清楚这二者之间的决定关系。目前的做法是，对于给定的模型大小，通常会有一个初步估计的achievable loss，这是由先前的初步实验得出的经验估计。  

而optimal batch size和optimal learning rate很可能并不是独立的。为了克服这种相关性，MiniCPM首先对learning rate进行了初步研究，然后选择一个最优learning rate来进行batch size实验，并使用batch size缩放再次进行learning rate调整。这有点像Coordinate Descent optimization method。  

细节上，MiniCPM分别对0.009B、0.03B和0.17B的模型进行了实验。每个模型大小都在6种不同的batch size上进行训练，使用了global learning rate=0.01和cosine learning rate scheduler。在C4数据集上，optimal batch size与loss的趋势如下图红线  

{% asset_img batch_size.png 超参搜索 %}  

这三条红线在log空间中很好连成一条直线，如下图。  

{% asset_img batch_size_2.png 超参搜索 %}  

这里就得到了在C4数据集上的训练loss和optimal batch size的关系。  

$$bs=\frac{1.21\times10^9}{L^{6.24}}$$  

## Optimal Learning Rate  

由于使用了Tensor Program，optimal learning rate在模型缩放的过程中应该不会有明显变化。为了验证这一点，MiniCPM在0.04B、0.1B、0.3B和0.5B的模型上进行了六组learning rate的实验。  

在下图中，可以发现尽管模型大小增加了十倍，但optimal learning rate并没有明显的偏移，基本上一致保持在0.01左右。  

{% asset_img learning_rate.png 超参搜索 %}  

MiniCPM进一步在2.1B规模的模型上进行了一个简单的验证，最终确认了0.01的learning rate确实实现了最低loss。  

# WSD  

## cosine learning rate scheduler的分析  

cosine scheduler的周期很重要，一般是设置降到最小learning rate的时间T和预训练的总step数S持平。为了验证这个设置的效果，用0.036B的模型的做了实验，按以下公式分别实现cosine和cosine loop两种scheduler。

{% asset_img cos_lr.png LR %}  

loss的变化如下图  

{% asset_img cos_loss.png LR %}  

可以看到确实总是T=S时效果最好。分析原因可能是：  
- 与T<S的scheduler相比，T=S的scheduler有更长的高learning rate持续时间。而这种高learning rate可能有助于模型找到更好的global optimum。  
- 与T>S的scheduler相比，T=S的scheduler有更彻底的learning rate decay。这种衰减可能涉及到training dynamics，使模型能够找到更好的 local optimum。  

## Warmup-Stable-Decay  

基于上面的分析，MiniCPM把训练过程显式分成high learning rate stage和learning decay stage，这个scheduler就叫Warmup-Stable-Decay scheduler，公式如下  

$$\left.WSD(T;s)=\begin{cases}&\frac{s}{W}\eta,\quad s<W\\&\eta,\quad W<s<T\\&f(s-T)\eta,\quad T<s<S\end{cases}\right.$$  

其中W是warmup的step数，T是stable training step数，$\eta$ 是maximum learning rate，$f\left(s-T\right)$ 是关于s的 decreasing function，取值在0到1之间。  

一般来说W只要足够，对训练的效果影响就不大，因此所有后面就忽略W了。  

继续做一些实验来探索WSD。  

（1）Loss Decreases Dramatically in Decay Stage  

首先在0.036B的模型上应用了WSD，并设置了不同的T和S（影响decay阶段的长度）。发现在decay阶段，随着learning rate的下降，loss出现了显著的快速下降，并迅速降低到等于或低于T=S时的Cosine LRS的loss，具体loss变化如下图  

{% asset_img wsd_exp1.png WSD %}  

由于stable training阶段learning是保持不变的，所以这里可以重用decay前的模型checkpoint，继续进行的高learning rate的训练。在原设置上增加了更多的stable training step之后，还可以再进行learning rate退火，并且能够实现与Cosine LRS在同样step下相同的loss。这进一步验证了“训练阶段可以明确地分为stable阶段和decay阶段”的假设。  

（2）10% Steps are Enough  

如上图所示，在40N、60N和80N训练数据的实验中，使用总token数的10%的进行learning rate decay就足以获得最好的结果，如果小于10%则效果会比较差。因此，在后续的训练实验中，都使用大约10%的step进行learning rate decay，以确保完全收敛。  

（3）Effective Data Scaling with WSD LRS  

使用WSD可以把模型训练到极致收敛的状态。为了展示WSD训练固定大小模型到收敛的潜力，MiniCPM对0.036B的模型进行持续训练，然后与使用40N数据的0.17B模型进行比较，loss如下图。  

{% asset_img wsd_exp2.png WSD %}  

0.036B模型在使用更多的数据后，超过Chinchilla Optimal，并且仍有收敛趋势，按这个趋势继续训练就能match 0.17B模型的loss水平。  

## Measuring the Scaling Law with WSD LRS  

利用WSD，可以把探索model size和data size的scaling关系的成本变成线性，因为stable stage阶段learning保持不变，可以把decay接在不同的step后面来获取不同数据量下的效果。  

通过训练从0.04B到2B共6种大小的SLM来测量scaling law。每种大小的模型都有从10N到60N数据共6个数据量开始decay的结果。  

这36个模型的训练结果在5个数据集上进行比较。为了可以比较不同tokenizer的模型的损失，按《GPT-4 technical report》里的做法，使用byte数的平均而非token数的平均来进行比较。然后用scipy curvefit function，按下面这个公式拟合model size N and data size D的关系。  

$$L(N,D)=C_NN^{-\alpha}+C_DD^{-\beta}+L_0$$  

实验结果和拟合结果如下图  

{% asset_img scaling_law.png scaling law %}  

然后参照《Scaling language models: Methods, analysis & insights from training gopher》、《Training compute-optimal large language models》、《Scaling laws for neural language models》的做法，推算出token数量应该是模型参数量的192倍，这比《Training compute-optimal large language models》中给出的20倍要大得多。  

MiniCPM还把LLAMA2的数据拿出来进行了验证。按LLAMA2报告中给出的数据计算出的token数应是模型参数量的70~100倍，这个值同样比20要大很多。

因此结论是，按照基于WSD的实验结果，语言模型比我们之前想象的可以吸收更多语料数据。  

# Two Stage Pre-training Strategy  

前面观察到WSD的衰减阶段loss有显著的减少，因此MiniCPM认为在learning rate的退火阶段整合高质量SFT数据，混合进预训练数据中可以SFT效果：  
- 一方面，在退火阶段使用SFT数据能获得预SFT更相关的loss下降  
- 另一方面，和在整个预训练阶段都使用SFT数据相比，只在learning rate decay阶段使用更不容易过拟合  

为了验证这个猜想，设计以下训练配置：  
- A-1: 2.4B模型，decay阶段仅用无标签数据，之后进行4B的SFT训练  
- A-2: 2.4B模型，decay阶段使用无标签数据+SFT数据，之后进行4B的SFT训练  
- B-1: 1.2B模型，decay阶段仅用无标签数据，之后进行6B的SFT训练  
- B-2: 1.2B模型，decay阶段仅用无标签数据，之后进行12B的SFT训练  
- B-3: 1.2B模型，decay阶段使用无标签数据+SFT数据，之后进行6B的SFT训练  

各个模型预训练+SFT之后的效果如下  

{% asset_img 2_stage.png 2阶段训练 %}  

可以看到在预训练learning rate退火阶段加入SFT数据的模型效果更好。  

# MiniCPM  

## 模型  

MiniCPM有2.4B和1.2B两个规模。其中2.4B模型的词表大小为122,753，1.2B模型词表大小为73,440，都是通过BPE进行构建。在测试数据集上评测，MiniCPM的tokenizer的效率是比较高的，具体数值如下  

{% asset_img tokenizer.png tokenizer %}  

MiniCPM模型的输入输出共享了矩阵，因为小模型共享输入输出矩阵可以节省很多参数。

在层数和hidden state的设计上，MiniCPM使用了相比Phi-2等SLM更深更瘦的模型结构，这和《Mobilellm: Optimizing sub-billion parameter language models for on-device use cases》的想法一致。具体的结构参数如下  

{% asset_img layers.png 更深更瘦的结构 %}  

1.2B模型上使用了GQA，可以进一步节省参数量。  

## 训练  

在WSD的stable阶段，使用1T预训练数据，batch size=3.93M，max lr=0.01。  

在decay阶段，decay的策略为 $f(s-T)=0.5^{(s-S)/T}$，其中T=5000 steps (20B tokens)。  

SFT阶段共使用了6B数据，learning rate和预训练阶段结束时的learning rate对齐，同样使用了WSD。  

预训练数据的分布如下  

{% asset_img data.png 训练数据 %}  

1.2B和2.4B模型的预训练loss如下图  

{% asset_img train_loss.png training loss %}  

左图loss的第一次突变是因为增大了batch size，效果相当于减小了learning rate。  

最终SFT模型在下游任务的评测结果如下  

{% asset_img eval.png evaluation %}  

## MiniCPM-DPO  

在SFT的基础上，MiniCPM用UltraFeedback数据集进行DPO训练。  

DPO训练使用了Cosine LRS, max learning rate=1e-5，一共训练了一个epoch。  

DPO使得模型在MT-bench上的得分从6.89提升到7.25，但是在原来通用benchmark的效果有所下降。  

## MiniCPM-128k  

长文本训练把MiniCPM支持的窗口大小从4k拓展到128k。在这一阶段的训练禁用了输入输出矩阵共享，这会使得模型的实际参数略有上升。训练的初始模型用的是预训练中stable阶段的最后一个checkpoint。  

MiniCPM将书籍、维基百科文章和论文分类为“长数据”，其他为“短数据”。那么在这一阶段的训练包含了44%的长数据和 56%的短数据。  

训练时不直接训练到128k，而是使用curriculum learning：先训练32k，再训练128k。4k-32k范围内应用ABF，32K到128K的范围内使用NTK-Aware RoPE scaling。  

如Yi的技术报告和《Zebra: Extending context window with layerwise grouped local-global attention》所指出的那样，使用合成的长QA数据，有助于提高模型在上下文感知任务中的性能，MiniCPM也使用了合成的长QA数据。  

MiniCPM-128k在∞Bench（《∞bench: Extending long context evaluation beyond 100k tokens》）上评测结果如下  

{% asset_img 128k_result.png 128k evaluation %}  

## MiniCPM-MoE  

MiniCPM-MoE使用Sparse Upcycling（《Sparse upcycling: Training mixture-of-experts from dense checkpoints》）进行初始化，使用了stable阶段的checkpoint。router用均值为0、方差为0.01的正态分布进行初始化。  

MiniCPM-MoE共有13.6B参数，激活2个专家，共激活4B参数。

训练时使用switch transformer的负载均衡函数，权重系数为0.01。

learning rate使用了WSD，在4M的batch size下共进行了130k步预训练，而在SFT阶段batch size减小了到2M。  

MiniCPM-MoE的效果评测如下  

{% asset_img moe_result.png moe evaluation %}  

# 小结  

MiniCPM站在很多前人结果的肩膀上，把目前各种比较先进的做法融合到了1B/2B模型上，获得了不错的效果。其中用到的参数搜索、对scaling law的刷新都挺有参考价值。  

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
[从loss视角理解大模型涌现能力](https://www.linsight.cn/f5fb75e4.html)  
[LLM长上下文的问题](http://www.linsight.cn/c4da56c0.html)  
[解锁大模型长上下文能力](http://www.linsight.cn/cc852861.html)  
[大模型推理窗口-从有限到无限大](http://www.linsight.cn/45ee1a6d.html)  
[理解Attention:从起源到MHA,MQA和GQA](http://www.linsight.cn/3dc22f96.html)  
[LLM的重复生成和ICL](https://www.linsight.cn/7381cae3.html)  
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

【1】MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training Strategies https://arxiv.org/abs/2404.06395  

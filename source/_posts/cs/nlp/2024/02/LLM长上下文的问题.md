---
title: LLM长上下文的问题
abbrlink: c4da56c0
date: 2024-02-28 15:19:28
tags:
  - NLP
  - LLM
  - transformer
  - 长上下文
  - 窗口外推
categories:
  - CS
  - NLP
  - LLM
---

最近长上下文的业务需求越来越多，刚好把这个能力现状和主流方案的基础内容简单梳理一下。  

跟长文本最相关的自然就是位置编码，现在很多模型都使用了RoPE这种位置编码，之前已经把RoPE的基础内容梳理了一遍：[博客](http://www.linsight.cn/a051710f.html) [知乎](https://zhuanlan.zhihu.com/p/684072868) [微信公众号](https://mp.weixin.qq.com/s?__biz=MzkyODY1MTA3Ng==&mid=2247483759&idx=1&sn=f7b59b879476b8687a340606b5568eae&chksm=c214c344f5634a52e299108c3deddfd2a0eccbf14d9392c205410723956c477925e89e791b9b&token=88551061&lang=zh_CN#rd)

# 关于长上下文  

2023年中开始，各大LLM厂商开始关注到长上下文的问题。2023年5月，Claude把长度支持到100k tokens；6、7月的时候，ChatGPT3.5也已经支持16k，而ChatGLM2-B最大长度已经可以到32k。  

（插一句，ChatGLM系列做得一直很不错，从基础模型、长窗口、工具调用、Agent都一直保持在比较前沿的水平，个人最近用ChatGLM3、ChatGLM4体验还是很不错的）  

差不多同时间还有LM-SYS的LongChat，MosaicLM的MPT也支持16k以及更长的上下文。

今年过年前刚出来的Qwen-1.5系列全家桶也都是32k起步了。还有一些支持超长窗口的模型  

<center>

| 模型 | 支持长度 |
| :----: | :----: |
| Baichuan2 | 192k |
| GPT4-turbo  | 128k |
| Yi | 200k |
| Kimi | 192k |
| Claude2 | 200k |

</center>

大厂商们卷完基础模型效果，把能刷的榜刷完，又盯上了长上下文能力（当然现在长上下文也有榜了）。  

为什么要那么长？  

# 长上下文的需求  

取决于语言和所使用的tokenizer，每个token对应编码的文本有所不同。以中文为例，大部分模型每个token对应的中文字数都>1.5个字（部分高效的tokenizer可以做到2个字以上）。那么200k的token就能对应处理30w字的上下文了。  

最近刚看了刘震云的长篇小说《一句顶一万句》，全书差不多27万字，也就是说现在这些长上下文大模型可以秒读完一部长篇小说，然后和我交流心得，或者告诉我全书的概要，又或者帮我找到一些文中的细节描写。  

上面这个场景对应的是大模型的<big><u>**工具化**</u></big>场景。我们可以借助大模型的能力，来阅读论文，总结研报或者阅读代码，这些场景都需要比较长的上下文输入。  

另外还有一个也比较火的大模型应用场景，RAG（Retrieval-augmented generation），也对长上下文输入有要求，只是在RAG中，大部分输入文本并不是直接来自于用户输入，而是通过检索得来的。  

除了工具化的应用场景，还有一些<big><u>**个性化**</u></big>的场景也会对长上下文有需求。举例来说，就是一些智能助手需要对用户的偏好和设置做长期记忆，这些偏好和设置可以以prompt或者对话的形式持久化存储下来，在进行新的对话的时候就把这些内容连同用户新的输入一起给到模型进行处理。  

实际上，哪怕是单次的聊天，也很有可能需要模型处理比较长的上下文。比如我们可能会让模型扮演一个特定的影视角色或者游戏角色和我们进行对话。这时通常会给模型一些设定，比如这是一个什么样的任务，故事背景世界观都是什么样的，以及现在要进行哪些方面的交流等。这些设定都会以prompt的形式在最开始输入给模型。而随着对话的进行，模型如果长文本能力比较差，就有可能忘记了我们之前给的设定，这样体验上就有问题了。  

上面这个例子实际引出了对长文本需求更具体的内容：（1）在文本比较长的时候，还能说人话，ppl要低（2）说人话之余，还要能attention到前面提过的细节，不能出现自我矛盾。

# 模型怎么支持长上下文

看来目前的很多应用场景确实对长上下文有需求，那怎么实现呢？  

如果我们直接训练2k/4k长度的模型，然后在推理的时候设定8k或者16k窗口，那么PPL会急剧上升，导致模型直接讲不了人话，原因之一在之前讲RoPE的时候也有提到，对于没有训练过的<u>**位置编码**</u>，模型不能很好地处理。  

## 直接训练

既然训练的时候用2k/4k不能很好地在8k/16k/32k+的上下文长度下推理，那直接在训练的时候用更长的数据进行训练不就可以了？  

这个思路理论上可行，只是实操的时候会遇到一些问题（壕可能觉得不是问题）。  

1.训练数据  

直观上来说，要训练长上下文的模型，就需要长文本。要达到32k或者更大的长度，基本都只能是书籍。  

当然，我们也可以通过把多个中等长度的文本进行拼接，再用来训练。比如筛选4k长度数据，那8条拼在一起也够长了。然后通过attention mask来限制各段文本之间注意力，让它们可以在各自的位置上各训各的，互不干扰。甚至实际上即使不做attention mask，效果也挺好。  

总的来说，就是【连续长文本】>【多个中等文本拼接】（也可用）  

2.资源消耗  

来简单看一下transformer在训练中所消耗的资源。  

假设模型有 $l$ 层，词表大小为 $V$ ，hidden size为 $h$ ，batch size为 $b$ ，训练窗口长度为 $s$ ，使用Adam优化器训练（需要存一阶和二阶动量），为简化估算，可以假设注意力头数为1。  

(1) 参数量

模型总参数量 $\Phi$  = 词向量参数量 + $l$ * decoder层参数量 = $Vh + l(12h^2 + 13h)$  

可以看到参数量和窗口长度 $s$ 无关，模型确定了就是一个固定值。  

(2) 计算量  

一次前向计算量 = 输出分类头logits计算 + $l$ * 每层计算量 $\approx2bshV + l*(24bsh^2+4bs^2h)$

看一下计算量和参数量的关系。忽略参数量和计算量中的低次项，则有

$$
\begin{equation}
\begin{aligned}
\frac{计算量}{参数量}
&=\frac{2bshV + l*(24bsh^2+4bs^2h)}{Vh + l(12h^2 + 13h)}\\
&\rightarrow bs\frac{6h+s}{3h}
\end{aligned}
\end{equation}
$$

可以看到，总计算量随着输入长度的增长是平方的。在 $s << h$ 的时候，基本还可以认为是线性的。目前大部分模型的 $h$ 是在1k到1w这个范围，基本上可以认为 $s$ 和 $sh$ 在不是超级长的情况下，还是可比较的。计算量算是长度的“弱”二次方关系  

(3) 显存  

训练过程中，显存主要有模型参数、梯度、optimizer状态值和中间激活值。

训练中，每个参数（$\Phi$）有一个对应梯度（$\Phi$），每个参数又对应优化器一个一阶动量和二阶动量（$2\Phi$）。在混合精度训练中，使用半精度进行前向计算和梯度计算，同时优化器备份一份单精度的优化器状态、梯度和参数用于更新参数，因此共有 $(\Phi + \Phi) \times 2 + (\Phi + \Phi + 2\Phi) \times 4 = 20\Phi = 20[Vh + l(12h^2 + 13h)]$ 的参数占用。

{% asset_img mix_precision_fp16.png 混合精度训练 %}  

这部分跟输入长度没有直接关系。

另外一个需要占用显存的部分是中间激活值。  

保存激活值是为了计算梯度，因此每个矩阵相乘、softmax、dropout都需要保存输入值的中间的激活值。  

对于attention层，输入时先要对 $x$ 做 $Q、K、V$ 投影，需要保存 $x$ 的中间值；计算权重的时候有 $Q、K$ 矩阵的相乘，需要保存 $Q、K$ 矩阵的值；做softmax的时候输入有 $QK^T$ 要保存；以此类推，则需要保存的所有中间激活值为 $11bsh+5bs^2+19sh+4bsh=34bsh+5bs^2$ 。对于 $l$ 层的模型，就再乘以 $l$ 。  

可以看到中间激活值随着 $s$ 增大，是以平方关系在增长。训练4k长度的模型和32k长度的模型，激活值所需的显存增长到了64倍。这种情况下，要么扩大集群，加入更多的GPU，要么减小batch size，或者提升gradient accumulation的值，无论如何，都会增加<big><u>**训练成本**</u></big>。  

小模型（比如2B、7B）可以硬刚，支持到16k或者32k长度，但是对于更大的长度（200k），或者更大的模型（34B、70B+），这么做就性价比就比较低了。  

现在一般的做法是分两阶段，第一阶段用2k或者4k训练一个基础模型，等到模型把文本内容和短位置关系都学好之后，再来用相比第一阶段小的数据量优化在长上下文情况下的效果。  

而第二阶段在如何用更少的训练量达到更好的效果这件事上，又有很多工作。  

## 线性插值 Position Interpolation

23年6月，Meta在[《EXTENDING CONTEXT WINDOW OF LARGE LANGUAGE MODELS VIA POSITION INTERPOLATION》](https://arxiv.org/pdf/2306.15595.pdf)中就提出了针对RoPE的线性插值方法PI（Position Interpolation），可以把2k的基础模型扩展到32k，并在1k个step的训练下就达到很好的效果。

{% asset_img meta_pi.png PI效果 %}  

>In contrast, LLaMA models that are extended via direct fine-tuning only saw a minimal increase of the effective context window size kmax from 2048 to 2560, even after fine-tuning for more than 10000 steps, with no clear indication of an acceleration in the increase of window size.  

相比之下，直接基于基础模型进行长文本微调的效率就比较低，训练1w步后，有效长度只是从2048提升到2560。

看来RoPE虽然拥有诸多优点，长上下文外推这个事情却不在其中。  

论文中对RoPE外推性能也进行了一些分析。本来RoPE是相对位置编码，而且具有远程衰减的特性，理论上应该具备一定的外推能力，但实际上却不是这样。简单地说，论文发现，在相对位置差 $\left|m-n \right|$ 不太大的时候（<2048），确实能保持远程衰减且attention值保持在较小的区间，但是一旦 $\left|m-n \right|$ 超过这个区间，还是有可能出现很大的值。

{% asset_img meta_rope_ext.png RoPE外推 %}  

看上图中间的这个图，在位置超过3000的时候，突然出现很大的attention score。而右边的图使用了插值的方式，就相对稳定。

（远程衰减上界问题具体推导的过程就不展开了，感兴趣的朋友可以看下论文原文）

而另一方面，PI甚至可以在只使用插值，而没有训练的情况下，就拥有一定的长窗口能力。  

{% asset_img meta_pi_nosft.png PI效果 %}  

插值的思路是这样的：如下图所示，左上部分表示预训练过的2k长度的位置编码，右上部分表示在这个基础上直接外推，这样就会出现很多之前没有训练过的值，模型的学习成本会比较高；下半部分表示在已经训练好的2k模型基础上进行插值，类似于在每两个位置编码之间，插入一个位置点，这样总的位置表示就从2k增加到4k。在这个基础上再进行少量的微调，模型就可以很快学到新的位置表示。

{% asset_img meta_pi_explanation.png PI效果 %}  

这个思路也很符合直觉，比如原来模型针对位置1，位置2，位置3...学到了一定的规律，现在告诉模型，位置不一定是整数，变成位置1，位置1.5，位置2，位置2.5...。虽然值变了，但是相对关系还在，因此模型也能借助原来学到的关系，快速推广到“0.5”的位置中。  

由于三角函数光滑的特性，我们可以重新定义attention score的计算，使得结果不要出现异常大的值，也就是 $\tilde{a}(s)=a(Ls/L^{\prime})$ ，$L$ 是原长度（也就是2048），$L^{\prime}$ 是我们想要增大的长度（8k/16k/32k等）。

更具体来说，就是对RoPE做一点修改  

$$
\begin{equation}
\begin{aligned}
\mathbf{f}'(\mathbf{x},m)=\mathbf{f}\left(\mathbf{x},\frac{mL}{L'}\right)
\end{aligned}
\end{equation}
$$

相当于位置 $m$ 的分辨率从1下降成了 ${L}/{L'}$。  

（一通分析，实现却很简单，工程师的福音，真希望以后论文都能给出这样的结果）  

然后使用几万到几十万条样本进行预训练，就可以了。

（文章里还埋有一个后续的关注点，在对长上下文进行训练后，在短上下文的场景下，效果略略有所下降）

## NTK-Aware Interpolation 

线性插值相当于直接线性拉伸了cos函数，减小了不同位置之间的差异，使得模型对于相近位置的区分度有所下降，多少有点简单粗暴，而NTK-Aware Interpolation就提出了理论上看起来更为“精致”方法，一种在RoPE上做<u>**非线性插值**</u>的方法，NTK-Aware Scaled RoPE。CodeLlama就是用这种方法把长度推广到1M。  

NTK，就是Neural Tangent Kernel，神经正切核。具体是什么，让GLM4帮忙解答一下  

>Neural Tangent Kernel (NTK) 是深度学习理论中的一个概念，它是在训练无限宽的神经网络时出现的一个理论工具。在深度学习领域，宽度通常指的是网络中每一层的神经元数量。NTK 的提出，为理解深度学习提供了一个新的视角，特别是在网络宽度趋于无穷大时，网络的行为可以用线性模型来近似。  
在理想情况下，当网络宽度趋于无穷大时，网络的训练过程可以用一个固定的核函数来描述，这个核函数就是 Neural Tangent Kernel。这个核函数描述了网络输出与输入之间关系的性质，使得我们可以用核方法来分析这类网络的训练和泛化能力。  
具体来说，NTK 使得我们能够预测在无限宽的网络中，随着训练的进行，网络权重的演化情况，以及网络的损失函数如何随时间变化。这些预测在数学上更加清晰和可分析，有助于我们深入理解深度学习的工作原理，特别是在泛化和优化方面。  
NTK 的研究对设计新的网络架构、训练算法和理论分析提供了重要的理论基础。它也表明，在一定的条件下，深度学习可以简化为传统的统计学习理论，如核方法，这为将深度学习与经典机器学习理论联系起来提供了桥梁。

这个了解各大概就行。那么具体在长上下文扩展这个事上，NTK有什么说法呢？  

它认为，线性插值把所有方向上的分量一视同仁，把旋转速度统一变慢，这里有问题。  

回顾一下在RoPE中，对位置 $m$ 的输入向量进行“旋转”的矩阵长这样  

{% asset_img rope_matrix.png RoPE旋转矩阵 %}  

它把输入向量的元素划分成2个2个一组，共有 $d/2$ 组，每组有两个元素，不同组分别旋转。这里可以发现每组的旋转速度并不相同，由于 $\theta_j=10000^{-2j/d}$ ，可以看到， $j$ 越小越靠前的组旋转越快，$j$ 越大的旋转越慢。这里 $base=10000$ ， $base$ 越大，整体的旋转速度越慢，反之越快。同一个位置下，由于旋转速度不同，位置向量的信号频率有高低，前面的部分是高频，越往后越低频。  

不加区分地对高低频信息进行拉伸，会丢失很多重要的高频信息，这样不是很好。高频信号应该外推，以防止分辨率太低，都挤在一起；而低频信号就适合插值。  

怎么实现“高频外推，低频内插”？  

先看回讲[RoPE](https://www.zhihu.com/people/us4ever)的时候，对于2维情况，有  

$$
\begin{equation}
\begin{aligned}
&\langle f_q(\boldsymbol{q}_m,m),f_k(\boldsymbol{k}_n,n)\rangle= \mathrm{Re}\left[\boldsymbol{q}_m\boldsymbol{k}_n^*e^{i(m-n)\theta}\right]
\end{aligned}
\end{equation}
$$  

推广到高维的情况，则有  

$$
\begin{equation}
\begin{aligned}
\langle f_q(\boldsymbol{q}_m,m),f_k(\boldsymbol{k}_n,n)\rangle=&\mathrm{Re}[\sum_j^{d/2}h_je^{is\theta_j}]\\
\end{aligned}
\end{equation}
$$  

其中 $h_j=\boldsymbol{q}_m\boldsymbol{k}_n^*$ ，$s=m-n$ 。  

在这个公式下，线性插值相当于把  

$$
\begin{equation}
\begin{aligned}
\mathrm{Re}[\sum_j^{d/2}h_je^{is\theta_j}]\\
\end{aligned}
\end{equation}
$$  

变成了  

$$
\begin{equation}
\begin{aligned}
\mathrm{Re}[\sum_j^{d/2}h_je^{i\frac{s}{\alpha}\theta_j}]\\
\end{aligned}
\end{equation}
$$  

其中 $\alpha=L'/L>1$ ，相当于把 $s$ 压缩了。  

而NTK-Aware Scaled RoPE则是对 $\theta_j$ 进行了改动，具体来说，是修改了其中的base值（RoPE中原来是10000）  

$$
\begin{equation}
\begin{aligned}
\hat{base}=base\times\alpha^{\frac{d}{d-2}}
\end{aligned}
\end{equation}
$$  

则有

$$
\begin{equation}
\begin{aligned}
\hat{\theta_j}=\hat{base}^{-2j/d}=base^{-2j/d}\times\alpha^{\frac{-2j}{d-2}}
\end{aligned}
\end{equation}
$$  

相当于 $\theta$ 乘了一个系数 $\alpha^{\frac{-2j}{d-2}}$ ，当 $j$ 比较小的时候， $\alpha^{\frac{-2j}{d-2}}$ 接近1，相当于直接进行了外推，而当 $j$ 比较大的时候（注意 $j$ 的取值是从0到 $d - 1$），$\alpha^{\frac{-2j}{d-2}}$ 就接近 $\alpha^{-1}$ ，这就和线性插值趋近了。

引用来自[知乎一篇文章](https://zhuanlan.zhihu.com/p/645770522)的一个视角来理解NTK-Aware Interpolation  

>有意思的解释一下，RoPE 的行为就像一个时钟。12小时时钟基本上是一个维度为 3、底数为 60 的 RoPE。因此，每秒钟，分针转动 1/60 分钟，每分钟，时针转动 1/60。现在，如果将时间减慢 4 倍，那就是二使用的线性RoPE 缩放。不幸的是，现在区分每一秒，因为现在秒针几乎每秒都不会移动。因此，如果有人给你两个不同的时间，仅相差一秒，你将无法从远处区分它们。NTK-Aware RoPE 扩展不会减慢时间。一秒仍然是一秒，但它会使分钟减慢 1.5 倍，将小时减慢 2 倍。这样，您可以将 90 分钟容纳在一个小时中，将 24 小时容纳在半天中。所以现在你基本上有了一个可以测量 129.6k 秒而不是 43.2k 秒的时钟。由于在查看时间时不需要精确测量时针，因此与秒相比，更大程度地缩放小时至关重要。不想失去秒针的精度，但可以承受分针甚至时针的精度损失。  

另外苏剑林从“进制”角度对RoPE作了分析，感兴趣的朋友可以看下[原文](https://kexue.fm/archives/9675)，也很巧妙。  

在YaRN的[论文](https://arxiv.org/pdf/2309.00071.pdf)中，对NTK的优缺点作了点评  

>Given the results from [6], this method performs much better at extending the context size of non-finetuned models compared to PI [9]. However, one major disadvantage of this method is that given it is not just an interpolation scheme, some dimensions are slightly extrapolated to "out-of-bound" values, thus fine-tuning with "NTK-aware" interpolation [6] yields inferior results to PI [9]. Furthermore, due to the "out-of-bound" values, the theoretical scale factor s does not accurately describe the true context extension scale. In practice, the scale value s has to be set higher than the expected scale for a given context length extension.  

NTK的优点是不用微调的情况下，能比线性插值做得好。但是由于低频部分还是会有部分被外推到超出范围的值，因此在设定系数的时候，要比需要的设得更大才行。比如想4k模型要在32k的时候取得比较好的效果，那 $\alpha=L'/L$ 就要选得比8更大一些，比如16。

## NTK-by-parts

NTK-by-parts的方法在NTK插值的基础上又多想了一层。它认为无论是线性插值还是NTK-aware插值，认为RoPE的所有分量都对网络有同样的重要性。而NTK-by-parts的思路认为，应该区别对待不同分量，他们对网络的影响有所不同。  

对于分量 $j$ ，RoPE嵌入的波长  

$$
\begin{equation}
\begin{aligned}
\lambda_j=\frac{2\pi}{\theta_j}=2\pi\cdot base^{\frac{2j}{d}}
\end{aligned}
\end{equation}
$$  

$\lambda_j$ 代表旋转一周所需的长度。当 $j$ 比较小时，波长短，反之波长长，这也对应我们前面说的，前面的分量高频，后面的分量低频。  

这里观察到，当 $j$ 比较大时，波长就可能比 $L$ 要大，这种情况下RoPE一圈都没有转完，会导致这个分量的分布不均匀（比如 $sin$ 只转了1/4圈，那值全都集中在0~1之间，-1~0的就没有值）。这种情况下，这个维度的编码相当于是绝对位置编码了，因为几乎每个位置都有自己独特的一个值。反之当 $j$ 比较小时，模型只能访问到相对位置信息。  

此外，插值会导致相邻或相近位置的关系更近（因为旋转量小，点积更大），文章认为这样会损害模型理解局部关系的能力，因此选择不对高频部分进行插值。NTK-by-parts的思路是  

- 如果维度 $j$ 的波长 $\lambda_j$ 远小于上下文长度 ，就不插值只外推  
- 如果波长 $\lambda_j\geq$ 上下文长度，就只插值不外推  
- 中间的部分就同时存在两种，类似NTK-aware interpolation  

引入一个比例 $r(j)=\frac{L}{\lambda_j}$ 来表示波长和上下文长度的关系。另外还需要两个阈值 $\beta_1、\beta_2$ 来区分以上三种情况。如果 $r(j)<\beta_1$ ，就认为波长大，如果 $r(j)\geq \beta_2$ ，就认为波长小。方便起见，定义一个斜坡函数  

$$
\begin{equation}
\begin{aligned}
\left.\gamma(r)=\left\{\begin{matrix}0&if&r(j)<\beta_1\\1&if&r(j)\geq\beta_2\\\frac{r-\beta_1}{\beta_2-\beta_1}&otherwise\end{matrix}\right.\right.
\end{aligned}
\end{equation}
$$  

NTK-by-parts插值可以定义为对 $\theta_j$ 的一个操作  

$$
\begin{equation}
\begin{aligned}
\hat{\theta_j}=\left(1-\gamma(r(j))\right)\frac{\theta_j}s+\gamma(r(j))\theta_j
\end{aligned}
\end{equation}
$$

这里有两个超参 $\beta_1、\beta_2$ 要定，文中根据实验给出的推荐值是 $\beta_1=1，\beta_2=32$ ，也就是当波长和上下文长度一样长的时候，认为波长大，就只插值，当波长小于上下文长度1/32时，认为波长远小于上下文，就只外推。   

## Dynamically NTK Scaled RoPE  

无论是线性插值还是NTK-Aware Interpolation，都是通过使用一个固定的系数，对原RoPE做了一个缩放，这样就会有一些局限。一方面，这种情况下，模型能支持的最大上下文就由使用的这个缩放系数来决定了，超出这个范围的，依然会出现attention score暴增的风险。另一方面，在解码过程中，当已解码的长度 $l$ 还没有达到训练长度 $L$ 时，就使用 $\alpha$ 来修改base，也可能带来一些损失。Dynamically NTK Scaled RoPE是在NTK插值的基础上，把固定的系数改成动态的系数。  

具体来说，就是  

$$
\begin{equation}
\begin{aligned}
\hat{\alpha}=max(1,\frac{l}{L})
\end{aligned}
\end{equation}
$$  

这样随着解码长度 $l$ 的增长，当 $l>L$ 之后 $\alpha$ 从1逐渐增大， $l\leq L$ 时则不需要改动。  

有一点要注意的是，使用动态的系数时要注意kv-cache的缓存机制是否正确，记得要缓存使用应用RoPE之前的值。  

## YaRN  

上面的方法都是使用插值，研究者发现，随着插值，token之间的距离变得更近（因为现在旋转角度变小了），平均最小距离在减小，这样注意力softmax的分布会变得更尖（也就是都集中在某个区间）。换句话说，就是RoPE原本远距离衰减的特性变弱了，衰减得更不明显，就会导致模型更平均地关注到更多的token，这样就削弱了注意力机制，导致输出质量下降。  

当将RoPE插值到更长的上下文时，注意力softmax分布中的熵会减少，因此研究者的目标是逆转这种熵减（即增加注意力logit的“温度”）。这可以通过在softmax之前，将中间注意力矩阵乘以温度 $t>1$ 来完成，但由于RoPE被编码为一个旋转矩阵，就可以简单地按常数因子 $\sqrt{t}$ 来扩展RoPE的长度。这样可以不必修改注意力的代码。  

$$
\begin{equation}
\begin{aligned}
\text{softmax}\left(\frac{\mathbf{q}_m^T\mathbf{k}_n}{t\sqrt{d}}\right)
\end{aligned}
\end{equation}
$$  

通过对Llama 1和Llama 2的实验，文章提出了建议值$\begin{aligned}\sqrt{\frac1t}&=0.1\ln(\alpha)+1.\end{aligned}$。这个值的效果再Llama各个版本和规模的模型都能有比较好的效果，这样说明这样的熵变在长文本中是常见的。  

YaRN最终的方法就是结合NTK-by-parts，以及使用这个温度值对attention score进行调整。  

YaRN在微调以及无微调的情况下，效果都比上面的几种都要好。

## logn  

logn指的是对attention计算中的缩放因子 $\sqrt{d}$ 进行通过logn进行改进的一个方法，苏剑林在[博客](https://zhuanlan.zhihu.com/p/678755776)中进行了分析。大致的思路和YaRN中的缩放颇有些相似。  

简单来说，依然是希望在长上下文的时候，引入了更多token的情况下，已有的token还能保持聚焦在原来哪些token上，而不要被过分分散了注意力。因此提出了一个新的attention score公式  

$$
\begin{equation}
\begin{aligned}
\text{Attention}_E(\boldsymbol{Q},\boldsymbol{K},\boldsymbol{V})=\text{softmax}\left(\frac{\log_{L}{L'}}{\sqrt{d}}\boldsymbol{Q}\boldsymbol{K}^\mathrm{T}\right)\boldsymbol{V}
\end{aligned}
\end{equation}
$$  

可以看到，当 $L'>L$ 时，其效果和YaRN中的放缩是类似的。

## 其他

在扩展推理长度上，还有很多其他有效的工作，比如各种window attention，streaming LLM，LongLoRA，Focus Transformer等，还有数据、评测等更方面的分析，待逐个梳理。

# 小结  

较短的预训练模型（2k、4k）应用在长上下文会因为训练和推理的两个不一致导致效果下降  

- 推理时用到了没训练过的位置编码  
- 推理时注意力机制所处理的token数量远超训练时的数量，导致注意力机制的崩坏  

这两个问题分别可以从位置编码和attention score的放缩来缓解。  

线性插值PI、NTK插值、分部NTK插值都可以缓解第一个问题，logn和YaRN则把第二个问题纳入的考虑。目前这些方法在实际应用中也有很多变体，包括超参的修改，函数的重定义等。  

# Reference  
【1】分析transformer模型的参数量、计算量、中间激活、KV cache https://zhuanlan.zhihu.com/p/624740065  
【2】EXTENDING CONTEXT WINDOW OF LARGE LANGUAGE MODELS VIA POSITION INTERPOLATION https://arxiv.org/pdf/2306.15595.pdf  
【3】Transformer升级之路：10、RoPE是一种β进制编码 https://kexue.fm/archives/9675  
【4】YaRN: Efficient Context Window Extension of Large
Language Models https://arxiv.org/pdf/2309.00071.pdf  
【5】详解基于调整RoPE旋转角度的大模型长度外推方法 https://mp.weixin.qq.com/s/RtI95hu-ZLxGkdGuNIkERQ  
【6】浅谈LLM的长度外推 https://zhuanlan.zhihu.com/p/645770522
【7】想让大模型在prompt中学习更多示例，这种方法能让你输入更多字符 https://cloud.tencent.com/developer/article/2330611  
【8】Transformer升级之路：8、长度外推性与位置鲁棒性 https://spaces.ac.cn/archives/9444  
【9】RoPE外推优化——支持192K上下文长度 https://zhuanlan.zhihu.com/p/678755776
***

博客：[http://www.linsight.cn/](http://www.linsight.cn/)  
知乎：[Linsight](https://www.zhihu.com/people/us4ever)  
微信公众号：Linsight  
![](/images/qrcode.jpg)

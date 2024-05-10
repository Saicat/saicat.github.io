---
title: MoE模型的前世今生
tags:
  - NLP
  - LLM
  - transformer
  - MoE
categories:
  - CS
  - NLP
  - LLM
abbrlink: 44e38c1b
date: 2024-03-30 09:56:05
---

【本文已在同名 微信公众号 / 知乎 / [个人博客linsight.cn](http://www.linsight.cn/) 上线】  

***

2024年3、4月这段时间，很多MoE模型扎堆发布，包括Qwen1.5-MoE、DBRX、Jamba和Mistral等。  

下面这个表格列出了部分近期发布的MoE工作  

<center>

| 模型 | 发布时间 | 备注 |
| :----: | :----: | :----: |
| GPT4 | 2023年3月 | 23年6月George Hotz爆料GPT4是8×220B模型 |
| Mistral-8×7B | 2023年12月 | Mistral AI，开源 |
| LLAMA-MoE | 2023年12月 | github开源项目 |
| DeepSeek-MoE | 2024年1月 | 幻方量化，国内首个开源MoE模型，有技术报告 |
| abab6 |2024年1月 | MiniMax，号称千亿MoE，无开源，无细节发布 |
| 天工2.0 | 2024年2月 | 昆仑万维，无开源，无细节发布 |
| Step-2 | 2024年3月 | 阶跃星辰，无开源，无细节发布 |
| MM1 | 2024年3月 | 苹果，多模态MoE，无开源，有技术报告 |
| Grok-1 | 2024年3月 | X，开源 |
| Qwen1.5-MoE-A2.7B| 2024年3月 | 阿里巴巴，开源 |
| DBRX | 2024年3月 | Databricks，开源 |
| Jamba | 2024年3月 | AI21，开源 |
| Mistral-8×22B | 2024年4月 | Mistral AI，开源 |
| WizardLM-2-8×22B | 2024年4月 | 微软，开源 |
| 天工3.0 | 2024年4月 | 昆仑万维，400BMoE |
| Arctic | 2024年4月 | Snowflake，480B，Dense-MoE Hybrid，开源 |

</center>  

MoE模型目前风头正劲，就连前不久小米汽车发布会上，雷总也弄了个多模态MoE大模型做汽车智能中控  

{% asset_img xiaomi_moe.jpg 小米汽车多模态MoE模型 %}  

相信今年接下来的这段时间，MoE还会给我们带来更多的大新闻。  

本篇将初步梳理MoE相关的一些经典工作和几个近期发布的中文MoE模型，从背景、思路和效果来了解MoE模型。  

到文章发出的2024年4月为止，个人认为DeepSeek-MoE和Qwen1.5-MoE是中文领域做得比较好的两个工作，赶时间的朋友可以优先关注这两个工作。

# 时间线  

这里先对后面会涉及的MoE相关工作，大致按时间线梳理一下，也列出一些关键信息包括模型结构、模型规模等。  

（很多经典的MoE工作都出自Google）

## 上古时代  

首先是很多MoE相关论文都会引用的，发表在1991年的论文[《Adaptive Mixtures of Local Experts》](https://www.cs.toronto.edu/~hinton/absps/jjnh91.pdf)，这篇文章出自Geoffrey Hinton和Michael I. Jordan两位大神之手。虽然在更早的时候就有MoE相关概念的工作，如原文所提到的，1988年这个概念就有了  

>This idea was first presented by Jacobs and Hinton at the Connectionist Summer School in Pittsburg in 1988.  

但是大部分MoE文章还是认为是这个工作奠定了MoE的基础。  

## RNN时代  

时隔二十多年，Google在2017年1月发布了[《Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer》](https://arxiv.org/abs/1701.06538)，把MoE带进了LSTM，训出了最大137B参数，专家数达到128k的LSTM模型。  

## Transformer时代  

1. 2020年6月，Google发布[《GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding》](https://arxiv.org/abs/2006.16668)，把MoE应用在encoder-decoder结构的transformer模型上，每两层将一个FFN层替换成一个MoE层，训出了模型参数量从12.5B到600B的一系列MoE模型，每层最大专家数也达到2048个。  

2. 2021年1月，Google发布[《Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity》](https://arxiv.org/abs/2101.03961) ，在T5（encoder-decoder结构）的基础上，把FFN层替换成MoE层，并简化了routing策略，训出了最大1.6T参数量的switch transformer。Switch Transformers对scaling、蒸馏等做了很多详细的探索，影响深远，是很重要的一个工作。  

3. 2022年2月，Google发布[《ST-MoE: Designing Stable and Transferable Sparse Expert Models》](https://arxiv.org/abs/2202.08906)，也是一个基于encoder-decoder结构的MoE模型，最大模型有269B的总参数，32B的激活参数。ST-MoE可以说不仅仅是一个MoE工作，对于模型结构、工程实现、训练策略等都做了很多分析，个人认为其重要程度相比Switch Transformer都有过之而无不及。  

## GPT时代  

1. 2021年12月，Google发布了GLaM，[《GLaM: Efficient Scaling of Language Models with Mixture-of-Experts》](https://arxiv.org/abs/2112.06905)，训出了最大为1.2T参数量的decoder-only模型。（从encoder-decoder到decoder-only，可以看到Google内部在模型结构方向上也有很多不同的尝试）  

2. 2024年1月，幻方量化发布[《DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models》](https://arxiv.org/abs/2401.06066)，对在23年12月开源的DeepSeekMoE，给出了一些细节。  

3. 2024年，Databricks的DBRX、阿里的Qwen1.5-MoE-A2.7B、Mistral AI的Mistral-8x22B等陆续发布。  

# 奠基工作  

Geoffrey Hinton和Michael I. Jordan的[《Adaptive Mixtures of Local Experts》](https://www.cs.toronto.edu/~hinton/absps/jjnh91.pdf)是大多数MoE论文都会引用的最早工作。  

1. 思路  

这篇文章大致的思路是这样的：对于比较复杂的任务，一般可以拆分为多个子任务。比如要求计算输入文本中有多少个动词和名词，那就可以拆分为“数动词”和“数名词”这两个子任务。  

而一个模型如果要同时学习多个子任务，多个子任务相互之间就会互相影响，模型的学习就会比较缓慢、困难，最终的学习效果也不好。  

因此这篇文章提出了一种由多个分开的子网络组成的监督学习方法。这些分开的网络，在训练过程中，分别学习处理整个训练数据集中的一个子集，也就是一个子任务。这个思路就是现代MoE的思路，每个子网络（也就是一个expert）学习处理一部分内容。  

文章里把这个MoE的方法应用于vowel discrimination task，即元音辨别任务，验证了MoE设计的有效性。元音辨别指的是语音学中区分不同元音的能力，在语音学中，模型需要学习辨别不同的元音因素，以便准确地理解和识别语音输入。通过让多个子模型分别学习分别学习不同元音（a、e、i、o、u）辨别的子任务，最终效果得到了提升。  

2. 模型设计  

下图展示的就是这个MoE的思路：各个expert network和gating network接收同样的输入，每个expert给出各自的处理结果；而gating network输出每个expert的权重，就像一个开关一样，控制着每个expert对当前输入的打开程度，只是这个开关不是离散的，而是stochastic的，给出的不是true和false，而是权重。  

{% asset_img vanilla_moe.png Vanilla MoE %}  

3. 损失函数优化  

实际上，MoE这个idea在这篇文章之前就有了。如论文中所提，Jacobs和Hinton在1988就讨论过。但是之前的工作在loss的设计上，和ensemble更相近，多个expert之间更倾向于合作，每个expert会学习其他expert的residual部分。  

具体来说，对于case $c$，假设第 $d^c$ 是对应的ground truth，第 $i$ 个expert的输出是 $o_{i}^c$，$p_{i}^c$ 是gating network给第 $i$ 个expert分配的权重，那么以前的工作所使用的损失函数 $E^{c}$ 计算如下

$$E^{c}=\left|\left|d^{c}-\sum_{i}p_{i}^{c}o_{i}^{c}\right|\right|^{2}$$

这样的损失计算方式，是把期望输出和所有expert输出的混合结果进行比较。  

这样做的结果是，在训练过程中，每个expert学习的其实是其他expert的组合结果所剩下的残差。这样的学习目标并不能很好迫使每个expert单独输出好的结果，因此不能得到稀疏的模型。  

从另一个角度来看，这个损失计算把所有专家耦合在了一起。即当一个expert的输出发生了变化，所有expert的组合结果也会变化，其他所有的expert也需要做相应的改动来适应这个变化。因此各个expert之间更加倾向于合作，而不是相互竞争并单独给出好的结果，让gating network输出稀疏的结果。  

虽然可以使用如增加辅助损失函数的做法，迫使模型给出稀疏激活的结果，但是这样相当于增加了很强的先验正则化，对模型最终效果也是有损害的。  

而Hinton和Jordan在这个工作里，提出更简单的做法是对loss计算进行修改，使得各个expert之间的关系从合作变成竞争。  

假设gating network每次随机选择一个expert，损失计算如下  

$$E^{c}=\langle\|\mathbf{d}^c-\mathbf{o}_i^c\|^2\rangle=\sum_{i}p_{i}^{c}\left|\left|d^{c}-o_{i}^{c}\right|\right|^{2}$$

在这个损失函数中，每个expert的输出结果会单独和期望结果进行对比，这就要求每个expert单独给出完整的结果，而不是仅学习其他expert的残差。  

这样的loss计算具有localization的特性，即如果一个训练case错了，那么会被修改的主要是被gating network选中且出错的expert，以及负责分配权重的gating network，而不会很大地影响其他expert。  

此外，localization还体现在，每个expert只会负责处理输入空间中某个特定子空间的向量，而不是完整的输入空间。  

这样一来，不同的expert之间不会直接相互影响，虽然还是有间接的影响，比如某个expert的输出变了，gating network可能会分配新的权重，但是至少不会改变其他expert error的符号（+，-），即优化的方向。  

最终的结果是，对于给定的输入，这样的系统会倾向于以高权重分配单一一个expert来预测结果（但其他权重还不是真正的0，不是真正的稀疏）。  

4. 实操技巧

上面提出的这个loss计算，理论上没有问题，实际上也能训练，但是为了得到更好的效果，作者把原loss计算作了如下变化：先指数化再求和，最后再取对数，得到了优化loss。看下变化前后的对比  

$$\text{原loss：}E^{c}=\sum_{i}p_{i}^{c}\left|\left|d^{c}-o_{i}^{c}\right|\right|^{2}$$  

$$\text{优化loss：}E^c=-log\sum_ip_i^ce^{-\frac12\|\mathbf{d}^c-\mathbf{o}_i^c\|^2}$$  

这样做有什么好处呢？来对比一下原loss函数和优化后的loss函数的求导结果  

$$\text{原loss导数：}\frac{\partial E^c}{\partial\mathbf{o}_i^c}=-2p_i^c(\mathbf{d}^c-\mathbf{o}_i^c)$$  

$$\text{优化loss导数：}\frac{\partial E^c}{\partial\mathbf{o}_i^c}=-\left[\frac{p_i^ce^{-\frac{1}{2}\|\mathbf{d}^c-\mathbf{o}_i^c\|^2}}{\sum_jp_j^ce^{-\frac{1}{2}\|\mathbf{d}^c-\mathbf{o}_j^c\|^2}}\right](\mathbf{d}^c-\mathbf{o}_i^c)$$  

相比原loss函数的导数，优化后的loss函数的导数，把当前第 $i$ 个expert的表现，和其他expert联系起来了。这样能够更好地衡量expert $i$ 对当前case的处理结果好坏。特别是在训练初期，gating network的权重是近似平均分配的，那么使用原loss函数的结果是，对当前case效果最好的expert，学习速度是最慢的（因为loss最小）；而优化的loss函数则可以让当前最好的expert的学习速度最快。相当于让“有天赋”的专家在对应的子任务上尽快提高水平。这样就强化了localization的特征，使得各个expert更快拟合到自己擅长的部分，加速训练。  

（BTW，优化后的这个loss导数，和现在的对比学习形式上看起来也很相似）  

这个工作在今天看来不很复杂，但是思路还是很踏实有效的，给MoE奠定了基础。  

# LSTM MoE  

Google在2017年1月发布了
[《OUTRAGEOUSLY LARGE NEURAL NETWORKS: THE SPARSELY-GATED MIXTURE-OF-EXPERTS LAYER》](https://arxiv.org/abs/1701.06538)，把MoE应用到了LSTM上，训出了最大137B的LSTM模型。这样规模的模型哪怕放在7年后的今天，也是巨无霸的存在，需要解决很多工程问题。  

相比1991年的工作，这里做到了真正的稀疏激活，从而可以在实际计算量较少的情况下，训练巨大的模型。  

## 背景

虽然当时Transformer还没出来，大规模模型的竞赛也还不像今天这么激烈，但是在多个领域中（文本、图像、音频），已经有不少工作反复证实了一件事：模型容量越大，能训出来的效果越好，上限越高。但是模型越大，需要的训练数据也就越多，二者共同作用下，就造成了训练开销基本是随着模型增大，以平方关系在增长。  

在这个背景下就出现一些conditional computation，条件计算的工作来解决这个问题。conditional computation就是根据输入，有选择地只激活部分网络模块。那么MoE其实就是一种条件计算的实现。由于不用激活全部参数，训练所需的计算量就大大减小，整体计算成本就不用以平方速度增长。  

虽然理论上计算量的成本下来了，不过实操起来还是会遇到几个问题：  

- 训练的时候，在MoE结构下，每个expert的batch size比整个模型的batch size小了。  
比如模型的batch size是32，一共有16个expert，那实际上一次迭代平均每个expert只能分到2个训练样本。而batch size对训练效率影响是很大的，大的batch size摊小了参数传输和更新的成本。如果直接增大模型的batch size，又会受显存和通讯效率的限制。  
- 训练数据量不足。  
要训大模型就需要大量的数据，让模型参数充分学习。在当时的背景下，大规模的NLP数据是比较缺的。当然如今数据集多了很多，特别是预训练数据，这个问题现在来看没有那么突出了。  
- 损失函数的设计。  
如何使用合适的损失函数来训练模型，提升效果，并且使得模型的负载比较均衡，这是一个不容易解决的问题。  
- 集群通讯问题。  
一个GPU集群的计算能力可能比设备间网络带宽的总和高出数千倍，因此设备间的通讯很可能成为训练效率的瓶颈。为了计算效率，就要使得设备内计算量和所需的通讯量的比值，达到相应的比例。  
- GPU计算特点。  
GPU做数学计算很快，但是并不擅长做branching（if/else），因此MoE的工作基本上都是用gating network来控制参数的激活。这个严格来说不算是新的挑战了，应该说是根据计算设备沿用下来的设计。  

要解决好这些问题，才能训出比较好的模型来。  

## 模型设计

1. 整体结构  

先看下模型结构的设计。  

论文里使用的是两个LSTM层，中间夹着一个MoE层，最上面和最下面分别还有一个embedding层和一个任务输出层，结构如下图所示  

{% asset_img rnn_moe.png LSTM MoE %}  

每个expert是一个简单的feed-forward neural network。一共有n个expert，gating network输出是一个稀疏的n维向量  

$$\begin{aligned}y=\sum_{i=1}^nG(x)_iE_i(x)\end{aligned}$$  

$E_{i}(x)$ 是第 $i$ 个expert的输出，$G(x)_{i}$ 是gating network给出的第 $i$ 个expert的权重。  

如果 $G(x)_{i}$ 为0，就不用计算对应的那个expert了，节省了计算。  

如果expert的数量特别多，可以用two-level hierarchical MoE，即使用两层gating network，第一层的gating network先选择一个包含一批expert的分支，每个分支又有一个单独的gating network来选择具体的expert。类似word2vec训练所用的hierarchical softmax。这样做可以节省一些计算。  

2. gating network  

那具体gating network怎么设计呢？  

如果对输入进行线性变换，再简单加上一个softmax，那得到的是一个非稀疏的gating function  

$$\begin{aligned}G_\sigma(x)=Softmax(x\cdot W_g)\end{aligned}$$  

在这个基础上，使用一个topk函数，只保留最大的k个值，其他都设为﹣∞（softmax之后变成0），这样就能只选择部分expert，得到了稀疏性。  

论文提到，虽然理论上这个形式的sparsity（topk）会造成gating function的不连续，不过在实操中暂时没有遇到相关问题。  

在这个基础上，在输入再加上一个Gaussian noise，这个noise的大小由另外一个可学习的参数来控制。整体的计算公式如下  

$$\begin{aligned}G(x)=Softmax(KeepTopK(H(x),k))\end{aligned}$$  

$$KeepTopK(v,k)_i=\begin{cases}v_i&\text{if }v_i\text{ is in the top }k\text{ elements of }v.\\-\infty&\text{otherwise.}\end{cases}$$  

$$\begin{aligned}H(x)_i=(x\cdot W_g)_i+StandardNormal()\cdot Softplus((x\cdot W_{noise})_i)\end{aligned}$$  

其中用来调整noise的非线性函数softplus是个类似ReLU的激活函数，但是更为光滑，函数图像如下  

{% asset_img softplus.png softplus %}  

这里添加噪声的原因和负载均衡有关，下面来分析下负载均衡。  

## 负载均衡  

在MoE模型训练的实验中观察到，如果不对gating network进行干预，任由模型自由学习，那么最终模型会倾向于收敛到“总是选那几个固定的expert”的状态，而其他expert几乎不会被使用。这就是负载不均衡的状态，如果这些专家分布在不同的计算设备上，结果就是有些设备输入排队特别长，而有些设备基本处于闲置状态，这明显不是我们想要的。  

这种负载不均衡的状态有自我加强的属性，因为一旦开始出现部分专家被较多选中激活，这些专家就会得到更充分的训练，从而获得更好的效果，进而又提升被选中激活的概率。  

针对这种情况，之前有一些工作使用hard constraint来缓解，比如当某个expert激活次数达到上限，就把它从候选集合中移除。hard constraint明显会对模型效果有影响。而这篇论文使用的是一种soft constraint。  

具体来说，对于每个expert，定义了一个它在当前这批输入数据里的重要性指标，如以下公式所示  

$$Importance(X)=\sum_{x\in X}G(x)$$  

$G(x)$ 是gating network给出的权重，是一个维度等于expert数量的向量。  

基于这个重要性指标，论文定义了一个辅助损失 $L_{importance}$，训练时和模型的交叉熵损失加到一起。$L_{importance}$ 的计算方式如下  

$$L_{importance}(X)=w_{importance}\cdot CV(Importance(X))^2$$  

其中权重 $w_{importance}$ 是手动设置的超参，实验的推荐值是0.1，CV是coefficient of variation。  

coefficient of variation离散系数，是概率分布离散程度的一个归一化量度，定义为标准差 $\sigma$ 和 均值 $\mu$ 的比值。  

对于MoE来说，确定激活的expert数之后，均值是固定的。如果expert的gating很不平衡，标准差就会很大，离散系数也会很大，使得 $L_{importance}$ 变大。  

但是这里还是有问题，虽然均衡的负载可以推导出 $L_{importance}$ 较小的结论，但是 $L_{importance}$ 较小却不能保证负载均衡。也就是说 $L_{importance}$ 较小只是负载均衡一个必要不充分条件。  

比如一个expert可能以很高的权重被分配到一个样本，而另一个expert可能以不太高的权重被分配到好几个样本。这种情况下对所有输入数据的gating权重进行求和，仍然可能呈现出均匀的表象（离散系数比较小），但这并不符合我们的要求。  

为了解决这个问题，需要额外再加上一个损失 $L_{load}$ 。这里就要用到添加在每个expert输出上的随机噪音了。  

我们想要各个expert的负载均衡，也就是每个专家需要处理的样本数基本一致，但是分配到各个专家的样本数是个离散值，因此没有办法直接用于back propagation，而 $L_{load}$ 就是对各个expert负载的一个平滑评估。  

回想一下前面在设计MoE的时候，定义了 $H(x)$ 为KeepTopK函数的输入  

$$\begin{aligned}G(x)=Softmax(KeepTopK(H(x),k))\end{aligned}$$  

$$\begin{aligned}H(x)_i=(x\cdot W_g)_i+StandardNormal()\cdot Softplus((x\cdot W_{noise})_i)\end{aligned}$$  

那么这里先定义一个 $kth\_excluding(H(x),k,i)$，表示在除去 $H(x)$ 中的第 $i$ 个分量之后，排在第 $k$ 大的值。基于这个，再定义 $P(x,i)$ 为：固定其他分量已经选取好的noise，重新给第 $i$ 个分量再添加一次noise，结果比 $kth\_excluding(H(x),k,i)$ 大的概率，公式如下  

$$\begin{aligned}P(x,i)=Pr\Big((x\cdot W_g)_i+StandardNormal()\cdot Softplus((x\cdot W_{noise})_i)\\>kth\_excluding(H(x),k,i)\Big)\end{aligned}$$  

通过这个noise，我们把“第 $i$ 个专家是否处理这个输入”的离散值，变成“第 $i$ 个专家处理这个输入的概率”这样一个平滑的估计，$P(x,i)$ 就表示这个概率。这个概率可以简化写成  

$$\begin{aligned}P(x,i)&=\Phi\Big(\frac{(x\cdot W_g)_i-kth\_excluding(H(x),k,i)}{Softplus((x\cdot W_{noise})_i)}\Big)\end{aligned}$$  

其中 $\Phi$ 是标准正态分布的CDF。  

接下来就可以把第 $i$ 个expert的负载定义为  

$$\begin{aligned}Load(X)_i=\sum_{x\in X}P(x,i)\end{aligned}$$  

有了每个expert的负载衡量，就可以和前面第一个负载均衡损失一样，计算新的负载均衡损失了  

$$L_{load}(X)=w_{load}\cdot CV(Load(X))^2$$  

$w_{load}$ 是手动设置的超参，实验的推荐值是0.1。  

相比前面的 $L_{importance}(X)$，$Load(X)$ 是对负载是否均衡更细粒度的评估。  

论文中提到一个细节，在刚开始训练的时候，希望模型分配的expert尽量均衡，因此把 $W_g$ 和  $W_{noise}$ 都设为0，这样相当于没有信号，也没有噪音。  

最终使用负载均衡之后的效果如下  

{% asset_img rnn_moe_load_function.png 负载平衡效果 %}  

使用这两个负载均衡损失之后，能达到接近完全平均分配的效果。  

## 实验  

1. 解决工程问题  

针对前面提出的一些工程问题，论文给出一些方案  

（1）batch size减小  

由于稀疏激活的原因，每个expert的batch size会变小。假设每次在n个expert中选择k个，模型训练的batch size为b，那么每个expert的batch size就是kb/n。论文通过以下这几种方法来提升每个expert的batch size：  
- 混合使用数据并行和模型并行。本来在使用数据并行的情况下，每个模型副本是异步处理各自的数据的。而这里做了优化，各个副本的batch是同步处理的，这样就可以把多个模型副本的batch组合起来。对于非MoE部分的参数，依然使用标准的数据并行机制；而对于每个expert，则在整个集群中只保留一个副本。如果模型分布在d个设备上，那每个expert就能得到一个kbd/n的batch size。
- 对于LSTM模型，在时间步上展开，就能把batch size提升相应的倍数。

（2）集群通讯问题  

另一个挑战就是平衡集群计算量和通讯量的关系。  

对于每个expert来说，主要的通讯就是input和output的传输。而每个专家的主要计算量就是两个全连接层，大小分别为[input_size, hidden_size]和[hidden_size, output_size]。对于GPU来说，计算速度可能是通讯速度的1000倍，那我们就需要把计算量设计得足够大。最简单的做法就是把hidden_size提高，使得每个expert的内部计算量比通讯量大1000倍，以保证通讯不会成为训练的瓶颈。  

2. 模型容量 & 参数效率  

为了验证模型容量提升带来的收益，以及MoE模型的参数效率（即和dense模型同样推理计算量下能达到的效果），训练了包含4/32/256个expert的flat MoE模型，和包含256/1024/4096个expert的hierarchical MoE模型。每个expert大约是1M参数量，对于所有flat模型都是激活4个expert，而对于hierarchical MoE是每层gating激活2个。  

效果如下图。左边的图显示，随着模型容量提升，测试的ppl有明显下降。右边的图将相近模型容量的dense模型和MoE模型的效果放在一起对比，可以看到MoE模型在相同模型容量下，效果更好

{% asset_img rnn_moe_perf.png 效果 %}  

3. 更大的模型  

前面几个模型训练用的数据量不是很大，模型最大也只有4B左右，训练不久就出现diminishing returns。  

为了验证更大数据集 + 更大模型的收益，在100B token的语料上，分别训了包含32, 256, 1024，4096, 16384, 65536, 和131072个expert的MoE模型，最大的模型达到了137B的参数量。  

各个模型对比如下表。整体来看，增加数据和模型容量，是可以继续获得提升的。  

{% asset_img rnn_moe_137b.png 137模型效果 %}  

从这里还可以看出，在专家数量不太多时，提升专家数量效果有提升，但是收益会慢慢减小，甚至会出现专家数量太多，效果反而下降的情况。  

4. Expert Specialization  

按照MoE的设计思路，不同的专家应该学习到不同的子任务，但是实际上是否是这样呢？

论文里把模型中不同的专家分配到token拿出看，发现确实有比较强的specialization效果，不同的专家处理不同的内容，如下所示  

{% asset_img rnn_moe_specilized.png RNN MoE 专门化 %}  

# GShard

1. 简介

2018年，随着Bert的发布，transformer结构彻底火了起来。2020年6月，Google发布《GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding》，把MoE用到了encoder-decoder结构的transformer模型上。MoE开始变成我们现在熟悉的样子了。  

GShard这个工作做了很多的实验，训了很多规模巨大的MoE模型，最大的达到了600B。训练的一系列模型的参数如下表  

{% asset_img gshard_moe_family.png GShard MoE family %}  

在expert数量的设计上，延续上面LSMT MoE工作的思路 -- expert越多，效果越好。（站在24年这个时间节点来看，太多的expert未必适合；但是也不能说这个思路一定错误，毕竟事物的发展是螺旋式的，就像ChatGPT出来之前大多数人都在魔改各种Bert，而GPT已经坐了几年冷板凳了。）  

GShard论文中很大的篇幅在介绍工程实现和优化，这也是MoE模型训练最大的痛点。关于工程框架的内容比较硬核，因此这里不会展开讲太多，而是关注在模型算法层面上。  

2. 模型设计

先来看下模型设计。  

Google在那段时间走的是encoder-decoder transfomer的技术路线，因此GShard也是基于encoder-decoder transfomer的模型结构。  

GShard的模型设计是，在encoder和decoder中，每两层把其中一个FFN层替换成MoE层。对于总共有N层的模型，则有N/2个MoE层，如下图  

{% asset_img gshard_model.png GShard模型结构 %}  

每层会选择最多top-2 expert来激活。为什么是最多，后面解释。  

GShard在上面这篇LSTM MoE论文的基础上，改进了gating function和auxiliary loss function。  

从公式来看，MoE层的具体计算如下

$$\begin{aligned}
\mathcal{G}_{s,E}& =\mathrm{GATE}(x_s)  \\
\mathrm{FFN}_e(x_s)& =wo_e\cdot\text{ReLU}(wi_e\cdot x_s)  \\
y_{s}& =\sum_{e=1}^E\mathcal{G}_{s,e}\cdot\mathrm{FFN}_e(x_s) 
\end{aligned}$$

其中 $x_s$ 是MoE的输入token，$w_i$ 和 $w_o$ 分别是输入输出的线性变换矩阵。向量$\mathcal{G}_{s}$ 就是gating function的输出。

GShard在gating function的设计上提出了两个要求：（1）负载均衡（2）高效扩展。  

负载均衡和前面讲的一样，很好理解。而为什么要高效扩展，因为如果要对N个token分别进行E个expert的分配，在N能达到百万甚至千万级别，而E也有几百上千的情况下，就需要一个高效的分布式实现，以免其他计算资源等待gating function。  

为了满足这些要求，gating function提出了以下机制  

（1）专家容量 expert capacity  

为了确保负载平衡，我们不希望有少量expert需要处理很多token，因此强制规定了每一个expert所负责处理的token数量有一个最大值，这个最大值就叫专家容量，在这里设置为2N/E，相当于平均分配的量。  

这个expert capacity通过GATE(·)给每个expert维护一个计数器 $c_e$ 来监控。如果一个token所选的两个专家当前处理量都已经超过设定的专家容量，那么这个token就不会被当前层的任何expert处理，而是直接通过残差链接透传到下一层。  

（2）分组分配 Local group dispatching  

给所有输入token分成了G组，不同的组并行处理，每个组相应地也把组内专家容量变成2N/EG。  

这样做相当于在前向推理时，把大的batch拆分成小的batch，每个小的batch就是一个group。这样做的好处是通讯的时候（特别是all2all）只需要在每个group内进行就可以了，减少了通讯量。  

而进行反向计算的时候这些group可以合起来一起用，相当于进行了gradient accumulation。  

（3）辅助损失函数 Auxiliary loss  

光设置专家容量并不能使得gating负载均衡，而且会导致大量溢出。参考前面LSTM MoE的工作，这里也定义了一个辅助损失函数，来帮助负载均衡。辅助损失函数设计如下  

$$\ell_{aux}=\frac1E\sum_{e=1}^E\frac{c_e}S\cdot m_e$$  

$S$ 是token数，$E$ 是专家数，$c_e$ 是分配给第 $e$ 个专家的token数，$m_e$ 是第 $e$ 个expert在 $S$ 个token中获得的平均权重。  

思路是，本来是要算 $\frac{c_e}S$ 的平方的，但这是离散值不可导，因此把平方中的一个 $\frac{c_e}S$ 换成了 $m_e$ ， $m_e$ 是第 $e$ 个expert在 $S$ 个token中获得的平均权重。在平均分配的情况下，这个loss达到最小。  

相比前面的负载均衡损失，这个loss的设计就简单许多。  

gating的整个算法如下  

{% asset_img gshard_algo_1.png GShard gating 算法 %}  

（4）随机路由 Random routing  

前面提到，每层会选择最多top-2 expert来激活，就是因为有随机路由的机制。直观来说，就是认为如果top-1专家的权重很高，而第二个专家的权重如果较小，那很有可能只用第一个专家就足够解决问题了。  

随机路由的机制是top-1的专家永远会被激活，而第二个专家如果权重很小，就认为它可以被忽略。具体来说，会以与第二个专家的权重g2成比例的概率激活第二个专家。  

3. 效果  

最后看一下模型在翻译任务上的效果

{% asset_img gshard_perf.png GShard效果 %}  

# Switch Transformer

2022年4月，距离ChatGPT发布还有半年，Google发布了《Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity》（实际上2021年Google就提出Switch Transformer了）。  

Switch Transformer和GShard一样，是encoder-decoder结构，基于T5开发的，具有1.6T的参数，2048个expert。  

和前面的很多工作一样，Switch Transformer有一个出发点，那就是参数量越大，模型效果越好，并且可以通过稀疏激活来减少总计算量。  

但是相比其他工作，Switch Transformer给出了一个更为具体的描述，那就是模型参数量可以是一个独立于总计算量的，单独的缩放轴。也就是说，在改变参数量的同时，（几乎）不改变训练和推理的计算量，就可以带来效果的提升。因此Switch Transformer关注在“同样的FLOPS/token的计算量”下，如何扩大模型，提升效果。  

Switch Transformer所做的工作还是比较多的，包括：  

（1）模型结构简化：简化了Transformer上的MoE架构，提出Switch Transformer架构。  

（2）MoE to dense：把训出来的效果较好的MoE模型蒸馏到dense模型，在压缩MoE模型99%的参数的情况下，效果还是比直接训练dense模型好。  

（3）训练和微调技术：  
- 首次使用bf16成功训练MoE模型  
- 更适合MoE结构的模型初始化  
- 增加的专家正则化，改善了稀疏模型的微调和多任务训练  

（4）训练框架：结合数据、模型和专家并行性，训练了超过1T参数的MoE模型。  

（5）多语言：在多语言数据集上训练，发现101种语言效果普遍有提升。  

（6）训练效率：在同样的FLOPS/token的计算量下，Switch Transformer模型收敛速度有数倍的提升。  

## 模型设计  

Switch Transformer的模型结构如下图，类似GShard，把transformer每层的FFN替换成MoE层  

{% asset_img switch_transformer_structure.png Switch Transformer 模型结构 %}  

Switch Transformer一个重要的改进点就是简化了gating function的做法（Switch Transformer论文里叫routing）。  

之前的工作大多探索了选择k个expert的做法，而Switch Transformer则直接把gating简化为只选择1个expert，即k=1。这样的MoE层叫做Switch layer。  

这样简化之后，routing的实现更简单，router的计算量小了，也减少了通讯量。  

## 负载均衡  

同GShard一样，Switch Transformer规定了一个专家容量expert capacity，来限制每个expert在一个batch里能处理的最大token数。  

如果一个token被分配到了一个已经满载的expert，就会出现overflow，那这个token在本层就不会被处理，而是直接通过残差链接，透传给下一层。这点也同GShard一样。  

在Switch Transformer，专家容量通过容量系数capacity factor来控制。  

$$\text{expert capacity}=\left(\frac{\text{tokens per batch}}{\text{number of experts}}\right)\times\text{capacity factor}.$$  

一个大的capacity factor意味着每个expert能够处理更多的token，从而减少overflow情况的发生，但是计算量和通讯量的压力也会增大，所以这是一个需要权衡的参数。

下图给出了一个不同capacity factor下的例子  

{% asset_img switch_transformer_diff_expert_capacity.png 不同的expert capacity %}  

那么如何设定expert capacity呢？

如果capacity factor为1的话，只有在完全平均分配的时候，才不会出现overflow的情况。而太大的capacity factor则可能造成算力和存储的浪费。  

首先，实验中发现expert的数量和overflow的数量之间没有什么关系，所以在所有实验中，所有MoE和Switch Transformer模型都用128个专家。  

不同的capacity factor对模型影响如下表。可以看到，大的容量系数相对来说能取得更好的效果（因为更少的overflow），但是相应地，大容量系数的模型处理速度就会慢一些。  

{% asset_img switch_transformer_capacity_effect.png expert capacity的效果 %}  

经验上，低的token丢弃率对模型的scaling很重要，想要训练超大规模的模型，就要解决这个问题。而通过负载均衡损失就可以确保良好的平衡，使得在使用较小容量系数的情况下，overflow尽量少，从而兼顾效果和计算速度。  

关键问题来到负载均衡损失怎么设计。  

给定 $N$ 个expert，和包含 $T$ 个token的batch $\mathcal{B}$，负载均衡损失是这么计算的 

$$\begin{aligned}\text{loss}&=\alpha\cdot N\cdot\sum_{i=1}^Nf_i\cdot P_i\end{aligned}$$  

$f_{i}$ 表示被分配到第 $i$ 个expert的token数，这个不可导  

$$\begin{aligned}f_i=\frac{1}{T}\sum_{x\in\mathcal{B}}\mathbb{1}\{\text{argmax }p(x)=i\}\end{aligned}$$  

$P_i$ 表示整个batch每个token分配给第$i$ 个expert的概率的总和，这个可导  

$$\begin{aligned}P_i=\frac{1}{T}\sum_{x\in\mathcal{B}}p_i(x).\end{aligned}$$  

这个损失的设计其实和GShard中的也是一样的。  

在完美平均分配的情况下，$f$ 和 $P$ 这两个向量都是 $1/N$，这个时候负载均衡损失是最小的。  

$\alpha$ 扫描了1e-5到1e-1，发现设为1e-2，已经足够大保持负载平衡，同时不过分影响模型收敛。  

观察到 $\sum_{i=1}^N(f_i\cdot P_i)=\sum_{i=1}^N(\frac1N\cdot\frac1N)=\frac1N$，所以负载均衡loss还乘了个 $N$，这样可以保持无论使用多少个expert，在平均分配的情况下，loss都能保持相同的常数。  

## 实验

1. 一些训练的trick  

（1）选择性地使用bf16  

半精度训练会带来一些训练的不稳定。因此选择性地使用bf16，具体来说，routing function内部使用单精度，其他部分使用半精度，这样既不影响通讯，也能提高效果。  

为什么选择在routing提高精度？因为softmax对误差特别敏感，exponential计算会极大放大输入中的rounding error，因此高精度对routing很重要。  

（2）较小的参数初始化  

从截断正态分布中抽取元素来初始化的模型参数，平均值 $\mu=0$，标准差$\sigma=\sqrt{s}/n$，其中s是超参，n是权重张量中的输入单元数量（e.g. fan-in）。  

论文建议将默认的Transformer初始化尺度s=1.0减少10倍。这个方案在实验中既提高了质量又降低了训练不稳定性的可能性。初始化实验对比如下表  

{% asset_img switch_transformer_init.png 初始化对比 %}  

（3）增大dropout  

由于Switch Transformer参数量很大，在微调的时候更容易过拟合，因此一个简单的方法就是增大dropout，效果如下  

{% asset_img switch_transformer_dropout.png dropout效果 %}  

可以看到大的dropout有效果，并且dense层保持0.1，只有expert层增大dropout效果更好。  

2. scaling  

对Switch Transformer结构预训练的scaling做了一些实验。  

（1）Step-Basis  

首先是验证在固定训练step的条件下，增大expert数量带来的提升，如下图所示。  

左边是不同规模的模型在相同step下收敛的结果，可以看到在保持相同计算量的条件下，只通过增大专家数量来提升规模，就有明显的收益。右边则展示训练过程中，不同规模的模型在各个step下的效果。  

{% asset_img switch_transformer_scaling_step.png step scaling %}  

（2）Time-Basis  

虽然Switch Transformer可以保持计算量不变的情况下提升模型规模，但是专家数量的增多会带来额外的通讯成本，所以即使训练的step数相同，实际的训练时间也不同。因此这里要回答的问题是，给定一个固定的训练时长，Switch Transformer是否相比dense模型仍有收益。  

答案是肯定的。下图展示以训练时长为横轴，Switch Transformer和dense模型的效果对比。Switch Transformer收敛到dense模型最终效果的时间只有dense模型的1/7。  

{% asset_img switch_transformer_scaling_time.png time scaling %}  

（3）和更大的dense模型对比

前面Switch Transformer和dense模型的比较，是基于相同计算量的前提。那么Switch Transformer是否具备超越更大规模dense模型的能力？  

下图在Step-Basis和Time-Basis对比了64个专家的Switch Transformer和T5-Large。无论是相同step还是相同时间下，Switch Transformer都有明显优势。  

{% asset_img switch_transformer_scaling_dense.png dense对比 %}  

3. SFT效果对比  

在GLUE和SuperGLUE等下游任务上微调，和dense模型对比。  

对于各个模型，每两百步进行一次eval，选最好的效果，尽量保证公平。结果如下表，大部分任务都有明显的提升。  

{% asset_img switch_transformer_sft_result.png sft对比 %}  

4. 模型蒸馏  

虽然Switch Transformer在相同计算量下效果更好，但是部署几百B甚至T级别的模型，还是不太方便，因此考虑把稀疏模型蒸馏到dense模型上来进行推理。  

论文中给出了几个蒸馏的技巧：  
- 初始化的时候，把Switch Transformer模型中的非稀疏部分用于初始化dense模型  
- 蒸馏所用的label，25%来自教师模型，75%来自ground truth，加权求和  

预训练模型的蒸馏效果如下，相比无蒸馏训练的dense模型，把同样计算量的稀疏模型蒸馏到dense模型，dense模型大约能获得Switch Transformer提升部分30%的增益。  

{% asset_img switch_transformer_distill.png 蒸馏 %}  

更进一步，用不同规模的稀疏模型下进行蒸馏，结果如下表，可以实现高达99%的压缩率  

{% asset_img switch_transformer_distill_diff_model.png 蒸馏 %}  

除了预训练模型，微调模型也可以蒸馏，效果如下，在SuperGLUE也有一定的提升  

{% asset_img switch_transformer_distill_sft.png sft蒸馏 %}  

# GLaM

1. 简介

2021年12月Google发表了《GLaM: Efficient Scaling of Language Models with Mixture-of-Experts》，训练出最大参数量为1.2T，每层包含64个专家，每个token激活参数量为96.6B的MoE模型。  

相比Switch Transformer，GLaM的训练数据量要大得多，达到了1.6T token。  

下表是论文中给出的，当时一些大规模模型的对比  

{% asset_img glam_related_model.png glam和相关模型 %}  

虽然模型总参数量比GPT-3（175B）大很多，但是训练成本却比GPT-3低很多，推理速度也更快，而且在多个NLP任务上的效果都超越了GPT-3，如下所示。  

{% asset_img glam_compare_gpt3.png glam和gpt3对比 %}  

{% asset_img glam_compare_gpt3_2.png glam和gpt3对比 %}  

2. 模型设计

模型设计上，和Switch Transformer一样，每两层把一个FFN替换成MoE层。但是和Switch Transformer不同，GLaM用回了每次激活两个expert的方案，模型结构如下图。  

{% asset_img glam_model.png glam模型 %}  

除此之外，模型在结构上海做了一些其他改动：  

（1）位置编码  

使用XLNET的相对位置编码。  

（2）激活函数

> In the non-MoE Transformer feed-forward sub-layers, we replace the first linear projection and the activation function with the Gated Linear Unit，which computes the component-wise product of two linear transformation of the input, followed by a Gaussian Error Linear Unit.  

3. 实验

训练中的一些trick：  

（1）参考《Lingvo: a modular and scalable framework for sequence-to-sequence modeling》，在梯度出现NaN或者Inf的时候就跳过那一步更新。  

（2）如果在BP更新的时候遇到NaN或者Inf，则重新加载更早的checkpoint并跳过有问题的数据来避免NaN或者Inf。  

论文训了一系列模型来探索MoE，这些模型的设置如下表  

{% asset_img glam_family.png glam模型系列 %}  

GLaM和dense模型的评测结果如下  

{% asset_img glam_perf.png glam模型效果 %}  

可以看到GLaM MoE的有效参数效率一致高于dense模型。  

# ST-MoE  

2022年2月，Google发表了《ST-MOE: DESIGNING STABLE AND TRANSFERABLE SPARSE EXPERT MODELS》。ST-MoE可以说不仅仅是一个MoE工作，对于模型结构、工程实现、训练策略等都做了很多分析，可以说是MoE的必读论文。  

ST-MoE最大模型包含269B总参数量，和与32B dense模型相当的激活计算量。论文中把模型称为称为Stable Transferable Mixture-of-Experts，或者ST-MoE-32B。  

在MoE层的使用上，ST-MoE比Switch Transformer更“节省”一点，每四层才替换1个MoE层。  

论文中主要训了两个规模的ST-MoE模型，分别有4B和269B的总参数量。ST-MoE以及其他用于对比的模型参数如下表  

{% asset_img st_moe_models.png ST-MoE模型及对比模型的参数 %}  

## 稳定性与效果分析  

论文通过对乘性操作、噪音和裁剪这几个内容进行探索，来指导模型的设计。  

1. 乘性操作对模型稳定性和效果的影响  

论文首先研究了乘性操作对模型的训练稳定性和最终效果的影响。  

之前已经有一些工作表明更多的乘法对模型效果有收益。  

> Some architectural improvements involve more multiplications than additions or do not sum many items at once

（1）GELU Gated Linear Units (GEGLU)  

第一个例子是关于激活函数的。GLU是一个对两个输入向量进行component-wise相乘的操作，之后被扩展成GELU-Linear FFN变体，用于替换transformer中的ReLU FFN变体，其计算如下  

$$\begin{aligned}FFN_{GEGLU}(x,W,V,b,c)=GELU(xW+b)\odot(xV+c)\end{aligned}$$  

这样在一些其他工作里已经被证明了对模型效果有提升。  

（2）RMSNorm  

第二个例子是RMSNorm中的缩放参数，也就是下面公式的 $g$。  

$$y_i=\frac{x_i}{\sqrt{\frac1d\sum_{i=1}^dx_i^2}}\cdot g_i$$  

ST-MoE针对GEGLU和RMSNorm这两个乘性操作，做了实验，结果如下表。  

{% asset_img st_moe_remove_multiplications.png 移除乘法操作的影响 %}  

发现移除乘性操作可以使模型稳定性更好（训练中发散的情况减少），但是最终效果变差了。  

（3）增加dense层  

ST-MoE还验证了在expert层增加更多dense层的效果。结果发现增加更多的乘法交互（增加dense层），可以在带来效果收益的同时，基本不影响推理速度，如下表所示。

{% asset_img st_moe_more_dense_layer.png 更多的dense层 %}  

（4）增加一个bias

在FFN层的第一个矩阵乘法后面增加一个可学习的bias B，分别通过加法和乘法加入  

$$\text{FFN}_{\text{GEGLU}}+\text{Add Bias}(x)=[(\text{GELU}(xW_{11})\odot xW_{12})+B]W_2$$  

$$\mathrm{FFN}_{\mathrm{GEGLU}}+\mathrm{Mult~Bias}(x)=[(\mathrm{GELU}(xW_{11})\odot xW_{12})\odot B]W_2$$  

乘法的收敛速度更快，效果也更好。  

上面这些实验显示，后续在模型效果的探索方向可以往多使用乘性操作去考虑。  

2. noise对模型稳定性和效果的影响  

接下来ST-MoE探索了“噪音可以提升模型稳定性”的假设。  

通过input-jitter，给router的输入logits乘以一个在[1e-2, 1e2]之间的均匀随机变量来添加噪音。  

{% asset_img st_moe_more_add_noise.png 增加noise %}  

结果是增加noise之后，有助于让模型的收敛更加稳定，但是对模型最终效果有负面影响。  

这里论文还提到，小模型上的结果不一定能直接推广到更大的模型上，比如在小模型上稳定的配置，在大模型就可能就不稳定了。因此还是需要在大模型上也进行充分实验。  

3. 限制激活值和梯度值对模型稳定性和效果的影响  

对activation和gradient进行限制是目前广泛应用的提升模型训练稳定性的手段。在反向传播过程中，通过裁剪梯度的范数来缓解梯度爆炸，就是一种常用的限制手段。  

但是在ST-MoE训练269B的大规模模型时，发现裁剪会使得模型收敛的效果很差。  

为了解决这个问题，ST-MoE在训练中引入了router z-loss，形式如下。  

$$L_z(x)=\frac{1}{B}\sum_{i=1}^B\left(\log\sum_{j=1}^Ne^{x_j^{(i)}}\right)^2$$  

$B$ 是token的数量，$N$ 是专家数，$x\in\mathcal{R}^{B\times N}$ 是router的输入。  

z-loss会对进入router的较大的logits值进行惩罚，以达到尽量减少进入指数函数的较大误差的目的。什么意思呢？后面来解释，先看下使用z-loss的效果。  

{% asset_img st_moe_z_loss_result.png z-loss效果 %}  

ST-MoE认为，在模型训练过程中，由于精度不足或者其他问题，会产生很大的值，从而引入误差。而对梯度进行裁剪是在误差发生之后，并且裁剪本身也造成了数据的不连续性，某种程度上，裁剪本身也是一种误差。相反地，z-loss自然地鼓励模型产生较小的对数值，因此可以更精确地建模。  

z-loss乘以一个权重超参 $c_z$ 加入到模型训练的总损失中，如下式所示。  

$$L_{tot}=L_{CE}+c_BL_B+c_zL_Z$$  

ST-MoE经过实验，选择了$c_z=0.001$。  

$L_B$ 是 auxiliary load balance loss负载均衡损失，ST-MoE这里使用了和GShard/Switch Transformer所用的相同的损失计算，这里回顾一下：  

$$\begin{aligned}\text{loss}&=\alpha\cdot N\cdot\sum_{i=1}^Nf_i\cdot P_i\end{aligned}$$  

$$\begin{aligned}f_i=\frac{1}{T}\sum_{x\in\mathcal{B}}\mathbb{1}\{\text{argmax }p(x)=i\}\end{aligned}$$  

$$\begin{aligned}P_i=\frac{1}{T}\sum_{x\in\mathcal{B}}p_i(x).\end{aligned}$$  

$N$ 是专家数， $\mathcal{B}$是包含 $T$ 个token的batch。$f_{i}$ 表示被分配到第 $i$ 个expert的token数，这个不可导；$P_i$ 表示整个batch每个token分配给第$i$ 个expert的概率的总和，这个可导。  

4. 数据精度对训练效率和训练效果的影响

目前大部分的大模型训练都使用混合精度训练：模型权重以float32格式存储以进行梯度更新，然后在正向和反向传播的矩阵乘法中转换为bfloat16；此外，所有激活值都以bfloat16存储和操作，而allreduce通信可以在bfloat16或float32数值精度中进行。  

对于ST-MoE-32B的训练，allreduce的数值使用半精度可以加速训练，然而这也会使训练变得不稳定，因此ST-MoE保持allreduce的数值精度为float32。  

bfloat16和float32在不同范围的舍入误差如下表所示  

{% asset_img st_moe_round_error.png bf16精度损失 %}  

可以看到，表达的数值越大，舍入误差越大。而z-loss限制了数值大小，也就将误差值限制在比较小的范围。  

MoE模型天生对舍入误差敏感，因为它们由于router的使用而有更多的指数函数，而指数函数会将小的输入误差放大很多，这就加剧舍入误差所导致的训练不稳定。  

另外，ST-MoE有一个策略：只有当排第二的专家的权重大于等于第一的专家的1/5时，token才会被路由到其第二位专家，否则第二个专家就会被忽略。  

因此虽然舍入误差不会改变softmax运算中各个概率的排序，但它确实会影响MoE中第二个专家的激活。  

## 模型设计

dense模型的设计有scaling law进行指导，但是MoE模型的设计比dense模型多出几个要考虑的点： 
 
（1）使用多少个expert  

（2）怎么routing  

（3）专家容量系数怎么定  

（4）硬件的影响  

（这里提到MoE模型的scaling law工作：《Unified scaling laws for routed language models》，可以了解一下）  

1. 使用多少个expert  

ST-MoE认为，从以往的经验来看，在总专家数量较少的情况下（如8/16/32），提升专家数量，能有收益。但是在特别稀疏的情况下（如激活专家数量<1%），或者总专家数较大（比如>256）之后，提升专家数量收益就很小了。  

从另一个角度来看，如果一个计算核心使用>1个专家，那么就会出现比较大的加载参数张量的成本，因此建议每个计算核心使用<=1个专家。  

2. routing和capacity factor  

论文做了一系列实验来探索capacity factor的选择，如下表所示  

{% asset_img st_moe_capacity_factor.png capacity factor %}  

从这些实验中得到几个结论：  

（1）训练和推理的capacity factor增大都会有收益  

（2）如果硬件资源足够，推理的capacity facotr可以设得比训练的时候大，会有进一步提升  
 
（3）激活的expert数量提升会有收益，但是收益随着capacity factor提升而越来越小  

当然，选择capacity factor还要看硬件的特性，如果通讯很快，可以适当增大capacity factor，否则就不能选择太大的。  

下表展示了不同capacity factor对推理速度的影响  

{% asset_img st_moe_capacity_factor_speed.png 不同capacity factor推理速度 %}  

## 实验  

1. ST-MoE效果  

ST-MoE-32B在下游任务上和以往最佳结果对比如下表，ST-MoE-32B刷新了超过一半任务的最佳效果  

{% asset_img st_moe_perf.png 不同capacity ST-MoE-32B效果 %}  

2. Expert Specialization  

论文还对各个专家的专业化进行了追踪，发现decoder中几乎没有专业化的迹象，各种类型的token近乎随机分配给不同的专家。而在encoder中则表现出了高度专业化的特征，如下表  

{% asset_img st_moe_encoder_specialization.png encoder专业化 %}  

此外，还发现在多语言的模型的encoder中，专业化的情况并不想原先预想那样，按不同语言划分，而是每个专家都会处理一种语言的一部分token，如下表  

{% asset_img st_moe_multiling_specialization.png 多语言专业化 %}  

# DeepseekMoE

2024年1月，幻方量化开源了DeepseekMoE，是国内首个开源的MoE大模型。幻方还发布了论文《DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models》，给出了一些DeepSeekMoE的细节内容，颇为实在了。  

DeepSeekMoE在其他MoE工作的基础上，进一步给出了2个模型设计的主要思路：  

（1）对expert的粒度进行细分，以提供更多样的expert激活组合；  

（2）对expert的类型进行区分，从所有expert中保留一部分作为shared expert共享专家，这部分专家对所有输入都保持激活。  

这样的做法可以帮助每个expert达到更高程度的专业化(specialization)的水平，更好地学习不同的专业知识。  

DeepSeekMoE先在2B的较小MoE模型上进行了充分的实验，然后把方案应用到16B参数的MoE模型上，并获得了较好的效果。其中DeepSeekMoE-16B不需要量化就可以在40GB显存的设备上运行。  

DeepSeekMoE-2B模型具有和稠密2B模型相当的性能，而DeepSeekMoE-16B则具有和7B稠密模型相当的性能，且计算量仅为稠密模型的40%。  

DeepSeekMoE-16B的参数效率相比稠密模型有明显的优势，如下图所示  

{% asset_img ds_moe_perf.png deepseek moe %}  

并且DeepSeekMoE-2B和16B模型都开源了。  

在前面实验的基础上，幻方还训练了DeepSeekMoE-145B的超大MoE模型，具有和稠密的DeepSeek-67B模型相当的表现，但计算量更小。这个后续也有机会放出来。  

## 模型设计  

MoE，mixture of expert，顾名思义，一个最初始的motivation就是让不同expert学习不同的内容，然后再混合起来。  

比如最上面提到的1991年的工作里，就是让不同的expert学习不同的元音特征，以此提升特征提取的准确率。  

但是当前大部分的MoE架构都会遇到“knowledge hybridity”和“knowledge redundancy”的问题，即知识的杂糅和冗余：  

（1）知识冗余  

有些基础的常识在不同的领域都需要用到，每个expert就都会学一点，这样这些常识就被多个expert重复学习了。  

（2）知识杂糅  

在expert数量不够多的情况下，一个expert就可能要负责学习多个领域的内容。以学习高中知识为例，在只有两个expert的时候，只能一个expert学习理科知识，另一个学习文科知识；当我们有8个expert的时候，不同expert就可以分别学习语文、英语、历史、地理、物理、生物、化学、数学知识。显然后者所学知识的专业化程度更高。  

知识的杂糅和冗余阻碍了专家专业化(expert specialization)的程度，也就阻碍了模型达到MoE结构理论上限性能。  

我们期望每个expert能够学习到non-overlap & foucusd knowledge的知识。  

针对上面的问题，DeepSeekMoE的架构设计有2个主要策略：  

（1）Fine-Grained Expert Segmentation  

参数总量不变的情况下，将expert分成更细的粒度（每个expert更小）。这样可以带来更灵活的激活组合，让每个expert可以有更强的specialization。比如原本是16个expert选择激活2个，那么总的组合数是120种；如果把每个expert缩小为原来的1/4，那在总参数量和激活数量不变的情况下，是64个expert选择激活8个，那么总的排列组合数就是 $\binom{64}8=4,426,165,368$ ，排列组合数比原来多了很多。   

（2）Shared Expert Isolation  

把部分expert分离出来，保持永远激活。我们期望这部分专家能够学到在多个领域间都通用的common knowledge。这样的策略同样可以使得其他expert能够提高专业化的程度，并且减少不同expert间的知识冗余。还是以学习高中知识为例，数学、物理和化学都需要算术能力，如果让学这三个领域的expert都学习算术技能，就会有冗余；我们可以把通用算术的技能剥离出来，由一个助手专门负责算术任务，相当于给他们发了一个计算器，这样学习数学、物理和化学的expert就能把更多的精力放在专业知识上，也就能达到更好的专业化效果。  

下图展示了在传统MoE结构上增加Fine-Grained Expert Segmentation和Shared Expert Isolation策略的设计  

{% asset_img ds_moe_structure.png deepseek moe 结构 %}  

（expert isolation的思路最早可以追溯到2022年1月发表的《DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training to Power Next-Generation AI Scale》，这里就不展开了。）  

假设传统的MoE模型每层的expert数量为 $N$，激活expert数为 $K$，DeepSeekMoE使用的细粒度expert大小为原来的 $1/m$，那DeepSeekMoE每层就有 $mN$ 个expert，激活的expert数量为 $mK$个。假设 $T$ 为输入长度，$L$ 为模型层数，$e_i^l$ 表示第 $i$ 个expert，DeepSeekMoE可以公式化为以下表示（忽略了layernorm）  

$$\mathbf{u}_{1:T}^l=\text{Self-Att}\left(\mathbf{h}_{1:T}^{l-1}\right)+\mathbf{h}_{1:T}^{l-1}$$  

$$\mathbf{h}_t^l=\sum_{i=1}^{mN}\left(g_{i,t}\text{ FFN}_i\left(\mathbf{u}_t^l\right)\right)+\mathbf{u}_t^l$$  

$$g_{i,t}=\begin{cases}s_{i,t},&s_{i,t}\in\text{Topk}(\{s_{j,t}|1\leqslant j\leqslant mN\},mK)\\0,&\text{otherwise,}\end{cases}$$  

$$s_{i,t}=\mathrm{Softmax}_i\left({\mathbf{u}_t^l}^T\mathbf{e}_i^l\right)$$  

## 负载均衡

如之前工作反复提及的，如果任由MoE模型自主学习gating，可能会遇到两个问题  

（1）routing collapse：专家分配的不均衡，也就是gating倾向于总是选择特定的少量expert，并且这种情况还会自我增强。  

（2）计算效率问题：多设备间，不平衡的负载可能会成为计算效率的瓶颈。  

针对routing collapse的问题，DeepSeekMoE引入一个expert-level balance loss，如下所示

$$\begin{aligned}
\mathcal{L}_{\mathrm{ExpBal}}& =\alpha_1\sum_{i=1}^{N'}f_iP_i
\end{aligned}$$  

$$\begin{aligned}
f_{i}& =\frac{N^{\prime}}{K^{\prime}T}\sum_{t=1}^T\mathbb{1}(\text{Token }t\text{ selects Expert }i)
\end{aligned}$$  

$$\begin{aligned}
P_{i}& =\frac1T\sum_{t=1}^Ts_{i,t} 
\end{aligned}$$  

$\alpha_1$ 叫做expert-level balance factor，是人工设定的超参。  

而 $f_i$ 和 $P_i$ 和Switch Transformer里的设定基本一样。  

在Switch Transformer里， $f_i$ 表示分配到第 $i$ 个expert的token数量。在DeepSeekMoE这里也是一样的含义，只是多乘了一个系数 $N'/K'$ ，其中 $N'=mN-K_s$，$K'=mK-K_s$，$K_s$ 是划分出来的共享expert的数量。这个系数是个常数，可以拿到求和符号外面，这样DeepSeekMoE里的 $f_i$ 就和Switch Transformer里的完全一样了。  

$N'/K'$ 这个系数可以使得在使用不同的数量的expert时，在完美平均分配的情况下，负载均衡loss都是相同的常数。  

$P_i$ 表示所有每个token分配给第 $i$ 个expert的权重的总和，和Switch Transformer里的含义一样。  

注意这里 $f_i$ 是不可导的，$P_i$ 是可导的。  

针对多设备间负载均衡的问题，DeepSeekMoE引入一个device-level balance loss，如下所示

$$\begin{aligned}
\mathcal{L}_{\mathrm{DevBal}}& =\alpha_2\sum_{i=1}^Df_i'P_i'
\end{aligned}$$

$$\begin{aligned}
f_i^{\prime}& =\frac1{|\mathcal{E}_i|}\sum_{j\in\mathcal{E}_i}f_j
\end{aligned}$$

$$\begin{aligned}
P_{i}^{\prime}& =\sum_{j\in\mathcal{E}_i}P_j
\end{aligned}$$

$\alpha_2$ 叫做device-level balance factor，是人工设定的超参。  

$\mathcal{E}_i$ 指第 $i$ 个设备。

device-level balance loss形式上和expert-level balance loss一样，只是 $f_i$ 和 $P_i$ 对应的对象从单个expert变成单个设备了。  

当我们的目标是缓解计算瓶颈时，我们不需要强制执行expert间的均匀分配，而只需确保设备之间计算量的平衡。比如我们每层有64个expert，均匀分布在8个设备上，我们只需要每个设备处理的token数平衡即可，在设备内部即使所有token都是同一个expert处理的，依然能满足设备间负载平衡的要求。  

相比expert间严格的负载平衡，只要求设备间的平衡是更松的限制条件，这样缓解了因为过度的负载平衡而损害模型性能的问题。

## 实验

1. 小规模模型验证

为了验证以上策略的有效性，先拿100B token的语料数据在DeepSeekMoE-2B模型做实验。词表也是通过BPE在语料上训练的8k词表，后面训练更大规模模型的时候再扩大词表。

DeepSeekMoE-2B模型参数初始化方差为0.006，使用multi-head attention，前向激活参数量约0.3B，具体参数如下表

{% asset_img ds_model_param.png 模型超参 %}  

relative expert size指的是DeepSeekMoE所用的细粒度expert的大小和正常FFN层大小的比值。

训练的具体参数设置如下  

<center>

| 属性 | 数值 |
| :----: | :----: |
| optimizer | AdamW |
| adam_beta_1 | 0.9 |
| adam_beta_2 | 0.95 |
| adam_weight_decay | 0.1 |
| warmup schedule | linear |
| warmup step | 2000 |
| max lr | 1.08e-3 |
| dropout | 0 |
| sequence length | 2k |
| batch size | 2k |
| total step | 25,000 |


</center>  

其他训练细节：  
- 所有expert放在单个GPU上，没有使用device-level balance loss  
- expert-level balance factor设为0.01  
- 训练到80%的时候，学习率乘以0.316，训练到90%的时候，再乘以0.316  

使用相同的100B训练数据，训了DeepSeekMoE-2B，在包含语言模型和下游任务的benchmark上和其他4个模型做对比：dense，hash layer（也是一种moe，《Hash layers for large sparse models》），Switch Transformer，GShard。效果对比如下

{% asset_img ds_moe_comparison.png deepseek moe 效果%}  

可以得到几个结论：  
- 更大的模型参数量和稀疏的架构，使得Hash Layer和Switch Transformer和具有同样激活参数的dense模型相比，有明显的优势。  
- 同样的模型参数下，GSshard比Hash Layer和Switch Transformer有更多激活参数，效果也更好  
- 同样的模型参数和激活参数下，DeepSeekMoE效果比GShard有明显优势。  

为了进一步探索DeepSeekMoE架构带来的收益，提升了dense模型和GShard模型的激活参数，直到效果和DeepSeekMoE-2B差不多。

结果dense模型和GShard模型需要分别扩大到16倍和1.5倍的参数量，才能达到DeepSeekMoE-2B相近的效果，如下表所示  

{% asset_img ds_moe_upper_bound_2b.png deepseek moe upper bound %}  

DeepSeekMoE的优势在更大规模的情况下，依然成立。训了DeepSeekMoE-13B, 对比参数量提升至1.2和1.5倍的GShard，DeepSeekMoE-13B依然能match，具体如下表  

{% asset_img ds_moe_upper_bound_13b.png deepseek moe upper bound %}  

2. DeepSeekMoE架构消融实验

针对DeepSeekMoE架构的两个主要设计，shared expert和fine-grained expert进行消融实验。使用不同数量的共享专家和不同粒度的expert进行效果对比，结果如下图。

{% asset_img ds_moe_ablation.png deepseek moe upper bound 消融实验 %}  

（1）对比蓝色和橙色，可以看到增加共享专家带来了收益

（2）绿色和红色在橙色的基础上进一步把专家颗粒分得更细，效果进一步提升

（3）共享专家和路由专家的比例：在总共64个expert的情况下，对比了1/2/4个共享专家的情况，结果并没有显著差别，在pile上的loss分别是1.808,1.806,1.811。最终选择了共享专家和激活路由专家1:3（2+6）的比例。

3. expert specialization的分析

通过实验来验证DeepSeekMoE中expert specialization的优化。

（1）前面实验看到DeepSeekMoE-2B和1.5倍参数量的GShard模型效果相当。在这个基础上，通过禁用不同数量的top专家，而只能从次优的专家中选择进行回答。  

实验结果如下

{% asset_img ds_moe_expert_specialization.png 专家专门化 %}  

发现DeepSeekMoE损失更大，说明DeepSeekMoE每个专家的专业化程度更好，必要性更高。  

（2）另外，通过禁用DeepSeekMoE的共享专家，而额外激活一个专家，发现loss也大大提升。这个结果突出了共享专家的关键功能，并表明共享专家捕捉到了与路由专家不共享的基本且重要的知识，使得它无法被路由专家替代。

（3）只激活更少专家，也能和GShard达到相同水平，这一观察结果支持了DeepSeekMoE可以更准确和高效地获取所需知识的观点。  

{% asset_img ds_moe_less_activated_expert.png 激活更少专家 %}  

此外还从零训了一个只用1个共享专家和3个激活专家的2b模型（正常是2个共享专家+6个激活专家），也比GShard好，说明DeepSeekMoE的有效参数效率更高

{% asset_img ds_2b_less_expert.png 2B激活更少专家 %}  

1. DeepSeekMoE-16B  

DeepSeekMoE-16B模型使用了2T数据训练（和LLAMA2-7B对齐）训练，并使用了100k的词表。其他参数如下表所示  

{% asset_img ds_model_param.png 模型超参 %}  

论文中提到，除了第一层以外，其他层都使用了MoE层。  

第一层不使用MoE是因为观察到第一层的负载均衡loss在训练中收敛得特别慢。  

DeepSeekMoE-16B每层有64个专家，其中有2个作为共享专家保持永远激活，加上6个通过gating function选择激活的，每个token共使用8个专家。每个token会激活16.4B中的2.8B参数。  

这里没有把专家的dimension再减小，是因为如果专家太小，计算效率就下降得太厉害。  

训练中使用的其他设置：  
- lr = 4.2e-4  
- 训练进行到80%和90%的时候，lr都会缩小到0.316倍  
- batch size = 4.5k，训练窗口长度是4k，因此每个batch有18M token，2T数据差不多是10.6w步  
- 使用了pipeline parallelism

expert level balance loss的系数设得比较小，0.001，因为实验中发现设得再大并不能进一步优化负载平衡，反而会损害模型效果。  

DeepSeekMoE-16B和DeepSeek-7B模型的对比如下  

{% asset_img ds_16b_perf_1.png 和DeepSeek-7B对比 %}  

DeepSeekMoE-16B和LLAMA2-7B模型的对比如下  

{% asset_img ds_16b_perf_2.png 和LLAMA2-7B对比 %}  

5. DeepSeekMoE-145B  

幻方还用245B的token训练了DeepSeekMoE-145B，模型效果上达到DeepSeek-67B的同等水平  

{% asset_img ds_moe_145b.png 145b %}  

# DBRX

2024年3月27日，Databricks开源了DBRX，一个拥有有132B参数，激活参数为36B的MoE模型。

结构上，DBRX使用了RoPE、GLU、GQA，采用了fine-grained expert的设计，每层有16个专家，每个token激活其中4个。相比Mixtral和Grok-1在8个专家中激活2个，DBRX有更多的专家组合方式。  

DBRX训练的上下文长度为32k，并使用了12T文本和代码token进行训练。DBRX在3072个H100上完成预训练，加上post-training、效果评估、red-team优化，整个过程耗费3个月时间。  

DBRX整体效果超过GPT-3.5，与Gemini 1.0 Pro相当，并且具有比较强的代码能力，甚至超过了在代码上专门优化过的模型，如CodeLLaMA-70B，如下图所示。  

{% asset_img dbrx_perf.png DBRX效果 %}  

推理效率效率上，DBRX也领先于其他模型。  

{% asset_img dbrx_infer_efficiency.png 推理效率 %}  

# Qwen1.5-MoE 

2024年3月28日，阿里放出了Qwen1.5-MoE-A2.7B，以2.7B的模型参数，达到了Qwen1.5-7B模型的相近效果。  

Qwen1.5-MoE-A2.7B参考了DeepSeekMoE和DBRX的工作，采用了fine-grained expert的做法，总共有64个专家，每个token激活8个专家，其中有4个为共享专家。  

Qwen1.5-MoE-A2.7B使用Qwen-1.8B进行初始化，并在初始化阶段引入随机性，这样可以显著加快收敛速度，并得到更好的收敛结果。  

Qwen1.5-MoE-A2.7B和其他模型效果对比如下  

{% asset_img qwen1.5_moe_perf.png Qwen1.5-MoE-A2.7B效果 %}  

虽然Qwen1.5-MoE-A2.7B总参数量较大，但激活的non-embedding参数量远小于7B模型，如下表所示  

{% asset_img qwen1.5_moe_params.png Qwen1.5-MoE-A2.7B参数量 %}  

实践中，Qwen1.5-MoE-A2.7B相比于Qwen1.5-7B，训练成本降低了75%。  

推理性能上，在A100-80G用vLLM部署Qwen1.5-7B和Qwen1.5-MoE-A2.7B模型进行了性能测试。  

输入/输出token数都设置为1000，输出token数设置为1000，TPS和throughput如下  

{% asset_img qwen1.5_moe_tps.png Qwen1.5-MoE-A2.7B TPS %}  

虽然MoE模型对内存需求更大，但是由于稀疏激活以及共享专家的设计，但是在速度和吞吐量上都比dense模型更好。Qwen1.5-MoE-A2.7B与Qwen1.5-7B相比，速度提高了约1.74倍。  

# Mistral

## Mistral 8x7B

2023年12月11日，Mistral AI开源Mistral-8x7B，每个token激活8个专家中的2个。  

Mistral-8x7B支持32k推理窗口和多语言，并且代码能力较好。和LLAM2-70B以及GPT-3.5的对比如下。  

{% asset_img mistral_8_7b_perf.png Mistral 8x7B效果 %}  

Mistral-8x7B在大多数任务表现优于LLAM2-70B，且推理速度提高了6倍。  

而和激活参数量相近的LLAM2-13B比，优势更为明显  

{% asset_img mistral_8_7b_active_perf.png Mistral 8x7B同样激活参数量下效果 %}  

## Mistral 8x22B

2024年4月17日，Mistral AI开源Mistral-8x22B模型，一个总参数为141B，激活参数为39B的超大MoE模型。  

Mistral-8x22B支持多语言，并且具有较强的数学和代码能力。此外，推理窗口长度也从Mistral-8x7B的32k增加到64k。Mistral-8x22B还具备function call的能力。  

在各个维度的评测结果如下  

{% asset_img mistral_8_22b_reasoning.png Mistral 8x22B reasoning效果 %}  

{% asset_img mistral_8_22b_multiling.png Mistral 8x22B 多语言效果 %}  

{% asset_img mistral_8_22b_code.png Mistral 8x22B 代码与数学效果 %}  

# 小结  

- 现有的工作都表明，MoE模型相比dense模型具有更高的参数效率，即同样的计算量下，MoE模型普遍能有更优的效果  
- 因此MoE不仅能支持更大规模模型的训练，在较小规模模型上使用MoE架构也有很大收益  
- 但是相比dense模型，MoE模型的训练也需要考虑更多内容，包括专家数量、激活数量和专家容量的设计，负载均衡的问题，如何在多设备上的并行等，训练难度更大  
- 结构上，共享专家和细粒度专家目前被验证效果较好  
- 负载均衡上，GShard和Switch Transformer的负载均衡损失被广泛采用  
- 推理时需要对底层框架进行优化以适配MoE机制，否则难以发挥MoE的性能优势  

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
[理解Attention:从起源到MHA,MQA和GQA](http://www.linsight.cn/3dc22f96.html)  
[Yi技术报告-划重点看细节](http://www.linsight.cn/41b6a819.html)  
[transformer中normalization的二三事](http://www.linsight.cn/6a40bfa5.html)  
[从代码实现看normalization-到底做了什么](http://www.linsight.cn/b70b4a2d.html)  
[稀疏注意力计算:sliding window attention](http://www.linsight.cn/c61d17e3.html)  
[理解LLM位置编码:RoPE](http://www.linsight.cn/a051710f.html)  
[大模型算法题(1)](http://www.linsight.cn/3345028a.html)  
[大模型算法题(2)](http://www.linsight.cn/ad0bba9d.html)  

***  

# Reference  
【1】Adaptive Mixtures of Local Experts https://www.cs.toronto.edu/~hinton/absps/jjnh91.pdf  
【2】Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer https://arxiv.org/abs/1701.06538  
【3】GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding https://arxiv.org/abs/2006.16668  
【4】Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity https://arxiv.org/abs/2101.03961  
【5】GLaM: Efficient Scaling of Language Models with Mixture-of-Experts https://arxiv.org/abs/2112.06905  
【6】ST-MoE: Designing Stable and Transferable Sparse Expert Models https://arxiv.org/abs/2202.08906  
【7】DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models https://arxiv.org/abs/2401.06066  
【8】Introducing DBRX: A New State-of-the-Art Open LLM https://www.databricks.com/blog/introducing-dbrx-new-state-art-open-llm  
【9】Qwen1.5-MoE: Matching 7B Model Performance with 1/3 Activated Parameters https://qwenlm.github.io/zh/blog/qwen-moe/  

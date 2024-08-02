---
title: 适合移动设备的语言模型--MobileLLM
tags:
  - NLP
  - LLM
  - transformer
  - Meta
  - 端侧模型
categories:
  - CS
  - NLP
  - LLM
abbrlink: 5ac36d34
date: 2024-08-02 22:46:21
---

【本文已在同名 微信公众号 / 知乎 / [个人博客linsight.cn](http://www.linsight.cn/) 上线】  

***  

移动设备对端侧模型的需求日益显现，语言模型先做大后做小的趋势和之前CNN的发展历程相似。Meta提出的MobileLLM就是专门为移动设备而生，主要是125M和350M规模的模型。（让人想起七八年前的MobileNet）  

{% asset_img mobilellm.png mobilellm %}  

# 背景  

大模型在生活中使用的频率越来越高，以后可能会占据到每个人每天活动时间的5%。按这个使用量，以GPT-4为准，可能需要1亿个H100才能支持推理所需的算力，这显然是不现实的。  

另外，对于在端侧使用的模型，虽然Llama-2-7B在4-bit量化之后可以运行到手机上了，但是按0.1J/token per million parameters的能耗计算（《Eie: Efficient inference engine on compressed deep neural network》，《Towards energyproportional
datacenter memory with mobile dram》），搭载4-bit Llama-2-7B的iPhone，电池大约只能支持2小时的模型推理使用。而如果使用125M或者350M这样规模的模型，则电量足够一整天的使用。  

而运行模型所需的空间对于移动设备也是个必须要考虑的问题。按下图的典型设备的memory hierarchy，7B规模的模型会占据大部分的空间，这会明显影响其他app和系统的运行。  

{% asset_img device.png mobilellm %}  

基于这些考虑，Meta把模型目标规模定在1B以下。  

# 结构设计  

小规模模型参数有限，怎么把这些有限的参数分配给不同的部分，以获得最好的效果，是核心问题。  

在标准transformer decoder的基础上，MobileLLM在结构设计上主要有4个设计：  
- 1、deep and thin的结构设计  
- 2、embedding sharing  
- 3、使用SwiGLU  
- 4、GQA  

各项设计下，模型在zero-shot common sense reasoning任务的效果如下  

{% asset_img structure.png mobilellm %}  

更详细的数据如下表  

{% asset_img structure_ablation.png mobilellm %}  

下面看一下各项改进的分析。  

## depth vs width  

在保持总参数量基本不变的条件下，设计不同层数的模型比较效果，结果如下  

{% asset_img deep.png mobilellm %}  

更详细的数据如下表  

{% asset_img deep_ablation.png mobilellm %}  

整体上，层数更多的模型在同样总参数量下，在多个zero-shot评测上都有较好的效果。基本上30层左右的模型就能得到一个比较好的效果了，而10层以下的模型对复杂问题明显较差。  

而目前大部分1B以下的模型层数都在20层以下，这可能限制了1B以下模型的潜力发挥。  

## embedding sharing  

对于比较大的模型，embedding占总参数量的比例比较低，比如Llama-7B是3.7%，而Llama-70B更是只有0.7%。因此对于这些模型，embedding sharing并不能在参数效率上带来多少好处，反而会因为共享参数对效果有损害。  

但是对于小规模的模型就不一样了。对于125M参数的模型，embedding的参数量甚至能占到总参数量的20%（embedding dimension = 512，vocab = 32k）。因此是否使用embedding sharing在小模型这里是一个需要重新考虑的事情。  

用30层的135M模型做实验，如下表所示：共享embedding减少了16M的参数量，同时带来了0.2的平均效果损失；把由于embedding共享减少的参数部分加上模型层数上去，从30层提升到32层，模型的参数量恢复到125M（仍然比135M小），而效果则提升了0.4。  

{% asset_img emb.png mobilellm %}  

也就是说独立的embedding参数的参数效率不如增加模型层数，因此这样的参数置换是划算。  

## GQA  

每个注意力头的大小要多大？  

> The trade-off between more semantics per
head dimension and more non-linear combinations of multiple
heads is a key consideration in choosing the head size.

此外，对于大模型来说，GQA的主要作用是减少推理时所需的KV cache，而对于小模型来说，GQA也是节省参数量的一个手段。  

改变head size和kv head的数量，125M和350M上的实验结果如下表  

{% asset_img head.png mobilellm %}  

从实验结果来看，16个query head的效果是比较好的。而kv head的数量为4的时候，350M模型的效果损失是0.2，但是参数规模能减少10%左右。  

## Layer Sharing  

除了以上的结构设计以外，Meta还实验了layer sharing的效果，即对模型一个层的参数重复使用，在不增加总参数量的情况下，通过提升计算量增加模型复杂度。  

这和Albert的做法是一样的，某种程度上算是耍赖，这样的设计和不使用layer sharing的模型比较是不公平的。使用了layer sharing的系列模型和没有使用的就分开来了，单独命名为MobileLLM-LS。  

文中提出三种layer sharing的方式，如下图：  

{% asset_img share.png mobilellm %}  

三种方式的效果如下表：  

{% asset_img share_2.png mobilellm %}  

从实验结果来看是repeat-all-over的方式最好，不过最后Meta选择使用immediate block-wise共享的方式，因为这种方式有个好处：相邻层参数共享，可以减少设备SRAM加载数据的次数，从而提高推理速度。  

此外，对block-wise共享的次数消融实验结果如下：  

{% asset_img repeat.png mobilellm %}  

随着共享次数增多，收益逐渐减小，因此选择使用repeat×2的方案。  

# 实验  

## 效果对比  

125M和350M这两个主力规模的模型和其他相近规模模型效果对比如下：  

{% asset_img result.png mobilellm %}  

看起来有比较明显的提升。不过所对比的模型很多都不是最新一代的了（当然也是因为最小模型的人少了），还需要更全面测试一下。  

## scale up  

为了看这个模型设计的方案在更大一些的参数上是否也有效，Meta在几个规模稍大一些的模型进行实验：

{% asset_img model.png mobilellm %}  

结果如下：  

{% asset_img zero_shot.png mobilellm %}  

基本上在1B左右，这个设计还能保持比较好的效果。  

## 蒸馏  

Meta试了把Llama-2-7B作为教师模型，给两个小规模模型进行蒸馏。  

但是从结果上看，蒸馏在效果上没有什么收益，而收敛时间还更长了：  

{% asset_img kd.png mobilellm %}  

# 小结  

- 端侧模型需求确实越来越大，最近苹果也已经出招，其他家要跟进，推理效率是最重要的问题之一  
- 业界从scaling law的大力出奇迹，慢慢又回到精细雕花的阶段，是否预示着，下一个潮流正在酝酿？  

***  

读到这了，来一发点赞收藏关注吧~

博客：[http://www.linsight.cn/](http://www.linsight.cn/)  
知乎：[Linsight](https://www.zhihu.com/people/us4ever)  
微信公众号：Linsight  
![](/images/qrcode.jpg)  

***  

【推荐文章】  
- MoE：  
[MoE模型的前世今生](http://www.linsight.cn/44e38c1b.html)  
[DeepSeek-V2和MLA](https://www.linsight.cn/83c49df0.html)  
[昆仑万维-SkyworkMoE](https://www.linsight.cn/1d5bcd45.html)  
[成本10w刀的JetMoE](https://www.linsight.cn/f3acf042.html)  
[MoE的top-p routing](https://www.linsight.cn/224c42da.html)  
[对MoE模型的一些观察](https://www.linsight.cn/5e1d14b3.html)  
[从dense到MoE -- sparse upcycling](https://www.linsight.cn/a0824e29.html)  
[MoE路由--expert choice routing](https://www.linsight.cn/2c8bbc7.html)  
- 端侧模型：  
[苹果智能系统模型--AFM](https://www.linsight.cn/1e34e252.html)  
- 预训练：  
[Llama3.1--预训练要点一览](https://www.linsight.cn/7d7294cb.html)  
[Qwen2技术报告](https://www.linsight.cn/a8f8b641.html)  
[Yi技术报告-划重点看细节](http://www.linsight.cn/41b6a819.html)  
[MiniCPM](https://www.linsight.cn/376db710.html)  
[GLM4报告的一些技术点](https://www.linsight.cn/a5206abd.html)  
[Gemma2](https://www.linsight.cn/cf3f1f81.html)  
[苹果的OpenELM](https://www.linsight.cn/f845f3e4.html)  
[从Yuan2.0到Yuan2.0-M32](https://www.linsight.cn/3df0cd42.html)  
[bilibili的index-1.9B](https://www.linsight.cn/770b63e1.html)  
[从loss视角理解大模型涌现能力](https://www.linsight.cn/f5fb75e4.html)  
- 数据：  
[预训练数据处理--长度分解](https://www.linsight.cn/210dbccd.html)  
- 长上下文：  
[LLM长上下文的问题](http://www.linsight.cn/c4da56c0.html)  
[解锁大模型长上下文能力](http://www.linsight.cn/cc852861.html)  
[大模型推理窗口-从有限到无限大](http://www.linsight.cn/45ee1a6d.html)  
- 推理加速：  
[大模型推理加速-投机解码](http://www.linsight.cn/f5c015c.html)  
[大模型推理加速-MEDUSA](https://www.linsight.cn/7bbe2df6.html)  
- 对齐：  
[Llama3.1--post-training要点一览](https://www.linsight.cn/93328a2a.html)  
[模型平均 -- model soup](https://www.linsight.cn/bb8fcf21.html)  
[大模型偏好对齐-DPO](http://www.linsight.cn/473f2b43.html)  
[大模型偏好对齐-ODPO](http://www.linsight.cn/da871ebe.html)  
[大模型偏好对齐-simPO](http://www.linsight.cn/280fa97a.html)  
[大模型偏好对齐-IPO](http://www.linsight.cn/4fe7b810.html)  
- Transformer：  
[理解Attention:从起源到MHA,MQA和GQA](http://www.linsight.cn/3dc22f96.html)  
[LLM的重复生成和ICL](https://www.linsight.cn/7381cae3.html)  
[transformer中normalization的二三事](http://www.linsight.cn/6a40bfa5.html)  
[从代码实现看normalization-到底做了什么](http://www.linsight.cn/b70b4a2d.html)  
[稀疏注意力计算:sliding window attention](http://www.linsight.cn/c61d17e3.html)  
[理解LLM位置编码:RoPE](http://www.linsight.cn/a051710f.html)  
[RoPE的远距离衰减](https://www.linsight.cn/f0902f1a.html)  
- 大模型算法题：  
[(1)](http://www.linsight.cn/3345028a.html)、
[(2)](http://www.linsight.cn/ad0bba9d.html)、
[(3)](http://www.linsight.cn/1736008.html)、
[(4)](http://www.linsight.cn/1736008.html)、
[(5)](http://www.linsight.cn/336f2f3e.html)、
[(6)](http://www.linsight.cn/7c04944d.html)、
[(7)](https://www.linsight.cn/dd614e12.html)、
[(8)](https://www.linsight.cn/e287b9c3.html)  

# Reference  

【1】MobileLLM: Optimizing Sub-billion Parameter Language Models for On-Device Use Cases https://arxiv.org/abs/2402.14905  

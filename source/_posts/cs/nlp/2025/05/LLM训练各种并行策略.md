---
title: LLM训练各种并行策略
tags:
  - NLP
  - LLM
  - 预训练
  - 分布式
  - 3D并行
categories:
  - CS
  - NLP
  - LLM
abbrlink: 4cd8532f
date: 2025-05-22 22:47:19
---

【本文已在同名 微信公众号 / 知乎 / [个人博客linsight.cn](http://www.linsight.cn/) 上线】  

***  

从一个搞数据和训练策略的LLM算法工程师角度，把LLM训练框架中的常用并行策略(的资料)大致理一下。  

数据并行之前已经写了：[LLM训练框架：从优化器和精度讲到ZeRO](https://mp.weixin.qq.com/s/tsQ40j_jm7VSnmNI2ShL0Q)。  

这里把张量并行（TP）、流水并行（PP）和序列并行简单整理一下。  

# 张量并行（TP）  

张量并行，Tensor Parallelism，TP（也有叫Model Parallelism，MP的）：LLM中，有的tensor或者layer很大，单卡放不下（或者单卡不够放整个模型），那么就需要用TP把tensor分割成多块，每一块放在一张卡上，分别使用和计算。仅当某些算子需要完整的张量时，才会进行聚合。  

## 分块矩阵乘法  

TP的基本思想是对矩阵乘法进行拆分：  

{% asset_img mat_mul.png 并行策略 %}  

那么矩阵乘法有两种拆分方法：（1）对矩阵A按列拆分（上图上）（2）对矩阵A按行拆分（上图下）。  

注意，当对矩阵A按行拆分的时候，也要对矩阵X进行列的拆分，保持维度的一致。  

当对矩阵A按行拆分的时候，X和A都是concat的关系，backward的时候可以分别计算X拆分出来的小矩阵的梯度，然后再拼接起来就可以得到X的完整梯度。  

而当对矩阵A按列进行拆分时，X同时参与了两块GPU上的前向计算，因此X的完整梯度等于两张卡上X的梯度相加。  

更加详细的说明可以参考：[图解大模型训练之：张量模型并行(TP)，Megatron-LM，https://zhuanlan.zhihu.com/p/622212228](https://zhuanlan.zhihu.com/p/622212228)。  

## MLP层的TP  

上面展示的是矩阵乘法的TP。那么如果我们的计算不仅是Y=XA，而还有个激活函数呢，比如Y=ACT(XA）。把矩阵A按行切分的方式，需要在进入激活函数的计算前，同步各个GPU得到的Y，这就有不少的通讯量；而把A按列切分的方式则可以直接进行激活函数的计算。  

那么再进一步，如果是MLP层，那么Y=ACT(XA)B，在上面的基础上又多了个B矩阵的计算，该怎么切分呢。理想的状况应该是尽量减少计算中的同步操作（从而减少通讯量），提升框架整体的计算效率。  

基于前面的分析，我们可以对A按列切割，那么各个GPU得到的Y就是concat的关系，为了和各个小Y能够直接进行计算，那么B应该是按行切分：  

{% asset_img mlp.png 并行策略 %}  

## Attention的TP  

那么多头注意力如何做TP呢？先回顾一下多头注意力的计算，多头注意力本身就对Q、K、V在dim维度做了切分，然后concat起来。也就是说这多个头本身，天然就是可以并行，独立进行计算的。那么只需要把不同的注意力头放到不同的GPU上，我们就得到了多头注意力的TP了。  

{% asset_img attention.png 并行策略 %}  

## Embedding层的TP  

最后还有embedding层。embedding层的做法是每块GPU维护一份embedding的子集，用id去gather向量的时候，各个GPU上分别获取，对于获取不到的id，则先用特殊向量比如零向量先表示，最后再allreduce各个GPU上的向量，替换掉零向量，就获得了完整的embedding输入了。  

# 流水并行  

流水并行，Pipeline Parallelism，PP：将网络按层切分，划分成多组，一张卡存一组。  

TP是对模型宽度进行切分，而PP是对模型的高度进行切分。  

```
# 假设模型有8层：L0~L7
# 两张卡：GPU0,GPU1
=====================   =====================
| L0 | L1 | L2 | L3 |   | L4 | L5 | L6 | L7 |
=====================   =====================
        GPU0                 GPU1
```

按这个思路，我们可以直接实现naive PP：假设模型有8层，把模型前4层放在一张卡，后4层放在另一张卡；前向的时候把中间激活数据从GPU0传给GPU1，反向的时候则把数据从GPU1传到GPU0。  

naive PP的问题是，当GPU0在跑前向的时候，GPU1是没事干的，反过来也有一样的问题，这就导致GPU有大量的空闲时间在等数据。而且随着PP的GPU数量的提升，这个空闲率就越来越高。比如设置8卡的PP，那么GPU0在做前向计算的时候，GPU1到7都在休息。真所谓是一卡有难，七卡围观。这些GPU的空余等待时间叫bubble。  

{% asset_img bubble.png 并行策略 %}  

有N张卡的PP，卡的计算利用率就只有1/N。  

那么怎么优化PP的GPU利用率呢。  

一个自然的想法是，能不能在GPU0算下一个batch的前向数据时，让GPU1在算上一个batch数据的反向呢？是可以的，并且还可以把batch切分成更小的micro-batch，这样就能减少GPU的空闲等待时间。  

这就是GPipe。GPipe单个batch进一步拆分为多个Micro-Batch，通过流水线调度不同Micro-Batch的前向和反向计算，减少设备空闲时间。  

还有很多别的方案，比如Interleaved Pipeline、1-Forward-1-Backward等，可以看看大佬们的做法。  

GPipe的Micro-Batch优化了bubble的问题，那还有显存问题呢。比如GPU1在接收来自GPU0的前向数据时，自己也还有反向传播的中间层数据，这么一来显存就很吃紧了。一个方法就是用activation checkpoint来减少显存的消耗。  

实际上个人感觉流水并行是比较复杂的，也有很多不同的实现方法，可以看看框架大佬们的资料。  

# 3D并行  

3D = DP + TP + PP。  

DP是对数据进行切分，TP是对模型宽度进行切分，而PP是对模型的高度进行切分。这三者是可以组合起来使用的。  

{% asset_img parallel.png 并行策略 %}  

层内使用TP，层间使用PP，多组TP+PP之间使用DP。一般来说DP可以跨机，而TP和PP的通讯更多，应尽量避免跨机。  

看下来自Bloom论文的图：  

{% asset_img bloom_3d.png 并行策略 %}  

每个白色方框表示一块GPU，每组机器有48块GPU，每组都复制了一份模型完整参数。左侧表示数据并行DP，有8组机器，每组输入一批数据；右侧图的竖向示意了PP过程，有12行，模型横跨了这12行GPU，例如模型有48层，则每4层放在一行中；右侧图横向示意了TP过程，一行4块GPU，表示这一行的模型参数被平摊到4块GPU上。  

看下DeepSpeed博客的版本：

下图是个三维的3D并行示意图。每种颜色表示一个节点，每个节点有4块GPU。上面16张卡和下面16张卡分别是一组，每组输入一份数据，这是数据并行。上面一组16张卡，假设模型有32 layer，一组GPU中每个节点存放8layer，每个节点的输出作为下一个节点的输入，例如GPU0的输出是GPU8的输入，这就是流水线并行。每个节点执行模型并行，意思是每个layer被分成了4分，放到一个节点的4个卡上。  

{% asset_img 3D.png 并行策略 %}  

下图是对上图的拓展示意。模型有32 layer，每8个layer放到一个节点，黄色框是一个节点，包含4个GPU。每个节点执行模型并行/张量并行， MP-0、MP-1、MP-2、MP-3表示同一layer中的张量被切分成4份，分别放到4个GPU上。Rank 0 和Rank 1是数据并行。节点之间执行流水线并行，0~7layer放在第一个节点，以此类推，最后的24~31layer放到最后一个节点。  

{% asset_img 3D_2.png 并行策略 %}  

# 序列并行  

序列并行主要是解决LLM的输入数据长的问题。由于attention的计算复杂度是平方增长，中间激活值的量随着输入输出长度增长而暴增，naive attention实现的情况下，比如10k长度的序列所需的显存是1k长度的100倍。  

前面TP和PP都是切模型，而序列并行就是切数据。  

主流的实现有这三种，对比一下：  

| **属性**           | **Colossal-AI**                                                                 | **Megatron-LM**                                                             | **DeepSpeed-Ulysses**                                                                 |
|---------------------|---------------------------------------------------------------------------------|-----------------------------------------------------------------------------|---------------------------------------------------------------------------------------|
| **核心目标**        | 突破序列长度限制，支持超长序列（如114K Token）                                  | 减少LayerNorm和Dropout的显存占用，优化张量并行下的显存效率                  | 高效支持超长序列（百万Token）和大模型训练，结合ZeRO-3参数分片                          |
| **通信机制**        | 环状通信（Ring Self-Attention），分块传递QKV，All-Gather聚合结果                | All-Gather和Reduce-Scatter聚合序列分片的中间激活值                          | All-to-All转置QKV矩阵，将序列分片转换为注意力头分片                                   |
| **兼容性**          | 兼容数据并行、流水线并行、张量并行                                              | 主要与张量并行结合使用                                                      | 与ZeRO-3和数据并行结合，支持FlashAttention优化库                                      |
| **无损性验证**      | 计算结果与单卡完全一致，实验验证Loss曲线和精度指标无差异                        | 分布式与单卡输出的均方误差（MSE）为浮点误差量级（<1e-7）                   | 生成文本的困惑度（Perplexity）与单卡一致，数学等价性通过矩阵分块转置严格保证          |

更详细的分析可以看这个：[LLM(31)：序列并行的典型方案与实现细节，https://zhuanlan.zhihu.com/p/14665512019](https://zhuanlan.zhihu.com/p/14665512019)。  

***  

博客：[http://www.linsight.cn/](http://www.linsight.cn/)  
知乎：[Linsight](https://www.zhihu.com/people/us4ever)  
微信公众号：Linsight  
![](/images/qrcode.jpg)
博主微信号(添加请注明来意)：  
![](/images/wechat.png)  

***  

【推荐文章】  
- Agent：  
[Agent完全手册(零)：三大模块，三个理念](https://www.linsight.cn/b242bfb3.html)  
- MoE：  
[DeepSeek-V3细节探索](https://www.linsight.cn/a9c496e3.html)  
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
[MiniCPM](https://www.linsight.cn/376db710.html)  
[适合移动设备的语言模型--MobileLLM](https://www.linsight.cn/5ac36d34.html)  
[phi系列模型](https://www.linsight.cn/fe13b56f.html)  
[Gemma2](https://www.linsight.cn/cf3f1f81.html)  
[苹果的OpenELM](https://www.linsight.cn/f845f3e4.html)  
[bilibili的index-1.9B](https://www.linsight.cn/770b63e1.html)  
- 预训练：  
[Qwen3实测&技术报告](https://www.linsight.cn/37ee84bb.html)  
[代码大模型(一)--业界现状](https://www.linsight.cn/a0b50049.html)  
[代码大模型(二)--OpenCoder](https://www.linsight.cn/7856bcc1.html)  
[LLM高效预训练(一)](https://www.linsight.cn/dcb57672.html)  
[LLM高效预训练(二)](https://www.linsight.cn/1e2e35a7.html)  
[Llama3.1--预训练要点一览](https://www.linsight.cn/7d7294cb.html)  
[Qwen2技术报告](https://www.linsight.cn/a8f8b641.html)  
[Yi技术报告-划重点看细节](http://www.linsight.cn/41b6a819.html)  
[InternLM系列模型](https://www.linsight.cn/7f3d361.html)  
[GLM4报告的一些技术点](https://www.linsight.cn/a5206abd.html)  
[从Yuan2.0到Yuan2.0-M32](https://www.linsight.cn/3df0cd42.html)  
[从loss视角理解大模型涌现能力](https://www.linsight.cn/f5fb75e4.html)  
- 数据：  
[训练数据合成(一)](https://www.linsight.cn/85132189.html)  
[训练数据合成(二)](https://www.linsight.cn/2a22baeb.html)  
[训练数据合成(三)](https://www.linsight.cn/e259c7b2.html)  
[LLM预训练数据策略(一)](https://www.linsight.cn/2c2cdc34.html)  
[预训练数据处理--长度分解](https://www.linsight.cn/210dbccd.html)  
- 长上下文：  
[Qwen2.5-1M技术解析](https://www.linsight.cn/6c0f6207.html)  
[LLM长上下文的问题](http://www.linsight.cn/c4da56c0.html)  
[解锁大模型长上下文能力](http://www.linsight.cn/cc852861.html)  
[大模型推理窗口-从有限到无限大](http://www.linsight.cn/45ee1a6d.html)  
[prompt压缩(一)](https://www.linsight.cn/4519eadd.html)  
[prompt压缩(二)](https://www.linsight.cn/ea2871bf.html)  
[reasoning压缩(一)](https://www.linsight.cn/bfa4f144.html)  
- 推理加速：  
[大模型推理加速-投机解码](http://www.linsight.cn/f5c015c.html)  
[大模型推理加速-MEDUSA](https://www.linsight.cn/7bbe2df6.html)  
- 对齐：  
[深度求索DeepSeek-R1详解](https://www.linsight.cn/9e4b4e6d.html)  
[基模型Cognitive Behaviors对RL的影响](https://www.linsight.cn/657a6d17.html)  
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
[LLM水印](https://www.linsight.cn/2dee4921.html)  
- 训练框架  
[LLM训练框架：从优化器和精度讲到ZeRO](https://www.linsight.cn/fe0adaa5.html)  
- 项目应用：  
[一个模型支持智能助手系统](https://www.linsight.cn/9c593ccd.html)  
[关于The Bitter Lesson](https://www.linsight.cn/d253d7b3.html)  
- CV：  
[CV入门--关于Vision Transformer](https://www.linsight.cn/a11e2633.html)  
[CV入门--无监督学习](https://www.linsight.cn/ae81a87b.html)  
- 多模态：  
[多模态入门(一)--CLIP](https://www.linsight.cn/3069051d.html)  
[多模态入门(二)--Flamingo,LLaVA系列和BLIP系列](https://www.linsight.cn/569d722c.html)  
[多模态入门(三)--MiniGPT4,DeepSeekVL,InternVL系列和QwenVL系列](https://www.linsight.cn/f16505b3.html)  
[多模态入门(四)--CogVLM,VILA,MM1,MM1.5和Pixtral-12B](https://www.linsight.cn/e00debee.html)  
[多模态入门(五)--InternVL系列](https://www.linsight.cn/52c8a4f9.html)  
[小米的移动UI多模态模型--MobileVLM](https://www.linsight.cn/96393d3b.html)  
[DeepSeek-VL2的细节](https://www.linsight.cn/b4d047c1.html)  
- 大模型算法题：  
[(1)](http://www.linsight.cn/3345028a.html)、
[(2)](http://www.linsight.cn/ad0bba9d.html)、
[(3)](http://www.linsight.cn/1736008.html)、
[(4)](http://www.linsight.cn/1736008.html)、
[(5)](http://www.linsight.cn/336f2f3e.html)、
[(6)](http://www.linsight.cn/7c04944d.html)、
[(7)](https://www.linsight.cn/dd614e12.html)、
[(8)](https://www.linsight.cn/e287b9c3.html)、
[(9)](https://www.linsight.cn/fb9c8882.html)  

# Reference  

【1】千亿参数开源大模型 BLOOM 背后的技术，https://zhuanlan.zhihu.com/p/615839149  
【2】图解大模型训练之：张量模型并行(TP)，Megatron-LM，https://zhuanlan.zhihu.com/p/622212228  
【3】大模型训练技术笔记总结，https://zhuanlan.zhihu.com/p/610139027  
【4】图解大模型训练之：流水线并行（Pipeline Parallelism），以Gpipe为例，https://zhuanlan.zhihu.com/p/613196255  
【5】https://zzqq2199.github.io/2021/04/02/DAPPLE/
【6】Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism，https://arxiv.org/abs/1909.08053  
【7】LLM(31)：序列并行的典型方案与实现细节，https://zhuanlan.zhihu.com/p/14665512019  

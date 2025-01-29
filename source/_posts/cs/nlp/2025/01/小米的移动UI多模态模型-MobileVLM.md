---
title: 小米的移动UI多模态模型--MobileVLM
tags:
  - 多模态
  - CV
  - NLP
  - transformer
  - 预训练
  - 无监督学习
  - SFT
  - UI
  - 小米
categories:
  - CS
  - 多模态
abbrlink: 96393d3b
date: 2025-01-29 17:40:32
---

【本文已在同名 微信公众号 / 知乎 / [个人博客linsight.cn](http://www.linsight.cn/) 上线】  

***  

图文多模态模型的一大应用场景就是手机上的智能助手，一个能听能看能写能说的智能助手还是颇有吸引力的。  

手机厂商里，除了苹果，小米也是一个不时能拿出一些实用AI技术和产品的厂商。（最近开出年薪千万挖人也是上了头条）  

今天就来了解一下小米关于手机UI多模态模型的一个工作 -- MobileVLM。MobileVLM算是多模态模型在手机UI垂域场景的一个应用工作了。这个工作主要做了两件事：  

- 针对手机UI场景，增加了对应的任务和训练stage  
- 构造了对应的数据集Mobile3M，用于训练 & 评测模型的手机UI理解和操作能力  

（不过模型的大小并不是很mobile啊...）  

# Mobile UI数据  

关于UI，特别是手机UI的数据集目前已经有一些了。现有的这些数据集在这里根据dataset structure被分为了两类：  

- Dot：这些数据集中的每个数据实例仅包含一个UI页面，以及不同的细粒度任务和相应的答案。这些数据集只关注单个UI的内容，无法捕捉到用户在使用这些app的操作过程。  
  - Rico（2017）：安卓UI数据集  
  - UIBert（2021）：发布了两个从Rico扩展来的数据集  
  - Ferret-UI（2024）：基于UI detection model打标的安卓 & 苹果数据集  
- Chain：包含a sequence of action-UI pages。  
  - AITW（2023）：有715k的数据  
  - Auto-UI（2023）：进一步过滤了AITW的GoogleApps子集，留下152k数据  

UI页面包括截图和结构化的文档，结构化文档能够给出UI中各个组件的层级关系，但是AITW和Auto-UI都没有结构化文档的信息。  

下表列出了现有的Dot和Chain类型的数据集：  

{% asset_img datasets.png MobileVLM %}  

# Mobile3M数据集  

Mobile3M专注在Chinese apps，总共包含49个下载量超过1000万的app：  
- 20,138,332 actions  
- 3,098,786 screenshots and corresponding XML  

XML就是每个UI截图对应的结构化信息。下面是一个例子：  

{% asset_img xml.png MobileVLM %}  

整个数据集被组织成49个directed graph，每个graph对应一个app。可以认为每个有向图就是一个app（几乎）所有可能操作的集合，有向图里的一条路径就是一个用户操作的sequence。UI截图就是节点，action就是有向图的边。  

49个app的选择中，确保AppStore中的每个主要品类至少包含两个app。Mobile3M的app分布如下：  

{% asset_img apps.png MobileVLM %}  

相比前面提到的Dot和Chain类型数据，Graph类型的Mobile3M能够提供更多样化更复杂的操作信息。下图是Dot、Chain和Graph的对比例子：  

{% asset_img graph.png MobileVLM %}  

## 收集方式  

那么这些UI和对应的结构化信息是怎么收集的呢？毕竟人工标注这么大规模的数据成本太高了。  

小米的方法是在模拟器上运行这些app，然后用Appium收集UI，收集的数据包括截图和结构化信息的XML，包括按钮和文本框等元素，以及边界框和分层树等布局信息；每个元素又包含名称和bbox。  

另外，人和UI的交互需要一些操作，这里定义了三种操作：  

- Click  
- Scroll  
- Input  

参考APPAgent的做法，小米采用random walk algorithm来探索app的空间，以此收集app的所有页面。在random walk期间，优先考虑输入动作，小米为每个app设计了10个相关的输入关键词，每次输入时随机从这些关键词选择输入。  

对于某一个UI page，它的action trace定义为从app主页到该页面的最短操作序列。action trace经过的每一个边（也就是操作）的id拼在一起，作为这个UI page的唯一标识。比如下面这个图里，Baicizhan0经过click（Edit）进入Edit界面，因此Edit页面的标识就是Baicizhan0_1。  

{% asset_img graph_example.png MobileVLM %}  

在随机游走手机UI数据的过程中，会遇到很多重复数据，因此需要设计一个UI数据去重的策略。对于新的UI，会用BM25检测库中最相似的前5个XML，如果Element Diff<5 & Pixel Diff <30%，那么就认为这是两个相似的页面。  

这样去重有几个好处：  

- 大大减少了搜索空间  
- 避免了死循环，因为如果重复访问一个页面可以检测并终止它  
- 这样可以把树状结构的数据构建为graph，不同的路径可以进入到相同UI，这有助于模型理解各个UI之间的连接关系  

# 训练  

小米使用的模型是Qwen-VL-Chat：  

- Qwen-7B  
- ViT-bigG，1.9B参数  
- 0.08B参数的adapter  

预训练和SFT过程的参数冻结设置如下：  

{% asset_img train.png MobileVLM %}  

## 预训练  

在通用预训练的基础上，小米增加了2个stage的预训练：  

{% asset_img ptm_stage.png MobileVLM %}  

1、stage 1  

这一阶段的预训练主要目标是增强模型对intro-UI page的理解。所用任务主要是这三个：  

- Element List Generation：要求模型识别页面中的所有interactive element，包括文本和bbox，要求模型具备对应的OCR和grounding能力  
- Element Grounding：给定一个element description，让模型输出bbox  
- Action Space Generation：生成当前UI的所有可能操作，模型需要分析每个元素的交互性，比如是可点击还是可输入等。这个能力对于stage 2的action prediction能力十分重要  

2、stage 2  

这个阶段主要是提升inter-UI page的理解能力，任务主要是Action Prediction：让模型输出从当前UI进入到目标UI的操作。  

下面是以上这几个任务的一个示例：  

{% asset_img task.png MobileVLM %}  

## SFT  

微调阶段，有3个任务：  

- Page Navigation：这个任务不再像stage 2一样提供两个UI，而是提供一个UI和一个指令，模型要给出应该进行什么操作  
- VQA：根据UI截图回答问题  
- Auto-UI  

各个阶段的训练量：  

{% asset_img train_data.png MobileVLM %}  

# 评测  

各个版本的MobileVLM效果：  

{% asset_img eval.png MobileVLM %}  

# 使用  

MobileVLM的使用上还有几个问题：  

- 目前模型太大，即使是4-bit的量化，也需要46G RAM和23G显存的设备才能跑  
- 推理速度不够快，比如作出滚动屏幕这一决定会导致UI变化，需要后续的理解能力能跟上屏幕滚动  
- 在手机上的使用存在权限问题：需要具有系统级操作的权限的工具来进行操作，但是，对于大多数闭源移动操作系统，向第三方应用程序授予系统级签名几乎是不可能的  

***  

博客：[http://www.linsight.cn/](http://www.linsight.cn/)  
知乎：[Linsight](https://www.zhihu.com/people/us4ever)  
微信公众号：Linsight  
![](/images/qrcode.jpg)
博主微信号(添加请注明来意)：  
![](/images/wechat.png)  

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
[MiniCPM](https://www.linsight.cn/376db710.html)  
[适合移动设备的语言模型--MobileLLM](https://www.linsight.cn/5ac36d34.html)  
[phi系列模型](https://www.linsight.cn/fe13b56f.html)  
[Gemma2](https://www.linsight.cn/cf3f1f81.html)  
[苹果的OpenELM](https://www.linsight.cn/f845f3e4.html)  
[bilibili的index-1.9B](https://www.linsight.cn/770b63e1.html)  
- 预训练：  
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
[LLM长上下文的问题](http://www.linsight.cn/c4da56c0.html)  
[解锁大模型长上下文能力](http://www.linsight.cn/cc852861.html)  
[大模型推理窗口-从有限到无限大](http://www.linsight.cn/45ee1a6d.html)  
- 推理加速：  
[大模型推理加速-投机解码](http://www.linsight.cn/f5c015c.html)  
[大模型推理加速-MEDUSA](https://www.linsight.cn/7bbe2df6.html)  
- 对齐：  
[深度求索DeepSeek-R1详解](https://www.linsight.cn/9e4b4e6d.html)  
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
- 项目应用：  
[一个模型支持智能助手系统](https://www.linsight.cn/9c593ccd.html)  
- CV：  
[CV入门--关于Vision Transformer](https://www.linsight.cn/a11e2633.html)  
[CV入门--无监督学习](https://www.linsight.cn/ae81a87b.html)  
- 多模态：  
[多模态入门(一)--CLIP](https://www.linsight.cn/3069051d.html)  
[多模态入门(二)--Flamingo,LLaVA系列和BLIP系列](https://www.linsight.cn/569d722c.html)  
[多模态入门(三)--MiniGPT4,DeepSeekVL,InternVL系列和QwenVL系列](https://www.linsight.cn/f16505b3.html)  
[多模态入门(四)--CogVLM,VILA,MM1,MM1.5和Pixtral-12B](https://www.linsight.cn/e00debee.html)  
[多模态入门(五)--InternVL系列](https://www.linsight.cn/52c8a4f9.html)  
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

【1】MobileVLM: A Vision-Language Model for Better Intra- and Inter-UI Understanding, https://arxiv.org/abs/2409.14818v2  

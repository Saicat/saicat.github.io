---
title: DeepResearch的报告生成方法
tags:
  - NLP
  - LLM
  - Agent
  - DeepResearch
categories:
  - CS
  - NLP
  - Agent
abbrlink: 44c62dc5
date: 2025-05-19 22:31:35
hidden: false
---

【本文已在同名 微信公众号 / 知乎 / [个人博客linsight.cn](http://www.linsight.cn/) 上线】  

***  

最近搞DeepResearch。  

DeepResearch的任务是为用户提供一份全面深入的研究报告。  

列一些典型的用户query：  

```
做一份端午从上海去东京的五天四夜购物攻略，预算八千
洛阳龙门石窟和老君山三天两夜文化摄影攻略
深入研究一下DeepResearch技术今年的发展趋势
三胎政策在成都改善型住房需求中的刺激效果，并预测一下明年四室户型供需缺口
```

都是一些较为复杂的，需要多步拆解处理的任务。用户中比较热门的任务类型包括「旅游攻略」，「专业研报」（如技术说明，专业分析："整理下MoE模型的演进过程"）还有「信息整合」（比如给出一份市面上20w以内的电车的对比资料）。  

可以粗暴认为DeepResearch主要就由DeepSearch + 报告生成这两大模块组成。当然这个过程还可以由planner + reflect循环调度。  

这篇略过DeepSearch，先看看「报告生成」的模块。  

假设我们已经有了比较合理、丰富的搜索结果了，那报告要怎么生成呢？  

# 报告的特点  

1、长度较长  

DeepResearch的报告首先长度是比较长的，一般至少在几千个token，甚至上万token或者更长，具体就取决于话题和任务的复杂度。  

2、图文并茂  

除了大量的文字，报告还应该是图文并茂的。这里的「图」包含图片和数据图表（比如折线图、扇形图、表格等）。  

3、排版和格式  

为了提供给用户提供更好的阅读体验，报告应该支持比较好的排版。具体的排版就和输出报告的格式有关，常用的就是html、pdf、ppt。  

比如我用coze空间做一份旅游攻略，prompt是：  

```
给我做一份端午节出国旅行的攻略，东南亚，两个人，悠闲一点，预算6000
```

跑了差不多半个小时之后就获得了一份html版本的图文攻略：  

{% asset_img dr_example.jpg 研报 %}  

上面这个图还没截完整，后面还有好几个不同国家的出行方案。  

# DeepSearch的结果  

DeepSearch是报告生成的起点，先看下它都提供了什么。  

1、general search  

目前DeepSearch的结果主要是一系列网页搜索的结果，每个网页包含以下字段：  

- title：网页标题  
- content：完整正文的「文本」内容  
- url：原文链接  
- summary：正文的简短总结  
- images：网页中的图片列表，包含图片的url和对应的caption  

其中title、content和url是常规的网页搜索结果字段，就不说了。  

summary是额外添加的，在报告生成的处理逻辑里，对于不需要网页细节内容的部分，就可以使用summary进行处理，从而减少处理的token，节省时间和成本。  

个人认为，成本问题是DeepResearch一个很重要的方向。如果DeepResearch要向大众推广，那么开发过程中，60%以上的时间都会在考虑怎么节省成本。  

图片的caption也是搜索到网页后增加的，用于后续在报告中添加图片。  

2、domain search  

除了general search，还会有一些常用场景需要的搜索源，比如：  

- 导航工具：用于获取特定地点之间的交通信息，包括驾车、航班和火车，一般旅游攻略对这个有强需求。  
- 美食工具：获取美食的价格评分和地点还有评价，也是旅游场景的所需要的。  

除了用现有工具，也可以针对自有数据建设向量搜索。  

这些都可以整合成general search的结果格式：标题、正文、摘要，url和图像是optional的。  

# step1：文字版的初稿  

从搜索结果到最终报告，中间需要多个步骤。（有没有大佬已经在做一步端到端的生成？把所有数据都塞给模型，要求一步到位生成图文结果。目前这样做的效果比较差，模型窗口长度限制也是个问题）  

报告的目标格式一般包含选择html、pdf和ppt，这些格式用户使用起来比较熟悉。  

转成目标格式之前，首先要生成一份逻辑通顺，行文流畅，内容完整，包含文字和图片的初稿。  

而这个初稿的生成又分为「生成文字」和「增加图像」两个阶段。  

第一步我们就是要获得文字版本的初稿。  

这里选择用markdown格式来生成文字初稿。因为markdown格式比较简单（能少用点token），模型生成的效果也好，支持多级的标题，公式以及图片的插入，基本能够满足我们的需求。  

## 直接生成的问题  

这个文字版本也没法一步直接生成。稍微讲一下直接生成的问题：  

- 搜索结果太多，假设一个网页平均有1000 token，那100个搜索结果就要100k token，已经超过或者接近很多模型的窗口上限了；参考秘塔AI，经常出现100个200个甚至更多的网页引用，所以100k级别的输入并不会是少见的情况。  
- 即使模型的窗口可以接受这个长度或者更长的输入，也容易出现lost in the middle的情况；对于需要使用到原文细节信息的情况（比如旅行规划中的车次/航班号，出发时间，或者经济研报中多个地区多个维度的数据），要在大量的文本中准确捞到正确的内容是一个容易出错的事情。  
- 目前大部分模型支持的生成长度在2k到4k，在更大长度的内容输出上，容易出现截断。  

## 大纲生成  

直接生成会遇到问题，那么更好一点的做法是先生成文档的大纲（即各级标题），再根据大纲去填充细节。  

生成大纲这一步就可以用上搜索结果中的summary了，因为生成大纲并不需要关注太多细节。  

比如在制定旅游攻略的任务下，我们搜索到的内容基本可以分为交通、住宿、美食、景点、通讯等，我们只要让模型根据搜索内容的summary指定report的大纲就可以了。类似地，研究NLP深度模型的发展模型也可以根据搜索结果分为embedding模型、Bert时代、GPT时代、Agent等。  

假设一个summary是30个token，那么即使有200个搜索结果，长度也只有6k token，模型可以轻松处理。  

生成大纲时，也有一些细节限制：  

- 要限定每级标题的数量，防止模型生成过多，并且限定标题级别数量，比如最多只能使用到3级标题。  
- 要求各级标题之间尽量不要有overlap。  
- 标题要起得明确清晰，让人单独看到这个标题也知道是什么意思（因为这些标题在设计上可能是要在报告完成前，展示给用户看的）。  

举一个例子。输入query = "上海至东京国庆购物攻略：8000元预算五天四夜经济型方案"  

制订的大纲各级标题是：  

```
["一、行程规划与交通安排\n1.1 机票选择策略\n1.2 机场至市区交通方案",
"二、住宿选择与区位分析\n2.1 银座商圈高性价比酒店\n2.2 新宿商圈经济型住宿",
"三、购物商圈深度攻略\n3.1 银座高端购物路线\n3.2 新宿平价消费指南\n3.3 表参道特色品牌挖掘",
"四、预算分配与消费控制\n4.1 8000元预算分解模型\n4.2 免税政策与退税实操",
"五、行程优化建议\n5.1 交通卡券组合方案\n5.2 错峰购物时段建议"]
```

上面这个例子共有5个一级标题，也就是5个大的chapter。  

大纲格式也可以自行设计，结构化的也可以，只要模型能准确遵循就行。  

这一步里其实有很多细节可以优化，比如传给LLM的搜索结果的排序和筛选，或者利用多次采样再合并获取更合理的大纲等。  

## 填充细节  

得到大纲的标题之后，就要根据搜索结果填充每个chapter的细节。  

这里可以并行来做：每个chapter调一个模型来填充细节。  

prompt是类似这样的（简化版）：  

```
你需要根据用户的要求，和文档的大纲，完成分配给你的章节的撰写。

你需要根据搜索结果来完成这一章节。

用户query: {query}

大纲: {outline}

分配给你的章节: {chapter}

搜索结果: {search_results}
```

前面分析了，全量的搜索结果过多，一起都塞给模型，可能导致结果不佳，成本也高。因此在这一步也不宜直接把所有搜索结果扔给模型去完成细节的编写，而是先从搜索结果里找到和当前要写的这个章节相关的条目。  

比如在旅游规划任务下，有一个chapter是交通相关的内容。200个搜索结果里有40个涉及了飞机火车的班次信息，以及景点之间的交通工具推荐。那么在写这一个chapter的时候，就只需要给模型输入这40个搜索结果，而不需要200个搜索结果都给。  

那怎么找到相关搜索条目呢？可以用BGE或者小的LLM给每个文档做一个打分或者匹配，以此筛选搜索结果。也可以在生成大纲的时候就要求模型把对应的条目编号和标题一同给出。  

这一步同样有很多细节可以优化，比如：  

- 如果觉得以一级标题进行搜索结果匹配还是有太多结果，那可以进行二级或者三级标题的匹配，把章节拆得更细，从而减少每个章节编写的难度。  
- 为了方便编写细节的模型理解，可以在生成大纲的时候增加一个长一点的解释，限定这一章需要补充的信息。  
- 把章节细节的编写也设计成迭代的模型，逐步完善。  

值得单独拎出来说的，是关于字母和数字的细节。涉及字母和数字的通常是比较严谨的信息，比如火车/航班的班次，出发/到达时间，或者路途的公里数，开车所需的时间和住宿价格等。一方面，这些内容错一个字母或者数字就会给用户带来比较大的困惑，另一方面，数字通常涉及计算，而LLM的"口算"并不是很可靠。针对这些问题，可以额外添加一个利用计算器或者python代码验证字母和数字的环节，并把结果提供给章节编写的模型，从而减少计算错误和幻觉带来的问题。  

最后，记得让模型给出reference，用于展示给用户。  

# step2：图文报告  

上面这几步做完之后，就有一个纯文本的report初稿了。但是呈现给用户，光有字不够，还得有图。  

## 图的类型  

report里都有什么图？先来分个类。  

1、来自检索结果（网页）的图  

检索结果中包含一些可以直接使用的图片，这些图片可以直接插入到report的适当位置。  

一种是如旅游景点的风景图，地标建筑照片等。这一类图片的特点是，插入到report时，在准确度上的要求相对比较低，只要别出现明显的图文不匹配（比如文字在介绍山，但是图片是海景），都还可以接受。  

另外，也有可能出现对准确度有一些要求的情况，比如路线导航，车次的信息表。这类信息如果出错（火车的章节配了个航班的图）可能就会让用户的体验大打折扣。  

再进一步，比如对于经济调研的研报，那么就有可能出现很多折线图、柱状图、扇形图或者信息密集的表格，这种图表每个字母每个数字都很重要，不能出错，不能和文本的信息对不上。  

这些来自检索结果文档的图片，插入report的关键在于
- 要用对图，比如搜索的时候有可能搜到有矛盾的信息，那么LLM在总结完文本之后，我们需要知道应该用哪些文档的图片，不应该用哪些文档的图片
- 插对位置，这就要求我们知道每张图片的主要信息是什么  

2、从其他来源获得的图  

有些时候搜索结果文档里只有文字，或者文档中的图不是我们想要的图，那我们就可能需要根据用户需求和文本报告内容，自己从另外的来源获取合适的图。  

（1）来源1：自己画数据图表  

如果report中有一系列数据，比如某地不同月份的温度，或者不同厂商的市场占比，那么这些数据就可以生成图表，方便用户直观阅读。比如不同月份的温度可以画成折线图，不同厂商的占比可以画成扇形图。根据数据的类型，也可以制成柱状图、表格或者其他图表。  

（2）来源2：图片搜索接口  

假设我们在给用户制作旅游攻略的时候，查到有一处古镇适合游玩，我们想把这个古镇的资料作为攻略的一部分进行介绍，但是恰好搜到的网页只有文字，那么我们可以在制作report的时候，拿这个古镇的文字介绍去搜索图片，然后把搜到的图片插入到report中。  

3、各种图的难度  

上面分出来的这几种图片和图表，按开发难度排个序：  

- level 1：常规的图表生成，如折线图、柱状图、表格等  
- level 2：插入来自文档的图片和图表  
- level 3：插入来自其他搜索源的图  

## 加入图的方法  

先说下插入「来自文档的图片」的方法。大致的思路就是和之前在[多模态入门(五)--InternVL系列](https://www.linsight.cn/52c8a4f9.html)中介绍的InternLM-XComposer类似。  

InternLM-XComposer生成图文并茂文档的做法是这样的：  

- （1）生成纯文本文档  
- （2）找到文本结果中可以/需要插入图像的位置，并生成对应的caption  
- （3）用上一步的caption进行图像检索，获得候选，并选择最符合的图像，获得最终结果  

稍微有点不同的是，InternLM-XComposer由于图片库比较大，所以它的做法是“假设某个位置需要图，并生成这张假想的图的caption”，然后根据这个caption去图库里找。

而在我们这个report生成的场景下，我们的图片库相对比较小。假设我们平均每个章节用到了30个搜索结果，每个搜索结果平均有3张图，那么我们的图库就有90张。如果按InternLM-XComposer的做法，很难在这么小的图库里找到对应的图，因此我们反过来，先跑出图库所有的图的caption，再把这些caption都提供给LLM，让模型来决定在哪里可以插入哪些图片。  

## 图表生成  

要生成图表，一个方法是要求模型在report中包含数字的地方，判断是否适合插入图表，适合插入什么图表，然后调用工具或者写python代码生成图表，最后把生成结果贴到对应位置上就行。  

而如果报告的目标格式是html，那么也可以在生成html的prompt中，直接要求模型判断和插入图表，html + css基本可以所有我们想要的图表。  

## 其他搜索源的图  

假设我们在旅游攻略的展示策略上，要求一定要有足够的景点图，而搜索文档中又刚好没有符合要求的，那我们可以单独去搜索我们想要的图。  

首先我们需要知道搜什么图。prompt可能是类似这样的：  

```
你是一个配图专家，你的任务是给文本配上合适的图。

你可以调用图片搜索工具，并利用关键字进行图片搜索。

{工具description}

请根据一下的文本，给出工具调用名称和关键词：

{chapter}
```

这部分的逻辑相对来说就比较定制化了。  

# 报告的格式  

报告常用的格式就是html，ppt和pdf了。其中html和ppt都可以转pdf，所以理论上只要支持html和ppt就可以了。  

1、html  

之前发现html的生成有一个不错的工具叫deepsite，[https://enzostvs-deepsite.hf.space/](https://enzostvs-deepsite.hf.space/)。可以根据输入prompt直接生成漂亮的页面。后来发现后台其实就是DeepSeek。  

试了在DeepSeek-R1和DeepSeek-V3上要求直接根据文案生成网页，效果不错，而且V3的效果比R1更好。前几天又发现Qwen Chat也专门针对WebDev做了优化，Qwen3能够直接给出比较好的网页设计了。  

随便给V3输了一组数据，生成的网页就挺漂亮的：  

{% asset_img v3_html.png 研报 %}  

不过直接大模型生成html目前也有一些问题：  

- 对指令的遵循会比较差，容易出现幻觉，比如上面这个图，下面那行字就是模型自己加的。  
- 复杂的页面设计，html代码很长，生成时间很久，还容易出现截断。  

2、ppt  

ppt的生成就得靠专业的接口了，这个头部的几家AI公司都有这个能力。  

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

【1】https://mp.weixin.qq.com/s/iPJ7eLa3O6zILXi1HESkCQ  

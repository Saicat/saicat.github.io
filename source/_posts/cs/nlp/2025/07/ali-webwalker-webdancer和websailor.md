---
title: '阿里通义Lab: WebWalker,WebDancer和WebSailor'
tags:
  - NLP
  - LLM
  - Agent
categories:
  - CS
  - NLP
  - Agent
abbrlink: f7d600f3
date: 2025-07-08 22:13:22
---

最近阿里通义Lab发布了WebSailor模型，顾名思义，这是一个专门优化「上网冲浪能力」的模型。而在WebSailor之前，他们已经发布过WebWalker和WebDancer。  

{% asset_img cover.png webagent %}  

（下一个模型会叫什么呢，WebPilot或者WebAstronaut？）  

整理下这几个工作的一些核心思路和内容。  

# WebWalker  

原文：《WebWalker: Benchmarking LLMs in Web Traversal》  

1、背景  

在RAG背景下，LLM缺乏系统性的网页遍历能力，无法处理深度、多步骤信息检索任务。  

网页遍历就是和人一样上网找资料的操作，回想我们自己找资料时，经常需要点开多个站点，以及各个站点下不同的子页面来找到我们想要的信息。  

具体来说，有几个原因：  

- 传统的搜索引擎执行“横向搜索”（horizontal search），即仅检索查询相关的最表层网页内容，无法深入挖掘网站内部的嵌套子页面。例如，回答一个复杂问题可能需要点击多个链接进入深层页面（如官网的会议日程或教育机构的课程详情），但现有RAG系统缺乏这种“垂直探索”（vertical exploration）能力。  
- 另外，网页包含大量无关信息（噪音），容易超出LLM的长上下文处理能力。  
- 复杂case需多源整合（如会议+教育领域信息），而传统RAG无法跨网站协同推理。  

而在训练数据和benchmark上，也没有针对这样细致的问题。  

2、方案  

（1）WebWalkerQA benchmark  

首先一个针对网页搜索能力的benchmark，WebWalkerQA。  

这个评测包含680个query（有中英双语），覆盖会议、组织、教育和游戏四大领域。  

其中又分为single-source和multi-source。single-source只需从一个网页开始，进行深度探索，而multi-source就需要从多个网站进行搜索。  

WebWalkerQA引入了“Web Traversal”任务：给定根URL和查询，要求遍历子页面提取信息。  

举个例子：  

query = 哈佛大学计算机科学系2024年春季学期人工智能导论课程的授课教师是谁？  

给定的root url = https://cs.harvard.edu（如果是multi-source的case，就会有多个url）  

搜索信息的路径：  

- 根页面点击 "Academics" → 进入课程目录页  
- 点击 "Spring 2024 Courses" → 进入学期课程列表  
- 点击 "Introduction to AI" → 在子页面找到教师信息  

找到答案 = "David J. Malan"  

（2）WebWalker multi-agent框架  

接下来就是构建具备这样能力的agent系统，WebWalker。  

WebWalker里有两个agent，Explorer Agent和Critic Agent，两个agent通过分工协作解决长轨迹导航问题。  

Explorer Agent负责决策每一步要找什么信息，并进行探索。Critic Agent从探索的页面找到有用信息，并加入存储。  

两个agent交替执行。  

看看原文的一个例子：  

{% asset_img webwalker.png webagent %}  

Thought相当于是常规的reasoning了，两个agent都有，可以先忽略。  

Explorer Agent会根据需求和当前的信息，决定下一步的浏览操作，比如step1里点击calls。  

接下来Critic Agent会给一个judge，判断当前新的信息有没有什么用，如果没用，那就不用加入到memory里，也就不会增加Explorer Agent的输入；如果有用，那就把信息加入到memory里。然后Explorer Agent会进行新一轮的浏览操作，直到完善需要的信息。  

# WebDancer  

原文：《WebDancer: Towards Autonomous Information Seeking Agency》  

1、要解决的问题  

目标是优化模型使用搜索工具的能力。搜索工具主要是大搜接口。  

现有的做法有：  

- prompt工程：开发快，上限有限，被模型能力所限制，而且依赖人工设计  
- SFT/RL训练：没有好的数据，泛化性差  

2、思路  

WebDancer针对三个问题来优化搜索效果：  

- 训练数据质量差：用CRAWLQA/E2HQA来获取高质量QA数据  
- 搜索轨迹不可靠：用双CoT采样 + 校验来提升质量  
- 泛化能力不好：SFT+RL训练  

3、方案  

（1）获取高质量QA训练数据  

E2HQA：从easy的单跳问题扩展到多跳，举个例子：  

- 第1轮：从一个easy case开始：搜索“爱因斯坦生平” → 获知出生地德国乌尔姆  
- 第2轮：改写为“德国乌尔姆1880年人口？”（引入新实体）  
- 第3轮：最终问题 → “爱因斯坦出生时乌尔姆的人口数有多少？”  

CRAWLQA  

具体流程：  

① 根URL收集：爬取知识性网站（arXiv/GitHub/Wiki等）的根页面。  
② 递归导航：通过超链接访问子页面，收集多层内容（如Wiki“气候变化”页 → 子页“温室气体列表”）。  
③ GPT-4o合成QA：基于子页面内容生成多跳问题，问题类型包括计数（COUNT）、多跳推理（MULTI-HOP）等。  

（2）双CoT采样 + 校验  

要被训练的模型是QwQ-32B。先进行两个采样，所用的数据就是第一步里获得的QA数据：  

- Short-CoT：用GPT-4o生成简洁轨迹（平均510 token），包含基础推理步骤。  
- Long-CoT：用QwQ-32B自主决策，记录完整推理链（平均1599 token）。  

先对QwQ-32B和GPT-4o生成的轨迹分别进行三级校验过滤，比如非法json、重复action、答案正确性等。  

然后把QwQ-32B和GPT-4o的答案和轨迹进行对比校验，保留通过正确性校验的数据（和GPT-4o对比）。  

（3）训练  

SFT  

SFT所用的数据就是上一步中通过校验的QA + 轨迹数据。  

训练的时候，只学习思考和action部分，而屏蔽observation的loss。  

RL  

使用DAPO，所用的数据是上一步中没有通过检验的数据。  

这些数据没有通过校验，说明难度较大，因此适合用来进一步提升模型能力。  

另外，这些数据来自于QwQ-32B自己生成，其分布一致性比直接使用外部的轨迹数据要好。  

训练中，会提高「部分正确数据」的采样比例，这些数据包含正确和错误推理，对强化学习更好。  

（4）流程图


```
                +-----------------+
                | CRAWLQA/E2HQA   |
                | (100K QA Pairs) |
                +--------+--------+
                        |
                +--------+--------+
                | 双CoT轨迹采样     |
                | [输入相同QA问题]  |
                +--------+--------+
                        |
        +---------------+---------------+
        |                               |
+---------+---------+           +---------+---------+
| Short-CoT         |           | Long-CoT          |
| (GPT-4o生成轨迹)   |           | (QwQ-32B生成轨迹)  |
+---------+---------+           +---------+---------+
        |                               |
+---------+---------+           +---------+---------+
| 三级过滤漏斗        |           | 三级过滤漏斗       |
| 1. 格式校验        |           | 1. 格式校验        |
| 2. 答案正确性       |           | 2. 答案正确性      |  # 独立校验
| 3. 质量评估        |           | 3. 质量评估        |
+---------+---------+           +---------+---------+
        |                               |
+---------+---------+           +---------+---------+
| 有效轨迹 → SFT数据  |           | 有效轨迹 → SFT数据  |
+---------+---------+           +---------+---------+
        |                               |
        +---------------+---------------+
                        |
                +--------+--------+
                | SFT训练          |  
                | (仅优化思考/动作) |  # 屏蔽observation损失
                +--------+--------+
                        |
                +--------+--------+
                | 被过滤QA对       |  
                | (部分正确+噪声)   |  # 注：含可挽救样本
                +--------+--------+
                        |
                +--------+--------+
                | DAPO优化        |
                | 1. 动态采样      |  # 过采样部分正确样本
                | 2. 奖励加权      |  # 格式10%+答案90%
                +-----------------+
```

# WebSailor  

原文：《WebSailor: Navigating Super-human Reasoning for Web Agent》  

1、要解决的问题  

目前大部分的agent可以解决简单搜索，或者多跳搜索的问题，但是在困难的问题，比如BrowseComp-en/zh上的level 3的问题上，效果不好。  

这些问题的特点是：  

- 高不确定性：问题涉及模糊描述，如“21世纪初”、“南美著名首都”。  
- 非线性推理路径：无预定义解决路径，需组合泛化（Compositional Generalization）能力。  

现有的方法，泛化性不够（因为level 3这种训练数据比较少）；另外即使能够有一定泛化搜索的能力，其推理轨迹也很长，导致上下文急剧增长，效果不好。  

2、方案  

（1）数据：SailorFog-QA  

WebSailor第一个要解决的问题依然是数据：需要收集level 3这种高难数据。  

step 1：QA收集  

首先，从wikidata获取稀有实体（如“5世纪匿名诗人”），通过随机游走关联多跳关系（诗人→创作的赞美诗→树轮年代学），形成网状拓扑（非链式结构）。  

每次游走，都有一定的概率选择走到新节点，或者回溯到旧节点，这样保证生成网状的结构，而不是偏线性的结构。  

从构造的图谱再随机采样一个子图，这个子图可以视为一个路径（path），即从一个实体到另一个实体的推理序列。  

这个路径上的信息会被模糊化处理，比如「2012年」就变成「21世纪初」，「Apple」就变成「以A开头的企业」，「0.8%」就变成「不足1%」。  

然后这样一个路径会被用于生成问题，比如路径「南美首都 → 21世纪荣誉获得者 → 哥伦比亚艺术学院」，就可以生成问题：  

```
问题：“某乐曲与南美首都相关，其歌词作者获21世纪荣誉，旋律创作者毕业于哥伦比亚艺术学院，求乐曲名称？”
答案：“Rue de Rivoli”
```

step 2：思考路径优化  

把上面收集到的数据直接给QwQ-32B来跑，推理路径会比较冗长，可能出现一些无效、重复或者错误的探索。因此需要对推理轨迹进行重构优化。  

具体做法是选择一个强大的模型，把问题和QwQ-32B的原始推理轨迹给它，让它对推理进行总结。  

比如QwQ-32B的原序列是(r0,a0,o0,r1,a1,o1...)，那么就要求重构模型根据r0,a0,o0生成重构思考R0；然后根据R0,a0,o0,r1,a1,o1生成R1，以此类推。  

比如把"分析太阳能冰箱的可行性，需考虑光伏效率...（150字）"变成简短的"验证'太阳能冰箱'与'洞穴探险'的关联"。这一步其实就是优化轨迹数据。  

（2）训练step 1：RFT冷启动  

类似SFT冷启动，这步从SailorFog-QA里筛选数据进行微调。筛选的标准包括：  

- 仅保留答案正确的轨迹  
- 剔除长度>32k tokens的轨迹  
- 筛选工具调用>5次的复杂样本  

过滤后SailorFog-QA样本量仅2k+，但覆盖高难度场景。  

计算loss时屏蔽了observation，专注优化决策（Thought/Action）。  

（4）训练step 2：DUPO  

DUPO = Duplicating Sampling Policy Optimization  

通常来说，on-policy RL在训练的时候，需要顺序rollout样本，在agent场景下还包括工具调用和结果返回，且有多步，耗时很长，达到分钟级。  

DUPO引入动态采样机制，避免顺序处理，从而加速训练。DUPO的分为两个主要步骤：训练前过滤和训练中复制样本。  

训练前过滤（Pre-training Filtering）：在RL训练开始前，过滤掉所有rollout结果完全一致的QA。具体地，如果一个QA的所有G个rollout都正确或都不正确，则会被移除。  

训练中复制样本（In-training Sample Duplication）：假设batch size为128，过滤后剩余100个case，其中20个空位，这时DAPO会重新rollout 20个新QA填充，而DUPO从剩余100个案例中随机复制20个方差非零的样本（例如复制高方差案例），填充空位。无需新rollout，直接复用现有数据。  

其他的做法大致上和DAPO类似。  

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
[DeepResearch的报告生成方法](https://www.linsight.cn/44c62dc5.html)  
[从RAG到DeepSearch](https://www.linsight.cn/7c2f9dcb.html)  
[Agent评测数据集](https://www.linsight.cn/72150a83.html)  
[Agent完全手册(零)：三大模块，三个理念](https://www.linsight.cn/b242bfb3.html)  
[agent调研(1)--MetaGPT,OpenManus和OWL](https://www.linsight.cn/226b059f.html)  
[Devin和Anthropic的Agent开发经验](https://www.linsight.cn/f93b3aaf.html)  
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
[LLM训练各种并行策略](https://www.linsight.cn/4cd8532f.html)  
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
- 论文阅读：  
[最近阅读--关于数据合成、agent、reasoning和多任务](https://www.linsight.cn/e96c7aac.html)  
[最近阅读2-关于自适应深度思考、context engineering和模型训练](https://www.linsight.cn/af7f9363.html)  
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

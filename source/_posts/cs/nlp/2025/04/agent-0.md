---
title: Agent完全手册(零)：三大模块，三个理念
tags:
  - NLP
  - LLM
  - Agent
categories:
  - CS
  - NLP
  - LLM
abbrlink: b242bfb3
date: 2025-04-11 22:44:15
---

【本文已在同名 微信公众号 / 知乎 / [个人博客linsight.cn](http://www.linsight.cn/) 上线】  

***  

打算把agent相关内容拉出来专门写一个系列，持续更新。  

作为第零篇，先看看Google的Agents白皮书和Anthropic开发者在agent开发上的一些经验。这里只整理一些关键的信息和自己的理解；更详细具体的内容可以看原文。  

# Google Agents whitepaper  

原文链接：[https://archive.org/details/google-ai-agents-whitepaper](https://archive.org/details/google-ai-agents-whitepaper)  

白皮书内容主要分三部分：  

- what is an agent，这部分是对agent的一个大致定义和理解  
- Tools，工具这块单独拉出来说，白皮书顺便也给G家的工具做了一下广告  
- 实践的例子，用LangChain和Vertex做的agent，Vertex也是G家的  

## what is an agent  

> In its most fundamental form, a Generative AI agent can be defined as an application that attempts to achieve a goal by observing the world and acting upon it using the tools that it has at its disposal.

现阶段，粗暴地说agent就是LLM + 工具构成的一个能和环境交互并完成任务的系统。  

一个典型的（Generative AI）agent可以分成三大块：  

- Model：系统的思考中枢  
- Tools：agent的“手脚”，提供专业的处理能力，以及和环境交互的桥梁  
- Orchestration：编排层，主责协调和管理系统中各个组件的交互和协同  

{% asset_img wp_agent.png agent_0 %}  

模型：现在可用的强大LLM已经很多，基本上开箱即用，不训练也是可以的。当然有些模型没有针对function call等工具调用能力做过优化，如果是涉及到较多function call，甚至有几百个上千个接口（这个数量已经足够构成一个单独的垂域了），那么针对场景进行一定的优化正常来说还是有收益的。  

工具：现在最火的就是MCP了，基本上也是开箱即用。如果有些私有工具，那么也可以按照MCP的方案套一层就行了。  

那么我们在开发的时候，操作空间比较大的应该就是在orchestration layer。  

> The orchestration layer describes a cyclical process that governs how the agent takes in information, performs some internal reasoning, and uses that reasoning to inform its next action or decision. 

简单来说编排层决定了模型获取信息和决策的方式。Orchestration中这个loop可长可短，具体取决于任务的难度和模型、工具的能力。  

对比一下model和agent，分别从知识范围、推理模式、工具能力和逻辑编排这几个维度来看：  

{% asset_img wp_model_agent.png agent_0 %}  

- 知识范围上，显然agent相比model有更动态&更加广泛的知识范围  
- 推理模式上，model一般是单次的，而agent能够和整个系统的其他模块进行多次交互  
- 工具能力上，agent能够调用外部工具完成任务  
- 逻辑编排上，agent有多种reasoning框架可以使用：CoT、ReAct等，当然model也能用，但是一般只能以prompt的形式单次线性地执行  

## Cognitive architectures: How agents operate  

Orchestration是cognitive architecture的核心，负责管理agent的记忆，状态，思考和规划。而记忆状态思考规划又都和prompt engineering息息相关。基于prompt来管理和规划的方法目前比较主流的方法有：  

- ReAct  
- CoT  
- ToT  
等  

当然PE发展很快，还有很多其他方法。利用这些方法，agent能够自主决定下一步应该干什么。  

prompt像是给agent提供了一个战略，在这个战略下，模型自发地根据当前情况设计具体战术。  

## 工具  

谷歌的框架里，模型能够与之交互的主要工具类型有三种：

- Extensions，扩展程序  
- Functions，函数  
- Data stores，数据存储  

Extensions对实际执行任务的API进行了一层封装，同时能够提供说明和样例，是模型和环境直接交互的桥梁。  

{% asset_img wp_extension.png agent_0 %}  

{% asset_img wp_extension2.png agent_0 %}  

而Functions和extensions相比，有两点主要的区别：  

- 1、模型给出function的调用命令和参数，但是并不实际执行  
- 2、extension的执行在agent-side，而function在client side  

{% asset_img wp_function.png agent_0 %}  

那么什么情况下会选择用function函数而不选择extension呢？举几个例子：  

- API的调用需要在程序的另一层进行，而不是agent架构  
- 安全或身份验证限制阻止agent直接调用API  
- API的调用涉及(人工)审核  
- 调用中包含额外定制的业务逻辑  

最后还有一个Data stores，通常是向量数据库之类的。  

几个工具的总结：  

{% asset_img wp_tools.png agent_0 %}  

## Enhancing model performance with targeted learning  

现在模型的通用能力都很强了，不过用到agent上，还是有可能出现一些场景和工具，超过了模型的预训练范围，那么就需要通过一些方法提升模型的领域能力。这些方法包括但不限于：  

- In-context learning：推理的时候加上few-shot example  
- Retrieval-based in-context learning：根据输入query，搜索一些最相关的工具和样例  
- Fine-tuning based learning：直接训练内化  

# Anthropic：Building effective agents  

Anthropic这篇博客介绍一些实际的agent经验，更加接地气一点。  

原文链接：[https://www.anthropic.com/engineering/building-effective-agents](https://www.anthropic.com/engineering/building-effective-agents)  

博客的作者还有一个相关的小演讲：[https://www.youtube.com/watch?v=D7_ipDqhtwk](https://www.youtube.com/watch?v=D7_ipDqhtwk)  

## workflows  

讲agent之前，先讲讲workflow。  

workflow我们日常已经用得很多了，写代码的逻辑，日常做饭的操作，都是workflow。这些工作一个特点就是任务相对是well-defined，大概有什么步骤，每个步骤干什么相对清晰，大不了加一个branch或者确定的loop。  

总结几种常用的workflow。  

1、Prompt Chaining  

{% asset_img an_prompt_chain.png agent_0 %}  

Prompt Chaining将任务分解为static的子step，前一步的输出作为下一步输入。  

Prompt Chaining适用于「任务可明确拆解」的场景（如先生成营销文案，再翻译），需通过切分子任务来降低单次LLM调用的复杂度的情况。  

2、Routing  

{% asset_img an_routing.png agent_0 %}  

Routing通过分类（LLM或传统算法）将输入导向不同的下游处理模块。  

Routing适用于「输入类型差异大且需专用处理」的场景，如客服问题分类：退款请求→财务工具，技术问题→知识库检索；或者「成本优化」的场景，如简单问题分到小模型如Haiku，复杂问题用Sonnet。  

3、Parallelization  

{% asset_img an_parallelization.png agent_0 %}  

Parallelization将任务同时下发给下游多路处理模块，并把多个下游模块的结果聚合起来，经过处理获得最终输出。  

Parallelization适用于「需加速处理或多样化视角」的场景，如多维度评估模型性能，又或者关键任务需冗余验证（如敏感内容过滤需多数表决）。  

Parallelization有两种变体：  

- Sectioning：独立子任务 & 执行（如内容生成与审核同步进行）  
- Voting：同一任务多次运行，聚合多次结果（如多LLM评审代码安全性）  

4、Orchestrator-Workers  

{% asset_img an_ow.png agent_0 %}  

Orchestrator动态分解任务，把子任务分配给Worker，最后合成结果。与Parallelization不同的是，这些子任务不是预先定义好的，而是根据输入动态生成的。这里其实已经有些agent的感觉。  

这种workflow适用于「复杂且不可预测的任务」的场景，如跨多文件代码修改，需动态分析依赖关系；或者「多源信息整合」，如研究任务需从不同数据库检索并交叉验证。  

5、Evaluator-Optimizer  

{% asset_img an_eo.png agent_0 %}  

Generator产出结果，evaluator提供feedback，并循环优化（类似人类写作的迭代修订）。  

Evaluator-Optimizer适用于「存在明确评估标准」的场景，比如翻译的语义保真度；
或者「需多轮改进」的场景，比如复杂搜索任务，evaluator根据已有搜索结果决定是否继续搜索。  

## Agents  

### workflow和agent  

LLM + workflow已经cover了很多我们日常使用的场景，比如手机上用语音助手操作闹钟就是一个LLM + workflow的流程。RAG也是一种workflow，先搜索后回答。这些使用LLM的workflow系统其内部的调度主要依赖人的设计。  

人设计的workflow相对固定，而agent则相对灵活。那是不是agent总是优于workflow，答案是no：  

> When building applications with LLMs, we recommend finding the simplest solution possible, and only increasing complexity when needed. This might mean not building agentic systems at all. Agentic systems often trade latency and cost for better task performance, and you should consider when this tradeoff makes sense.

灵活的调度，其背后是复杂度和成本的增加，对应的风险也有提升，如果业务没有那么复杂，那么单次的大模型调用基本可以满足需求，没有必要非要上agent。  

### 从workflow到agent  

一个典型的agent流程图：  

{% asset_img an_agent.png agent_0 %}  

和workflow相比，agent有这些特征：  

- 自主性：LLM自主规划步骤、调用工具，无需预先定义路径  
- 环境交互：常常需要依赖工具执行结果作为事实依据  

agent适用于「开放性」的问题，比如GitHub Issue的修复，期间需动态分析代码库。  
agent在执行过程中，需要做一些风险控制：  

- 设置停止条件（如最大迭代次数），防止agent陷入死循环，这个在目前阶段是比较常见的情况  
- 沙盒测试和监控工具调用（防错误累积）；正常来说agent会有至少两三次调用，期间有多次操作比如工具调用，这些可能需要结构化输出的场景模型还是有可能出错的，需要对格式和内容进行一定的修正；另外模型写代码是比较天马行空的，有可能写出移除路径之类的代码，这在生产环境风险很大，最好搞个沙盒环境，免得被删库  

### Agents in Practice  

Anthropic从agent的实践中总结了一些实际经验，给了两个适用agent的案例。  

1、案例一：Customer Support  

适用原因：天然对话流与工具调用结合（调取订单数据、触发退款等操作），可量化成功指标（如按解决率计费）。  

2、案例二：Coding Agents  

适用原因：结构化问题空间（代码可通过测试验证），自动化反馈驱动迭代（测试失败→重新修改代码）。

不过代码agent目前也还有一些问题，比如功能性验证≠系统兼容性，仍需人工审核（如架构设计一致性）。这个目前的复杂度还是比较难由agent自己完全handle。  

Anthropic用sonnet + agent搞了个优化SWE-bench效果的项目，地址在：[https://www.anthropic.com/engineering/swe-bench-sonnet](https://www.anthropic.com/engineering/swe-bench-sonnet)。  

另外还有一个让Claude操作电脑解决问题的project：[https://github.com/anthropics/anthropic-quickstarts/tree/main/computer-use-demo](https://github.com/anthropics/anthropic-quickstarts/tree/main/computer-use-demo)。  

### Prompt Engineering your Tools  

Anthropic给出一个重要经验：工具设计的质量直接影响agent效果，需要像设计人机交互（HCI）一样重视agent-computer接口（ACI）。在把工具给模型使用前，先让人类的使用者试下，看是否会在使用时遇到问题，比如参数的理解有没有问题，接口的设计是否合理，如果人类使用起来有困难，那么模型大概率也会遇到问题。  

在SWE-bench的project中，工具优化的耗时占了大部分开发时间。  

有几条具体的实践原则：  

- 格式优化：选择LLM易处理的格式，如避免JSON转义，优先Markdown；或者其他模型能在internet上看到的格式，保持阅读和生成的格式一致性  
- 绝对路径：SWE-bench代理中，用绝对路径替代相对路径能减少一些错误  
- 开发文档：包含示例、边界说明、常见错误，像给人类开发者用一样  
- 区分相似工具：比如"文件编辑"需明确是全量重写还是差分更新，并考虑哪个方法对模型更友好（比如查分更新，只修改文档的一部分，那么就需要模型具备补全的能力）

## 开发agent的一些建议  

博客作者的演讲给了三个agent开发中的观点。  

1、Don't build agents for everything  

agent不是workflow的升级版，不要把原有的workflow都替换成agent。workflow和agent虽然略有重叠，但是更主要的应该是互补的关系，合作的关系，不是上下级的关系。  

workflow可以处理确定性高的任务，而agent擅长处理模糊的问题。  

workflow成本更低，而agent可能会在一个任务消耗极其大量的token（百万甚至更多），所以要好好考虑成本问题（以及耗时）。  

{% asset_img an_checklist.png agent_0 %}  

2、Keep it simple  

> Agents are models using tools in a loop

Anthropic认为重要的agent组件：  

- environment  
- prompt  
- tools  

把精力放在优化这几个重要组件上，不要过度设计。  

3、Think like your agents  

agent可以说每次处理都是从零背景知识开始，模型能够看到的信息全都在prompt里了，所以请在prompt里把重要的相关信息都提供清楚，包括详细清晰的背景和任务描述，准确的工具和环境说明，还有详尽的历史记录和反馈。  

如果人类无法在模型的环境下工作（能看到的prompt，能操作的工具），那么模型效果不好也就可以理解了。因此记得「设身处地」，跟模型「换位思考」。  

# 小结  

- agent的三大模块：模型，工具，和调度（prompt + 相关配套）  
- agent开发三个理念：（1）不要拿着锤子看啥都是钉子，agent不是workflow的升级版，agent和workflow解决的问题是不一样的，不要到处套用（2）不要过度设计agent系统，先尝试从环境、工具和prompt入手优化（3）换位思考，从agent的角度出发思考，如果一个任务人难以完成，agent一样会遇到问题  

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

【1】Google Agent whitepaper，https://archive.org/details/google-ai-agents-whitepaper  
【2】Building effective agents，https://www.anthropic.com/engineering/building-effective-agents  
【3】Building effective agents作者演讲，https://www.youtube.com/watch?v=D7_ipDqhtwk  
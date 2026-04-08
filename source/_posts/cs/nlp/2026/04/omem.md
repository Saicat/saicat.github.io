---
title: O-MEM
tags:
  - NLP
  - LLM
  - Agent
categories:
  - CS
  - NLP
  - Agent
abbrlink: b3f8d798
date: 2026-03-31 20:27:42
---

# 简单的Chunk-Retrieve记忆系统

Agent在长时间，跨对话的交互中，就需要使用记忆系统来保存重要的历史信息，否则每次用户每次使用都相当于新认识一遍。  

记忆系统最简单的做法就是把所有历史交互都切成chunk，然后入库。当来了新的交互，就从数据库中检索相似/相关的历史交互。这种方式主要有两个问题：  

- 依赖交互内容的相似度/相关性，对于有逻辑联系的内容效果不佳  
- 简单粗暴的检索会引入很多“相似”内容，一般提升召回量能提升效果，但是也会引入很多噪音，记忆系统的整体效率就不高。  

# O-MEM：Active User Profiling

O-MEM的思路：记忆系统不应只是历史交互的存储容器，而应是用户特征的持续构建者。  

简单来说，O-MEM设计三个记忆系统的组件：  

- Persona Memory（人格记忆）：存储长期、抽象的用户知识，包括用户的稳定属性（如性格、偏好、身份特征）和关键事实事件（如职业变更、健康状况）。这是关于"用户是什么样的人"的总结性知识。  
- Working Memory（工作记忆）：存储与当前交互主题相关的所有历史交互。不同于生理学中短期工作记忆的概念，O-MEM中的Working Memory是主题关联的长期记忆库，用于提供当前话题的上下文背景。  
- Episodic Memory（情景记忆）：建立从显著线索词到具体交互情境的映射，类似于人类通过"关键词"触发完整回忆的能力。当用户提到"上次那个项目"时，Episodic Memory能通过"项目"这个线索快速定位到相关的完整对话记录。  

## 信息提取

对于每次交互，首先用LLM提取当前交互的：  

- 主题
- 当前交互揭示的用户属性
- 当前交互揭示的过往事件

比如用户的输入是：Attended an LGBT rally yesterday and received massive supports from the public. Transitioning has been a really difficult journey for me, and I am grateful for everyone's help in the past year.  

那么提取的信息：  

- 主题：LGBT Activity  
- 属性：Transgender  
- 事件：  
  - Attended an LGBT Rally Yesterday  
  - Underwent sex reassignment surgery  

这里论文用的是GPT-4.1 和 GPT-4o-mini作为提取模型。  

## Persona Memory

Persona Memory又包含两类：Persona Attributes（属性）和Persona Facts（事实）。  

两者都是以自然语言文本列表的形式存储。Attributes是用户稳定特征的抽象，Facts则是用户经历的具体事件记录：

- Persona Attributes = ["素食主义者"、"软件工程师"、"性格内向", ...]  
- Persona Facts = ["2023年9月完成手术", "去年辞职创业", ...]  

这两类信息都经过LLM的提炼，是高层次的结构化知识。  

### Persona Attributes的储存

用户输入提取出属性attr之后，会和现有的属性库通过相似度进行聚合，决定是更新现有数据库（update），新增属性（add），还是已有可忽略（ignore）。  

比如有五条属性：  

- 我喜欢周末打篮球  
- 篮球是我最热爱的运动  
- 每周六下午都打球  
- 我是素食主义者  
- 不吃肉类食物  

计算相似度发现前三条相似，后两条相似，就聚合成两条结果：  

- 用户是篮球爱好者，固定在周末进行该项运动  
- 用户是严格的素食主义者  

### Persona Attributes的检索

检索的时候，直接把用户输入和属性库中的数据计算相似度。  

比如query = 这周末有什么运动建议？

属性数据库有3条数据：  

- 用户是篮球爱好者，固定在周末进行该项运动  
- 用户是严格的素食主义者  
- 用户对机器学习有深入研究  

### Persona Facts的储存

同属性类似，对于用户输入提取到的事件，基于和现有数据的关系，有三种操作：  

- update  
- add
- ignore  

只是少了一步，不进行聚合，因为事件都是独立的。  

### Persona Facts的检索

检索的时候和属性检索一样，通过相似度召回。  

## Working Memory

### 储存

Working Memory储存结构式一个map，key是topic，也就是输入的时候提取的主题，value是原始交互数据。  

随着交互越来越多，每个value中储存的交互数据也越来越多。  

### 检索

当一个新的用户输入进来的时候，就会从Working Memory检索内容。  

具体来说，就是通过语义相似度，从Working Memory中找到所有和当前交互的话题相似的话题，然后获取对应的交互数据返回。  

## Episodic Memory

### 储存

Episodic Memory的数据储存是一个map。其中key是词，来自于用户输入的分词结果，而value则是原始的交互。  

具体来说，用户的输入会被分词，每个词会作为key，然后去找Episodic Memory里对应的key，把原始交互数据加到value里，这是一个增量更新。  

### 检索

要从Episodic Memory检索的时候，也是用关键词作为key，找到对应的原始交互。  

关键点在于选择什么词来搜索。原文的方法是参考inverse document frequency的思路，从用户的输入里找到最关键的一次词用于搜索。  

# Reference

【1】O-MEM: OMNI MEMORY SYSTEM FOR PERSONALIZED, LONG HORIZON, SELF-EVOLVING AGENTS，https://openreview.net/pdf?id=K3bOz7oYec

【推荐文章】  
- Agent：  
[Harness Engineer](http://localhost:4000/24a6683a.html)  
[字节的M3-Agent](https://www.linsight.cn/5f0d3c82.html)  
[DeepResearch的报告生成方法](https://www.linsight.cn/44c62dc5.html)  
[从RAG到DeepSearch](https://www.linsight.cn/7c2f9dcb.html)  
[阿里通义Lab: WebWalker,WebDancer和WebSailor](https://www.linsight.cn/f7d600f3.html)  
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
[VeRA，LoRA-XS和TinyLoRA](https://www.linsight.cn/cc1c31d.html)  
[腾讯的Training-Free GRPO](https://www.linsight.cn/9cb56255.html)  
[深度求索DeepSeek-R1详解](https://www.linsight.cn/9e4b4e6d.html)  
[基模型Cognitive Behaviors对RL的影响](https://www.linsight.cn/657a6d17.html)  
[Llama3.1--post-training要点一览](https://www.linsight.cn/93328a2a.html)  
[模型平均 -- model soup](https://www.linsight.cn/bb8fcf21.html)  
[大模型偏好对齐-DPO](http://www.linsight.cn/473f2b43.html)  
[大模型偏好对齐-ODPO](http://www.linsight.cn/da871ebe.html)  
[大模型偏好对齐-simPO](http://www.linsight.cn/280fa97a.html)  
[大模型偏好对齐-IPO](http://www.linsight.cn/4fe7b810.html)  
- Transformer：  
[Attention Residuals](https://www.linsight.cn/5b81d487.html)  
[理解Attention:从起源到MHA,MQA和GQA](http://www.linsight.cn/3dc22f96.html)  
[LLM的重复生成和ICL](https://www.linsight.cn/7381cae3.html)  
[transformer中normalization的二三事](http://www.linsight.cn/6a40bfa5.html)  
[从代码实现看normalization-到底做了什么](http://www.linsight.cn/b70b4a2d.html)  
[稀疏注意力计算:sliding window attention](http://www.linsight.cn/c61d17e3.html)  
[理解LLM位置编码:RoPE](http://www.linsight.cn/a051710f.html)  
[RoPE的远距离衰减](https://www.linsight.cn/f0902f1a.html)  
[LLM水印](https://www.linsight.cn/2dee4921.html)  
- 训练框架  
[Muon优化器](https://www.linsight.cn/f25d614e.html)  
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

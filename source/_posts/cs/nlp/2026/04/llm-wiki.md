---
title: Karpathy所说的LLM Wiki
tags:
  - NLP
  - LLM
  - Agent
categories:
  - CS
  - NLP
  - Agent
hidden: false
abbrlink: 2d60084c
date: 2026-04-07 22:23:13
---

这AI圈的新概念真是比新概念英语的新概念还多。  

Andrej Karpathy最近又提了一个概念，叫LLM Wiki。这是一个使用LLM构建个人知识库的方法，Karpathy在他的post开篇就说了，这是一个high level idea，每个人在操作层面都可以有一些自己的具体设计。  

大致上来说，LLM Wiki是一套人与LLM/Agent（用Claude Code就行）协作，逐步构建个人知识库的“动态过程”。为什么强调动态过程，因为LLM Wiki不是一次构建就固化使用的概念，而是不断迭代，持续更新的方案。  

比如说，我现在要系统学习LLM，那我首先要去搜索一些论文，学习一些概念，形成体系。在这个过程中，一个经典的做法可以把这些文献构造成一个数据库，然后借用RAG的能力来学习，比如用腾讯的IMA。  

RAG确实可以解决很多问题，但是也存在一些约束。比如说我要学习模型架构的发展，一开始怎么在RNN里开始加上attention，然后出了Transformer，之后有BERT和GPT，期间Pre-Norm和Post-Norm如何对比，MHA怎么发展到MQA、GQA和MLA，MoE又是什么起源，位置编码怎么从绝对迁移到相对。这些概念都交织到一起，如果想获得这些概念的全景图，或者顺着一个概念一直联系相关概念学习下去，仅用RAG可能就没法获得很好的效果。  

RAG为什么处理不好这些问题呢？因为离散式的搜索召回，本身就很难把这些相关概念正确地组织起来。可能有人说，不会啊，模型自己就能知道这些概念都和模型结构相关，因此搜索的时候可以通过生成覆盖面足够大的query list来召回。且不说模型能不能在规划搜索的时候做到完善，如果面对的概念不是预训练的数据库中已经较好学习过的，比如是最新最前沿的科学概念，那模型明显就缺失了相关背景知识，从而难以覆盖应有的概念。  

另外，即使搜索到了足够多的信息，要从离散的片段里，正确地梳理好这些知识，本身就不容易。再者，问过的问题好不容易用了很多知识回答好了，但是下次再问又得从头做了。  

那么能不能把这些知识联系和组织的工作提前做好，并且还能不断更新呢。诶，LLM Wiki就是为此而生的。与其在查询时从原始文档中检索，LLM Wiki增量式地构建和维护一个持久的wiki——一个结构化的、相互链接的 markdown 文件集合。当你添加新资料时，LLM Wiki不只是为后续检索建立索引。它会阅读资料，提取关键信息，并将其整合到现有wiki中——更新实体页面，修订主题摘要，标注新数据与旧数据的矛盾。知识被重新整合，保持最新，而不是在每次查询时重新推导。  

爽的是，这一切操作都不用自己动手，而是让LLM来干就行了。用户只需要把操作的原则设定好就行。  

具体来说LLM Wiki的架构有三部分：  

- 原始资料：源文档集合，文章、论文、图片、数据文件。这些是不可变的，LLM 从中读取但从不修改它们。这是你的真相来源。  
- Wiki：LLM 生成的 markdown 文件目录。摘要、实体页面、概念页面、比较、概览、综合等。LLM 完全拥有这一层的权限。  
- Schema：一个文档（例如 Claude Code 的 CLAUDE.md 或 Codex 的 AGENTS.md），告诉 LLM wiki 如何结构化，约定是什么，以及在摄入资料、回答问题或维护 wiki 时要遵循什么工作流程。这是关键配置文件——它让 LLM 成为一个有纪律的 wiki 维护者，而不是一个通用聊天机器人。你和 LLM 随时间共同演进这个文件。  

一个schema的样例：  

````
# LLM Wiki Schema

## 项目结构
- `raw/` — 不可变的源文档。严禁修改。
- `wiki/` — LLM 生成的 Wiki。你拥有其完全所有权。
- `wiki/index.md` — 主目录。每次摄取时更新。
- `wiki/log.md` — 仅限追加的活动日志。

## 页面规范
每个 Wiki 页面必须包含 YAML frontmatter：
```
---
title: Page Title
type: concept | entity | source-summary | comparison
sources: [引用的 raw/ 文件列表]
related: [链接的 Wiki 页面列表]
created: YYYY-MM-DD
updated: YYYY-MM-DD
confidence: high | medium | low
---
```

## 摄取工作流
当我说 "ingest [文件名]" 时：
1. 读取 raw/ 中的源文件
2. 与我讨论核心要点
3. 在 wiki/sources/ 中创建/更新摘要页面
4. 更新 wiki/index.md
5. 更新所有相关的概念和实体页面
6. 在 wiki/log.md 中追加一条记录

## 查询工作流
当我提问时：
1. 阅读 wiki/index.md 以查找相关页面
2. 阅读这些页面
3. 综合答案并附上 [[wiki-link]] 引用
4. 如果答案有价值，提议将其归档为
   一个新的 wiki 页面

## Lint 工作流
当我说 "lint" 时：
1. 检查页面之间是否存在矛盾
2. 查找没有入站链接的孤立页面
3. 列出已提及但缺少独立页面的概念
4. 检查被新来源取代的陈旧主张
5. 建议下一步要调研的问题
````

一开始收集资料，就让LLM/Agent按照schema的要求对所有资料进行处理，后续再进来新的资料，也同样操作。  

所构建成的wiki directory样例：  

```
wiki/
  index.md                 # Master catalog of all pages
  log.md                  # Chronological activity record
  overview.md             # High-level synthesis
  concepts/
    attention-mechanism.md
    mixture-of-experts.md
    scaling-laws.md
    tokenization.md
  entities/
    openai.md
    anthropic.md
    google-deepmind.md
  sources/
    summary-attention-revisited.md
    summary-scaling-update.md
  comparisons/
    gpt4-vs-claude-vs-gemini.md
    rag-vs-finetuning.md
```

schema里的raw和wiki就是三部分架构里的另外两个。schema里还提到index.md和log.md两个部分。log.md很好理解，就是wiki的变化日志，用于追溯每个变化都是哪个文献引起的。  

index.md则是wiki中所有内容的目录。有query的时候，LLM/Agent首先就看index.md，然后找相关的内容。index.md长这样：  

```
# Wiki Index

## Concepts
- [[attention-mechanism]] — 自注意力、多头
  注意力及其变体 (12 个来源)
- [[mixture-of-experts]] — 稀疏 MoE 架构，
  路由策略 (8 个来源)
- [[scaling-laws]] — Chinchilla、Kaplan 定律，
  计算最优训练 (15 个来源)
- [[tokenization]] — BPE、SentencePiece、tiktoken
  (3 个来源)

## 实体
- [[openai]] — GPT 系列，组织历史
  (20 个来源)
- [[anthropic]] — Claude 系列，宪法 AI
  (14 个来源)
- [[google-deepmind]] — Gemini, PaLM, AlphaFold
  (18 个来源)

## 来源摘要
- [[summary-attention-revisited]] — 2026-03-15
- [[summary-moe-efficiency]] — 2026-04-01
- [[summary-scaling-update]] — 2026-04-02

## 对比
- [[moe-routing-strategies]] — 归档自查询 2026-04-04
- [[rag-vs-finetuning]] — 权衡与使用场景
```

在操作上，Karpathy 定义了三个核心操作。  

1、Ingest  

这是最重要的操作。单次 ingest 不仅仅是创建一个新页面 — 它会波及整个 wiki。如果你 ingest 了一篇关于新 transformer 变体的论文，LLM 可能会：  

- 为该论文创建一个新的摘要页面  
- 使用新变体更新 “Attention Mechanism” 概念页面  
- 如果论文包含新的基准测试，则更新 “Scaling Laws” 页面  
- 更新论文作者或其所属机构的实体页面  
- 如果论文针对已知模型进行了基准测试，则更新对比页面  
- 从引用新内容的现有页面中添加链接  
- 使用新页面更新索引  
- 在log中记录导入 (ingest)  

示例：  

```
# 你将一篇新文章放入 raw/ 目录

> 我刚刚在 raw/articles/ 中添加了一篇新文章
>   2026-04-mixture-of-experts-efficiency.md
> 请导入它。

# LLM 读取它，然后回复：

本文的核心要点：
1. MoE 模型在相同质量下实现了 4 倍的吞吐量
2. 路由效率比 Switch Transformer 提高了 23%
3. 与我们 wiki 中关于稠密模型 (dense models) 的说法相矛盾：
   即在 10B 参数以下总是更高效

我将执行以下操作：
- 创建 wiki/sources/summary-moe-efficiency-2026.md
- 更新 wiki/concepts/mixture-of-experts.md
- 更新 wiki/concepts/scaling-laws.md (新增基准测试)
- 在 wiki/concepts/dense-vs-sparse.md 中标记矛盾点
- 更新 wiki/index.md
- 追加到 wiki/log.md
```

2、query  

直接对LLM/Agent提问就行，但是有一个关键点：优秀的答案可以作为新页面归档回 wiki 中。  

示例：  

```
> 对比我们所有来源中
> MoE 模型的路由策略。哪种方法具有最佳的效率/
> 质量权衡？

# LLM 读取 wiki/concepts/mixture-of-experts.md，
# wiki/sources/summary-moe-*.md，并生成分析

[LLM 生成带有表格的详细对比]

> 太棒了。将其归档为 wiki 页面。

# LLM 创建 wiki/comparisons/moe-routing-strategies.md
# 更新 index.md，从相关页面添加链接
```

3、Lint  

定期让 LLM 对 wiki 进行健康检查。比如看有没有矛盾的信息、孤立的信息、缺失的信息。  

Karpathy提出了一些检查类型：  

```
> Lint the wiki.

Wiki Health Report (2026-04-04):

CONTRADICTIONS (2):
- concepts/dense-vs-sparse.md claims dense > sparse
  below 10B, but sources/summary-moe-efficiency.md
  shows opposite. Recommend: update with nuance.
- entities/openai.md says GPT-5 is 200B params,
  but sources/summary-gpt5-leak.md says 300B.

ORPHAN PAGES (3):
- concepts/tokenization.md (no inbound links)
- sources/summary-old-bert-paper.md (no references)
- comparisons/old-gpu-benchmark.md (outdated)

MISSING PAGES (4):
- "RLHF" mentioned 12 times, no concept page
- "Constitutional AI" mentioned 8 times, no page
- "KV Cache" referenced in 5 sources, no page
- "Speculative Decoding" mentioned 3 times, no page

SUGGESTED INVESTIGATIONS:
- No sources on inference optimization post-2025
- Entity page for Meta AI is thin (only 1 source)
- The "Scaling Laws" page hasn't been updated in 3 weeks
```

LLM Wiki大致的构建和维护思路就是这样。schema的设计，directory的设计也仅供参考，用户完全可以自定适合自己的一套。  

Karpathy提到他是用Obsidian来做文档转换和链接可视化的，并且最好一点一点构建数据库，这样可以边构建边学习。  

# Reference

【1】https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f  
【2】Karpathy's LLM Wiki：其 Idea File 全方位指南，https://antigravity.codes/zh/blog/karpathy-llm-wiki-idea-file  

---

【推荐文章】  
- Agent：  
[Harness Engineer](https://www.linsight.cn/24a6683a.html)  
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
[预训练经验](https://www.linsight.cn/e996bf25.html)  
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

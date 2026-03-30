---
title: Harness Engineer
tags:
  - NLP
  - LLM
  - Agent
categories:
  - CS
  - NLP
  - Agent
abbrlink: 24a6683a
date: 2026-03-23 21:04:43
hidden: false
---

# 新概念

最近都在聊一个概念，harness engineer。  

查了下，最早是有个老哥在博客[《My AI Adoption Journey》](https://mitchellh.com/writing/my-ai-adoption-journey)里提到，大致是说chatbot已经不能满足工作的需求，agent才是适合形态。而要让agent听话，就要harness engineer。  

然后是OpenAI发布了实验报告[《工程技术：在智能体优先的世界中利用 Codex》](https://openai.com/zh-Hans-CN/index/harness-engineering/)，介绍了内部一个纯用AI完成百万级项目代码编写的实验，就用了harness engineer来描述这个开发的方法。  

# 从Prompt Engineer到Harness Engineer

Prompt Engineer -> Context Engineer -> Harness Engineer  

## Prompt Engineer

在LLM的使用上，最早有prompt engineer，针对的情况是如何让模型能够更好地遵循输入，进行单次的任务执行。这时关注的主要还是语言格式，表达的结构，以及调试输出的技巧（比如lost in the middle，prompt重复等）。  

## Context Engineer

然后大概在2025年，就有了context engineer。因为随着任务的复杂程度增加，比如deepresearch任务，agentic search或者更复杂的形式，上下文的长度越来越长。这时怎么在整个流程里组织信息，压缩历史，提取关键内容，甚至确保KV cache的命中以提升效率，防止上下文腐坏/退化等，就很重要，因此就需要更多地engineering来处理context。  

context engineering是一个把prompt engineering涵盖在内的大概念。除了写prompt，在这些任务里我们往往还要搜索信息、读取大文件，调用工具获取进行操作，更新环境结果。这个过程要处理大量的信息，也就是大量的上下文context。按照这些信息的功能，context可以分成多类，比如：  

- guiding context：规定模型行为准则的部分，比如system prompt。跟prompt engineering最为相关的部分。  
- information context：外部知识，比如从RAG搜索到的内容，也包括用户画像（agent记忆）等，和RAG跟为相关。  
- actionable context：比如tool description，以及工具调用的轨迹、结果等。  

(from [《Context Engineering，一篇就够了。》](https://zhuanlan.zhihu.com/p/1938967453951571269))  

context engineering的目的就是设计合理的规则和流程，协调好各种信息输入给模型的方案。  

- 当搜索结果太多时，要怎么简化？可以做摘要压缩，可以分层展示，也可以分批处理。  
- 当要跨长上下文进行对话时，可能还需要一个专门设计的记忆系统。现在很多对话agent都有做记忆系统，比如问一下豆包“我是什么样的人”，你就可以看到豆包的记忆系统记了你的什么信息。广义来说，这个记忆系统也算是context engineer的。  
- 任务复杂时，agent往往需要拆解出子任务来。那么这些子任务之间怎么排布，信息怎么流通和交互，就更是重要的一环。直接都塞给main agent是一种方法，但是这样可能倒是长度太长，成本过高，或者效果不好，或者细节太多无法处理。  

之前在[《最近阅读2-关于自适应深度思考、context engineering和模型训练》](https://www.linsight.cn/af7f9363.html#factor-agents)里也分享过一些context engineer的设计建议：  

- 结构化  
- 无状态  
- agen设计：小而精 > 大而全  
...  

coding agent，比如claude code和cursor，也包含了很多context engineer。  

## Harness Engineer

为什么说那么多context engineer？因为其实harness engineer也还是在context engineer的概念范畴内，但是在实现上更工程更细致更具体了。  

大体来说，context engineer是一个很通用的概念，只要有上下文，就需要context engineer。而且之前更多关注在面向普通用户的场景。比如做一份研报，做一份PPT，做一个旅游规划，通过对话做一个小游戏等。这些任务可能上下文也很长，但是大部分的处理逻辑其实通用的，或者对于模型来说，标准是相对宽松的。而如果放到做一个商业级的软件项目，所要做的事情则“工程”得多，也细致得多：开发者需要用工程语言准确描述自己的需求，给出详细的设计文档，技术框架，任务拆解，每一个模块的输入输出，依赖关系，技术选型。当然这些事也可以用AI来做，但是归根到底，需要人来确认这些“脚手架”。只有合理的脚手架才能长出高质量的AI实现。  

我认为harness engineer其实就是context engineer在各个垂直领域专业化的结果。比如说软件工程，这是AI开发者最熟悉的领域，因此也最早被探索。为了在软件开发的过程更好更高效地使用AI，开发人员把人类的经验应用到了AI coding里，让AI能够在一个好的环境下进行开发。  

以后可能会发展成每个领域有自己一套harness engineer的框架。比如一个法律案件怎么来分析和准备，那么有律师已经把经验和规则都定好，AI能够自己按照sop去搜索资料，分析结果，对比条文，撰写文书，甚至能造一个agent进行模拟辩论。  

### OpenAI的经验

总结一下OpenAI在软件开发工程上的harness engineer经验[https://openai.com/zh-Hans-CN/index/harness-engineering/](https://openai.com/zh-Hans-CN/index/harness-engineering/)：  

- 范式变更：人类掌舵，AI执行。工程师主要工作不再是编写代码，而是设计环境、明确意图和构建反馈回路。这与 Chad Fowler 提出的 "Relocating Rigor"（重新定位严谨性） 呼应——严谨性不再主要体现在手动编写每行代码的谨慎，而是体现在设计约束系统、反馈机制和控制环境上。  
- 任务拆解：将更大的目标拆解为更小的构建模块。不要对AI说"给我做个电商平台"这种大黑盒任务。要像流水线一样拆开：先设计数据库表结构 → 再写用户登录API → 再写登录页面组件 → ...。每完成一小块，确认没问题，再解锁下一块。  
- 知识管理：给AI"导航地图"，别给"百科全书"，要做知识分层。不要把内容全写到AGENTS.md里，AI看不过来，而且这周写的规则可能下周就过时了。正确做法：所有文档放在docs/目录里，有质量评分和验证状态（是否过期）。还有专门的"doc-gardening"智能体，定期扫描哪些文档和代码对不上了，自动发起修复。AGENTS.md：  
  - 需要并发控制？→ 看 docs/architecture/concurrent.md  
  - 需要数据验证？→ 看 docs/standards/zod-validation.md  
  - ...  
- 可观测性：让环境对AI可读。比如接入Chrome DevTools，AI能自己截图、看DOM元素，发现"登录按钮点不了"，或者提供LogQL（查日志）和PromQL（查指标），AI能验证"服务启动是否<800ms"。这样AI就能自己闭环修bug，不用人每次来帮忙看修好了没。  
- 架构约束：给AI画好车道线，把开发的要求和规范用可明确检测的方式实现，比如写linter规则，如果UI层的代码文件试图import数据库驱动，直接报错，而不是仅仅在规则中要求。这是一种权衡，放弃"生成任何东西"的灵活性，换取可预测性和可靠性。  
- 技术债务：人类高频（每天）跟踪AI是否犯错，之前给定的规则是否有漏洞，免得积累太多技术债。  
- 反馈分层：小感冒AI自己吃药，大病人类建医院。测试挂了、编译错了、指标不达标——AI自己读错误日志、查监控、改代码、重试，自己迭代几十次直到成功。而如果AI在某类问题上反复卡壳（比如总是无法通过某项安全扫描），说明系统缺能力，需要人类接入，帮助诊断问题所在，增加工具，或者优化文档等。不是让人类和AI结对编程，而是"人类设计控制回路，AI在回路里跑，人类只在回路失效时修回路"。  
- 代码审查：从人审AI，到AI审AI。项目的规则完善之后，人类的投入减少，90%以上的PR由AI审AI，人类只看关键架构决策。  
- 现实限制：不要以为随便一个项目扔给AI就能自动跑。需要大量的尝试和经验。Martin Fowler 提醒，不应幻想"下一代模型会自动解决可维护性问题"。OpenAI 的实践表明，即使拥有最先进的模型，仍需大量工程工作构建确定性的 Harness 工具。[https://martinfowler.com/articles/exploring-gen-ai/harness-engineering.html](https://martinfowler.com/articles/exploring-gen-ai/harness-engineering.html)  

新旧世界的分化 -- 假设 Harness 技术成熟，能把 AI 自主性调到最高档：  

- 新建应用：从第一天就按 Harness 设计，享受高自主性维护  
- 遗留系统： retrofitting（加装）Harness 可能成本过高，特别是当代码库充满技术债、缺乏标准化时（就像在从未跑过静态分析的代码库上第一次启用，会被警报淹没）  
- 结果：可能出现"前 AI 时代"与"后 AI 时代"应用维护的两极分化。  

### Ralph Loop

看资料的过程中，看到有Ralph Loop这个东西。来源是《Everything is a Ralph Loop》。据说这篇文章的作者角色代码已死，现在已经去澳洲放羊了，也不知道真假。  

Ralph Loop大致的思路是说，只要给定好任务，定好目标，定好测试，那么剩下的就是让AI不断地循环开发，直到实现目标就可以了。  

Ralph 取自《辛普森一家》中的 Ralph Wiggum——笨拙、易错但永不放弃的角色。这隐喻了该方法的哲学：接受 AI 会不断失败，但通过暴力循环让其最终收敛到正确结果。  

具体实现上，有一些技巧。  

Ralph Loop 通过状态外化实现跨实例持久化，避免依赖 AI 记忆（Context Rot）：  

| 文件                    | 角色             | 关键细节                                                                  |
| --------------------- | -------------- | --------------------------------------------------------------------- |
| **PRD.md / prd.json** | **契约（What）**   | 人类编写的结构化任务清单，包含用户故事、验收标准、依赖关系。AI 只更新 `passes: true/false` 状态，不修改规格本身。 |
| **PROMPT.md**         | **操作手册（How）**  | 模板化指令，告诉 AI 如何读取 PRD、选择任务、运行测试、提交代码、何时标记完成。                           |
| **progress.txt**      | **记忆（Memory）** | 追加式日志，记录失败模式与成功经验。每个全新 AI 实例通过它了解历史尝试，但需人工定期审查以防止错误累积。                |

文件关系流程：  

- AI 实例启动 → 读取 PROMPT.md 获得行为指令  
- 按指令读取 PRD → 找到最高优先级且 passes: false 的任务  
- 实现任务 → 更新 PRD 状态 → 追加 progress.txt  
- 提交代码（git commit）→ 进程退出  
- 循环重启，全新 AI 实例重复上述流程  

另外还有个叫上下文高压锅（Context Pressure Cooker）的概念，这是 Ralph Loop 对抗"上下文腐烂"（Context Rot）的核心机制：  

- 机制：每次迭代都重新加载完整的 PRD，而非依赖 AI 的记忆  
- 目的：强制 AI 在"高压"下面对自己的错误，防止长会话中的遗忘与幻觉  
- 效果：通过"故意低效"的上下文分配，确保早期约束不被后期迭代遗忘  

Ralph Loop的执行模式是严格的多实例隔离。每个任务由完全独立的 AI 实例处理，实例间完全隔离。  

```
迭代 1: 全新 AI 实例 → 实现任务 A → 提交 → 写入 progress → 进程结束
迭代 2: 全新 AI 实例 → 读取 PRD + progress → 实现任务 B → 提交 → 进程结束
```  

每次迭代都是全新 CLI 进程（甚至 Docker 容器），上下文窗口清零。完全通过 Git 历史 + 文件系统，而非 AI 内部记忆。失败任务被记录到 progress，下次由全新实例用新策略重试，避免在同一会话中陷入死循环。  

每个任务必须小到能在一次上下文窗口内完成（约 15 分钟内），且修改文件数 ≤ 3 个。  

Ralph Loop可以说是Spec Coding的暴力版了。总体来说，Ralph Loop比较适合小型的项目。  

（不过具体的实现反而没那么重要，中心思想是，只要环境合适，AI就能给你干完了。）  

# Reference  

【1】everything is a ralph loop，https://ghuntley.com/loop/  
【2】工程技术：在智能体优先的世界中利用 Codex，https://openai.com/zh-Hans-CN/index/harness-engineering/  
【3】Harness Engineering 深度解析：AI Agent 时代的工程范式革命，https://zhuanlan.zhihu.com/p/2014014859164026634  
【4】Harness Engineering，https://martinfowler.com/articles/exploring-gen-ai/harness-engineering.html  
【5】A Guide to Which AI to Use in the Agentic Era，https://www.oneusefulthing.org/p/a-guide-to-which-ai-to-use-in-the  
【6】My AI Adoption Journey，https://mitchellh.com/writing/my-ai-adoption-journey  
【7】Context Engineering，一篇就够了，https://zhuanlan.zhihu.com/p/1938967453951571269  
【8】最近阅读2-关于自适应深度思考、context engineering和模型训练
，https://www.linsight.cn/af7f9363.html#factor-agents  
【9】我的Harness Engineering实践心得，https://zhuanlan.zhihu.com/p/2020528036827636560  

【推荐文章】  
- Agent：  
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

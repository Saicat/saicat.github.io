---
title: Devin和Anthropic的Agent开发经验
tags:
  - NLP
  - LLM
  - Agent
categories:
  - CS
  - NLP
  - Agent
abbrlink: f93b3aaf
date: 2025-06-17 20:25:18
---

看下最近两篇关于Agent开发经验的文章。  

# Don’t Build Multi-Agents  

来自Devin开发团队：[https://cognition.ai/blog/dont-build-multi-agents#a-theory-of-building-long-running-agents](https://cognition.ai/blog/dont-build-multi-agents#a-theory-of-building-long-running-agents)  

引入了Context Engineering的概念。  

一般的agent设计流程是这样的：planner把输入task拆分成sub-task，然后分配给不同的sub-agent「独立」执行。这种并行方案可能出现的问题是，各个sub-agent在处理不同的任务时，理解可能有偏差，处理任务的风格也不同，从而可能导致最终sub-task的结果不能合理地合在一起。  

比如一个做小游戏的任务，做背景的sub-agent做了个超级玛丽的背景，但是设计人物的sub-agent搞了个黑悟空的形象设计，这就不统一。  

要缓解这个问题，有人会把原始prompt也带给子agent，希望能提供更多一致信息。这种信息共享的方式有效，但还不够好，因为信息共享得太少。  

Prompt Engineering → Context Engineering：把原始的planner信息，包括原始输入query，planner的对话、思考和action（都是context）等，都带给各个sub-agent，以尽量提供更多信息。  

但这仍然无法完全解决问题。更进一步的方法是，各个sub-agent不要并行处理任务，而是串行处理。上一个agent的处理结果会传给下一个agent。  

这又会引起另一个问题，随着链路上的agent增加，context越来越多，模型逐渐处理不了了。一个优化方法是使用一个小模型来压缩context(this is in fact something we’ve done at Cognition)。  

{% asset_img devin.png agent经验 %}  

# How we built our multi-agent research system  

来自Anthropic：[https://www.anthropic.com/engineering/built-multi-agent-research-system](https://www.anthropic.com/engineering/built-multi-agent-research-system)  

1、MAS的优势  

Research类的工作不确定性很大，需要case by case设计方案，甚至step by step调整。这种unpredictability就很适合Agent来处理。  

单个人的智慧是有上限的，而集体的智慧更好scale，就像人类社会，集合在一起分工协作，进步的速度就比每个人单干快很多。MAS > sing agent。  

The essence of search is compression: distilling insights from a vast corpus. Sub-agent可以从不同的角度预先做好这种compression，然后把最重要的token给到lead research agent。实践上，MAS特别擅长广度优先搜索，每个agent关注不同的方向，比单agent系统的效率更高。通过给多个agent分配资源，MAS的推理能力也得到了扩展。但是这样的缺点也很明显：token消耗很快，MAS是普通对话交互的15倍。所以，MAS适合用来处理「价值足够覆盖成本」的任务。另外需要所有agent共享信息，或者有复杂依赖关系的任务并不适合MAS(这和《Don’t Build Multi-Agents》对上了)，比如多数coding任务（上下文强烈依赖）。  

适合MAS是三种场景：（1）子任务高度可并行的（2）信息量超出single context window的（3）有大量复杂工具调用的。  

MAS架构：  

{% asset_img anthropic.png agent经验 %}  

2、Research Agents的Prompt Engineering经验  

- Think like your agents：设身处地，观察agent在prompt下的行为  
- Teach the orchestrator how to delegate：分配的任务要详细，目标明确，可执行  
- Scale effort to query complexity：定一些资源分配规则，难度越大的任务，分配的资源越多，反之越少，别让agent杀鸡用牛刀  
- 工具设计和选择：不用说了，重中之重  
- Let agents improve themselves：这点有意思，把badcase给模型看，它能提出合理的方案，比如重写prompt和工具描述  
- 搜索策略：学习人的方法，先广度后聚焦  
- Guide the thinking process：用深度思考模型，不用说了  
- 并行起来：sub-agent并行，多工具并行  

一个大原则：prompt engineering注重high-level的策略启发，而不是硬性的规则（prompt要写抽象点）。最佳prompt并非严格指令，而是定义分工架构、解题方法和资源预算的协作框架。参考：（开源Cookbook中的提示词样例）Research MSA的prompt：[https://github.com/anthropics/anthropic-cookbook/blob/main/patterns/agents/prompts/research_lead_agent.md](https://github.com/anthropics/anthropic-cookbook/blob/main/patterns/agents/prompts/research_lead_agent.md)。  

Anthropic提供的样子有三个agent（整体结构看上图）：  

- （1）citations_agent（主要用来加citation）  
- （2）research_lead_agent（相当于manager）  
- （3）research_subagent（内部也有单独的逻辑）  

research_lead_agent：  

```
#### 一、**任务指导框架：结构化研究流程**
**核心机制**：通过四阶段流程（评估分解→query分类→计划制定→执行监控）系统化处理query。

1. **评估分解阶段**  
   - 要求agent将用户query拆解为：核心概念、关键实体、所需数据点、时间/上下文限制（例：分析"2025年AI金融agent最佳方案"时需识别"AI技术趋势"、"金融监管环境"等实体）  
   - 明确用户深层需求：通过反问"用户最关心什么？期望的最终形式是详细报告还是对比分析？"（例：当query要求"比较欧盟国家税制"时，需预判用户需要可视化对比图表）

2. **查询类型判断**  
   - **Depth-first**：单主题多视角（例："肥胖症成因"需基因/环境/心理/社会经济多维度分析）  
   - **Breadth-first**：多子问题并行（例："财富500强CEO信息收集"需按公司分段委托子agent）  
   - **Straightforward**：单线程解决（例："东京当前人口"只需一个子agent验证权威数据源）

3. **计划制定原则**  
   - 深度优先需定义3-5种方法论（例：分析"2008金融危机原因"时部署经济模型/监管漏洞/行为金融学三个子agent）  
   - 广度优先强调子任务边界清晰化（例："前端框架对比"需严格划分React/Vue/Angle的评估维度）  
   - 每个步骤需通过必要性测试："该步骤能否拆分？是否需要多视角？输出形式是否明确？"

---

#### 二、**行文约束机制**
**核心目标**：确保输出专业、高效、无冗余。

1. **子agent指令规范**  
   - 必须包含：单一研究目标、输出格式定义、可靠来源白名单（例：半导体供应链分析需指定TSMC财报/SEMI行业报告为优先来源）  
   - 禁止任务重叠：每个子agent需有独特研究领域（例：欧盟税制比较时，北欧/西欧子agent不得重复研究同一国家）

2. **最终报告控制**  
   - 严格禁止Markdown引用：由独立agent专门处理文献引用  
   - 强制使用`complete_task`工具提交报告，禁止子agent参与最终撰写  
   - 信息密度要求：在Slack/Asana工具集成场景中，需明确指导子agent使用`slack_search`等内部工具（例：用户任务涉及内部文档时，需创建专属Google Drive子agent）

3. **伦理约束**  
   - 禁止部署可能产生有害内容的子agent（例：涉及种族/暴力等敏感query时立即终止研究）  
   - 数据验证机制：对关键数值/日期进行多源交叉验证（例：CEO年龄信息需对比公司年报/LinkedIn资料）

---

#### 三、**流程控制策略**
**核心方法**：动态资源分配与效率优化。

1. **子agent数量控制**  
   - 复杂度分级机制：  
     - 简单查询（1个子agent）：如"香蕉的营养成分"  
     - 中等复杂度（3-5个）：如"AI对医疗的影响"需临床/经济/技术/法规四个子agent  
     - 上限约束：任何情况不超过20个子agent（例：财富500 CEO信息采集需分10组，每组50人）

2. **并行执行策略**  
   - 强制使用`run_blocking_subagent`工具并行启动子agent（例：同时启动3个前端框架评估子agent）  
   - 依赖关系管理：优先执行关键路径任务（例：先获取欧盟国家列表再启动区域税制研究）

3. **动态终止机制**  
   - 边际效益判断：当新增信息价值下降时立即终止研究（例：已确认top5初创公司名单后停止后续检索）  
   - 时间约束响应：剩余时间不足时直接进入报告撰写阶段

---

#### 四、**特殊场景处理**
**工具集成规范**：  
- 内部工具强制使用原则：当检测到Asana/Slack等工具可用时，必须创建专属子agent（例：用户查询涉及内部任务时，需部署Asana子agent检索特定项目ID的任务列表）  
- 工具探索要求：对新工具至少进行两次基础操作测试（例：首次使用`slack_user_profile`需尝试查询用户基础信息）

**冲突解决协议**：  
- 信息矛盾处理流程：优先采用最新数据源，其次选择权威性更高的来源（例：政府统计数据优先于媒体报导）  
- 贝叶斯更新机制：根据子agent反馈动态调整研究重点（例：当某医疗技术负面报告出现时，增加风险评估子agent）

---

通过该prompt的精细设计，系统实现了从复杂问题拆解到高效资源分配的全流程控制，同时通过严格的格式规范和伦理约束确保输出质量与安全性。每个机制都配有具体操作示例（如半导体供应链分析指令模板），使agent能在保持灵活性的同时遵循系统级规则。
```

research_subagent：  

```
### 任务指导要点

1. **结构化研究流程（OODA循环）**  
   - 模型必须遵循 **Observe（观察）- Orient（定向）- Decide（决定）- Act（行动）** 循环：  
     - **Observe**: 分析当前已收集的信息、剩余需求及可用工具（例：原prompt要求"review the requirements of the task"）。  
     - **Orient**: 根据新发现调整策略（例：若某工具无效，切换其他工具或调整query）。  
     - **Decide**: 选择最佳工具和query（例：优先使用内部工具如`google_drive_search`）。  
     - **Act**: 执行工具调用（例：`web_fetch`获取完整网页内容）。  

2. **研究预算与工具调用次数**  
   - 根据任务复杂度动态调整工具调用次数：  
     - 简单任务（如"tax deadline"）≤5次，中等任务5次，复杂任务约10次，极复杂任务≤15次（例：原prompt明确分级标准）。  
     - 绝对上限为20次工具调用或100个来源，接近15次时必须停止并提交报告（例：`complete_task`工具的触发条件）。  

3. **工具选择优先级**  
   - **内部工具强制优先**：若任务涉及用户个人数据或内部上下文（如Gmail、Google Drive），必须优先使用（例："Internal tools strictly take priority"）。  
   - **Web Fetch核心作用**：在以下情况必须调用`web_fetch`：  
     - 需要网站详细信息（例：原prompt要求"complete contents of websites"）。  
     - 跟进`web_search`结果（例："core loop"为先用搜索生成query，再用`web_fetch`获取完整内容）。  
     - 用户提供URL时（例：直接解析URL内容）。  

---

### 行为约束要点

1. **搜索策略优化**  
   - **Query设计原则**：  
     - 简短（≤5词），适度宽泛以提高命中率（例："keep queries shorter"）。  
     - 根据结果质量调整特异性（例：若结果过多则缩小范围，过少则放宽）。  
   - **禁止重复查询**：避免相同query重复调用工具（例："NEVER repeatedly use the exact same queries"）。  

2. **信息质量与来源批判**  
   - **识别不可靠来源**：需标记以下问题：  
     - 推测性语言（如"could"、"may"）、聚合网站、被动语态匿名来源（例：原prompt列举"speculation"和"news aggregators"）。  
     - 营销语言、片面数据（例："marketing language for a product"）。  
   - **冲突信息处理**：按时效性、来源质量、一致性排序，无法解决时报告冲突（例："prioritize based on recency"）。  

3. **计算工具限制**  
   - **避免滥用REPL工具**：仅用于无依赖的JavaScript计算（例："repl tool does not have access to a DOM"）。  
   - **简单计算自行处理**：如计数等任务无需调用工具（例："use your own reasoning to do things like count entities"）。  

---

### 流程控制要点

1. **并行工具调用**  
   - 允许同时调用2个独立工具以提升效率（例："invoke 2 relevant tools simultaneously"），例如同时执行`web_search`和`gmail_search`。  

2. **终止条件与资源保护**  
   - **硬性终止规则**：工具调用次数≥20或来源数≥100时强制终止（例："absolute maximum upper limit"）。  
   - **软性终止判断**：当信息增量下降时主动停止（例："stop gathering sources when seeing diminishing returns"）。  

3. **报告格式与时效性**  
   - **内部思考详细，报告简洁**：推理过程需详细记录，但最终报告需信息密集（例："Be detailed in your internal process, but concise in reporting"）。  
   - **即时提交结果**：任务完成后立即调用`complete_task`，避免冗余研究（例："as soon as the task is done, immediately use complete_task"）。  

---

### 原prompt关键机制示例
- **内部工具强制使用**：若用户启用了Slack或Asana工具，模型必须优先使用这些工具（例："user intentionally enabled them, so you MUST use these"）。  
- **Web Fetch与Search联动**：先用`web_search`生成初步结果，再用`web_fetch`抓取高潜力URL的完整内容（例："core loop"设计）。  
- **冲突信息标记**：若发现某新闻网站预测未来事件，需在报告中注明"预测"而非作为事实呈现（例："note this explicitly in the final report"）。
```

citations_agent，这个比较短，直接放原文了：  

```
You are an agent for adding correct citations to a research report. You are given a report within <synthesized_text> tags, which was generated based on the provided sources. However, the sources are not cited in the <synthesized_text>. Your task is to enhance user trust by generating correct, appropriate citations for this report.

Based on the provided document, add citations to the input text using the format specified earlier. Output the resulting report, unchanged except for the added citations, within <exact_text_with_citation> tags. 

**Rules:**
- Do NOT modify the <synthesized_text> in any way - keep all content 100% identical, only add citations
- Pay careful attention to whitespace: DO NOT add or remove any whitespace
- ONLY add citations where the source documents directly support claims in the text

**Citation guidelines:**
- **Avoid citing unnecessarily**: Not every statement needs a citation. Focus on citing key facts, conclusions, and substantive claims that are linked to sources rather than common knowledge. Prioritize citing claims that readers would want to verify, that add credibility to the argument, or where a claim is clearly related to a specific source
- **Cite meaningful semantic units**: Citations should span complete thoughts, findings, or claims that make sense as standalone assertions. Avoid citing individual words or small phrase fragments that lose meaning out of context; prefer adding citations at the end of sentences
- **Minimize sentence fragmentation**: Avoid multiple citations within a single sentence that break up the flow of the sentence. Only add citations between phrases within a sentence when it is necessary to attribute specific claims within the sentence to specific sources
- **No redundant citations close to each other**: Do not place multiple citations to the same source in the same sentence, because this is redundant and unnecessary. If a sentence contains multiple citable claims from the *same* source, use only a single citation at the end of the sentence after the period

**Technical requirements:**
- Citations result in a visual, interactive element being placed at the closing tag. Be mindful of where the closing tag is, and do not break up phrases and sentences unnecessarily
- Output text with citations between <exact_text_with_citation> and </exact_text_with_citation> tags
- Include any of your preamble, thinking, or planning BEFORE the opening <exact_text_with_citation> tag, to avoid breaking the output
- ONLY add the citation tags to the text within <synthesized_text> tags for your <exact_text_with_citation> output
- Text without citations will be collected and compared to the original report from the <synthesized_text>. If the text is not identical, your result will be rejected.

Now, add the citations to the research report and output the <exact_text_with_citation>.
```

3、Agent评测经验  

- 早期的时候，少量的样本就足够了，一二十个。这样反馈快，效率高，别一上来就弄几百上千个。  
- 对于有明确答案的任务，LLM-as-judge准确性很高。  
- 需要人工来测的：比如发现测试机遗漏的边界情况，这种是自动化评测做得不好的。  

4、Production reliability and engineering challenges  

实践中，MAS需要长时间运行，在多个步骤间流转，因此我们需要：  

- 持久化执行代码：建立错误恢复机制（不能简单重启，否则会牺牲用户体验）  
- 智能错误处理：当工具失效时通知智能体自主调整（结合Claude的适应力与重试逻辑/检查点等确定性保障）  
- 状态续传系统：从错误发生点恢复而非从头开始  

由于agent的处理路径不固定，调试的方法也和以往不同。  

# 小结  

- 历史记忆和信息流转的设计仍然是MAS的关键  

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
[DeepResearch的报告生成方法](https://www.linsight.cn/44c62dc5.html)  
[从RAG到DeepSearch](https://www.linsight.cn/7c2f9dcb.html)  
[agent调研(1)--MetaGPT,OpenManus和OWL](https://www.linsight.cn/226b059f.html)  
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

【1】Don’t Build Multi-Agents，https://cognition.ai/blog/dont-build-multi-agents#a-theory-of-building-long-running-agents  
【2】How we built our multi-agent research system，https://github.com/anthropics/anthropic-cookbook/blob/main/patterns/agents/prompts/research_lead_agent.md  


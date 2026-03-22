---
title: 腾讯的Training-Free GRPO
tags:
  - NLP
  - LLM
  - GRPO
  - RL
categories:
  - CS
  - NLP
  - LLM
abbrlink: 9cb56255
date: 2026-03-18 21:07:42
---

腾讯优图提出的Training-Free Group Relative Policy Optimization（Training-Free GRPO）方法，用8-18美元和100条样本，让671B参数的DeepSeek-V3.1-Terminus在AIME数学竞赛任务上获得提升，超过微调的32B模型。  

# 从"调参"到"给经验"

## RL的困境

当前的LLM Agent优化，主流路径是参数层面的学习：  

1. SFT（监督微调）：准备高质量轨迹数据进行训练  
2. RL（强化学习）：用PPO/GRPO等算法，基于奖励信号优化policy  

这两条路径都有明显的成本门槛：  

- 数据门槛：需要数千到数万条高质量训练样本  
- 计算门槛：需要多卡并行训练数小时到数天  
- 过拟合风险：微调后的模型往往在特定任务上过拟合，通用能力下降  
- 领域隔离：在一个领域训练的模型，难以直接迁移到另一个领域  

## 知识 vs 参数

想象这样一个场景：你是一个资深程序员（相当于基础模型），现在要接一个金融量化项目（没有训练过的新任务）。你有两种学习方式：  

方式A（传统微调）：通过某种"手术"直接改变你的大脑神经连接（修改参数），让你懂量化。  

方式B（Training-Free）：给你一本经验手册（外部知识库），里面记录着："当遇到波动率计算时，应该先查X数据再算Y"；"当策略回撤超过5%时，应该考虑Z因素"。你不需要改变大脑结构，只需要在做项目时查阅这本手册，输出分布自然就变了。  

Training-Free GRPO的核心洞察就是：LLM的输出分布可以通过"注入经验知识"来改变，而不需要调整任何参数：  

> Large language models (LLMs) can utilize knowledge as a token prior to shift their output distributions, eliminating the need for parameter updates  

# 从数值优势到语义优势

## GRPO的回顾

先回顾下GRPO。  

GRPO是DeepSeek提出的RL算法，解决的是PPO中critic模型难以训练的问题。它的关键 trick 是组内相对优势估计：  

对于同一个问题 $q$，采样 $G$ 个输出 $\{o_1, o_2, ..., o_G\}$，每个获得奖励 $r_i$。第 $i$ 个输出的优势值计算为：  

$$\hat{A}_{i}=\frac{r_{i}-\text{mean}(r)}{\text{std}(r)}$$  

这个数值 $\hat{A}_i$ 告诉模型：相对于同组的平均水平，这个输出好多少。然后通过梯度上升，让策略网络 $\pi_\theta$ 倾向于生成高优势的输出。  

这里 $\hat{A}_i$ 是一个数值，用于更新参数 $\theta$。  

## Training-Free GRPO的范式转换

Training-Free GRPO 做了一个的替换：

不再计算数值 $\hat{A}_i$，而是生成自然语言的"语义优势" $A_{\text{text}}$。  

Training-Free GRPO对模型的优化具体流程分四步。  

### Step 1: 组内采样与奖励判定

对于问题 $q$，从当前策略采样 $G=5$ 条轨迹（实验发现采样5条效果比较好），每条轨迹 $o_i$ 包含推理过程、工具调用和最终答案。  

奖励信号 $r$ 的获取：  

- 默认设置：二元奖励 $r \in \{0, 1\}$（正确/错误）  
- 判断者（Verifier）：  
  - 数学任务：与Ground Truth比对（Exact Match）  
  - 代码任务：Sandbox执行验证（单元测试通过与否）  
  - 搜索任务：Programmatic metrics（如是否找到目标信息）  

虽然论文实验主要使用二元奖励，但实际身上也可用多种奖励的加权结果：  

$$R_{\text{total}} = \lambda_{\text{fmt}} R_{\text{fmt}} + \lambda_{\text{cls}} R_{\text{cls}} + \dots$$  

常见组件包括格式奖励（是否遵循`<think>`标签规范）、正确性奖励等。但与传统GRPO不同，这些奖励不直接代入数值公式，而是用于确定组内相对优劣（哪条轨迹更好），进而引导语义提取。  

### Step 2: 轨迹摘要（Trajectory Summarization）

由于完整轨迹可能很长（包含多步工具调用），因此先进行压缩：  

$$s_{i}=M(p_{\text{summary}}, q, o_{i})$$  

这里 $M$ 是LLM（保持参数冻结），$p_{\text{summary}}$ 是摘要prompt。摘要过程中会通过`<evaluation>`标签注入奖励信号（"correct"或"wrong"），让模型知道这条轨迹最终是成功还是失败。  

### Step 3: 提取语义优势

将组内所有摘要 $\{s_1, s_2, ..., s_G\}$ 和当前经验库 $E$ 输入模型：  

$$A_{\text{text}}=M(p_{\text{extract}}, q, s_{i}, E)$$  

提示词 $p_{\text{extract}}$ 的设计让模型扮演"分析师"角色，比较组内轨迹的优劣。这里的组比较是关键——模型会看到"Attempt 1（正确）做了X，Attempt 2（错误）做了Y"，从而提取出：  

> "在解决几何问题时，先画图标注已知条件（如用户X的做法）比直接列公式（如用户Y的做法）更容易得到正确答案。建议：遇到几何题先调用绘图工具可视化。"  

这个 $A_{\text{text}}$ 就是语义优势——它编码了"为什么这个输出更好"的可解释知识。  

### Step 4: 更新经验库

经验库 $E$（初始化为 $\emptyset$）通过四种操作动态更新：  

- Add：追加 $A_{\text{text}}$ 描述的新经验  
- Delete：基于 $A_{\text{text}}$ 移除低质量经验  
- Modify：利用 $A_{\text{text}}$ 洞察精炼现有经验（替换内容，保留ID）  
- Keep：保持 $E$ 不变  

更新后的 $E$ 作为token prior在后续LLM API调用时注入上下文，实现输出分布 $\pi_\theta(y|q,E)$ 的偏移，而基础模型参数 $\theta$ 保持冻结。  

# 实现细节：Prompt

Training-Free GRPO的本质是通过Prompt，将RL的信用分配机制转化为自然语言操作。  

## Prompt 1: 轨迹摘要

功能：将原始轨迹压缩为结构化文本，定位关键决策点。  

输入参数：  

| 参数 | 来源 | 说明 |
|------|------|------|
| `query` | 训练集 | 当前问题 |
| `full_output` | LLM生成 | 完整轨迹 $o_i$（含推理过程、工具调用） |
| `evaluation` | 奖励模型 | "correct" 或 "wrong"（基于$r_i$） |
| `groundtruth` | 训练集 | 标准答案（用于对比分析） |

Prompt模板：  

```
An agent system may be provided with some experiences, and then it produces the following trajectory to solve the given problem. Please summarize the trajectory step-by-step:

1. For each step, describe what action is being taken, and which experience has been used in this step.
2. Given the grading of this rollout and the correct answer, identify and explain any steps that represent detours, errors, or backtracking.
3. Maintain all the core outcome of each step, even if it was part of a flawed process.

<trajectory>{full_output}</trajectory>
<evaluation>{evaluation}</evaluation>
<groundtruth>{groundtruth}</groundtruth>

Only return the trajectory summary of each step.
```

设计要点：  

- 步骤级分解：强制模型回顾每步决策，而非仅看结果  
- 错误定位：要求指出弯路（detours）和回溯（backtracking）  
- 经验关联：追踪使用了哪条经验（用于后续Modify/Delete）  

输出去向：生成文本摘要 $s_i$，进入下一阶段。  

## Prompt 2: 语义优势提取

功能：比较组内轨迹，提取可泛化的经验（即语义优势 $A_{\text{text}}$）。  

输入参数：  

| 参数 | 来源 | 说明 |
|------|------|------|
| `query` | 训练集 | 当前问题 |
| `trajectory_summaries` | Stage 1输出 | 组内所有轨迹摘要及正确性标记 |
| `current_library` | 全局状态 | 当前经验裤 $E$ |

Prompt模板：  

```
You are an expert analyst reviewing problem-solving attempts. Given a group of trajectory summaries (some correct, some wrong) and the current experience library, 
analyze the patterns and extract semantic advantages.

<problem>{query}</problem>

<attempts>
Attempt 1 (Result: Correct): {s_1}
Attempt 2 (Result: Wrong): {s_2}
Attempt 3 (Result: Correct): {s_3}
...
</attempts>

<current_experience_library>{E}<current_experience_library>

Analyze:
1. What key decisions distinguish correct attempts from wrong ones?
2. Which existing experiences were helpful or misleading?
3. What new strategic insights can be extracted?

Output your analysis as a natural language description (semantic advantage) focusing on generalizable problem-solving patterns. Do not output JSON yet.
```

输出示例：  

```
The correct attempts successfully recognized that for Diophantine equations, enumerating possible factorizations after algebraic rearrangement is more reliable than direct formula application. Wrong attempts immediately applied 
quadratic formulas without considering integer constraints. Current experience G17 about "checking discriminants" was insufficient - we need specific guidance on integer constraint handling.
```

## Prompt 3: 批次整合与经验库更新

功能：输出的语义优势 $A_{\text{text}}$，决定如何操作经验库。  

输入参数：  

| 参数 | 来源 | 说明 |
|------|------|------|
| `semantic_advantage` | Stage 2输出 | 自然语言分析结论 $A_{\text{text}}$ |
| `current_library` | 全局状态 | 当前经验库 $E$ |
| `problem` | 训练集 | 当前问题（用于上下文） |

Prompt模板：  

```
Based on the following semantic analysis of problem-solving attempts, decide how to update the experience library.

<semantic_analysis>{A_text}</semantic_analysis>

<current_library>{E}</current_library>

Decide on actions:
- Add: Create new experience if novel insight found
- Delete: Remove if experience proven wrong or obsolete  
- Modify: Refine existing experience if partial match
- Keep: No change if existing experiences sufficient

Requirements:
- Each experience ≤32 words
- Focus on strategic patterns, not specific numbers
- Begin with context (e.g., "For integer equations...")

Return JSON format:
[
  {"option": "add", "experience": "..."},
  {"option": "modify", "modified_from": "G17", "experience": "..."},
  {"option": "delete", "delete_id": "G5"}
]
```

- Prompt 2输出的是分析性文本（为什么好/坏）
- Prompt 3输出的是操作性JSON（具体增删改哪条）

**这两个阶段也可以合并为一个阶段，用一个prompt完成任务。**论文实现中采用的是合并方式（end-to-end）：  

```
You are an expert analyst reviewing an AI agent's problem-solving attempts. 
Your task is to analyze a group of trajectories (some correct, some incorrect), identify key patterns, and update the experience library accordingly.

<problem>
{query}
</problem>

<attempts>
Attempt 1 (Answer: Correct):
{summary_1}

Attempt 2 (Answer: Wrong):  
{summary_2}

Attempt 3 (Answer: Correct):
{summary_3}
...
</attempts>

<ground_truth>
{correct_answer}
</ground_truth>

<current_experience_library>
{E}
</current_experience_library>

Analyze the trajectories:
1. Compare correct vs wrong attempts: What key decisions or strategies distinguish success from failure?
2. Review existing experiences: Which were helpful? Which were ignored or led astray? What's missing?
3. Extract generalizable insights: Focus on strategic thinking patterns applicable to similar problems.

Then update the experience library:
- **add**: Create new experience for novel successful strategies not in current library
- **delete**: Remove experiences proven wrong, misleading, or obsolete  
- **modify**: Refine existing experiences that were partially correct but incomplete
- **keep**: No action if current library is sufficient

Requirements for each experience:
- Begin with context (e.g., "For geometry problems...", "When using tools...")
- Focus on strategic decisions, NOT specific calculations or numbers
- Maximum 32 words per experience
- Use action verbs (Check, Verify, Use, Apply, Consider)

Output strictly in JSON format:
[
  {
    "option": "modify",
    "modified_from": "G17", 
    "experience": "For polynomial equations, verify discriminant before solving and ensure solution count matches theoretical maximum"
  },
  {
    "option": "add",
    "experience": "When encountering integer constraints, prioritize factorization over direct formula application"
  },
  {
    "option": "delete",
    "delete_id": "G5"
  }
]
```

输出约束（Output Constraints）：  

- 强制JSON格式（便于程序化更新 $E$）  
- 32词限制（强制抽象化）  
- 动词开头（保证可操作性）  

### 四种操作的参数详解

LLM输出JSON格式的操作指令，具体参数如下：  

| 操作 | 必要字段 | 类型 | 具体行为 |
|------|----------|------|----------|
| **Add** | `experience` | string | 新增经验，系统自动分配ID（如G42） |
| **Delete** | `delete_id` | string | 移除指定ID的经验条目（如"G17"） |
| **Modify** | `modified_from` | string | **源经验ID**（被修改的对象） |
| | `experience` | string | **新经验的完整内容**（替换原内容，保留ID） |
| **Merge** | `merged_from` | array | 源经验ID数组（如["G5", "G12"]） |
| | `experience` | string | 合并后的通用经验文本 |

Modify操作的本质：  

- 不是增量更新，而是全量替换  
- 保留原ID（`modified_from`指向的旧ID），更新内容  
- 实现经验的渐进式精炼（generalized or refined）  

示例：  

```json
// 原经验 G17: "For quadratic equations, check discriminant"
{
  "option": "modify",
  "modified_from": "G17",
  "experience": "For quadratic and polynomial equations, verify discriminant before solving and ensure solution count matches theoretical maximum"
}
```

系统会将G17的内容更新为新的更长版本，实现从"二次方程"到"多项式方程"的泛化。

# 其他细节

## 无Ground Truth（w/o ground truths）机制

消融实验中的一项，即没有标准答案的情况。指奖励信号不依赖标准答案，而是通过组内自举（bootstrapping）获得：  

1. 答案聚类：对 $G$ 个输出进行答案提取（如从 `\boxed{}` 中提取）  
2. 多数表决：出现频率最高的答案被视为"伪正确答案"  
3. 二元标记：  
   - 与多数答案一致的轨迹 $r=1$（正确）  
   - 与多数答案不一致的轨迹 $r=0$（错误）  
4. 语义提取：基于这个"伪标签"进行组间比较  

发现即使没有标准答案（w/o ground truths），在AIME24上仍达到79.1%（对比有GT的79.6%），证明其对奖励噪声具有鲁棒性，可利用模型自身的Self-Consistency作为监督信号。  

这个很适合用于没有标签的情况。  

## 实验效果：成本与性能的双重验证

### 与传统RL的成本对比

| 方法 | 训练成本 | 模型规模 | AIME24 Mean@32 | AIME25 Mean@32 |
|------|----------|----------|----------------|----------------|
| ReTool（传统RL） | ~$10,000 | 32B | 低于基线 | 低于基线 |
| DeepSeek-V3.1基线 | API费用 | 671B冻结 | 74.8% | 61.5% |
| **Training-Free GRPO** | **~$8-18** | **671B冻结** | **79.6%** | **68.5%** |

成本：从约10,000美元降至8-18美元（3个epoch，100条样本）  

### 消融实验：验证每个组件

| 配置 | AIME24 | AIME25 | 说明 |
|------|--------|--------|------|
| ReAct基线 | 74.8% | 61.5% | 无经验注入 |
| ReAct+自生成经验 | 78.0% | 65.1% | 无组比较（$G=1$），直接生成 |
| Training-Free GRPO（无Ground Truth） | 79.1% | 67.7% | 仅用多数投票/自判别 |
| Training-Free GRPO（完整版） | 79.6% | 68.5% | 使用标准答案奖励 |

关键洞察：  

1. 组内比较是必要的：单轨迹自生成经验（第二行）效果显著差于组比较（第三、四行），证明"有对比才有鉴别"  
2. 奖励信号鲁棒：即使没有标准答案，仅靠组内多数投票也能达到79.1%  

### 效率提升：不仅准，而且快

在3轮学习过程中，平均工具调用次数从8.4次降至6.7次，同时准确率提升。这说明模型学会了更简洁高效地使用工具，而非盲目增加计算量。  

## 为什么需要"组"（Group）？

核心原因在于相对性。  

单条轨迹的奖励 $r$ 是绝对的（比如0.8分），但0.8分到底算好还是差？没有参照系很难判断。而组内比较提供了相对参照系：  

- 如果组内其他输出都得0分，你的0.8分就是优秀  
- 如果组内其他输出都得1分，你的0.8分就是不足  

这也是GRPO的核心思想。  

## 经验库的规模与收敛性

经验库的增长遵循边际递减规律：  

- 100条训练样本通常生成 20-40条经验（去重后）  
- 收敛曲线：  
  - Epoch 1：快速增长（0 → ~25条），覆盖主要错误模式  
  - Epoch 2：增长放缓（25 → ~35条），开始精细化  
  - Epoch 3：趋于稳定（35 → ~40条），以Modify和Delete为主  

性能饱和点：经验库规模在 30-50条 时，AIME24/25性能趋于 plateau。  

# 适用场景与局限性

## 最佳适用场景

1. 长尾细分场景：任务有价值但数据量不足以支撑传统微调（如特定领域的客服、专业工具使用）  
2. 快速迭代场景：策略需要频繁更新，但是频繁训练成本高的情况  
3. 预算受限团队：个人开发者、中小企业、学术研究者  
4. 黑盒API场景：无法获取模型权重（如调用第三方API）进行微调  

## 局限性

1. 基础模型能力依赖：在QwQ-32B上效果不佳（Pass@1仅25.5%），说明方法依赖基础模型的推理能力  
2. 上下文长度：经验库 $E$ 会占用token（通常限制在50条以内），对长上下文模型更友好  
3. 提示工程敏感：$p_{\text{summary}}$ 和 $p_{\text{extract}}$ 的设计影响效果，需要实验  

# Reference  

【1】Training-Free Group Relative Policy Optimization，https://arxiv.org/pdf/2510.08191  

***

【推荐文章】  
- Agent：  
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

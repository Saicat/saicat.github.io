---
title: VeRA，LoRA-XS和TinyLoRA
tags:
  - NLP
  - LLM
  - LoRA
categories:
  - CS
  - NLP
  - LLM
abbrlink: cc1c31d
date: 2026-03-21 11:36:24
---

# VeRA：随机投影的极限压缩

时间：2023.10  

VeRA = Vector-based Random Matrix Adaptation  

LoRA虽然相比全量微调，训练的参数量大大减少，不过随着模型的增大（d增大），AB矩阵的规模也会线性增加。再加上场景增多，所需的LoRA参数也会增加。如果以后发展到每个用户一个LoRA，那这个储存量还是很可观的。  

面对LoRA的存储瓶颈，VeRA提出了一个洞察：如果低秩适配的方向可以预先确定，我们只需学习如何"调制"这些方向，而非学习方向本身。  

## VeRA核心机理：冻结随机矩阵+可训练缩放

VeRA的核心结构如下：  

$$
h = W_0 x + \Lambda_b B \Lambda_d A x
$$  

与LoRA不同，VeRA中的 $B \in \mathbb{R}^{m \times r}$ 和 $A \in \mathbb{R}^{r \times n}$ 是冻结的随机矩阵，且跨所有层共享。真正可训练的是两个对角缩放向量：  

- $b \in \mathbb{R}^{m}$：调制 $B$ 的输出行  
- $d \in \mathbb{R}^{r}$：调制 $A$ 的输入列  

这相当于用 $\Lambda_b = \text{diag}(b)$ 和 $\Lambda_d = \text{diag}(d)$ 对随机投影进行"重新加权"。通过将某些缩放因子推向零，VeRA可以"关闭"随机矩阵中不重要的行或列；通过调整符号和幅度，它可以微调投影方向。  

这样一共只有2r个训练参数。  

## 初始化策略细节

VeRA的性能对初始化高度敏感。消融实验发现：Kaiming Uniform显著优于Kaiming Normal和Uniform[0,0.1]。  

- Kaiming Uniform：从均匀分布 $U[-a, a]$ 采样，$a = \sqrt{6 / \text{fan\_in}}$，取值有确定边界，无极端值  
- Kaiming Normal：从正态分布 $\mathcal{N}(0, \sigma^2)$ 采样，$\sigma = \sqrt{2 / \text{fan\_in}}$，存在长尾极端值  

VeRA选择Uniform，或者说Uniform更佳的原因在于：其随机矩阵是冻结的，模型只能依赖训练缩放向量来"选择"投影方向。Uniform分布的确定性边界确保了所有投影向量的范数在一定范围内，避免了Normal分布可能出现的极端值，使得缩放向量的优化空间更加"均匀"和可控。

论文发现 $d_{\text{init}} = 10^{-1}$ 或 $10^{-7}$ 显著优于默认的 $1.0$。$d \in \mathbb{R}^r$ 是调制矩阵 $A$ 输入列的缩放向量。$d_{\text{init}}$ 是训练开始时的初始值。  

这背后的机理是：较小的初始值允许优化早期发生符号翻转（sign changes），为选中行提供更大的优化灵活性。如果 $d_i$ 初始为较大的正值（如1.0），优化过程中它可能始终维持正号，限制了"关闭"（推向零）或"反转"（变负）该行投影的灵活性。而从小值（如0.1或$10^{-7}$）开始，梯度可以更容易地推动 $d_i$ 穿过零点，实现更精细的方向调制。  

## 双向量必要性

VeRA的消融实验证明b和d两个缩放向量都必要，但重要性不同：  

- 仅训练 $d$（固定 $b$）：MRPC 89.7，RTE 67.0  
- 仅训练 $b$（固定 $d$）：MRPC 81.6，RTE 64.3  
- 训练 $b + d$（完整VeRA）：MRPC 90.5，RTE 85.8  

（MRPC和RTE都是自然语言推断任务。）  

这表明 $d$ 向量（调制 $A$ 矩阵输入侧）比 $b$ 向量更具表达能力，因为它影响两个低秩矩阵的行。  

## 参数效率的提升

| 方法 | 可训练参数 | 与模型维度关系 | LLaMA-7B示例 | 100万模型存储 |
|------|-----------|---------------|-------------|--------------|
| LoRA | $O(d \cdot r)$ | 线性依赖 | 159.9M | 144TB |
| **VeRA** | **$O(r)$** | **与$d$无关** | **1.6M (100×↓)** | **96GB (1500×↓)** |

在GLUE基准上，VeRA以0.061M参数（LoRA的1/13）达到持平的性能（87.8%）；在指令微调任务中，VeRA以1%参数（1.6M vs 159.9M）接近LoRA的表现（MT-Bench 4.77 vs 5.03）。  

## 共享vs独立矩阵的权衡

VeRA还探索了随机矩阵在所有层共享的影响：  

- Shared（跨层共享）：MRPC 90.0，存储仅需保存随机种子  
- Unique（每层独立）：MRPC 90.7，需存储所有矩阵  

## 局限与权衡

VeRA的代价在于表达能力。由于随机矩阵固定且缺乏预训练先验，VeRA通常需要比LoRA更高的秩才能达到同等性能。此外，对初始化敏感，缩放向量的初始值选择对最终性能有显著影响。  



# LoRA-XS：引入SVD

时间：2024.05  

LoRA-XS：Low-Rank Adaptation with eXtremely Small number of parameters  

VeRA证明了"冻结投影+训练缩放"的可行性，但它使用随机投影，没有利用预训练权重中的先验信息。  

LoRA-XS提出了一个想法：既然LoRA的更新方向与预训练权重高度相关，为何不直接用预训练权重的奇异向量作为投影基？  

而《LoRA Learns Less and Forgets Less》中提到：LoRA的低秩更新不仅减少了参数，还通过"更新量少"的特性减缓了灾难性遗忘，保留了预训练阶段学到的通用知识。那么既然低秩更新本身有优势，我们能否进一步压缩秩的规模？  

## 核心机理：SVD分解+核心矩阵训练

LoRA-XS的结构如下：  

$$
h = Wx + BRAx
$$

- $A = U_r \Sigma_r$，$B = V_r^T$：来自预训练权重 $W$ 的截断SVD（冻结）  
- $U_r, V_r$：前 $r$ 个左右奇异向量；$\Sigma_r$：前 $r$ 个奇异值  
- $R \in \mathbb{R}^{r \times r}$：唯一的可训练矩阵  

这相当于在预训练权重的主导奇异方向上学习一个"重组矩阵" $R$，对奇异值进行重新加权组合。论文将这种方法称为latent editing——在预训练权重的潜在空间中进行微调。  

## 可训练矩阵R的初始化

LoRA-XS中 $R$ 矩阵的初始化采用高斯分布：  

$$
R \sim \mathcal{N}(0, \sigma^2), \quad \sigma = 10^{-5}
$$  

使用极小的初始化值（$10^{-5}$）确保微调从接近原始预训练模型的状态开始，避免初始阶段过大的偏离。  

## 为何SVD初始化更优？

消融实验比较了SVD初始化与随机初始化：  

| 任务 | 秩 | SVD初始化 | 随机初始化 |
|------|---|----------|-----------|
| CoLA | 4 | **60.52** | 56.65 |
| SST-2 | 4 | **94.84** | 93.92 |
| QNLI | 4 | **90.94** | 88.83 |

SVD初始化不仅提升了最终性能，还加速了收敛——SVD初始化在前1-2个epoch即显现出优势，而随机初始化需要更多epoch才能稳定。  

## 参数效率

对于RoBERTa-large（$d=1024, r=16$）：  

- LoRA：1,572,864参数  
- VeRA：50,400参数  
- LoRA-XS：12,288参数（比LoRA少128倍，比VeRA少4倍）  

LoRA-XS的可训练参数规模为 $O(r^2)$，与模型维度$d$无关。因此模型越大，效率优势越大。  

## 参数分配策略

LoRA-XS的经验：当参数预算受限时，优先使用较低的秩 $r$ 并增加适配模块的覆盖范围，而非集中参数于少数模块。  

论文将LoRA-XS应用于Query、Key、Value、Attention Output和MLP的投影矩阵（共7个模块 per layer），使用较小的 $r$（如4-16），而非仅应用于Q/V矩阵。这种"广撒网"策略在总参数量相近时表现更优。  



# TinyLoRA：跨层共享与RL的协同

时间：2025.02  

LoRA-XS将参数压缩至 $r^2$ 级别，但仍受限于"每层至少$r^2$参数"的下界。对于80层的LLaMA-3 70B，即使 $r=1$ 也需要560参数。TinyLoRA提出了更激进的问题：能否用单个参数微调整个模型？以及，什么训练范式能支撑这种极限压缩？  

另外，随着参数规模缩减，学习率的调整变得至关重要：  

- 《LoRA Learns Less and Forgets Less》里提到：不同rank配置影响学习动态和超参数选择。  
- 《LoRA Without Regret》做了LoRA在不同rank下的训练动态分析，指出低rank需要调整学习率以补偿参数容量的减少。  

## 核心机理：向量投影+极端共享

TinyLoRA的结构如下：  

$$
W' = W + U\Sigma \left(\sum_{i=1}^u v_i P_i\right) V^\top
$$  

这是LoRA-XS的进一步压缩：  

- $U, \Sigma, V$：来自SVD（冻结）  
- $P_i \in \mathbb{R}^{r \times r}$：固定随机张量（冻结），从标准随机初始化（如Kaiming Uniform或正态分布）生成后冻结  
- $v \in \mathbb{R}^u$：唯一的可训练向量，通过 $\sum_{i=1}^u v_i P_i$ 重构出 $r \times r$ 的变换矩阵  

更激进的是，TinyLoRA采用跨层、跨模块类型的全共享策略：所有Transformer层的所有适配模块（Attention的Q/K/V/Output + MLP的up/gate/down，共7个模块 per layer）共享同一个 $v$。  

## 冻结秩r的选择：r=2

TinyLoRA的消融实验确定了冻结SVD秩 $r$ 的最优值：  

| 冻结秩 $r$ | 性能趋势 |
|-----------|---------|
| $r=1$ | 基线可用 |
| $r=2$ | 最佳（主实验采用） |
| $r>2$ | 性能下降 |

增加 $r$ 从1到2带来适度增益，但更大的值会引入过多自由度，使得小向量 $v$ 的优化变得困难。因此，$r=2$ 是所有主实验的固定配置。  

## 关键发现：RL vs SFT

TinyLoRA还发现训练范式影响：  

| 训练方式 | 13参数配置 | 120参数配置 | 1M参数配置 |
|---------|-----------|------------|-----------|
| RL (GRPO) | 91% GSM8K (+15%) | 95% GSM8K (+19%) | ~95% |
| SFT | 83% (+7%) | 84% (+8%) | ~90% |

在极低参数（<1000）场景下，RL比SFT高效100-1000倍。  

洞察：  

- SFT需要吸收完整demonstration序列的信息，通过next-token prediction损失：  
  $$
  \mathcal{L}_{\text{SFT}} = -\mathbb{E}_{(x,y)} \sum_{t=1}^{|y|} \log \pi_\theta(y_t | x, y_{<t})
  $$
  模型无法区分demonstration中的任务相关特征与无关噪声，必须存储所有token的信息。  

- RL仅依赖稀疏的reward信号。策略梯度为：  
  $$
  \nabla_\theta J(\theta) = \mathbb{E}_{x,y} \left[ \sum_{t=1}^{|y|} \nabla_\theta \log \pi_\theta(y_t | x, y_{<t}) \cdot R(y) \right]
  $$
  有用的信息仅来自奖励 $R(y)$（二元奖励仅需1 bit per sample），通过重采样放大与奖励相关的信号、抵消无关噪声。  

## 学习率

由于参数规模变化影响有效学习率，TinyLoRA对每个更新规模进行网格搜索：  

$$
\text{LR} \in \{10^{-7}, 5 \times 10^{-7}, 10^{-6}, 5 \times 10^{-6}, 10^{-5}, 10^{-4}, 2 \times 10^{-4}\}
$$  

取每个更新规模下表现最佳的学习率，平均3个随机种子。极低参数场景通常需要$10^{-3}$或$10^{-4}$级别，远高于全量微调的$10^{-5}$。  

## 参数分配：u与n_tie的权衡

实验表明，在固定参数预算下要提升效果：  

- 优先增加 $u$（每模块的投影维度）  
- 减少 $n_{\text{tie}}$（共享因子，即减少共享程度）  

> "Performance generally improves with larger $u$ and smaller $n_{\text{tie}}$—that is, more expressive per-module updates with less sharing."

## 模型规模的规律

TinyLoRA发现，模型越大，达到同等性能所需的adapter参数越少。  

| 模型规模 | 达到95%峰值性能所需参数 |
|---------|----------------------|
| 较小模型 | 较多 |
| Qwen2.5-7B | 13-120参数 |
| 更大模型（理论） | 更少 |

这暗示大模型在预训练阶段已内化推理能力，微调只是"激活"这些能力的开关。随着模型规模增长，"激活开关"所需的参数反而减少。  

# 对比

| 维度 | VeRA (2023.10) | LoRA-XS (2024.05) | TinyLoRA (2025.02) |
|------|----------------|-------------------|-------------------|
| **核心思想** | 随机投影+缩放向量 | SVD先验+核心矩阵 | 向量投影+极端共享+RL |
| **数学形式** | $h=W_0x+\Lambda_b B \Lambda_d A x$ | $h=Wx+BRAx$ | $W'=W+U\Sigma(\sum v_i P_i)V^\top$ |
| **投影矩阵** | 随机，跨层共享 | SVD分解，层内冻结 | SVD分解+随机张量 $P_i$ |
| **可训练参数** | $b \in \mathbb{R}^m, d \in \mathbb{R}^r$ | $R \in \mathbb{R}^{r \times r}$ | $v \in \mathbb{R}^u$ |
| **参数复杂度** | $O(r)$ | $O(r^2)$ | $O(u)$（可降至1） |
| **与模型维度关系** | 无关 | 无关 | 无关 |
| **初始化关键** | $d_{\text{init}}=10^{-1}$或$10^{-7}$（便于符号翻转） | $R \sim \mathcal{N}(0, 10^{-10})$（极小值） | $P_i$随机初始化后冻结 |
| **随机矩阵类型** | Kaiming Uniform > Normal | - | 标准随机（推测Kaiming） |
| **最优配置** | $r=256$（需较高秩） | $r=2$（主成分对齐） | $r=2, u=13$或$120$（全共享vs轻共享） |
| **学习率策略** | 标准 | 标准 | **必须搜索调整**（参数规模影响有效LR） |
| **训练范式** | SFT/RL均可 | SFT/RL均可 | **RL必需**（SFT在<1K参数下失效） |
| **最佳场景** | 大规模个性化部署 | 通用PEFT，需快速收敛 | RL训练，超大规模模型，极限压缩 |
| **典型压缩比** | 10-100× | 100-1000× | 1000-10000× |

# Reference

【1】VeRA: Vector-based Random Matrix Adaptation，https://arxiv.org/pdf/2310.11454  
【2】LoRA-XS: Low-Rank Adaptation with Extremely Small Number of Parameters, https://arxiv.org/pdf/2405.17604  
【3】Learning to Reason in 13 Parameters, https://arxiv.org/pdf/2602.04118v1  

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

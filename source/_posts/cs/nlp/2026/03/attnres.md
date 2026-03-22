---
title: Attention Residuals
tags:
  - NLP
  - LLM
categories:
  - CS
  - NLP
  - LLM
abbrlink: 5b81d487
hidden: false
date: 2026-03-22 11:51:44
---

# 残差连接的隐性成本：PreNorm Dilution

标准的 PreNorm 残差连接通过恒等映射为梯度提供高速公路，使深层网络的训练成为可能，[post-norm & pre-norm](https://www.linsight.cn/6a40bfa5.html?highlight=pre+norm#post-norm-pre-norm)。但残差连接同时定义了信息在深度方向上的聚合方式。  

展开递推公式可见，第 $l$ 层的输入实际上是前面所有层输出的等权累加：  

$$
h_l = h_1 + \sum_{i=1}^{l-1} f_i(h_i)
$$  

这种直接相加的均匀聚合带来了三个隐性成本：  

- 第一，早期层的信息被持续稀释。随着深度 $l$ 增加，每一项 $f_i(h_i)$ 对 $h_l$ 的贡献以 $1/l$ 的速度衰减，早期特征被淹没在后续的累加和中。  
- 第二，隐藏状态幅度随深度线性增长。实证观察表明，PreNorm 架构中 $\|h_l\|$ 随层数 $l$ 呈现 $O(L)$ 增长，深层网络必须学习产生越来越大的输出来维持影响力，这增加了训练的不稳定性。  
- 第三，缺乏选择性访问机制。每一层只能看到压缩后的单一状态 $h_{l-1}$，而非前面各层的独立输出 $f_i(h_i)$。这与序列建模中 RNN 的困境如出一辙：信息在传递过程中被强制压缩，丢失的细节无法在后续恢复。  

# 从时间到深度的对偶

Transformer 在序列维度上通过注意力机制解决了 RNN 的信息压缩问题。与其让当前位置只能读取前一个隐藏状态，不如让它直接 attend 到前面所有位置的表示。  

AttnRes 将这一思想迁移到深度维度。既然残差连接在深度上的累加与 RNN 在时间上的递推形式对偶，那么深度维度同样可以从线性累加升级为 softmax 注意力。  

具体地，第 $l$ 层的输入不再固定为累加和，而是前面所有层输出的加权组合：  

$$
h_l = \sum_{i=0}^{l-1} \alpha_{i \to l} \cdot v_i
$$  

其中 $v_i = f_i(h_i)$ 表示第 $i$ 层的输出（$v_0 = h_1$ 为嵌入层），$\alpha_{i \to l}$ 是满足 $\sum_{i=0}^{l-1} \alpha_{i \to l} = 1$ 的注意力权重。  

权重通过 softmax 计算：

$$
\alpha_{i \to l} = \frac{\exp\left(q_l^\top \text{RMSNorm}(k_i)\right)}{\sum_{j=0}^{l-1} \exp\left(q_l^\top \text{RMSNorm}(k_j)\right)}
$$

这里 $q_l = w_l$ 是第 $l$ 层独有的可学习向量，$k_i = v_i$ 是历史层的输出作为 key。RMSNorm 的引入防止了幅度较大的层输出在注意力计算中占据主导地位。  

整体的思路上，感觉有点像DenseNet。  

# Full Attention Residuals 与工程现实

上述形式被称为 Full AttnRes。它在理论上完成了从 depth-wise linear attention 到 depth-wise softmax attention 的跃迁，但在大规模训练场景下面临工程挑战。  

## 激活重计算的冲突

LLM训练普遍采用重计算（activation recomputation）来节省显存。标准残差连接中，前向传播完成后可以立即释放中间激活，反向传播时通过重计算恢复。但 AttnRes 要求在第 $l$ 层前向传播时保留所有历史输出 $\{v_0, \ldots, v_{l-1}\}$，因为这些向量不仅是反向传播所需的梯度计算中间量，更是当前层前向计算的输入依赖。  

这种依赖性破坏了重计算的前提：即使能在反向时重算出 $v_i$，前向传播时也必须显式保存这些向量供后续层使用。Full AttnRes 的内存复杂度为 $O(Ld)$，在 $L$ 达到数百甚至上千时难以承受。

## Block AttnRes：分块压缩

为解决扩展性问题，Block AttnRes 将 $L$ 层划分为 $N$ 个块（通常 $N=8$），每块包含 $S = L/N$ 层。

块内采用标准残差累加，维护部分和（partial sum）$b_n = \sum_{j \in B_n} f_j(h_j)$。块间则执行 Full AttnRes，但只关注 $N$ 个块级表示而非 $L$ 个独立层输出。  

对于第 $n$ 块中的第 $i$ 层，其 value 矩阵为：  

$$
V = 
\begin{cases} 
[b_0, \ldots, b_{n-1}]^\top & \text{if } i = 1 \\
[b_0, \ldots, b_{n-1}, b_n^{i-1}]^\top & \text{if } i \geq 2
\end{cases}
$$  

其中 $b_n^{i-1}$ 表示当前块内前 $i-1$ 层的累加和。通过这种方式，内存复杂度从 $O(Ld)$ 降至 $O(Nd)$，通信复杂度在流水线并行中也相应降低。  

{% asset_img attnres.png attnres %}  

# 优化细节

## 流水并行优化

流水并行可以优化到基本没有额外成本，这里不太了解，先略过了。  

## 两阶段计算：推理时的内存访问优化

推理阶段面临的问题是：如果每层都重新读取所有块表示，内存访问次数为 $O(L \cdot N)$。两阶段计算策略通过摊销（amortization）解决这一问题。  

Phase 1（并行阶段）：利用伪查询 $w_l$ 与层前向计算解耦的特性，将块内 $S$ 层的查询批处理，一次性读取所有历史块表示并计算注意力，得到 $S$ 个中间结果及 softmax 统计量（max 和 log-sum-exp）。  

Phase 2（顺序阶段）：逐层处理块内部分和，使用在线 softmax（online softmax）算法将 Phase 1 结果与当前部分和合并。在线 softmax 允许在不重新读取历史块的情况下，动态加入新的 attention 分量。  

这种设计将块表示的读取次数从 $S$ 次降至 1 次，每层内存访问仅为 $(\frac{N}{S} + 3)d$ 读取和 $2d$ 写入。当 $N=8, S=16$ 时，相比标准残差的 $3d$ 读取，增幅不足 17%，端到端推理延迟增加控制在 2% 以内。  

# 训练动态与实验验证

## 初始化策略

所有伪查询向量 $w_l$ 必须初始化为零。这确保训练初期 $\alpha_{i \to l}$ 在所有源层上均匀分布，AttnRes 退化为等权平均，避免因初始注意力分布不均导致的训练震荡。  

## PreNorm Dilution 的缓解

在 48B 总参/3B 激活参数的 MoE 模型上，AttnRes 展现出对 PreNorm Dilution 的显著缓解：  

- 输出幅度：各层输出在深度方向上保持有界，不再随深度线性增长  
- 梯度分布：梯度范数在各层间分布更加均匀，避免了标准残差中常见的深层梯度衰减  

## 计算效率

Scaling Law 实验表明，Block AttnRes 达到与基线相同验证损失所需的计算量减少约 20%，等效于在相同计算预算下获得 1.25 倍的训练效率。  

下游任务评估显示，AttnRes 在多步推理（GPQA-Diamond 提升 7.5 分）和代码生成（HumanEval 提升 3.1 分）等复杂任务上收益尤为显著，这表明选择性访问历史层的能力对复杂模式匹配至关重要。  

# Reference  

【1】ATTENTION RESIDUALS，https://arxiv.org/pdf/2603.15031  

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

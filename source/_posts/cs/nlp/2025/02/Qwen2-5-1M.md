---
title: Qwen2.5-1M技术解密
tags:
  - NLP
  - LLM
  - transformer
  - SFT
  - pretrain
  - 长上下文
  - 窗口外推
categories:
  - CS
  - NLP
  - LLM
abbrlink: 6c0f6207
date: 2025-02-18 22:28:39
---

【本文已在同名 微信公众号 / 知乎 / [个人博客linsight.cn](http://www.linsight.cn/) 上线】  

***  

在看Qwen2.5-1M的方案之前，先把Qwen2.5-1M中用到的技术，DCA，MInference 1.0和chunked prefill学习一下。  

其他长文本相关文章：

[LLM长上下文的问题](https://mp.weixin.qq.com/s/Ci9tAMIER0Aj96sK81HNcw?token=1318369845&lang=zh_CN)  
[解锁大模型长上下文能力](https://mp.weixin.qq.com/s/FTewPxSr5fcwkxAgRZm7Wg?token=1318369845&lang=zh_CN)  
[大模型推理窗口-从有限到无限大](https://mp.weixin.qq.com/s/NaTtwURRw7lsG55QTIaVsA?token=1318369845&lang=zh_CN)  
[理解LLM位置编码:RoPE](https://mp.weixin.qq.com/s/QEHdtJKsY7lIU0aK8CeEkg?token=1318369845&lang=zh_CN)  

# Dual Chunk Attention (DCA)  

DCA是一个不用进行训练，就可以有效进行窗口长度外推的方法。（LLM位置编码现在默认都是基于RoPE的了，DCA也是。）  

DCA可以不用训练进行外推，也是可以用在训练中的，而且有训练的效果肯定比不训练的外推更好。在不训练的情况下，DCA就可以把在4k窗口训练的模型外推到32k，这相比其他主流外推方案（PI、NTK、YaRN等）都算是比较强的。  

## 方案  

来看下DCA是怎么做的。假设现在有一个模型，训练窗口的长度为6，在处理sequence length = 12的输入时，relative position matrix M是这样的：  

{% asset_img dca_ori.png Qwen2.5-1M %}  

M中出现了 ≥ 6的相对距离，这些距离的值在训练的时候模型没有见过，这就导致了输出效果变差。  

DCA的大致思路就是重新构造这个relative position matrix M：把长输入（超过预训练窗口长度）拆分成多个chunk，在这个基础上计算三种attention：（1）intra-chunk attention（2）inter-chunk attention（3）successive-chunk attention。  

1、intra-chunk attention  

既然使用没有训练过的相对距离值会影响模型效果，那在M中就只使用不大于训练长度的距离值。把输入切分成多个大小固定的chunk，保证chunk size s ≤ pretrain window size c。这样每个chunk内部的token间距离单独计数，就不会出现超过预训练长度的距离了。  

下面这个例子就是pretrain window size c = 10，chunk size s = 6，当前输入长度为12时，intra-chunk attention对应的M的值：  

{% asset_img dca_intra_chunk.png Qwen2.5-1M %}  

实验中，通常把s设置为 $\frac{3}{4}c$ 的大小。  

实现上只要修改q和k的position index就可以了。比如原来q和k的位置下标都是  

0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11  

现在切分成两个长度为6的chunk，每个chunk内部的token index都是  

0, 1, 2, 3, 4, 5  

对于整个input就是  

0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5  

公式化来说就是  

$$
P_{q}^{Intra} = P_{k} = [0, 1, \cdots, l - 1] \bmod s
$$  

$$
M[i][j] = P_{q}^{Intra}[i] - P_{k}[j]
$$  

2、inter-chunk attention  

仅使用intra-chunk attention相当于把各个chunk视为独立（相当于截断了），每个token只能看到和自己同个chunk的信息，这当然就会损失不同chunk之间的信息关联，因此就要加入inter-chunk attention来让模型捕捉多个chunk之间的关系。  

由于要保证M中的值 < c，所以这里不能直接使用原始的token距离。观察到在原始的M中，有效的距离值构成下三角矩阵，而q的位置总是大于k的位置的，因此inter-chunk attention对应的值总是 > 0的。  

在符合这两个条件的前提下，一个方法就是把q的位置index全部设到最大，即预训练窗口长度大小（= c - 1），对应的M的值如下：  

{% asset_img dca_inter_chunk.png Qwen2.5-1M %}  

$$
P_{q}^{Inter} = [\underbrace{c - 1, c - 1, \cdots, c - 1}_{l \text{ elements}}]
$$  

$$
M[i][j] = P_{q}^{Intra}[i] - P_{k}[j] = c - 1 - P_{k}[j] \geq c - s
$$  

inter-chunk attention没有对chunk的具体位置纳入考虑，而只给出两个chunk的相对前后关系。比如“chunk1和chunk3”的inter-chunk attention与“chunk1和chunk4”的inter-chunk attention所使用的相对位置是一样的，模型没法据此区分chunk3和chunk4的相对位置。（不过decoder-only的模型本身具有一定的位置编码能力，这里只是说没有显式地在位置编码中体现）  

（其实inter-chunk attention可不可以有别的设计呢，个人感觉是可以的，只是原文中使用的方案如此；总之只要保证使用不超过训练窗口大小的位置编码的情况下计算chunk之间的attention就可以）  

3、successive-chunk attention  

上面这种inter-chunk attention的设计在计算「不相邻」的chunk的时候是没有问题的，但是对于两个相邻的chunk就有问题了。  

比如看上面图中，红色子矩阵的第一行，inter-chunk距离是  

9，8，7，6，5，4  

继续往后数就进入到intra-chunk attention的范围，看上上图，下一个距离是0。  

连起来就是  

9，8，7，6，5，4，0  

从4到0这里有个突变，这样的距离不连续，会加重模型对attention机制理解的负担。因此对于相邻的chunk，它们之间的inter-chunk attention要稍微修改一下。  

对于每个chunk，position的index变成：  

$$P_{\mathbf{q}}^{\mathrm{Succ}}=[\overbrace{\underbrace{s,s+1,\ldots,s+w-1}_{\text{the same for all chunks}}}^{w\mathrm{~elements}}]$$  

这里有一个local window size w，w的物理意义可以认为是：在这两个相邻chunk的之内，每个token都可以保证正常看到的窗口到小。在这个窗口内，距离的计算和原生的RoPE是保持一致的。而在w之外，则会出现分辨率的降低。  

文中提到w的值可以直接设置为c - s，比如s = $\frac{3}{4}c$，那就有w = $\frac{1}{4}c$。  

在这个具体的例子里，原来的q的position indices是：  

$$P_{\mathbf{q}}^{\mathrm{Inter}}=[\underbrace{9,9,9,9,9,9}_{\mathrm{chunk~}0},\underbrace{9,9,9,9,9,9,9}_{\mathrm{chunk~}1}]$$  

现在变成  

$$P_{\mathbf{q}}^{\mathrm{Succ}}=[\underbrace{6,7,8,9,9,9}_{\mathrm{chunk~}0},\underbrace{6,7,8,9,9,9}_{\mathrm{chunk~}1}]$$  

{% asset_img dca_successive_chunk.png Qwen2.5-1M %}  

## 完整DCA  

结合上面这三种方式的attention，最终的relative position matrix是这样的：  

$$
M[i][j] = 
\begin{cases}
P_{q}^{Intra}[i] - P_{k}[j] & \text{if } \lfloor i / s\rfloor - \lfloor j / s\rfloor = 0 \\
P_{q}^{Succ}[i] - P_{k}[j] & \text{if } \lfloor i / s\rfloor - \lfloor j / s\rfloor = 1 \\
P_{q}^{Inter}[i] - P_{k}[j] & \text{if } \lfloor i / s\rfloor - \lfloor j / s\rfloor > 1
\end{cases}
$$  

注意inter-chunk attention只用在不相邻的chunk之间了。  

再看一个例子配合理解：在s = 4，c = 8，w = 3，输入长度为12的情况下：  

{% asset_img dca_example.png Qwen2.5-1M %}  

在ppl和几个长文本benchmark上，DCA（CHUNKLLAMA）也能比其他放好一些：  

{% asset_img dca_perf.png Qwen2.5-1M %}  

## DCA和Flash Attention一起使用

DCA还有一个好处，就是可以和Flash Attention一起用。  

Pseudocode of DCA with FlashAttention：  

```python
# q: 1 x d query vector (tensor with shape [1, d])
# i: the absolute index of q (integer)
# K, V: i x d matrices for keys and values (tensors with shape [i, d])
# s: chunk size (integer)
# P_k, P_q_intra, P_q_succ, P_q_inter: position ids (lists of integers)
n = math.floor(i/s) # Number of chunks before the current chunk
# Apply rotary position embeddings to the entire key matrix K
K = apply_rotary_pos_emb(K, P_k) # K is [i, d] after embedding
# ------------- Intra-chunk Attention, casual=True -------------
q_intra = apply_rotary_pos_emb(q, P_q_intra[i]) # q_intra is [1, d]
# Select intra-chunk keys and values
K_intra = K[s*n:i] # K_intra is [(i - s*n), d]
V_intra = V[s*n:i] # V_intra is [(i - s*n), d]
# Compute output and softmax attention map for intra-chunk attention
o_intra, map_intra = Flash(q_intra, K_intra, V_intra) # o_intra is [1, d], map_intra is [1, i - s*n]
# ------------- Successive-chunk Attention, casual=False -----------
q_succ = apply_rotary_pos_emb(q, P_q_succ[i]) # q_succ is [1, d]
# Select successive-chunk keys and values
K_succ = K[s*(n-1):s*n] # K_succ is [s, d]
V_succ = V[s*(n-1):s*n] # V_succ is [s, d]
# Compute output and softmax attention map for successive-chunk attention
o_succ, map_succ = Flash(q_succ, K_succ, V_succ) # o_succ is [1, d], map_succ is [1, s]
# ------------- Inter-chunk Attention, casual=False -----------
q_inter = apply_rotary_pos_emb(q, P_q_inter[i]) # q_inter is [1, d]
# Select inter-chunk keys and values
K_inter = K[:s*(n-1)] # K_inter is [s*(n-1), d]
V_inter = V[:s*(n-1)] # V_inter is [s*(n-1), d]
# Compute output and softmax attention map for inter-chunk attention
o_inter, map_inter = Flash(q_inter, K_inter, V_inter) # o_inter is [1, d], map_inter is [1, s*(n-1)]
# Normalization
# Sum the attention maps for each attention type to get normalizers
sum_intra = map_intra.sum(-1) # sum_intra is a scalar
sum_inter = map_inter.sum(-1) # sum_inter is a scalar
sum_succ = map_succ.sum(-1) # sum_succ is a scalar
normalizer = sum_intra + sum_inter + sum_succ # normalizer is a scalar
# Concatenate attention outputs and divide by normalizer
output = (sum_intra*o_intra, sum_succ*o_succ, sum_inter*o_inter) / normalizer # output is [1, d]
```

# MInference 1.0  

MInference 1.0是一个理论有损的推理加速框架，加速的是pre-filling的阶段。在1M上下文长度的情况下，首字的推理速度相比Flash Attention-2快10倍（8B模型，单卡A100条件下）：  

{% asset_img minfer_speed.png Qwen2.5-1M %}  

虽然理论有损，但是在下游任务上的实验，可以做到很接近完全attention计算的效果。  

（作者在知乎上有亲自解读MInference 1.0，写得比较接地气，挺实在；原文在 [https://zhuanlan.zhihu.com/p/707815545](https://zhuanlan.zhihu.com/p/707815545)）  

## 长上下文的推理瓶颈  

先看下长上下文情况下推理效率的问题。这里说的长上下文，是几百k甚至M级别的长度。在这样的长度下，pre-filling的耗时就很长，其中大部分是花在attention上的。  

{% asset_img minfer_prefill.png Qwen2.5-1M %}  

一个8B的模型，在单卡A100上，对1M的输入进行pre-filling，耗时就要30分钟，说明pre-filling在很长的上下文时是一个效率瓶颈。MInference 1.0针对的就是pre-filling阶段的问题。  

细看长文本下的attention计算，作者发现attention计算在long-context场景下是很稀疏的，也就是少部分的token贡献了大部分的attention score。这也很符合直觉，毕竟在几万甚至几十万token的上下文中，不可能每个token都跟当前token紧密相关。  

对于一个128k的prompt，它的attention matrix大小是128k×128k。对模型每层的每个头，如果仅保留attention中top-4k的column，就可以召回96.4%的attention score，说明这4k token就贡献了大部分的注意力得分，而剩下的124k基本上都是near-zero element，在attention计算中贡献率很低。  

{% asset_img minfer_sparse.png Qwen2.5-1M %}  

如果把这条prompt的top4k indice应用到另一个128k prompt，召回率就只有83.7%了。  

{% asset_img minfer_dynamic_sparsity.png Qwen2.5-1M %}  

这说明不同的prompt有不同的topk pattern，attention的稀疏分布是input-dependent的，其根据输入的不同，稀疏的分布也大不相同。因此之前一些人为设计的，固定的稀疏attention pattern都无法很好解决这个dynamicity的问题：  

{% asset_img minfer_attention_pattern.png Qwen2.5-1M %}  

那么综合起来，长文本推理的瓶颈之一就是pre-filling阶段复杂的attention计算，而attention计算的特点有两个：「Attention is Dynamically Sparse」。  

一个理想的efficient long-context attention需要兼顾attention的sparsity与dynamicity：根据attention的输入，动态地估计出一个input-dependent的稀疏mask，从而完成attention的稀疏运算。  

## 注意力的稀疏模式  

虽然attention的稀疏分布是input-dependent的，但是也不是完全没有规律，还是有一些模式的。参考StreamLLM和LM-infinite（[大模型推理窗口-从有限到无限大](https://mp.weixin.qq.com/s/NaTtwURRw7lsG55QTIaVsA?token=1318369845&lang=zh_CN)）中的分析，attention的sparse pattern主要可以分成A-shape、Vertical-Slash (VS), 和 Block-Sparse三类。  

A-shape就是StreamLLM中所用的，集中在local和initial token的计算；Vertical-Slash (VS)中的vertical则是对应某些特殊token，而slash斜线则是强调相对位置关系；而Block-Sparse的分块聚集模式则相当于有多个局部重点。  

为什么这些pattern可以高效运算，但Top-K sprase attention不可以？因为这三种pattern都呈现出了空间聚集性，这就为GPU加速提供了条件。GPU kernels可以使用 64×64（A-shape和block-sparse heads）或 64×1（v-s heads）的block来高效地完成稀疏运算。而Top-K sprase attention (i.e., 对每个Q只计算top-k个K向量)，由于其过分fine-grained的sparse分布，需要花费很长的时间build sparse index，且在GPUs上使用block进行运算时会产生大量的无效运算。  

{% asset_img minfer_pattern.png Qwen2.5-1M %}  

## 实现加速  

接下来的问题就是怎么对这三种pattern的attention计算进行加速。  

> 在MInference明确三种稀疏pattern后，其将完成以下三个步骤：（1）给定一定的FLOPs budget下，为每个attention head寻找最优的pattern。（2）对每一个input，动态计算最优的sparse分布（e.g., 竖线-斜线的位置，或block的index）。（3）根据第二步得到的sprase index，进行attention的稀疏运算。  

在给定的FLOPs预算下，为了搜索最优的sparse pattern，MInference提出了Kernel-Aware Sparse Pattern Search。它能够给出（1）当前的head属于三种pattern中的哪一种（2）计算当前head的最优稀疏率。  

这里有一个关键发现，就是作者发现attention head的sparse pattern种类是input-independent的，因此sparse pattern search是可以offline提前完成的。  

## 效果  

在效果上，虽然MInference理论上是有损的，但是具体任务上效果挺好：  

{% asset_img minfer_perf.png Qwen2.5-1M %}  

# chunked prefill  

原文是《SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills》，现在在很多推理框架都用上了，比如vllm。  

（这块比较底层，我只能写个大概）

LLM在推理的时候可以分成prefill和decode。在输入很长的情况下，prefill的时候要处理所有token，显存的占用就很高，而GPU的利用率就没那么高；而在decode的时候，只处理一个token，显存的占用就没那么高。  

无论是显存占用高，利用率低还显存占率低，利用率高，都不是对GPU资源的最好利用。最好的情况就是显存占用和计算利用率都高。那么一个方法就是在一个batch里既有prefill又有decode。这样就能榨干GPU的所有能力。  

基于这个思路，大概的做法就是把各个输入prompt都切分成小一点的chunk。在推理的时候，这些chunk有些事prefill，有些是decode，通过现有显存和算力的余量来调度二者的比例，从而提升GPU整体的利用率和吞吐量。  

# Qwen2.5-1M  

终于来到了Qwen2.5-1M。  

听名字就知道，Qwen2.5这次把模型窗口提升到了1M。文中共report了三个模型，其中两个开源的，分别是wen2.5-7B-Instruct-1M和Qwen2.5-14B-Instruct-1M，另外还有一个通过API形式提供的Qwen2.5-Turbo，是个MoE模型（猜测应该是Qwen2.5-Max）。  

模型结构上，Qwen2.5-1M集成Qwen2.5模型的设计，这个没什么好说的。  

## 多阶段预训练  

Qwen2.5-1M的预训练分成5个阶段，渐进式提升训练的窗口长度。随着长度的提升，RoPE的base frequency也跟着增大：  

| Phase      | window size | base frequency |
| ----------- | ----------- | ----------- |
| 1 | 4,096 | 10,000 |
| 2 | 32,768 | 1,000,000 |
| 3 | 65,536 | 1,000,000 |
| 4 | 131,072 | 5,000,000 |
| 5 | 262,144 | 10,000,000 |

在各个阶段中，使用的数据里有75%和当前的窗口长度相同，而另外25%的数据则是较短的。这样可以保证模型长短文本能力的平衡。  

各个阶段的模型在评测benchmark RULER上的指标如下：  

{% asset_img 1m_ptm.png Qwen2.5-1M %}  

可以看到，在最后训练长度为256k的阶段之后，128k的评测指标也有明显的提升。这点和《Why does the effective context length of llms fall short?》中观察到的一致。  

## 预训练数据  

真实世界的数据很少有达到128k甚至256k长度的，而那些达到这个长度的，往往也没有在信息上有真实的长距离依赖。因此需要借助合成数据的力量。合成数据主要包括下面这三种。  

1、Fill in the Middle  

FIM起源于训练代码补全能力，这个在之前讲代码的篇章《[代码大模型(一)--业界现状](https://mp.weixin.qq.com/s/OllCcuxugOqf0aLkc9K-Mg?token=1318369845&lang=zh_CN)》中也有提到。FIM要求模型根据跟定的上文和下文，补充中间的部分。  

2、Keyword-Based and Position-Based Retrieval  

让模型根据关键词检索对应的段落，增强其识别和连接文本不同部分相关信息的能力，同时提高其对序列中位置关系的理解。  

3、Paragraph Reordering  

段落被随机排列，模型需要对它们重新排序。这对于模型生成连贯的长文本文本很重要。  

## SFT数据和Qwen-Agent  

跟Llama-3和LongAlign类似，Qwen2.5-1M从预训练语料中选择长文档，并根据这些长文档来生成QA，来进行SFT。生成的任务有很多，包括summarization, information retrieval, multi-hop question answering, reasoning, coding等等。  

那么具体怎么从长文档获得QA呢？这里就借助了Qwen-Agent框架。Qwen-Agent又是啥呢？简单来说，就是通过RAG的方式，让较短窗口的模型（比如8k），可以处理长文档的一个框架。  

Qwen-Agent包括三个level。  

1、level 1：RAG  

最naive的做法就是RAG。  

假设现在有一个输入query，以及相关的1M长度的文档，要根据这个文档应答这个query。由于模型的窗口长度只有8k，因此第一步就要把文档切分成较短的段落，比如512 token的长度。接下来，就要从这些段落里找到和query相关的部分：  

- step 1：把query中的信息和非信息部分分开。简单来说，信息部分就是需要检索的，非信息部分就是不需要检索的。比如query="回答时请用2000字详尽阐述，我的问题是，自行车是什么时候发明的？请用英文回复。"，就分为{"信息": \["自行车是什么时候发明的"\], "指令": \["回答时用2000字", "尽量详尽", "用英文回复"\]}  
- step 2：要求模型从query的信息部分推导出多语言关键词。例如，短语"自行车是什么时候发明的"会转换为{"关键词_英文": \["bicycles", "invented", "when"\], "关键词_中文": \["自行车", "发明", "时间"\]}  
- step 3：用BM25找到最相关的chunk  

{% asset_img agent_level1.png Qwen2.5-1M %}  

实践上，基于向量的检索在这里并没有太大的优势，而且带来的计算负担会大很多，因此还是用BM25比较实惠。  

2、level 2：分块阅读  

分块阅读相当于是RAG的进化版。  

上面这样的RAG很方便快捷，但是也有问题。有时会出现一些相关chunk和query重叠不足（比如一个关键信息刚好被切分成前后两个chunk），导致检索失败的情况。为了解决这个问题，分块阅读采用了一种暴力的检索方式，具体来说分为三步：  

- step 1：对于「每个」chunk，让模型评估它和query的相关性。如果相关则输出相关的句子。  
- step 2：把相关句子拿出来作为搜索输入，用BM25检索出最相关的chunk。  
- step 3：基于检索到的上下文生成答案。  

{% asset_img agent_level2.png Qwen2.5-1M %}  

3、level 3：逐步推理  

在基于文档的问题回答中，一个典型的挑战是多跳推理。  

比如这么一个问题：“与第五交响曲创作于同一世纪的交通工具是什么？”。模型首先需要确定子问题的答案，“第五交响曲是在哪个世纪创作的？”，得到“19世纪”，然后才可以知道“自行车于19世纪发明”跟问题相关。  

Tool-calling agent或ReAct agent是经典的解决方案。因此，将level 2封装为一个工具，由工具调用智能体（Lv3-智能体）调用。tool-calling agent进行多跳推理的流程如下：  

```python
Ask the Lv3-Agent a question.
while (the Lv3-Agent cannot answer the question based on its memory) {
    The Lv3-Agent proposes a new sub-question to be answered.
    The Lv3-Agent asks the Lv2-Agent the sub-question.
    Add the Lv2-Agent's response to the Lv3-Agent's memory.
}
The Lv3-Agent provides the final answer to the original question.
```

{% asset_img agent_level3.png Qwen2.5-1M %}  

为了验证这个Agent的有效性，做了下实验。使用一个32k训练+外推256k的模型，以及4k-RAG和4k-Agent三个方案。在LV-Eval和NeedleBench上，效果是这样的：  

{% asset_img agent_perf.png Qwen2.5-1M %}  

实验结果说明了以下几点：  

- 在短上下文场景中，4k-RAG的表现可能不如32k模型。这可能是由于RAG方案难以检索到正确的信息或理解多个片段造成的。  
- 随着文档长度的增加，4k-RAG越发表现出超越32k模型的趋势。这一趋势表明32k模型在处理长上下文方面并没有训练到最优的状态。  
- 4k-Agent始终表现优于32k模型和4k-RAG。它分块阅读所有上下文的方式使它能够避免原生模型在长上下文上训练不足而带来的限制。  

## SFT  

SFT也分为两个阶段。第一阶段为短文本SFT，最大长度为32768，而第二阶段使用长短数据混合，长度从32768到262144不等。  

Qwen2.5-1M还做了强化学习，不过只在8k的样本上机型了DPO。从结果上看，8k的DPO对长文本也有提升：  

{% asset_img 1m_rl.png Qwen2.5-1M %}  

不过这里的RL感觉有点匆忙，更像是跟了一下RL的风，似乎没有做一些深度的探索。  

## 推理  

1、Length Extrapolation  

推理时，首先就是使用前面提到的DCA，可以把256k的训练窗口再往外推。  

另外就是使用YaRN的注意力缩放：  

$$\mathrm{softmax}\left(\frac{\mathbf{q^Tk}}{t\sqrt{D}}\right),\mathrm{where}\sqrt{\frac{1}{t}}=0.1\ln(s)+1$$  

这个在长窗口模型中也算是标配了。Qwen2.5-1M中始终是吧缩放和DCA一起使用的。  

在NIAH和Passkey Retrieval上验证DCA的效果，还是不错的：  

{% asset_img 1m_extrapolation.png Qwen2.5-1M %}  

2、使用MInference  

模型已经在256k的窗口训练过了，通过MInference可以把推理的窗口提升到1M。为了提升吞吐量，MInference会配合chunked prefill使用。  

{% asset_img 1m_minfer_chunk.png Qwen2.5-1M %}  

3、sparse attention配合DCA  

在结合MInference和DCA的时候，发现有些case出现了performance drop。猜测可能的情况是DCA中非连续的距离影响了MInference pattern的感知，一个解决方法就是在处理这些pattern的时候，恢复距离值的连续。  

4、Sparsity Refinement  

还记得，MInference需要离线先计算每个头的pattern，但是在长度为1M的情况下，计算每个头的attention score所需要的显存太大了。Qwen2.5-1M的方法是sparsity refinement。简单来说，就是随着长度增加，跟踪MInference的pattern的attention score召回值，如果召回值低于阈值，那么久增加vertical或者slash的预算，把更多的值纳入计算。这样虽然会稍微增加MInference在推理时的计算量，但是在1M长度下能大大提升召回率：  

{% asset_img 1m_attn_refine.png Qwen2.5-1M %}  

## 效果  

{% asset_img 1m_perf.png Qwen2.5-1M %}  

# 小结  

- 在long CoT火热的背景下，长文本能力的重要性再次被强调  

- 不仅需要模型能够完成大海捞针这的任务，在几百k甚至更大的长度下进行reasoning也是一个必要的需求  

- Qwen2.5-1M用了DCA、MInference、chunked prefill和Sparsity Refinement等方案，结合很多效率和效果的优化，看起来效果是不错的  

- 数据和算法上的篇幅占比相比工程优化减少了，只搞数据和调参远远不够了  

- 工程上的有损加速方案应该是未来有前途的一个方向  

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
[DeepSeek-V3细节探索](https://www.linsight.cn/a9c496e3.html)  
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
[LLM长上下文的问题](http://www.linsight.cn/c4da56c0.html)  
[解锁大模型长上下文能力](http://www.linsight.cn/cc852861.html)  
[大模型推理窗口-从有限到无限大](http://www.linsight.cn/45ee1a6d.html)  
- 推理加速：  
[大模型推理加速-投机解码](http://www.linsight.cn/f5c015c.html)  
[大模型推理加速-MEDUSA](https://www.linsight.cn/7bbe2df6.html)  
- 对齐：  
[深度求索DeepSeek-R1详解](https://www.linsight.cn/9e4b4e6d.html)  
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
- 项目应用：  
[一个模型支持智能助手系统](https://www.linsight.cn/9c593ccd.html)  
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

【1】使用Qwen-Agent将上下文记忆扩展到百万量级，https://qwenlm.github.io/zh/blog/qwen-agent-2405/  
【2】Training-Free Long-Context Scaling of Large Language Models，https://arxiv.org/abs/2402.17463  
【3】Qwen2.5-1M Technical Report，https://arxiv.org/abs/2501.15383  
【4】MInference 1.0: Accelerating Pre-filling for Long-Context LLMs via Dynamic Sparse Attention，https://arxiv.org/abs/2407.02490  
【5】单卡可Million-context推理TTFT 10倍加速 - MInference 1.0，https://zhuanlan.zhihu.com/p/707815545  
【6】SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills，https://arxiv.org/abs/2308.16369  
【7】vLLM调度器解密（下）：chunked prefill是如何进一步优化的？，https://zhuanlan.zhihu.com/p/6144374775  

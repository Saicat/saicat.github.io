---
title: Qwen2技术报告
tags:
  - NLP
  - LLM
  - transformer
  - 技术报告
  - Qwen
  - MoE
  - 预训练
  - 对齐
categories:
  - CS
  - NLP
  - LLM
abbrlink: a8f8b641
date: 2024-07-17 22:01:21
---

【本文已在同名 微信公众号 / 知乎 / [个人博客linsight.cn](http://www.linsight.cn/) 上线】  

***  

不久前Qwen2发布了4个dense模型和1个MoE模型，模型规模从0.5B到57B，实用效果都还不错。现在技术报告终于来了，来看下技术报告里披露了那些有用的信息。  

# 模型  

Qwen2的5个模型结构和训练token数如下表  

{% asset_img model.png Qwen2 %}  

## tokenizer  

Qwen2使用和Qwen1一样的tokenizer，压缩率比较好，也支持多语言。  

词表包含151,643个常规token和3个control token。而在训练的时候，为了方便分布式训练，实际的vocab size设到了151936，多出来的部分实际是没有用到的。  

## dense model  

- 和之前的版本不同，Qwen2都使用了GQA而不是MHA。  
- SwiGLU、RoPE、RMSNorm + pre-norm和之前一样，属于常规设置。  
- 参考了苏神在《Bias项的神奇作用：RoPE + Bias = 更好的长度外推性》里的做法，在QKV加上bias以提升RoPE长度外推的能力。  
- 参考《Training-free long-context scaling of large language models》，实现了Dual Chunk Attention（DCA），并使用YaRN对attention weights进行rescale以获取更好的长度外推效果。  

## MoE model  

Qwen2-57B-A14B使用了fine-grained expert和shared expert，都是已经证明效果比较好的做法。  

Qwen2-57B-A14B是从Qwen2-7B初始化的，类似《Sparse upcycling: Training mixture-ofexperts from dense checkpoints》的做法，但是更加强调了在细粒度专家之间实现多样化。  

假设专家大小为h_E, 专家数量为n，用于初始化MoE模型的原始FFN层大小为h_FFN，那么FFN层将被复制 ⌈n × h_E / h_FFN⌉ 次。这样可以确保和任意专家大小和专家数量兼容。  

为了促进每个FFN copy内部的多样性，参数在intermeidate维度会做shuffle。这样使得后面得到的每个细粒度专家都能从不同特征初始化。  

在这个基础上，每个细粒度专家内部有50%的参数会用随机初始化覆盖，只保留50%训练过的参数。这样可以增强模型在训练中探索的能力。

# 预训练  

## 预训练数据  

Qwen2预训练语料的处理包含了几个关键领域：  
- Quality Enhancement：包括使用之前版本的Qwen模型来过滤掉低质量数据，和合成高质量的预训练数据。  
- Data Expansion：相比Qwen1.5，Qwen2多收集很多代码数据、数学数据和囊括30种语言的多语言数据。  
- Distribution Improvement：在小规模的数据上做了数据配比的实验，优化不同来源和领域的数据混合。  

基于以上的工作，Qwen2最终得到了7T高质量数据。除了0.5B模型，其他dense模型都是在7T数据上训练，Qwen2-57B-A14B则是在4.5T数据上训练的。而在0.5B模型上，尝试使用了放松清洗阈值而得到的12T数据进行了训练，但是相比7T高质量数据，12T数据的训练并没有带来进一步的提升。  

## 长上下文训练  

在预训练的最后阶段，把训练窗口从4096提升到32,768以提升模型的长上下文能力。这个阶段使用了长度更长的文本。  

除了数据的变化，RoPE的base也从10,000提升到1,000,000。再加上YaRN和Dual Chunk Attention，Qwen2模型可以在131,072长度的窗口上保持比较好的效果。  

# POST-TRAINING  

Qwen2的对齐阶段包括SFT和RLHF。主要针对coding，mathematics，logical reasoning，instruction following 和 multilingual comprehension 提升效果。  

## 数据  

在数据的获取上，Qwen2的方法更多关注在“scalable alignment with minimal human annotation”（《Towards scalable automated alignment of LLMs: A survey》）。  

构建数据的过程主要包括两个步骤，collaborative data annotation 和 automated data synthesis。  

1、collaborative data annotation  

- 本体提取：借助InsTag（《#InsTag: Instruction tagging for analyzing supervised fine-tuning of large language models》）的tagger，再加上人工精炼保证本体提取的准确。  
- 指令选择：每条带有tag的指令都根据tag多样性、语义丰富性、复杂性和意图完整性进行了评估。基于这些标准选择有代表性的指令（《How abilities in large language models are affected by supervised fine-tuning data composition》）。  
- 指令进化：为了丰富指令数据集，采用了self-evolution策略（《Tree-Instruct: A preliminary study of the intrinsic relationship between complexity and alignment》），用Qwen模型对现有指令添加约束或要求，增加其复杂性，并确保数据集中难度级别的多样性。  
- 人类打标：使用不同的生成策略和不同规模的Qwen模型获取一条指令的多个response。标注者这些response进行排序，确保最佳response符合标准，最终得到demonstration和preference数据。  

2、automated data synthesis  

- 拒绝采样：对于数学或类似的有明确最终答案的任务，应用了拒绝采样（《Scaling relationship on learning mathematical reasoning with large language models》）来提高solution的质量。LLM被用来为每条指令生成多个response。那些准确且被模型认为是合理的response就保留下来。通过对比正确和错误的response还可以获得偏好数据。  
- 执行反馈：对于编程任务，LLM被用来生成solution和相关测试用例。这些solution的有效性通过执行测试用例来评估。这种方法也适用于评估指令遵循情况（《Self-play with execution feedback: Improving instruction-following capabilities of large language models》）。比如对有长度限制的指令，LLM的任务是生成一个Python验证函数，以确保response的长度要求。  
- 数据再利用：对于没有专门训练的标注者来说，在文学写作任务中给出好的答案是很困难的。为了解决这个问题，收集了高质量的文学作品，并使用LLM开发不同详细程度的指令。这些指令与原作品配对，作为训练数据。比如为了获取角色扮演数据，先从知识库（如维基百科）中获取详细的角色资料，并指导LLM生成相应的指令和response（《Large language models are superpositions of all characters: Attaining arbitrary role-play via self-alignment》）。这个过程类似于阅读理解任务，确保了角色资料的完整性。  
- Constitutional Feedback：参考《Constitutional AI: Harmlessness from AI feedback》的做法，制定了response要遵循的规则和原则，用于知道模型生成合情合理合法合规的response。  

## SFT  

- 数量>500,000条样本上训练  
- 训练2个epoch  
- lr = 7e-6，最终decay到7e-7  
- weight decay = 0.1  
- gradient clip = 1.0  
- seq length = 32,768  

## RLHF  

强化学习使用DPO，并参照《Online merging optimizers for boosting rewards and mitigating tax in alignment》，用Online Merging Optimizer以缓解alignment tax的影响。  

# 评测  

## base模型  

各个规模的base模型评测结果如下。  

1、0.5B模型和1.5B模型  

{% asset_img eval_base_small.png 评测 %}  

2、7B模型  

{% asset_img eval_base_7B.png 评测 %}  

3、32B模型和57B-A14B模型  

{% asset_img eval_base_large.png 评测 %}  

## INSTRUCTION-TUNED模型  

各个规模的it模型评测结果如下。  

1、0.5B模型和1.5B模型  

{% asset_img eval_chat_small.png 评测 %}  

2、7B模型  

{% asset_img eval_chat_7B.png 评测 %}  

3、32B模型和57B-A14B模型  

{% asset_img eval_chat_large.png 评测 %}  

## 长窗口  

Qwen2模型的长窗口能力在3个评测集上进行了评估。  

1、the Needle in a Haystack   

{% asset_img eval_needle.png 评测 %}  

2、NeedleBench（OpenCompass）  

（见下图）  

3、LV-Eval  

{% asset_img eval_long.png 评测 %}  

# 小结  

- Qwen2 MoE模型的初始化思路可以作为从dense模型upcycling的一个参考。  
- 预训练数据量来到10T token，这里12T训练数据没有更大收益的原因，除了数据质量外，猜测可能是0.5B模型本身容量有限导致。  
- 在数据配比上，报告没有给出太多信息，但这块很重要，各家应该有些压箱底信息没有舍得给出来。  

***  

读到这了，来一发点赞收藏关注吧~

博客：[http://www.linsight.cn/](http://www.linsight.cn/)  
知乎：[Linsight](https://www.zhihu.com/people/us4ever)  
微信公众号：Linsight  
![](/images/qrcode.jpg)  

***  

【往期文章】  
[MoE模型的前世今生](http://www.linsight.cn/44e38c1b.html)  
[DeepSeek-V2和MLA](https://www.linsight.cn/83c49df0.html)  
[昆仑万维-SkyworkMoE](https://www.linsight.cn/1d5bcd45.html)  
[成本10w刀的JetMoE](https://www.linsight.cn/f3acf042.html)  
[MoE的top-p routing](https://www.linsight.cn/224c42da.html)  
[对MoE模型的一些观察](https://www.linsight.cn/5e1d14b3.html)  
[从loss视角理解大模型涌现能力](https://www.linsight.cn/f5fb75e4.html)  
[LLM长上下文的问题](http://www.linsight.cn/c4da56c0.html)  
[解锁大模型长上下文能力](http://www.linsight.cn/cc852861.html)  
[大模型推理窗口-从有限到无限大](http://www.linsight.cn/45ee1a6d.html)  
[理解Attention:从起源到MHA,MQA和GQA](http://www.linsight.cn/3dc22f96.html)  
[大模型推理加速-投机解码](http://www.linsight.cn/f5c015c.html)  
[大模型推理加速-MEDUSA](https://www.linsight.cn/7bbe2df6.html)  
[LLM的重复生成和ICL](https://www.linsight.cn/7381cae3.html)  
[大模型偏好对齐-DPO](http://www.linsight.cn/473f2b43.html)  
[大模型偏好对齐-ODPO](http://www.linsight.cn/da871ebe.html)  
[大模型偏好对齐-simPO](http://www.linsight.cn/280fa97a.html)  
[大模型偏好对齐-IPO](http://www.linsight.cn/4fe7b810.html)  
[Yi技术报告-划重点看细节](http://www.linsight.cn/41b6a819.html)  
[MiniCPM](https://www.linsight.cn/376db710.html)  
[GLM4报告的一些技术点](https://www.linsight.cn/a5206abd.html)  
[Gemma2](https://www.linsight.cn/cf3f1f81.html)  
[苹果的OpenELM](https://www.linsight.cn/f845f3e4.html)  
[从Yuan2.0到Yuan2.0-M32](https://www.linsight.cn/3df0cd42.html)  
[bilibili的index-1.9B](https://www.linsight.cn/770b63e1.html)  
[transformer中normalization的二三事](http://www.linsight.cn/6a40bfa5.html)  
[从代码实现看normalization-到底做了什么](http://www.linsight.cn/b70b4a2d.html)  
[稀疏注意力计算:sliding window attention](http://www.linsight.cn/c61d17e3.html)  
[理解LLM位置编码:RoPE](http://www.linsight.cn/a051710f.html)  
[RoPE的远距离衰减](https://www.linsight.cn/f0902f1a.html)  
[大模型算法题(1)](http://www.linsight.cn/3345028a.html)  
[大模型算法题(2)](http://www.linsight.cn/ad0bba9d.html)  
[大模型算法题(3)](http://www.linsight.cn/1736008.html)  
[大模型算法题(4)](http://www.linsight.cn/1736008.html)  
[大模型算法题(5)](http://www.linsight.cn/336f2f3e.html)  
[大模型算法题(6)](http://www.linsight.cn/7c04944d.html)  
[大模型算法题(7)](https://www.linsight.cn/dd614e12.html)  

***  

# Reference  

【1】QWEN2 TECHNICAL REPORT https://arxiv.org/abs/2407.10671  

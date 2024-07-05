---
title: GLM4报告的一些技术点
tags:
  - NLP
  - LLM
  - transformer
  - 技术报告
  - 预训练
  - 微调
  - 强化学习
  - agent
categories:
  - CS
  - NLP
  - LLM
abbrlink: a5206abd
date: 2024-06-27 14:51:38
---

【本文已在同名 微信公众号 / 知乎 / [个人博客linsight.cn](http://www.linsight.cn/) 上线】  

***  

智谱的GLM系列模型在中文领域一直是比较受关注的，特别是最新的GLM-4，个人在使用体验上，感觉已经可以满足大部分日常需求。  

最近智谱开源了GLM-4-9B，也发了相关的技术报告，总结了整个系列从预训练到agent的各个方案。  

{% asset_img glm.png GLM %}  

这里简单梳理一下一些技术相关的点。  

# 数据  

报告中提到的关于预训练数据的几个内容：  
- 数据的处理分为deduplication, filtering和tokenization。  
- deduplication包括exact deduplication和fuzzy deduplication。  
- 去重提升了数据分布的多样性，这对模型训练结果有很大的影响。  
- tokenization采用byte-level BPE，基于tiktoken的cl100k_base进行训练，获得150k的词表。  
- 最后的训练数据中re-weight了各个来源的数据，提高了高质量数据如wiki、books的比例。  
- 最终获得10T的token。  

# 模型结构  

模型结构上的改动和发现：  
- No Bias Except QKV：除了QKV之后都不使用bias，这样可以提升训练速度。此外还发现不使用bias在长度外推的能力有微微提升。  
- 使用RMSNorm和SwiGLU。  
- 使用2D的RoPE。  
- 使用GQA，以减少推理的KV cache需求。由于GQA相比MHA有更少的参数，把FFN的大小增加到10/3的hidden size来保持模型总参数基本不变。  

# Alignment  

SFT中，发现真实的人类prompt和交互比template-based的人造数据和模型生成的答案要好得多。  

# ChatGLM Techniques  

在训练ChatGLM的路上，智谱总结了不少经验：  
- LongAlign：《Longalign: A recipe for long context alignment of large language models》能把GLM-4的推理长途提升到128k，并且效果达到Claude 2和GPT-4 Turbo (1106)的水平。  
- ChatGLM-Math：《Chatglm-math: Improving math problem-solving in large language models with a self-critique pipeline》给出了一套通过self-critique提升数学能力的方法。  
- ChatGLM-RLHF：《Chatglm-rlhf: Practices of aligning large language models with human feedback》总结了PPO和DPO的应用。  
- Self-Contrast：《Extensive self-contrast enables feedback-free language model alignment》给出了Self-Contrast的策略，用于让模型自动生成大量负样本用于RLHF，避免了投入大量的人力。  
- AgentTuning：《Agenttuning: Enabling generalized agent abilities for llms》包括agent的训练框架和AgentInstruct instruction-tuning数据集。  
- APAR：《Apar: Llms can do auto-parallel auto-regressive decoding》总结了auto-parallel auto-regressive的并行解码生成策略。  

# GLM-4 All Tools  

GLM-4 All Tools可以说是前面这些技术探索的集大成者，效果确实不错。  

{% asset_img all_tools.png All Tools %}  

# 评测  

GLM在agent、function call、all tools的评测整理了一些数据集和方案，后续可以参考使用。  

***  

读到这了，来一发点赞收藏关注吧~

博客：[http://www.linsight.cn/](http://www.linsight.cn/)  
知乎：[Linsight](https://www.zhihu.com/people/us4ever)  
微信公众号：Linsight  
![](/images/qrcode.jpg)  

***  

【往期文章】  

[MoE模型的前世今生](http://www.linsight.cn/44e38c1b.html)  
[昆仑万维-SkyworkMoE](https://www.linsight.cn/1d5bcd45.html)  
[成本10w刀的JetMoE](https://www.linsight.cn/f3acf042.html)  
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

【1】ChatGLM: A Family of Large Language Models
from GLM-130B to GLM-4 All Tools https://arxiv.org/abs/2406.12793  

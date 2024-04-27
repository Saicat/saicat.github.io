---
title: 实现大模型(超)长上下文能力之路
abbrlink: 812c93f3
date: 2024-04-14 14:41:31
tags:
  - NLP
  - LLM
  - transformer
  - 长上下文
  - 预训练
  - attention
categories:
  - CS
  - NLP
  - LLM
---

【本文已在同名 微信公众号 / 知乎 / [个人博客linsight.cn](http://www.linsight.cn/) 上线】  

***

步入2024年Q2，大模型在RAG、文档对话、大模型Agent能力等方向的发展持续升温。在我平时的日常生活和工作中，大模型工具提供的文档总结、文本润色、代码生成等能力已经是我离不开的帮手，甚至在一些复杂或者不熟悉的场景上，大模型也已经能提供一些比较专业的帮助。  

在这些方向上，大模型(超)长上下文的能力都是基础。无论是使用详细的CoT/ToT，还是通过多篇检索文档提供专业知识，抑或是使用相关样例提升回复质量，都需要模型具备处理很长的输入信息的能力。这不仅要求模型在较长的位置编码下依然具有良好的语言建模能力，而且还需要模型能够进行长距离的、细致的阅、准确的阅读和理解。  

本篇将梳理关于提升模型长上下文能力的一些主要工作，提供大模型长上下文能力的big picture。  

# 

直接训 --> 数据 --> 训练效率 --> 推理成本&效率，kv cache

改结构：MHA-->GQA、MQA，swa，ring attention、线性注意力

技巧型，PoSE

低资源做法：外推，插值、ntk插值、logn、yarn，rope abf

无限长：

StreamingLLM

Leave No Context Behind:Efficient Infinite Context Transformers with Infini-attention

# 小结  

***

读到这了，来一发点赞收藏关注吧~

博客：[http://www.linsight.cn/](http://www.linsight.cn/)  
知乎：[Linsight](https://www.zhihu.com/people/us4ever)  
微信公众号：Linsight  
![](/images/qrcode.jpg)  

***  

往期文章  

[Yi技术报告-划重点看细节](http://www.linsight.cn/41b6a819.html)  
[transformer中normalization的二三事](http://www.linsight.cn/6a40bfa5.html)  
[稀疏注意力计算:sliding window attention](http://www.linsight.cn/c61d17e3.html)  
[理解Attention:从起源到MHA,MQA和GQA](http://www.linsight.cn/3dc22f96.html)  
[LLM长上下文的问题](http://www.linsight.cn/c4da56c0.html)  
[理解LLM位置编码:RoPE](http://www.linsight.cn/a051710f.html)  
[大模型算法题(1)](http://www.linsight.cn/3345028a.html)  
[大模型算法题(2)](http://www.linsight.cn/ad0bba9d.html)  

***  

# Reference  
【1】  
【2】
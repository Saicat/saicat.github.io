---
title: 成本10w刀的JetMoE
tags:
  - NLP
  - LLM
  - transformer
  - MoE
categories:
  - CS
  - NLP
  - LLM
abbrlink: f3acf042
date: 2024-06-26 11:22:35
---

【本文已在同名 微信公众号 / 知乎 / [个人博客linsight.cn](http://www.linsight.cn/) 上线】  

***  

JetMoE是由MIT、Princeton等几个学术机构发布的MoE模型，其总参数量为8B，激活参数量为2B。  

训练JetMoE的总花费约为10w美元，而JetMoE在各个benchmark上都有不错的效果，这样看训练这个模型算是比较经济实惠的了。  

MoE的基础内容可以看之前梳理的 [MoE模型的前世今生](http://www.linsight.cn/44e38c1b.html)。  

# 模型设计  

## 结构  

在模型结构上，和目前一些主流的模型如Deepseek MoE、Mixtral 8x7B、Qwen-MoE等有点不同，JetMoE不仅在FFN层应用Sparsely-gated Mixtureof-Experts（SMOE）的设计，而且参考了《Moduleformer: Learning modular large language models from uncurated data》的做法，把attention层也设计成了混合专家的结构，如下图所示。  

{% asset_img structure.png 结构 %}  

attention层的混合专家结构也叫MoA（ Mixture of Attention heads (MoA)，是由《Mixture of Attention Heads: Selecting Attention Heads Per Token》提出的。  

MoA和FFN层的MoE一样，每个attention层包含多个attention expert。而每个attention expert e包括4个形状为 $\mathbf{R}^{D_{emb}\times D_{att}}$ 的矩阵： $\mathbf{W}_q^e,\mathbf{W}_k,\mathbf{W}_v,\mathbf{W}_o^e$。其中 $D_{att}=H\times D_{head}$，H是每个attention expert的attention head数量。每个attention expert内部和常规的注意力层是一样的。  

每层attention expert中的 $\mathbf{W}_k$ 和 $\mathbf{W}_v$ 这两个矩阵的参数在同个attention层的多个expert之间共享，这样可以减少一些参数量和计算量，提升计算效率。而每个attention expert保留各自的 $\mathbf{W}_q^e$ 和 $\mathbf{W}_o^e$。  

对于一个输入的vector x，首先用2个共享的矩阵获得k和v  

$$\begin{aligned}\mathbf{k}&=\mathbf{W}_{k}\mathbf{x}\\\mathbf{v}&=\mathbf{W}_{v}\mathbf{x}\end{aligned}$$  

而在gating function选择了expert之后，再在attention expert内部进行标准的attention计算：  

$$\begin{aligned}&\mathbf{q}_{e}=\mathbf{W}_{q}^{e}\mathbf{x}\\&\mathbf{a}_{e}=\mathrm{МНА}\left(\mathbf{q}_{e},\mathbf{k},\mathbf{v}\right)\\&\mathbf{o}_{e}=\mathbf{W}_{o}^{e}\mathbf{a}\end{aligned}$$  

JetMoE的FFN层的设计和gating的设计就是常规的top-k gating MoE，就不再赘述。  

JetMoE的具体模型参数如下  

{% asset_img model_param.png 模型参数 %}  

每层有8个expert，每个token激活2个expert。  

## 负载均衡  

在负载均衡上，参考Switch Transformer，加入了frequency-based auxiliary loss：  

$$loss_b=N\sum_{i=1}^Nf_iP_i$$  

其中N是expert数量，$f_i$ 是分配给expert i的token占比，$P_i$ 是router分配给expert i的概率占比。  

此外还加入了ST-MoE中的z-loss来提升训练稳定性：  

$$loss_z=\frac1B\sum_{i=1}^B\left(\log\sum_{j=1}^N\exp(x_j^i)\right)^2$$  

x是router给出的logits，B是token数。  

通过两个超参把这两个负载平衡的loss加入到训练loss中  

$$loss=loss_{lm}+\alpha loss_b+\beta loss_z$$  

训练中 $\alpha=0.01$，$\beta=0.001$。  

# 训练数据  

JetMoE预训练数据使用了真实数据和合成数据两种。  

真实数据：  
- RefinedWeb：从总共5T的token里抽取了600B来训练  
- StarCoder：包含86种代码语言  
- Dolma：包含3T token的英文数据集  
- The Pile：825GB的英文数据集  
- 其他：还使用了Proof-Pile-2、OpenWebMath、StackMathQA、OpenAssistant、xP3x、CommitPackFT这些规模比较小、质量比较高的数据集。  

合成数据：  
- OpenHermes 2.5  
- UltraTextbooks  
- UltraChat 200k  
- 其他：还使用了TemplateGSM、Magicoder-Evol-110K、Evol-Code Alpaca、Code-290k-ShareGPT这些规模比较小、质量比较高的数据集。  

# 训练  

JetMoE基于Megatron框架进行训练，仅使用pipeline parallelism而不expert parallelism。训练过程用了96个H100，消耗约30,000个GPU hour，训练了大概1.25T token的数据。  

一些训练设置：  
- 使用AdamW优化器  
- maximum learning rate = 5e-4  
- batch size = 4M  
- sequence length = 4096  
- learning rate schedule = WSD，warmup = 10B token，decay = 250B token  

参考MiniCPM的做法，把训练分为两个阶段：
- phase1：warmup and stable learning rate；使用的数据集包括RefinedWeb, Starcoder, The Pile, peS2o from Dolma, and OpenWebMath  
- phase2:decay learning rate；使用了更多的高质量数据。  

phase1和phase2的具体数据混合情况如下  

{% asset_img data1.png 数据 %}  

{% asset_img data2.png 数据 %}  

# Alignment  

JetMoE用Distilled Supervised Fine-Tuning（dSFT）的方法对模型进行微调。dSFT就是用prompt获取更强模型的应答结果，用来训练别的模型。  

JetMoE使用Zephyr-7b-beta的chat template获取GPT-4和Claude的答案用来训练JetMoE，所用的数据有：  
- UltraChat 200k  
- Airoboros-3.2  
- Code-Feedback  
- Orca-math-word-problems-200k  
- SystemChat  
- Capybara  

训练配置：  
- lr = 2e-5  
- batch size = 128  
- epoch = 3  

此外，在SFT的基础上，还用了Distilled Direct Preference Optimization (dDPO)进一步优化模型。  

所用的数据集是UltraFeedback，包含了preference数据对。  

训练配置：  
- lr = 5e-7  
- batch size = 128  
- epoch = 1  

# 效果  

在各个benchmark的效果如下  

{% asset_img evaluation.png 评测 %}  

{% asset_img mtbench.png 评测 %}  

# 小结  

JetMoE算是一次比较低成本的MoE训练实践，其中大部分的训练设置都是沿用了之前多个工作总结下来的经验。这些经验基本上可以保证训练不出什么大问题了，是相对比较成熟的了。  

常规的内容之外，attention expert可能是一个可以探索的方向。  

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

【1】JetMoE: Reaching Llama2 Performance with 0.1M Dollars https://arxiv.org/abs/2404.07413  

---
title: 苹果的OpenELM
tags:
  - NLP
  - LLM
  - transformer
  - 技术报告
  - 苹果
categories:
  - CS
  - NLP
  - LLM
abbrlink: f845f3e4
date: 2024-07-02 17:14:36
---

【本文已在同名 微信公众号 / 知乎 / [个人博客linsight.cn](http://www.linsight.cn/) 上线】  

***  

苹果开源的OpenELM系列大模型包括整套规模（0.27B、0.45B、1.08B、3.04B）的最终模型、一些模型的中间checkpoint，training log以及训练框架和具体配置等，这些资源都在https://github.com/apple/corenet可以找到，训练信息算是给得比较全面的了。  

在苹果给出的效果对比中，OpenELM-1B算是同规模模型效果比较好的，比训练数据量多一倍的OLMo也要好一些，具体如下表  

{% asset_img intro.png OpenELM %}  

这里整理一下OpenELM的方案细节。  

# 模型设计  

OpenELM模型设计：  
- 所有linear层都没有使用bias  
- pre-norm + RMSNorm  
- RoPE  
- GQA  
- SwiGLU FFN  
- 和LLAMA相同的tokenizer  

这些都是比较常规的设计。和目前其他主流模型比较不同的是，苹果参考《Delight: Deep and light-weight transformer》，在OpenELM采用了layer-wise scaling的设计。  

通常来说，主流模型的设计是每层都使用一样的超参，比如注意力头的数量，和hidden size的大小等。但是他们认为这样的设计在参数量的分配上不够高效，不能很好发挥这么多参数的效果，因此对每层的超参进行scaling。  

假设模型共有N层，每层的宽度为 $d_{model}$，MHA有 $n_{h}$ 个头，每个头的大小为 $d_h=\frac{d_{model}}{n_h}$，FFN层的大小为 $d_{\mathrm{FFN}}=m\cdot d_{model}$，其中m是FFN multiplier。  

layer-wise scaling使用 $\alpha$ 和 $\beta$ 两个参数来调整模型每层的attention head数量 $n_{h}$ 和FFN multiplier m。  

具体调整如下：对于第i层，有  

{% asset_img formula.png 公式 %}  

其中 $\alpha_{min}$ 和 $\alpha_{max}$ 是调整注意力头数量的超参，$\beta_{min}$ 和 $\beta_{max}$ 是调整宽度的超参。实践中，使用了 $\alpha_{min}=0.5$，$\alpha_{max}=1.0$，$\beta_{min}=0.5$，$\beta_{max}=4.0$。  

（后面有整理各个模型结构和训练超参的表）  

# 数据  

预训练数据来源包括：  
- RefinedWeb  
- deduplicated PILE  
- a subset of RedPajama  
- a subset of Dolma v1.6  

总共1.8T token，各个来源具体token数量如下表  

{% asset_img data.png 数据 %}  

数据中小于200 character或者小于256 token的数据都会被筛出来不使用。  

# 训练  

训练超参：  
- 总共约350k step  
- AdamW optimizer  
- cosine learning rate schedule，warmup=5k  
- weight decay = 0.1  
- gradient clipping = 1.0  

各个规模模型的结构超参、训练设置和资源消耗如下表  

{% asset_img pretrain_hp.png 超参 %}  

微调超参如下  

{% asset_img sft_hp.png 超参 %}  

# evaluation  

下游在三大类型任务上评测了OpenELM：  
- Standard zero-shot tasks  
- OpenLLM leaderboard tasks  
- LLM360 leaderboard tasks  

{% asset_img eval_1.png 评测 %}  

pretrained model在Standard zero-shot tasks的7个任务上，不同checkpoint的效果如下图  

{% asset_img eval_2.png 评测 %}  

可以看到随着训练步数增加，效果有上升趋势。  

此外，研究人员观察到使用最后5个相隔5000 step的模型checkpoint进行滑动平均获得的模型参数，比单纯使用最后一个checkpoint要好，这可能是因为多个checkpoint平均可以去掉一些noise的影响。  

后续评测都是使用平均后的模型。  

OpenELM预训练模型在各个类型的任务上和其他模型对比如下  

{% asset_img eval_3.png 评测 %}  

OpenELM微调前后的效果如下  

{% asset_img sft_result.png 评测 %}  

对OpenELM分别使用LoRA和DoRA微调的效果对比如下  

{% asset_img peft_eval.png 评测 %}  

# 小结  

OpenELM相比其他模型，最主要的变化是使用了layer-wise scaling，看起来效果不错，但是苹果并没有在消融实验整个各个变化的有效性。  

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
[GLM4报告的一些技术点](https://www.linsight.cn/a5206abd.html)  
[Gemma2](https://www.linsight.cn/cf3f1f81.html)  
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

【1】OpenELM: An Efficient Language Model Family with Open Training and Inference Framework https://arxiv.org/abs/2404.14619  

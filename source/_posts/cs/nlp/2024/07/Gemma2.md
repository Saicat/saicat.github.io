---
title: Gemma2
tags:
  - NLP
  - LLM
  - transformer
  - 技术报告
  - Gemma2
categories:
  - CS
  - NLP
  - LLM
abbrlink: cf3f1f81
date: 2024-07-01 16:30:28
---

【本文已在同名 微信公众号 / 知乎 / [个人博客linsight.cn](http://www.linsight.cn/) 上线】  

***  

Google发布Gemma2了，包括2B、9B和27B三个规模。  

其中9B和27B的模型在huggingface上已经可下载，包括base模型和fine-tuned模型，2B模型晚点也会放出来。  

{% asset_img intro.png Gemma2 %}  

来看看有啥可关注的技术点。  

# 结构设计  

3个规模的模型结构设计如下  

{% asset_img model.png 模型 %}  

一些设计点和Gemma1一样：  
- decocer-only  
- RoPE  
- context length = 8192  
- GeGLU  

除此之外相比一代也有一些变化点，下面一一看下。  

## sliding window attention  

Gemma2每两层使用一个sliding window attention层，sliding window的大小为4096。  

关于sliding window的内容，可参考[稀疏注意力计算:sliding window attention](http://www.linsight.cn/c61d17e3.html)。  

理论上，这样的设计可以在减少计算资源需求的情况下，保持一定的长文本能力（得益于这部分没有使用sliding window的层）。  

Mistral的早期版本也用了sliding window attention，但后来又去掉了。感觉是否使用sliding window attention还得看下游场景的需求。  

## logit soft-capping  

参考Gemini 1.5，Gemma2使用了logit soft-capping。  

soft-capping是一种在不进行truncation的情况下，防止logits过度增长的方法。  

具体来说，就是对logits进行如下的操作：  

```python  
logits = soft_cap ∗ tanh(logits / soft_cap)  
```  

这样logits的最终值就可以保持在(-soft_cap, +soft_cap)区间上，就能够在不损失太多信息的情况下稳定训练。  

soft-capping应用在模型的final layer和每个attention layer。对于9B和27B模型，final layer和attention layer的soft_cap值分别是30.0和50.0。  

这里有个问题是Flash Attention / SDPA不支持soft-capping，因此微调训练的时候推荐使用eager attention而非SDPA。  

至于推理，研究人员发现去掉soft-capping对结果影响不大，因此推理的时候可以去掉然后用原先的加速方案加速。当然，这样依然有小概率出现结果被改变的情况，所以推理的时候是否移除soft-capping，可能需要根据下游任务来定。  

## 其他  

Gemma2报告还提到：  

（1）post-norm 和 pre-norm 都使用 RMSNorm  

（2）使用group num = 2的GQA  

# 训练  

## 预训练数据  

2B模型总共训练了2B token，9B模型训练了8T token，而27B模型训练了13T，是第一代的两倍。data mixture通过和Gemma1类似的消融方法确定，这里没有给出具体的数据。  

Gemma2所用的tokenizer和Gemma1、Gemini一样，基于BPE，大小为256k。  

## knowledge distillation  

Gemma2 27B模型是直接进行预训练的，而2B和9B模型没有通过next token prediction的任务训练，而是使用了知识蒸馏的方法：  

$$\min_{P_S}\sum_x-P_T(x\mid x_c)\log P_S(x\mid x_c)$$  

实操时，teacher model先离线跑出每个token的概率保存下来。由于vocabulary太大了，所以保存的时候只保存一个subset。（长尾部分置零，头部重新归一化概率？这里报告没有细说）  

而在SFT的时候，通常的做法是把synthetic data和真实prompt数据喂给teacher模型，获取对应的response，然后用常规的distillation的方式进行训练。Zephyr和OpenHermes就是这样的做法。  

这样的训练方式虽然有效，但是有可能出现train-inference mismatch的问题，即student model在推理的时候出现和训练时不同的分布。  

为了解决这个mismatch的问题，这里Gemma2参考《On-policy distillation of language models: Learning from self-generated mistakes》，使用on-policy distillation的方法。  

具体来说，就是由student对prompt生成response，然后最小化teacher和student在这个response上的KL divergence。这样就不会出现train-inference mismatch的问题了。  

得到SFT模型之后，这里还进行了RLHF进一步提升模型效果。  

post-training所用的特殊token和格式样例如下  

{% asset_img format.png formatting %}  

{% asset_img example.png formatting %}  

Gemma2报告中还提到了参考《Warp: On the benefits of weight averaged rewarded policies》进行了model merging。  

以前训练Bert的时候就用了Exponential Moving Average对多个checkpoint进行平均，整体来说确实是略有提升。  

# 消融实验  

Gemma2还做了一些消融实验。  

（1）distillation versus from scratch  

相比直接从零训练，蒸馏的效果略好一些，如下所示  

{% asset_img ablation_1.png 消融实验 %}  

（2）impact of distillation w.r.t. model size  

使用相同的7B模型作为teacher model，不同规模的student模型都可以有相对稳定的收益，没有明显衰减  

{% asset_img ablation_2.png 消融实验 %}  

（3）GQA versus MHA  

在9B模型上对比GQA和MHA的效果，GQA要略好一些（这就有点反直觉了）  

{% asset_img ablation_3.png 消融实验 %}  

（4）wide versus deep  

在相同参数量下，更深的9B模型比更宽的9B模型更好，这个和以往的认知的相同的：模型深度对效果影响更大  

{% asset_img ablation_4.png 消融实验 %}  

（5）changing sliding window size  

使用不同大小的sliding window，在评测集上的ppl差别并不大  

{% asset_img ablation_5.png 消融实验 %}  

（6）impact of formatting  

相对Mistral，Gemma2的得分方差相对更小一些  

{% asset_img ablation_6.png 消融实验 %}  

# 评测  

在各个benchmark的效果：  

{% asset_img eval1.png eval %}  

{% asset_img eval2.png eval %}  

# 小结  

Gemma2集合了一些模型、训练上的改进，最大的点应该就是知识蒸馏，而结构上的soft-cappint看来也有一些效果。另外巨大的数据量再次证明了中/小模型还能吸收更多的数据。  

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

【1】Gemma 2: Improving Open Language Models
at a Practical Size https://storage.googleapis.com/deepmind-media/gemma/gemma-2-report.pdf  
【2】https://huggingface.co/blog/gemma2  
【3】稀疏注意力计算:sliding window attention http://www.linsight.cn/c61d17e3.html  

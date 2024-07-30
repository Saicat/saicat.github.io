---
title: 模型平均 -- model soup
tags:
  - NLP
  - LLM
  - transformer
  - 微调
  - 模型融合
categories:
  - CS
  - NLP
  - LLM
abbrlink: bb8fcf21
date: 2024-07-30 20:33:25
---

【本文已在同名 微信公众号 / 知乎 / [个人博客linsight.cn](http://www.linsight.cn/) 上线】  

***  

最近苹果的DCLM和Llama-3.1技术报告都提到了model soup：《Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time》。  

拿model soup出来和SWA已经EMA一起看下。  

# 背景  

一般来说，模型微调的过程是这样的：  
- 1、用不同的超参训练多个模型，每个配置下得到一系列模型checkpoint  
- 2、选择在验证集上最佳的checkpoint，其他的就丢弃掉了  

这样的常规做法方便易操作，但是有几个缺点：  
- 多个微调得到的模型如果进行合适的ensemble应该能有超过单个模型的效果，直接选择一个”最佳模型“浪费了一部分算力  
- 微调之后对于out-of-distribution data的效果可能变差，而这在验证集是看不出来的；而模型在实际使用中，很容易遇到有分布偏移的数据  

# SWA & EMA  

针对单次训练的模型平均方法主要有SWA和EMA。  

Stochastic Weight Averaging（SWA）算是模型微调里比较常见、普遍有效的方法了。  

SWA一般取训练后期的等间隔checkpoint，然后计算他们的参数平均。  

{% asset_img swa_1.png swa %}  

{% asset_img swa_2.png swa %}  

SWA为什么有效呢？  

一般SGD会让模型收敛到loss平面的一个wide flat region。这个空间的维度很高，所以wide flat region的大部分volume都集中在边界附近（类似碗口附近），所以SGD得到的解更容易出现在边界附近。  

另外，train loss和test error的曲面并非完全对齐。位于wide flat region中心的解不像边界附近的解那样容易受到训练和测试曲面之间的偏移影响，也就是靠近中间的解有更好的泛化性。  

SWA对多个解进行平均，能使其能够朝着区域的中心移动，因此得到的模型有更好的效果。  

下图是SWA和SGD解的train loss和test error曲面，虽然SWA得到的位置，train loss较大，但是它在收敛区域的中心，有更好的泛化性，在test error上更好。

{% asset_img swa_3.png swa %}  

EMA和SWA类似，只是对模型进行平均的方法不一样，细节可以参照《【炼丹技巧】指数移动平均（EMA）的原理及PyTorch实现》([https://zhuanlan.zhihu.com/p/68748778](https://zhuanlan.zhihu.com/p/68748778))。  

# model soup方法  

关于model average的一些工作：  
- 《What is being transferred in transfer learning?》里观察到，从同一个预训练模型进行微调的下游模型，会收敛到同一个error landscape basin。  
- 《Rethinking the inception architecture for computer vision》和《Averaging weights leads to wider optima and better generalization》（SWA）的结果显示在单个微调训练路径上进行weight average有效果。  
- 《No one representation to rule them all: Overlapping features of training methods》中观察到，把使用不同超参微调出来的模型进行ensemble有效果提升。  

受上面这些方法和观察的启发，model soup把model average扩展到使用多个超参的independent run，而不仅是如EMA/SWA那样的单次训练。  

假设使用多套超参 $[h_1,...h_k]$ 对预训练模型（$\theta_0$）进行微调，得到 $[\theta_1,...,\theta_k]$ 共k个模型checkpoint，分别是各自超参下，在验证集上取得最佳结果的checkpoint。通过对这k个checkpoint的模型参数进行平均，获得比单次微调的模型更好的效果，这就是model soup。  

文中提出了3种具体model soup方法：uniform soup、greedy soup和learned soup：  

{% asset_img method_soup.png model soup %}  

其中uniform soup把所有模型都用起来，计算均值。  

而greedy soup的做法则是把k个checkpoint按在验证集上的效果排序，按从高到低的顺序逐个验证checkpoint，只有当前checkpoint的加入对最终效果有提升时，才会保留它，否则就丢弃。算法如下：  

{% asset_img algo.png model soup %}  

uniform soup和greedy soup都比较直接，learned soup方法则需要额外训练。假设 $\alpha\in\mathbb{R}^k$ 是mixing coefficients，$\beta$ 是temperature scaling parameter，learned soup基于以下目标解出 $\alpha$ 和 $\beta$：  

$$\arg\min_{\alpha\in\mathbb{R}^k,\beta\in\mathbb{R}}\sum_{j=1}^n\ell\Bigg(\beta\cdot f\Bigg(x_j,\sum_{i=1}^k\alpha_i\theta_i\Bigg),y_j\Bigg)$$  

当k比较大时，learned soup对显存的需要会很大。  

综合来看，greedy soup应该是比较方便有效，性价比高的做法。  

# model soup实验  

图像上，用CLIP、ALIGN和BASIC模型做了验证，而文本则是用文本分类transformer模型。  

1、Error landscape visualizations  

用CLIP在ImageNet上使用不同超参进行多次微调，training loss和test error的可视化如下：  

{% asset_img angle.png model soup %}  

x和y轴是二维化的模型参数空间。多次的的微调模型本身并不在error landscape的最低点，而是分布在边缘上。  

这结果说明：  
- 对多个finetuned solution取平均能获得超过单个模型的效果  
- 越不相关的solution -- 参数空间上和initialization模型构成的连线之间的夹角越大 -- 的平均效果可能更好（个人这点感觉不是很直观）  

为了验证solution相关性对model average效果的影响，分别改变随机数种子、学习率和图像数据增强，得到多对结果。model soup的准确性增益随着solution之间的差异增大而增加，如下图：  

{% asset_img angle_2.png model soup %}  

2、Ensemble comparison  

model soup和ensemble方法，在不同learning rate下的对比如下：  

{% asset_img compare.png model soup %}  

观察到：  
- 当lr较小时，ensemble和model soup的效果同样，都比较差  
- 当lr适中时，ensemble和model soup的效果都较好  
- 当lr较大时，ensemble比model soup好，但都比适中lr差  
- 整体上，在in-distribution的数据上，ensemble效果更好，而在distribution shift数据上，则model soup更好  

3、One dimensional hyperparameter grids  

仅改变一个超参，获得的多个模型进行平均，效果是否有提升？  

针对这个问题，文章在optimizer、augmentation和lr上分别做了实验，结果是除了太大或者太小的lr，其他都有正收益。  

4、效果  

图像和文本模型在下游任务上使用model soup的效果如下：  

{% asset_img result.png model soup %}  

从结果上来看，都有比较稳定的收益，但是文本任务的收益没有图像那么明显。  

# 小结  

- model soup中性价比比较高的就是greedy model soup，操作简单，不影响推理成本，大部分任务都能获得提升  
- model soup的方法可以和adapter比如LoRA结合起来使用，还是比较有可扩展性的  
- 是和对抗训练、r-drop之类的方式一样，涨点好用，但是水文不多的方案  

***  

读到这了，来一发点赞收藏关注吧~

博客：[http://www.linsight.cn/](http://www.linsight.cn/)  
知乎：[Linsight](https://www.zhihu.com/people/us4ever)  
微信公众号：Linsight  
![](/images/qrcode.jpg)  

***  

【往期文章】  
- MoE：  
[MoE模型的前世今生](http://www.linsight.cn/44e38c1b.html)  
[DeepSeek-V2和MLA](https://www.linsight.cn/83c49df0.html)  
[昆仑万维-SkyworkMoE](https://www.linsight.cn/1d5bcd45.html)  
[成本10w刀的JetMoE](https://www.linsight.cn/f3acf042.html)  
[MoE的top-p routing](https://www.linsight.cn/224c42da.html)  
[对MoE模型的一些观察](https://www.linsight.cn/5e1d14b3.html)  
[从dense到MoE -- sparse upcycling](https://www.linsight.cn/a0824e29.html)  
[MoE路由--expert choice routing](https://www.linsight.cn/2c8bbc7.html)  
- 预训练：  
[Llama3.1--预训练要点一览](https://www.linsight.cn/7d7294cb.html)  
[Qwen2技术报告](https://www.linsight.cn/a8f8b641.html)  
[Yi技术报告-划重点看细节](http://www.linsight.cn/41b6a819.html)  
[MiniCPM](https://www.linsight.cn/376db710.html)  
[GLM4报告的一些技术点](https://www.linsight.cn/a5206abd.html)  
[Gemma2](https://www.linsight.cn/cf3f1f81.html)  
[苹果的OpenELM](https://www.linsight.cn/f845f3e4.html)  
[从Yuan2.0到Yuan2.0-M32](https://www.linsight.cn/3df0cd42.html)  
[bilibili的index-1.9B](https://www.linsight.cn/770b63e1.html)  
[从loss视角理解大模型涌现能力](https://www.linsight.cn/f5fb75e4.html)  
[预训练数据处理--长度分解](https://www.linsight.cn/210dbccd.html)  
- 长上下文：  
[LLM长上下文的问题](http://www.linsight.cn/c4da56c0.html)  
[解锁大模型长上下文能力](http://www.linsight.cn/cc852861.html)  
[大模型推理窗口-从有限到无限大](http://www.linsight.cn/45ee1a6d.html)  
- 推理加速：  
[大模型推理加速-投机解码](http://www.linsight.cn/f5c015c.html)  
[大模型推理加速-MEDUSA](https://www.linsight.cn/7bbe2df6.html)  
- 对齐：  
[Llama3.1--post-training要点一览](https://www.linsight.cn/93328a2a.html)  
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
- 大模型算法题：  
[(1)](http://www.linsight.cn/3345028a.html)、
[(2)](http://www.linsight.cn/ad0bba9d.html)、
[(3)](http://www.linsight.cn/1736008.html)、
[(4)](http://www.linsight.cn/1736008.html)、
[(5)](http://www.linsight.cn/336f2f3e.html)、
[(6)](http://www.linsight.cn/7c04944d.html)、
[(7)](https://www.linsight.cn/dd614e12.html)、
[(8)](https://www.linsight.cn/e287b9c3.html)  

# Reference  

【1】Averaging Weights Leads to Wider Optima and Better Generalization https://arxiv.org/abs/1803.05407  
【2】Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time https://arxiv.org/abs/2203.05482  
【3】Stochastic Weight Averaging in PyTorch https://pytorch.org/blog/stochastic-weight-averaging-in-pytorch/   
【4】【炼丹技巧】指数移动平均（EMA）的原理及PyTorch实现 https://zhuanlan.zhihu.com/p/68748778  

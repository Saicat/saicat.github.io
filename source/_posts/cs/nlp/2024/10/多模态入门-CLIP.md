---
title: 多模态入门--CLIP
tags:
  - 多模态
  - CV
  - NLP
  - transformer
  - 预训练
  - CNN
  - 无监督学习
categories:
  - CS
  - 多模态
abbrlink: 3069051d
date: 2024-10-01 12:40:06
---

【本文已在同名 微信公众号 / 知乎 / [个人博客linsight.cn](http://www.linsight.cn/) 上线】  

***  

放假了，小小水一篇多模态的经典之作，CLIP。  

论文：《Learning Transferable Visual Models From Natural Language Supervision》  

时间：2021年3月  

机构：OpenAI  

又是Ilya参与的一个工作。  

CLIP = Contrastive Language-Image Pre-training，顾名思义，这是一个基于对比学习的语言图像多模态学习方法。CLIP训练的目的其实主要还是获得通用的图像表征模型，因此在CLIP框架里，语言数据可以认为是作为监督信号存在的，类似图像分类任务中的类别信号，只是从一个one hot label扩展成了自然语言的形式。使用自然语言作为监督信号的好处是，自然语言信号更加灵活，可以支持扩展到zero-shot的推理，并且能够提供更加丰富的监督信息。  

# 数据  

其实在CLIP之前就有好些多模态训练的工作，但是效果没有这么好，原因主要是数据量不够大，另外就是对自然语言数据使用不够好，未能充分发挥自然语言的作用。因此一个很重要的工作就是构建数据集。CLIP是这么干的：  
- 以英文维基百科中出现至少 100 次的所有单词为基础构建词集，并增加了双词组合和所有 WordNet 同义词  
- 爬取网上的数据，试（图像，文本）数据对中的文本包含词集中的一个词  
- 为了尽可能覆盖广泛的视觉概念，对结果进行平衡，每个概念最多包括 20,000 个（图像，文本）对  
- 构建的 WIT（WebImageText） 数据集包含 4 亿个（图像，文本）对  

WIT数据集比之前很多多模态数据集都大，包含的内容也更丰富。  

# 训练框架  

CLIP预训练框架如下图：  

{% asset_img contrastive_pt.png CLIP %}  

text encoder和image encoder分别对文本和图像进行编码。text encoder通过对比学习，把文本的表征向match的图像靠拢，而和batch内其他图像，也就是负样本的距离尽量拉大。image encoder也是同样地学习图像表征。  

训练的pseudo-code如下：  

```python
# image_encoder - ResNet or Vision Transformer
# text_encoder - CBOW or Text Transformer
# I[n, h, w, c] - minibatch of aligned images
# T[n, l] - minibatch of aligned texts
# W_i[d_i, d_e] - learned proj of image to embed
# W_t[d_t, d_e] - learned proj of text to embed
# t - learned temperature parameter
# extract feature representations of each modality
I_f = image_encoder(I) #[n, d_i]
T_f = text_encoder(T) #[n, d_t]
# joint multimodal embedding [n, d_e]
I_e = l2_normalize(np.dot(I_f, W_i), axis=1)
T_e = l2_normalize(np.dot(T_f, W_t), axis=1)
# scaled pairwise cosine similarities [n, n]
logits = np.dot(I_e, T_e.T) * np.exp(t)
# symmetric loss function
labels = np.arange(n)
loss_i = cross_entropy_loss(logits, labels, axis=0)
loss_t = cross_entropy_loss(logits, labels, axis=1)
loss = (loss_i + loss_t)/2
```

在clip的训练框架中，text encoder和image encoder的地位是对称的。  

和之前的对比学习一样，为了提升学习的效果，负样本需要尽量多，因此实验中使用32,768的batch size。  

理论上，text encoder和image encoder可以是任意模型。OpenAI选择了ResNet/EfficientNet-style的模型和几个ViT（ViT-B/32、ViT-B/16、ViT-L/14）作为image encoder进行实验，而text encoder则是使用GPT-2的结构，最后一层的 [EOS] token 就作为text representation。  

训练中，image encoder和text encoder都是随机初始化的，不需要预先训练。  

# 使用  

完成预训练之后，一个常规的用法是基于image encoder进行微调，包括仅训练classifier，和完整模型的训练。  

CLIP另一个强项就是可以做zero-shot predictor。比如我们想要知道让预训练模型对一张图片的类别进行预测，可以把所有可能的类别填进一个prompt里：“A photo of {object}”，然后让text encoder给出所有representation，并计算不同类别下的text representation和image representation的相似度，取最高的那个就是预测结果了：  

{% asset_img clip_infer.png CLIP %}  

当然CLIP的用法不仅是可以做zero-shot的图像分类，后续还有很多其他应用方法，挖个坑后面来填。  

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

【1】Learning Transferable Visual Models From Natural Language Supervision https://arxiv.org/abs/2103.00020  

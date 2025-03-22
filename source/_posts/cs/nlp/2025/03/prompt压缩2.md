---
title: prompt压缩(二)
tags:
  - NLP
  - LLM
  - transformer
  - prompt压缩
categories:
  - CS
  - NLP
  - LLM
abbrlink: ea2871bf
date: 2025-03-22 15:12:54
---

【本文已在同名 微信公众号 / 知乎 / [个人博客linsight.cn](http://www.linsight.cn/) 上线】  

***  

前文：[prompt压缩(一)](https://mp.weixin.qq.com/s/7ugiuuhRaXV4P62C7GsA1w)   

# Nano-Capsulator  

Nano = NAtural laNguage prOmpt，吐槽一下，这又是硬凑了一个缩写名字。  

之前提到的soft prompt需要针对生成模型进行一定的训练，无法在多个LLM之间通用；而类似selective contexts这样，根据self-information或者ppl选择一些token删除的方法在压缩效果（保留原prompt效果）上略差一些，因此Nano-Capsulator就被设计出来，「在保留自然语言可读性的情况下压缩输入（获得Capsule Prompt），既要保证效果，又能够在多个LLM之间通用」。  

要达到这些目的，需要训练一个模型，Nano-Capsulator。Nano-Capsulator也是一个LLM，实验中用的是Vicuna-7B。推理的时候，Nano-Capsulator就用来根据long prompt生成short prompt。  

那么现在问题就是怎么训练Nano-Capsulator，让它能够在减少生成长度的同时，最大程度保留原prompt的语义信息。  

一个直觉是，如果两个prompt的语义很相近，那么以这两个prompt为LLM的输入，输出的d-dimension embedding应该也要相近。  

具体来说，文中借助两个的instruction，replicating instruction和summarizing instruction。replicating instruction要求模型重复输出接收到的原始prompt，而summarizing instruction就是用来压缩原prompt的指令。  

假设原始prompt是$K=\{k_{1},\cdots k_{n}\}$，压缩后的prompt是$C=\{c_{1},\cdots c_{m}\}$。replicating instruction + 原prompt在Nano-Capsulator的输出embedding $e_K$ 就代表原prompt的语义，而summarizing instruction + 原prompt在Nano-Capsulator的输出embedding $e_C$ 就代表压缩prompt的语义，那么二者的semantics loss就是  

$$\mathcal{L}_{\mathrm{Comp}}=\mathbb{E}_C\left[D_{\mathrm{dist}}(e_K\parallel e_C)\right]\quad(1)$$  

distance理论上可以采样各种合理的计算方式，原文使用的是mean square error。  

到这里主要是为「原始prompt」和「压缩prompt」的相似度建了模，接下来还需要对「原始prompt+question的LLM生成结果」和「压缩prompt+question的LLM生成结果」的效果进行评估。  

假设用于生成的LLM是 $\mathcal{G}$，那「原始prompt+question的LLM生成结果」就是 $\mathcal{G}\left(K_{i} \oplus Q_{i}\right)$，「压缩prompt+question的LLM生成结果」就是 $\mathcal{G}\left(C_{i} \oplus Q_{i}\right)$。有了这两个生成结果，就可以使用一个reward function $I$，来评判使用压缩prompt后生成结果的效果。理论上 $I$ 可以是任意reward function，原文中使用的是两个输出的hidden state embedding之间的mean square error。能够使用mean square error是基于能够拿到生成模型的参数的情况，如果使用的是API，那也可以使用其他类型的reward function（注意这里的reward function概念和RL中的reward function不太一样），比如可以用GPT4等模型对两个输出结果的差异度进行打分，总之就是要求两个输出结果的差异越小，$\mathcal{R}_{cap }就应该越小$。  

另外，不要忘记我们压缩prompt的一个主要目的是缩短输入，因此压缩prompt的长度不能太长。怎么把长度限制加入reward function里呢？可以对 $C$ 做一个简单的cut-off，获得 $\Phi(C)$。如果 $C$ 的长度超过了设置的长度，它就会被截断，那正常来说生成结果就不会好，获得的reward自然就低。  

保持输出效果这个能力，原文叫prompt utility preservation。最终，prompt utility preservation的loss完整版本是  

$$\mathcal{R}_{cap }=\mathbb{E}_{Q}\left[I\left\{\mathcal{G}\left(\Phi\left(C_{i}\right) \oplus Q_{i}\right) \| \mathcal{G}\left(K_{i} \oplus Q_{i}\right)\right\}\right]\quad(2)$$  

最后结合semantics loss和prompt utility preservation，获得最终的训练loss  

$$\mathcal{L}_{Nano }=\mathcal{L}_{Comp }\left(\cdot | \theta_{C}\right) * \mathcal{R}_{cap }\left(\cdot | \theta^{*}\right)$$  

这里用的是相乘。思路是，如果压缩prompt生成的结果不好，$\mathcal{R}_{cap }$就会大，这就给nano-capsulator的损失项$\mathcal{L}_{Comp }$施加一个大的惩罚系数。  

Nano-Capsulator的整个训练流程如下：  

{% asset_img nano.png prompt_compression2 %}  

# LLMLingua-2  

LLMLingua-2的目标是做task-agnostic prompt compression，也就是可以处理任意的prompt。  

之前介绍的selective contexts根据self-information选择token来删除，还有一些使用PPL来评估token的信息量的，这样做存在问题：（1）单向计算token的重要度并不是很可靠，缺乏完整的双向信息（2）这样的做法和prompt压缩的最终目标并不完全一致：prompt压缩是要在减少输入的情况下，保持模型的输出，仅仅在一定程度上保留输入token的信息并不能保证输出效果。  

LLMLingua-2的方法是直接收集一批（原prompt，压缩prompt）的数据对，把原prompt中每个token是否保留当做一个二分类任务，训练一个Bert模型输出每个token「保留」的概率，选择「保留」概率高的token组合成压缩prompt。这个工作里两个主要事情就是（1）收集/处理数据（2）训练模型。  

1、数据的收集和处理  

收集数据主要是用GPT4：利用下面这个prompt要求GPT4对输入的prompt进行压缩：  

{% asset_img gua2_prompt.png prompt_compression2 %}  

这个prompt没有给GPT4施加太多的硬性规定，因为不同的文本在不同位置的token是否要压缩的情况都不一致，所以大思路上是都交给模型来处理，而只要求保留重要的文字，同时不要修改、打乱或者添加原本的文本（这些其实都是GPT4的幻觉问题）。  

实际使用中发现GPT4会给很长的文本使用更大的压缩率，这其实是由于模型处理能力随着输入文本长度的增加而下降，这点和人一样，信息一多就overwhelm了。这样一来压缩的效果就不好。为了缓解这个问题，会按句子把原prompt切分成长度不超过512的段落，分别进行压缩。  

获得GPT4的压缩结果之后，还需要进一步处理。主要是要解决由于压缩和GPT4指令遵循不好带来的几个问题：（1）Ambiguity（2）Variation（3）Reordering。这三个问题的示例和处理方法：  

{% asset_img gua2_algo.png prompt_compression2 %}  

为了评价GPT4的压缩文本的质量，引入两个指标：Variation Rate和Alignment Gap。

Variation Rate其实就是看压缩prompt里出现了多少原prompt没有的token/word。这些词肯定就是来自于GPT4的幻觉，变异的token/word越多，说明这条prompt的幻觉越严重，质量就越差。VR最高的5%数据就删掉不用了。  

Alignment Gap稍微复杂一点，它由matching rate (MR)和 hitting rate (HR) 计算得到。MR是指原prompt中能在压缩prompt中找到对应token的比例，而HR是指压缩prompt能在原始prompt找到的token的比例。HR的理想值是1，因为一个好的压缩prompt所有token都是来自原始prompt的。AG = HR - MR，如果AG大，说明MR小，太小的MR说明压缩prompt所包含的原始prompt的token比例不高，可能无法很好地表征原始prompt，因此AG越小越好。实践中把AG最高的10%数据去掉了。  

2、压缩模型  

有了（原prompt，压缩prompt）数据之后，就可以用Bert模型训练每个token的二分类了，这和用Bert做NER类似。  

推理的时候，每个原始prompt的token都能获得一个得分，表示模型认为这个token需要保留的概率。一般来说我们有一个目标的压缩率τ（比如0.2），如果原prompt有N个token，那么我们就需要保留τN个token组成压缩prompt。因此我们就根据Bert模型输出，选择概率最高的τN个token，保持原顺序组成压缩prompt。  

# CPC  

CPC = Context-aware Prompt Compression。  

CPC的思路和LLMLingua-2有点相似，都是构建一个数据集，标出一个输入中哪些内容重要，哪些内容不重要，然后用一个模型来学习这种分布。在推理的时候就只保留那些重要的内容。不过和LLMLingua-2不同的是，CPC不以token或者word为单位，而以句子为单位，这样可读性和连续性更好一些。另外就是构建数据集和训练模型的细节的不同。  

1、构建训练数据集  

CPC构建的训练数据集叫Context-aware Question-Relevance (CQR) 。CQR数据集中每条样本是包含（上下文，问题，正例(positive)，负例(negative)）的四元组。其中上下文+问题其实就是完整的原始输入，而正例和负例都是来自上下文的句子。正例就是重要的要保留的句子，而负例就是可以删去的句子。  

压缩模型训练的时候就要学习根据上下文和问题来区分正例和负例句子。  

要构建这样的数据集，首先从WikiText的文档开始，这些文档都是比较长的。第一步就是要从这些上下文里获取一些positive的句子。  

positive的句子至少得是一致连贯的：如果一个句子英文单词的数量达到一定的比例，并且都由ASCII字符组成，那么就认为这个句子是一直连贯的。把符合要求的句子和上下文一起，用下面这个prompt生成QA对：  

```python
Prompt 1 (Question Prompt):
Here is a text to consider: TEXT: "text"
Read the sentence in double brackets,
namely, [[sentence]].
Ask questions to this sentence, and make
sure the question is not answerable from
this sentence alone without knowing the
context.
Reply in this format:
Q: {question 1}
A: {answer 1}
Q: {question 2}
A: {answer 2}
```

常有的一种情况是，一个句子虽然不能直接回答问题，但是包含了回答问题所需要的元素，因此压缩模型需要关注这种情况：只有这一个positive不足以回答的问题。因此用下面这个prompt来检验生成的QA是否符合这种情况：  

```python
Prompt 2 (Verification Prompt):
You are given a piece of text, a question
and an answer. Verify whether it is
possible to derive such an answer by
considering only the given piece of text
(you should rely only on the piece of
text). Think step by step and finish
your thoughts with one word: "Yes" or
"No". Answer "Yes" if and only if ALL the
necessary information is contained in the
text. If anything is missing, then state
what is missing and answer "No". Answer
"Yes" ONLY if there is no such information
in the answer that is missing in the text.
Otherwise, answer "No"!!
{A demonstration}
Text: {context sentence}
Question: {question}
Answer: {answer}
Verification result: Yes/No
```

只有验证结果为No的QA会保留。  

下一步就是获取negative句子。获取negative句子有两步。第一步先用一个sentence embedding模型，计算正例句子和问题Q的相似度η，然后计算上下文中所有其他句子和问题Q的相似度，对于相似度小于η的句子，都放到负例候选集里。  

第二步会对负例候选集里的所有句子做一个校验。校验方法就是获取有这个句子和没有这个句子这两种情况下，模型对问题Q回答的情况。如果两种情况下答案A的KL散度大于阈值，则说明这个句子对答案还是有影响的，否则就说明它确实是一个不重要的句子（对于回答这个问题来说）。  

收据收集的整体流程：  

{% asset_img cpc_datapipeline.png prompt_compression2 %}  

2、训练  

接下来就要训练一个模型，用于在推理的时候判断每个句子是否要保留。这里用的是对比学习的方法来训练。具体来说，计算给定上下文C下，正例P和问题Q的相似度，以及负例N和问题Q的相似度，然后最大化正例的相似度，最小化负例的相似度：  

$$Sim_{P}=\exp (cosine(\xi_{Q_{b}}, \xi_{(P_{b}, C_{b})}))$$  

$$Sim_{N}=\exp (cosine(\xi_{Q_{b}}, Neg_{(b, ext)}))$$  

$$\mathcal{L}_{SC}=-log \frac{exp \left(Sim_{P}\right)}{exp \left(Sim_{P}\right)+\sum exp \left(Sim_{N}\right)}$$  

3、推理  

推理的时候，把输入拆分成句子，计算每个句子和Q的相似度，并按照需要的压缩率保留相似度最高的句子。  

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
[Qwen2.5-1M技术解析](https://www.linsight.cn/6c0f6207.html)  
[LLM长上下文的问题](http://www.linsight.cn/c4da56c0.html)  
[解锁大模型长上下文能力](http://www.linsight.cn/cc852861.html)  
[大模型推理窗口-从有限到无限大](http://www.linsight.cn/45ee1a6d.html)  
[prompt压缩(一)](https://www.linsight.cn/4519eadd.html)  
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
[LLM水印](https://www.linsight.cn/2dee4921.html)  
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
[DeepSeek-VL2的细节](https://www.linsight.cn/b4d047c1.html)  
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

【1】Learning to Compress Prompt in Natural Language Formats  
【2】LLMLingua-2: Data Distillation for Efficient and Faithful Task-Agnostic Prompt Compression  
【3】Prompt Compression with Context-Aware Sentence Encoding for Fast and Improved LLM Inference  

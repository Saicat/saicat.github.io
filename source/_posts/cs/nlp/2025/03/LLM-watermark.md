---
title: LLM水印
tags:
  - NLP
  - LLM
  - transformer
categories:
  - CS
  - NLP
  - LLM
abbrlink: 2dee4921
date: 2025-03-01 11:00:54
---

【本文已在同名 微信公众号 / 知乎 / [个人博客linsight.cn](http://www.linsight.cn/) 上线】  

***  

给图片加水印大家都很熟悉了：  

- 比如给身份证照片加水印以确定用途保护个人信息，或者画师给自己的画加水印保护产权，想要无水印版本的就需要购买；这种水印一般比较明显，甚至会覆盖在图片上的关键位置；  
- 也有些图片水印是隐形的，人眼极难发现；这种一般是用来确定数据来源（比如你要是给公司的数据文件截屏上面很可能就有你工号的隐形水印，一下就能定位到操作人），或者保护知识产权的；隐形水印可能需要借助工具自动识别，人眼的分辨能力没法处理。  

其实不只图片可以加水印，LLM生成的文字也可以加水印。LLM生成结果加水印有什么用呢？目前来看主要是两种主要应用场景：  

（1）帮助人类辨别哪些文本是LLM生成，哪些是人类写的（比如检测AI生成的作业或者论文）  

（2）对于开源协议不可以商用的模型，开发者能够有一定手段判断是否有人在未授权的情况下使用了自己的模型  

不过个人感觉这两种应用场景其实都不是很solid：  

（1）现在已经有大量开源模型，没法保证使用这些开源模型的人会愿意加入水印（或者说大概率不愿意），只有用户使用API生成，且模型提供商加入了水印，且水印的检测方式给到你的情况下，你才有机会检测水印；  

（2）如果使用者进行了一定的微调，模型的参数变化了，那么水印检测的方法也可能会失效；不过这个场景应该会有一些开源大厂会关注：如何在保持开源的情况下，又能不被滥用，如果不能在一定程度上保持这点，那就有可能对关键大小的模型选择不开源，这对AI开发者就是个坏消息；另外如果是通过API提供的，那模型供应商自己插入水印就变得方便了。  

来看下LLM水印具体是怎么做的。  

# Red List 和 Green List  

这个方法出自23年初的《A Watermark for Large Language Models》，这是一种比较简单朴素的LLM水印方式。  

首先，我们需要知道，LLM watermarking分成两部分：水印的添加和水印的检测。添加水印和检测水印形成一个完整的闭环，我们才能够追溯这些文字的源头。  

文中提出理想水印的几个要求，还是比较合理的：  

- 水印的检测不需要知道模型参数和获取model API  
- 加水印的模型不需要经过训练  
- 只要有一个连续的文字片段就可以检测水印，甚至无需完整的上下文  
- 水印无法被只修改少量文字就移除  
- 有一个统计量能够判断水印检测的置信度  

更长远来看，个人认为可能还有一些更深层次的要求，比如：  

- 水印难以被迁移到无水印的文本上  
- 水印的添加和检测成本显著地低于生成文本  

## hard red list watermark  

看下文中提出的第一种最简单的加水印的方法，使用hard red list。  

正常来说，LLM decode的时候首先根据prompt计算出下一个token在vocab上的概率分布；如果是贪心解码，那就下一个token就直接选择概率最高那个，如果是top-k或者top-p解码，那就会在概率最高的几个token里选择一个。  

现在如果在每次解码，都把vocab随机、均匀地分成red list和green list两组，并且限定下一个token只能从green list里选，那么后续就可以通过检测生成的文本中是否包含red list的token来判断这段文本是否是这个LLM生成的了：如果这段文本很好地遵循了只从green list选择的规则，那么就是LLM生成的（检测到了水印），反之如果这段文本根本不遵从这个rule，那么就很可能是人类写的（没有水印）。可以使用z检验作为指标来判断。  

加入水印的具体方案：  

{% asset_img algo1.png watermark %}  

在这个方案下，检测水印需要两个前提：  

- 需要知道这个模型的vocab  
- 需要知道随机数生成逻辑和split词表的具体逻辑  

有了这两个，就可以检测出来插入了这个水印的文本。  

这种插入水印的方式显然会对生成的文本质量造成很大的影响，毕竟有一半的token不能使用了，运气最差的情况下，概率最高的token都被加入到red list里，那下一个token就是概率很低的token，导致句子完全不通顺了。  

我们知道无论哪种语言都有一些概率很高，近乎固定组合的sequence，比如成语、俗语、诗歌、歇后语。比如本来“落霞与孤鹜齐”的下一个字是“飞”，但是很不好运，使用hard watermark之后“飞”字被ban了，那生成的结果肯定就出问题了。  

这种搭配很固定的情况称为low entropy sequence，因为只要你看到了上文就基本可以确定下一个字是什么。  

## soft red list  

既然hard watermark会对生成的文本质量造成太大影响，那就尝试把hard变成soft，缓解一下这个问题。  

直接看下方案：  

{% asset_img algo2.png watermark %}  

前两步和hard方案是一样的。生成完随机数之后，就要根据随机数，切分red list和green list了，在soft方案里，red list和green list不再是对半分，而是有一个超参数γ，只有γ|V|个token会被放到red list里。明显γ越大，水印越强，但是对生成质量的影响也越大。  

（其实再进一步，还可以对每个token设定predefine的red list。）  

分出red list之后，也不直接把red list里的token禁用，而是对在green list里的token的logits增加一个正数δ，这相当于人为降低了red list里的token得分，但是并不100%禁用。δ越大，red list中的token被压制得越多，水印强度越大(更容易检测)，同样也会生成质量影响更大。  

soft方案对于low entropy的情况也有比较好的缓解，因为low entropy sequence中，下一个token的概率是非常高的，因此即使加了一个δ，这个固定搭配的token仍然有很大的机会被选中，无论是什么解码方式。  

γ和δ选择和模型是高度相关的，这就需要做实验确定了。文中在OPT模型上做了实验，随着γ和δ的增强，ppl是提升的：  

{% asset_img influence.png watermark %}  

在检测的时候同样用z检验判断是否加了水印。  

## private watermark  

前面的hard watermark和soft watermark在生成red list的时候都是根据当前最后一个token来生成的，这样简单的机制在抗破译抗攻击方面还是不够强。那么一个增强的方法就是使用pseudorandom function (PRF)。  

PRF使用一个密钥，根据当前token和前h个token作为输入，生成随机种子，以决定vocab里的token是否要加入red list。这样在检测水印的时候，同样需要密钥和相同的PRF。  

{% asset_img algo3.png watermark %}  

最后看一个水印例子：  

{% asset_img example.png watermark %}  

# Undetectable Watermarks  

上面的方法中，无论是hard还是soft，都会明显地影响生成结果，那有没有办法再减小一点水印对生成结果的影响呢？《Undetectable Watermarks for Language Models》就基于密码学的工具引入了undetectable watermarks。  

首先一个思路是，并不是所有文本都适合插入水印。比如对于low entropy的文本，“团结就是力量，这力量是铁，这力量是钢”这段文字根本就没法判断是人写的还是LLM生成的，因为它的搭配太确定了。如果非要改变这个搭配，就会明显影响了生成内容。因此low entropy sequence就不适合插入水印。  

再进一步，一段文本，有些entropy高（词表上token的得分分布比较均匀），有些entropy低（词表上token的得分只有一个0.99999，其他基本都是0），那就只在entropy足够高的时候才插入水印。  

原文有很多数学证明，这里就直接用一个例子来说明下。  

## 插入水印  

比如对一个模型输入prompt=“今天天气怎么样”，正常情况下模型会输出“今天天气很好，适合去公园散步”。  

step1：检测熵

如果我们要插入水印，首先就要计算empirical entropy。经验熵是基于已生成文本的实际概率分布计算的熵值，反映当前生成过程的“不确定性”。对于序列x1、x2、...、xt，p(x)是这个token生成时的概率，经验熵计算所有x的log(p(x))的平均值。经验熵越大，说明当前序列的不确性高，反之则说明序列的确定性很高（就是前面说的固定搭配）。这里就需要一个阈值来判别当前的熵是高还是低。  

比如生成到“今天”，发现下一个token是“天气”的概率特别高，经验熵很低，那么这个时候就不插入水印；把“天气”放进序列之后，发现“今天天气”后面的熵开始高了，p(“很好”)=0.6，p(“不错”)=0.4，这两者比较接近，导致经验熵高于阈值，那么就可以开始插入水印了。  

step2：生成随机数

我们需要一个PseudoRandom Function，来生成随机数。PRF的特点是，只要输入相同，输出就相同，这也是后面我们可以检测水印的前提。PRF的输入有两个：  

- 密钥，这个是生成水印的人加入的，只有持有这个密钥才能检测水印，密钥可以是一个字符串，比如"技术博客linsight.cn"  
- 当前的上下文  

根据这两个输入，我们获得一个随机数u=0.55  

step3：选择token  

根据随机数，在两个候选token “很好”和“不错”之间进行选择：  

- 如果u ≤ p("很好")，选择"很好"  
- 如果u > p("很好")，选择"不错"  

这一步在外部看来，概率的分布不变。因为这里我们只是替换了随机选择的函数，把“所有人都不知道的随机逻辑”变成“持有密钥的人知道，其他人不知道的随机逻辑”。  

## 检测水印  

先要需要验证句子 "今天天气很好，适合去公园散步" 是否含有水印。  

step1：重建PRF随机数  

使用相同的密钥"技术博客linsight.cn"和上下文"今天天气"，通过PRF重新生成u₁=0.55。  

step2：计算单个token验证分数  

实际生成的token是"很好"，基于对数似然：  

如果选"很好"：s(x₁, u₁) = -ln(u₁)

如果选"不错"：如果选0：s(x₁, u₁) = -ln(1 - u₁)

s = -ln(0.55) ≈ 0.597  

step3、累加所有token分数  

假设整个句子有5个高熵token，每个token的分数如下：  

s₁=0.597, s₂=0.8, s₃=0.3, s₄=0.9, s₅=0.4  

总分数：C = 0.597 + 0.8 + 0.3 + 0.9 + 0.4 = 2.997  

step4、阈值判断  

假设一段文本高熵token的数量为L。  

对于自然文本：若文本无水印，伪随机数uj服从均匀分布 U(0,1)，此时每个 s(xj,uj) 的期望值为1，总C值的期望为 E(C)=L。  

对于含水印文本：若文本含水印，uj的分布被密钥约束（例如选词1时 uj≤pj(1)），导致每个 s(xj,uj) 的期望值大于1，总C值 E(C)>L。  

因此在上面这个例子里，2.997 < L，可以认为没有包含水印。  

## 思路  

总体来说，这种方法有两个核心思路：  

- 只对高熵部分的文本添加水印，减少对生成质量的影响  
- 把高熵token的选择从「不可知的随机选择」变成「基于密钥的PRF选择」，让掌握密钥的人可以检测到水印。  

# Unbiased Watermark  

《UNBIASED WATERMARK FOR LARGE LANGUAGE MODELS》提出两种不影响生成质量的水印方案，δ-Reweighting和γ-Reweighting。  

## δ-Reweighting  

δ-Reweighting前半部分和上面的undetectable watermark类似，主要在后面选择包含水印的token的策略有所不同。还是直接看一个例子。  

比如模型根据prompt“你喜欢吃什么水果？”，现在生成到了“我喜欢吃”，再下一个token的选择有：  

- 苹果，p=0.35  
- 香蕉，p=0.25  
- 橘子，p=0.20  
- 梨子，p=0.15  
- 葡萄，p=0.05  

还是通过一个密钥和PRF生成一个随机数，根据这个随机数落在的区间，选择一个token。比如现在生成随机数0.66，落在了“橘子”的区间，那就输出“橘子”。  

从单次采样来看，这是一个delta分布：只有“橘子”的概率为1，其他token都是0。因为只要PRF是同样的，context和密钥也是同样的，那么每次都会输出相同的“橘子”。  

而从多次采样取平均的结果来看，生成苹果的概率依然是0.35，生成橘子的频率依然是0.20。  

{% asset_img delta.png watermark %}  

## γ-Reweighting  

相比δ-Reweighting，γ-Reweighting主要是在调整概率的方案上有所不同：  

γ-Reweighting将词表随机打乱（还是使用包含密钥的PRF）之后分成前后两段，每次decode都固定对词表后半段的token概率进行提升（翻倍），而对词表前半段的token进行缩减（变成0）。  

同样地，单次来看，有一半的token被ban了，但是整体多次统计来看，还是保持正常的概率。  

{% asset_img gamma.png watermark %}  

那检测水印的时候，就看打乱后的词表后半段是否概率更高。  

整体来看γ-Reweighting的实现和检测更为简单，可能被逆向破解。而δ-Reweighting更加动态随机，抗攻击性更强。  

# 其他  

- 清华等⾼校联合推出了开源的⼤模型⽔印⼯具包 MarkLLM，支持多种水印的嵌入和检测方式  

# 小结  

- 目前加水印都多多少少会对生成质量产生影响  
- 加水印 & 检测水印是高成本的事情，不是特殊场景恐怕不会使用  

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

【1】A Watermark for Large Language Models, https://arxiv.org/abs/2301.10226  
【2】UNBIASED WATERMARK FOR LARGE LANGUAGE MODELS, https://arxiv.org/abs/2310.10669  
【3】Undetectable Watermarks for Language Models, https://arxiv.org/abs/2306.09194  
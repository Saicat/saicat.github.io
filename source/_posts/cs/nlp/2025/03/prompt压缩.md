---
title: prompt压缩(一)
tags:
  - NLP
  - LLM
  - transformer
  - prompt压缩
categories:
  - CS
  - NLP
  - LLM
abbrlink: 4519eadd
date: 2025-03-15 17:16:52
---

【本文已在同名 微信公众号 / 知乎 / [个人博客linsight.cn](http://www.linsight.cn/) 上线】  

***  

现在长思考模型助力agent，能够循环进行复杂任务的拆解和执行。为了告诉模型任务和能够调用的工具/当前观察到的信息等，输入prompt也是越来越长，10k甚至更长的输入已经是普遍的状况。虽然推理框架也越来越强，不过本着能省一点是一点想法，prompt压缩也是个值得考虑的方向，特别是在偏垂域的场景。  

其实垂域任务的微调也算是prompt压缩的一种。比如对于通用模型，做法是加上任务描述：  

"判断这段文本的情感类型，正向输出1，负向输出0。{text}"  

现在我们直接构造数据对(text, label)，而不需要加额外的任务描述；用这些数据微调模型，让它专注做情感分析。回想一下，这正是Bert的拿手任务。  

这相当于把任务描述训进了模型参数中，从而减少了每次的输入文本量，即"判断这段文本的情感类型，正向输出1，负向输出0。"这段文字。  

当然这种做法在现在看来太低效了。针对每个任务或者每个prompt微调一个模型在大规模模型的场景下成本太高，也不可能部署多个模型来处理各种任务，更重要的是没法顾及到所有的输入prompt。  

今天先来学习下几个prompt压缩的工作。  

# Conditioning Contexts（CC）  

Conditioning contexts是prompt压缩比较早期的工作了。  

CC属于soft prompt compression。既然有soft prompt compression，那肯定有hard prompt compression。简单来说，hard compression主要研究一个输入里哪些token可以保留哪些token可以删掉，是相对离散的；而soft compression则是把压缩的prompt带到了参数层面，是连续的，有点像prompt tuning。  

## 方法  

CC具体方案就是在输入question前面append一些embedding（这就是压缩过的prompt），训练目标就是最小化原prompt的输出和压缩prompt的输出之间的KL散度：  

{% asset_img cc_method.png prompt_compression %}  

方案确实和prompt tuning很像，区别就是prompt tuning是学习正确label，而CC学习的是原prompt下的输出结果。  

研究人员拿GPT2系列各个规模的模型做实验，在不同规模的模型上，KL散度的走势和值都很相近，这说明这样的soft prompt是有在多个规模的模型使用的潜力的（比如在小模型上训练，在大模型上使用）；另外随着soft prompt的长度的增加，KL散度越来越低：  

{% asset_img cc_kl.png prompt_compression %}  

## 观察  

另外，研究人员通过对general类型问题和细节问题在压缩前后准确率的比较，发现压缩过的soft prompt更倾向于记住general的内容，而遗忘细节内容：  

{% asset_img cc_specific.png prompt_compression %}  

那么这样的方式训练出来的soft prompt可以在多大程度上保留原prompt的信息呢？研究人员用一个reconstruction task，即在压缩的soft prompt后加上“Now repeat the text:”的要求，让模型尝试给出它看到的信息。注意即使只给“Now repeat the text:”这句话，模型也会输出，因此需要进行一定的归一化。把输出的分布在原prompt + repeat和no prompt + repeat之间进行归一化，把原prompt的token的概率可视化：  

{% asset_img cc_repeat.png prompt_compression %}  

heatmap中黄色是1（说明soft prompt很好地保留了原prompt信息），紫色是0。可以看到大致的趋势是：（1）随着soft prompt的长度n减小，损失越来越大（2）soft prompt更倾向于记住原prompt中靠前部分的信息。  

最后，文中还提出一个想法：soft prompt更能让模型遵循相应的要求。这里做了一个实验：有两个prompt，一个要求模型谈论猫，另一个要求模型输出负面情感的内容，然后分别使用原prompt和压缩过的soft prompt对模型的输出结果进行指令遵循情况的检验。结果发现，soft prompt比原prompt的指令遵循情况更好一些：  

{% asset_img cc_follow.png prompt_compression %}  

## 小结  

整体来看，CC所使用的soft prompt和prompt tuning很像，依然是一次训练只能针对一条prompt。因此如果要使用的话，比较适用于有超长固定system prompt的场景，这样在推理的时候可以节省一些推理成本。当然，在效果上是有一些损失的，而且损失的是现在大家比较关注的细节信息。  

不过原文也提出了几个有意思的点：

- 压缩过的soft prompt可以提升模型指令遵循的效果  
- reconstruction task可用于检验prompt的压缩效果  

# Selective Context  

前面的CC是soft prompt compression，这里要讲的selective context就是hard prompt compression。  

selective context的大思路：输入prompt中不是每个token都一样重要，有些知识模型已经知道，就不必重复说，因此可以删掉一些token。  

## self-information  

那怎么判断一个token重不重要呢，就是使用self-information。  

Information theory中，self-information表征在给定的分布下，一个event携带的信息量。  

在language modeling的context下，生成一个token就可以看作一个event。那self-information就可以写作：  

$$I(x)=-\log_2P(x_t|x_0,x_1,...,x_{t-1})$$  

I越大，x的信息量越多。  

题外话一下，language modeling中和self-information相关的还有entropy和perplexity：  

$$H(S)=\frac{1}{N}\Sigma_tI(x_t)$$  

$$PP(S)=2^{H(S)}$$  

对于连续的token，有  

$$\begin{aligned}I(x_0,x_1)&=-\log_2P(x_0,x_1)\\&=-\log_2P(x_0)P(x_1|x_0)\\&=-\log_2P(x_0)-\log_2P(x_1|x_0)\\&=I(x_0)+I(x_1)\end{aligned}$$  

这说明token级别以上的lexical unit（比如words、phrases和sentences）都可以通过token的自信息得分相加而得到，这点很重要。  

## 方法  

selective context的方法有三步。  

（1）计算token的self-information  

这一步可以选用小一些的模型比如Llama-7B，而生成模型则是更大规模的Llama模型。(那么这里就有一个问题：用于压缩的模型和用于生成模型之间的关系是否支持这种对应)  

在实操上有一个发现，LLM倾向于给靠后的lexical unit打低分，因此实操中不把整个prompt一次输入，而是改成一个句子一个句子计算，这样就缓解了靠后的lexical unit分数偏低的问题。  

（2）（optional）聚合lexical unit  

在token层面删除，可能导致文本的不连续，因此可以改为在phrase或者sentence级别删除内容。不过这也引入了新的复杂性：phrase和sentence边界的检测。实操上可以依赖传统的NLP工具来分割phrase和sentence。  

在消融实验中，phrase level的效果最好，而sentence level的效果最差：  

{% asset_img sc_level.png prompt_compression %}  

（3）eliminate不必要的部分  

删除的时候，不是使用自信息的threshold，或者固定保留top-k个unit，而是按self-information从高到低排序，保留总和为top-p的lexical unit。p相当于限定了保留信息的量，而从高到低排序保证了所用的unit是最少的，也就是最大的compression rate。  

p设为0.5时的一个例子：  

{% asset_img sc_example.png prompt_compression %}  

## 实验  

既然删除了部分lexical unit，那模型输出结果就会变化，模型的效果很可能会下降。研究人员用4类任务实验，验证压缩的效果：（1）original context reconstruction（2）summarisation（3）question answering（4）conversation task。  

各个任务的指标都是和原prompt相比。具体来说，用original prompt下的模型输出作为标准，计算压缩prompt的输出和原输出的BLEU, METEOR, ROUGE, and BERTScore。  

在不同的压缩率（删除的lexical unit比例）下，模型的在各个任务的平均结果：  

{% asset_img sc_result.png prompt_compression %}  

和random compression的比较  

{% asset_img sc_compare.png prompt_compression %}  

## 小结  

实验没有测原task的得分变化，感觉这里有点不完善。  

Selective context的好处是不用训练生成模型，而可以应用到所有的输入prompt。不过一个问题是，用于压缩的prompt的小模型和生成模型在分布上也存在一些对不齐的情况，因此效果是有一些损失的。  

# LLMLingua  

LLMLingua是prompt压缩比较经典的工作了。  

他们观察到prompt里不同的部分 -- instruction、demonstration和question三者所能用的压缩率是不同的。demonstration通常是一些示例，是instruction和question的具象化，因此会包含比较多的信息冗余，而instruction和question本身是和answer是更加相关的，因此不能压缩太多。  

{% asset_img lingua_framework.png prompt_compression %}  

## 方法  

1、coarse compression  

基于上面的思路，LLMLingua首先对demonstration做一个coarse的compression。具体来说就是以完整的demonstration为单位，删掉一部分demonstration。  

instruction和question的压缩率是预定义的（实操中这两个压缩率分别是τ_ins=0.85,τ_que=0.9），可以根据这两个部分的压缩率、整体的target压缩率和各个部分的原始长度计算coarse demonstration compression这一步要删掉多少demonstration。  

那么怎么决定保留哪些demonstration呢？就是用一个小模型计算demonstration的PPL，然后保留PPL大的文本。  

{% asset_img lingua_algo1.png prompt_compression %}  

k表示选择多少个demonstration。  

（由于demonstration是粗粒度的选择，最终选的token数量和target的压缩率有出入，因此需要重新计算一下inteructino和question的压缩率）

2、fine compression  

在粗粒度的删除之后，就要进行细粒度的Iterative Token-level Prompt Compression（ITPC），把（instruction，删减过的demonstration，question）再进一步进行压缩。在这一步，token-level的dropout可能造成更多的信息损失，因此应该使用sentence-level的dropout以保持一定的lingusitic integrity。  

（1）分段  

首先把文本切成segment（实操中segment的长度是100token）。  

（2）计算条件概率

使用小模型 $\mathcal{M}_s$ 计算每个段 $s_j$ 中token的条件概率：

$$
p\left(s_{j, i} \mid s_{j,<i}, \widetilde{s}_{<j}\right)
$$  

其中 $\widetilde{s}_{<j}$ 表示前 $j-1$ 个段压缩后的结果。   

（3）动态计算压缩阈值  

根据段 $s_j$ 的压缩比例 $\tau_{s_j}$，动态计算阈值 $\gamma_j$，保留满足 $p(s_{j,i}) > \gamma_j$ 的令牌。  

（4）迭代压缩  

将压缩后的段 $\widetilde{s}_j$ 拼接至后续段，重复步骤2-3，直到所有段被压缩。  

## 效果  

LLMLingua在长文本上的效果还是可以的：  

{% asset_img lingua_perf1.png prompt_compression %}  

{% asset_img lingua_perf2.png prompt_compression %}  

# GIST  

## 思路  

长的prompt，比如system prompt占用大量重复计算；通过cache可以减少计算，但是prompt很长的话还是需要比较多的缓存；通过finetune可以把prompt内化，不过这样每个prompt都需要训一个模型。context distillation就是这样，不需要额外数据，内化prompt，一个模型学一个prompt：  

$$
\mathcal{L}_{CD}\left(p_{CD}^{t}, t\right)=\mathbb{E}_{x}\left[D_{KL}\left(p_{LM}(y | t, x) \| p_{CD}^{t}(y | x)\right)\right]
$$

t是prompt。  

那么更好的情况应该是只训一个模型，可以处理所有的prompt；学习G(t)这样一个映射，让G(t)更短，并且G有足够的泛化性。GIST方法就是学习G(t)的一种方法。泛化的G，只训练一次，就可以支持各种prompt的压缩：  

$$
\mathcal{L}_{G}(p_{G}, T)=\mathbb{E}_{t \sim T, x}[D_{KL}(p_{LM}(y \mid t, x) \| p_{G}(y \mid G(t), x))]
$$  


## 训练  

GIST方法首先在词表增加一个gist token，注意只有一个。训练的时候在prompt和answer中间夹k个gist token的copy，这k个gist token就用来学习怎么压缩prompt。  

把gist token加在prompt和answer中间之后，还要修改attention mask，让gist token后面的token只能看到gist token，而看不到原始的prompt token；而gist token可以看到原始的prompt，相当于让gist token成为把信息从prompt传递到answer的唯一桥梁，这就强制把prompt的信息都压缩到gist token里。  

各个模型结构下，attention mask的修改：  

{% asset_img gist_mask.png prompt_compression %}  

## 实验  

几个对照模型：  

- positive control：不修改attention mask，等价于用原prompt微调  
- negative control：不输入task，相当于random gist token  
- discrete compression with TF-IDF：用TF-IDF把prompt里的最关键词提取出来，比如Write a letter to your boss asking for an increase in salary → salary，Given two integers, find their average → average  

结合模型打分（GhatGPT）和Rouge-L，各个模型的效果：  

{% asset_img gist_perf.png prompt_compression %}  

GIST方法还是比较接近原prompt的效果的。  

# 小结  

- prompt压缩在特定的场景下，还是有比较大的收益的；对于目前输入普遍很长的情况，如果考虑成本，这是一个值得考虑的方向  
- prompt压缩的评测还是更多关注怎么和原prompt输出对齐，这里要记得做归一化  

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

【1】Compressing Context to Enhance Inference Efficiency of
Large Language Models  
【2】Prompt Compression and Contrastive Conditioning for Controllability and Toxicity Reduction in Language Models  
【3】Learning to compress prompts with gist tokens  
【4】LLMLingua: Compressing prompts for accelerated inference of large language models  

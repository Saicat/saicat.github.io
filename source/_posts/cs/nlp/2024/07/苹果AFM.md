---
title: 苹果智能系统模型--AFM
tags:
  - NLP
  - LLM
  - transformer
  - 技术报告
  - 苹果
  - post-training
  - SFT
  - DPO
  - RM
  - RS
  - 端侧模型
  - 蒸馏
categories:
  - CS
  - NLP
  - LLM
abbrlink: 1e34e252
date: 2024-07-31 22:28:10
---

【本文已在同名 微信公众号 / 知乎 / [个人博客linsight.cn](http://www.linsight.cn/) 上线】  

***  

之前苹果在WWDC24发布了包含多个强大模型的Apple Intelligence系统，苹果刚刚最新发出来的技术报告《Apple Intelligence Foundation Language Models》介绍了关于其中两个模型的一些细节 -- 端侧使用的，大小约3B的AFM-on-device，和云侧使用的更大模型AFM-server（AFM=Apple Foundation Model）。报告里没有给出AFM-server的规模。  

# 模型  

模型的设计比较常规（没有和OpenELM一样玩底大头尖的设计）：  

{% asset_img afm.png afm %}  

几个细节：  
- 共享了输入输入的embedding，减少参数量  
- 参考《Small-scale proxies for large-scale transformer training instabilities》，使用Query/key normalization，提升训练稳定性  
- RoPE的base frequency为500k  

tokenizer是基于SentencePiece用BPE训的，所有数字都切分为单个数字。AFM-server模型的词表大小为100k，AFM-on-device则小一些，只有49k。  

# 预训练  

## 数据  

- 数据的来源主要包括：开源数据，从出版商获得使用许可的数据，已经通过苹果的爬虫Applebot爬取的数据  
- 苹果认为相比数量，预训练数据的质量对模型在下游任务的影响更大  

另外苹果（自称）特别看重隐私和安全性，因此所有数据的几乎全部流程都有大量移除有害数据、personally identifiable information（PII）、成人内容的处理工作。  

下面罗列一些预训练数据的处理细节。  

1、网页数据  

处理pipeline包括：  
- 结合Safari的reader mode和Boilerpipe算法提取网页的主体内容  
- 规则+model based的安全过滤  
- 基于locality-sensitive n-gram hashing的模糊去重  
- 质量过滤（《Large language model-guided document selection》，《Datacomp-lm: In search of the next generation of training sets for language models》）  
- Decontamination：从预训练数据按n-gram删除和811个benchmark过度相关的数据，避免测试集污染  

2、授权数据  

从出版社获取的高质量长文本数据，主要用在二阶段和三阶段的预训练（各阶段方案在后面）。同样做了避免测试集污染的操作。  

3、代码  

来自github的开源仓库，包含14种语言，经过去重、PII过滤、质量过滤和Decontamination处理。  

4、数学  

包括3B数学QA内容，和14B数学相关的文档，来自数学论坛、博客、tutorial和seminar等。为了提取这些数据，苹果专门开发了对应的模板、数学符号filter、数学相关的quality filter以及领域filter。  

5、公开数据  

从公开数据里挑了一部分高质量数据。  

## 训练  

AFM的预训练分为3个stage：  
- core：大部分训练预算都在这一个阶段消耗  
- continued：上采样高质量数据，更多的code、math等内容  
- context-lengthening：和continued类似，使用更大的训练窗口和长文本数据  

三个stage在调参的时候，用了和《Small-scale proxies for large-scale transformer training instabilities》中的“μParam (simple)”类似的方法。  

### Core pre-training  

规模较大的AFM-server模型是从0开始训的，而较小的AFM-on-device则是从更大的模型蒸馏+剪枝来的。  

1、AFM-server  

- 使用6.3T数据  
- sequence length = 4096  
- batch size = 4096  
- weight decay = 3.16e-4  
- cosine lr schedule, max lr = 0.01, min lr = 0.5% max lr  
- warmup step = 5000  

batch size是通过scaling law的实验决定的，不过实践中发现，下游任务的效果对预训练的batch size并不敏感：batch size增大一倍或者缩小一半下游任务效果没有影响，因此虽然scaling law给出的预测最佳batch size是3072，实际训练的时候，为了效率还是使用了4096。  

通过proxy model的lr扫描，定了最佳lr在0.01~0.02，最终选择了0.01。（这里使用类似μParam的方法，各参数初始化和前向计算的时候应该都有缩放，所以这个lr会相对大一些）  

苹果训练的时候选择的优化器是RMSProp with momentum，而其他大部分大模型基本都是使用AdamW。  

对于训练设置的问题，苹果做了消融实验，把上面的core training和以下配置的训练（baseline）进行对比：  
- 使用AdamW，beta_1 = 0.9，beta_2 = 0.95  
- weight decay = 1e-4  
- lr 最小decay到0.0001  
- batch size = 1024  

其他设置保持一致，用AFM-on-device模型结构训练3.1T数据。  

二者的对比如下：  

{% asset_img core_ablation.png afm %}  

整体上AFM的core training比baseline略略好一点，基本上可以认为是持平。  

2、AFM-on-device  

AFM-on-device模型不是从零训练的，而是基于一个6.4B的模型（使用和AFM-server一样的训练方法得到的），使用了structural pruning和distillation得到的。  

所用的structural pruning和《Structured pruning of large language models》、《Sheared llama: Accelerating language model pre-training via structured pruning》相似，除了几点变化：  
- 只对FFN层做prune  
- 使用Soft-Top-K masking（《Conditional adapters: Parameter-efficient transfer learning with fast inference》）  
- 用了和core training一样的data mix训练了188B得到pruning mask  

以得到的模型的为初始化，进行知识蒸馏：把原来core训练的target label替换成：0.9 * teacher top-1 prediction + 0.1 * true label。  

同样进行了6.3T的蒸馏训练。  

相比直接从零训练，pruning和distillation在数据效率和最终结果上都有收益。使用不同方法训练出来的模型效果对比如下：  

{% asset_img distill.png afm %}  

整体来看，prune + distill能比多5倍training cost的从零训练baseline更好一点，训练效率更高。  

### Continued pre-training  

这一stage提高了math和code的比例，而降低了低质量的爬虫数据比例，进行了1T token的训练。  

训练设置：  
- sequence length = 8192  
- max lr = 3e-4，min lr = 0.1% max lr  
- weight decay = 1e-5  
- warmup step = 1000  

其他和core training保持一致。  

这一阶段数据蒸馏没有什么收益，所以AFM-on-device和AFM-server一样，采用直接训练的方式。  

3、Context lengthening  

最后这一阶段使用100B的长窗口训练来提升模型的长文本能力：  
- sequence length = 32768  
- RoPE base frequency 500k --> 6315089（《Scaling laws of rope-based extrapolation》）  
- 在二阶段数据的基础上，增加长的QA合成数据  

### 评测  

三个阶段后，AFM-on-device和AFM-server的评测效果如下（报告提到，使用了internal的formulation，所以没法和其他模型直接比较）  

{% asset_img pretrain_1.png afm %}  

{% asset_img pretrain_2.png afm %}  

continued pre-training和预期的一样，对math和code的能力有比较大的提升。  

# Post-Training  

AFM的post-training包括SFT和RLHF两个阶段，并使用了两个新方法iTec和MDLOO。  

## 数据  

post-training的数据包括人类真实数据和合成数据。  

### 合成数据  

一个好的reward model是合成高质量数据的关键，同时扩展prompt set提高多样化和覆盖范围也很重要。  

苹果介绍了数学、工具使用和代码这3个领域的数据合成。  

1、Mathematics  

数学数据的合成包括两个stage：  
- 生成数学问题  
- 生成对应答案  

基于一些种子prompt，通过以下方法获取数量更大、更多样化的prompt：  
- Problem rephrase and reversion：参考《Metamath: Bootstrap your own mathematical questions for large language models》，进行问题重述  
- Problem evolution：和指令进化类似（《WizardLM: Empowering large language models to follow complex instructions》），深度进化提升指令的复杂度，而广度进化提升话题的覆盖范围  

2、Tool use  

先从简单的single-tool数据开始，训练模型。然后逐步包含multi-tool和multi-step的问题，提升模型能力。此外，还会在数据里混入oracle tool和其他相似同居，增加工具选择的难度。  

另外还增加了tool intent detection数据，以减少过度使用工具的问题。  

3、Coding  

从71个话题的种子数据开始，通过self-instruct和rejection sampling让模型自动化学习。  

对于每个问题，模型会生成单元测试和多个solution，通过执行这些solution能够检验结果的正确性，组中选择通过测试最多的solution。  

另外还会给通过的单元测试设定一个阈值，只要高于这个阈值才会被使用。最终得到了12k的高质量代码数据。  

## SFT  

1、数据选择  

在质量过滤、去重之外，通过数据合成 + rejection sampling来提供大量合成数据，提升SFT训练数据规模。  

2、比例调整  

对不同数据组成部分的权重进行训练，然后调整比例。对此进行了大量实验，移除掉一些影响较小的数据。  

3、训练超参  

模型使用constant lr训练，AFM-server和AFM-on-device的lr分别为5e−6和2e−5。  

和其他家做法比较不同的，苹果使用的0.1的dropout rate。  

由于不同checkpoint的eval指标会有波动，因此使用RM选择best-of-N的方式来挑选最佳checkpoint。  

## RLHF  

苹果的RLHF有多轮，迭代提升模型。  

### RM  

用前面收集的偏好数据训练RM：  
- 每条prompt有两个response（对比一下，Llama-3可能有3条）  
- 偏好数据分为significantly better, better, slightly better, negligibly better四个等级  
- 除了综合的对比之外，每条response还有细粒度的打分，维度包括指令跟随、真实性、有害性、简明程度，每个维度的打分有3个等级  

RM取最后一层的最后一个non-padding token的embedding，再加上一个linear层和4个MLPhead来输出打分。linear层输出偏好奖励，而4个MLP层分别输出4个细粒度打分的分类结果，四个分类头的输出分别是 $u_\phi^\mathrm{if},u_\phi^\mathrm{verb},u_\phi^\mathrm{truth},u_\phi^\mathrm{harm}$。  

RM训练时，使用soft label loss function，这样可以把偏好的程度也纳入考虑。同时细粒度的打分也作为regularization term加入训练，实验发现这些细粒度打分能提升RM的准确性。  

1、Soft label loss  

基于Bradley-Terry model，y_c（c=chosen）比y_r（r=rejected）的概率是

$$\sigma(r_\phi(x,y_c)-r_\phi(x,y_r))$$  

直观来说，如果两条response的质量差距越大，这个值应该越大。  

因此对于不同的偏好程度l，设计了一个超参，target preference probability p_l，并构造一个soft label loss：  

$$\begin{aligned}
L_{\mathrm{ranking}}(\phi)=& -p_\ell\log(\sigma(r_\phi(x,y_c)-r_\phi(x,y_r))  \\
&-\left(1-p_\ell\right)\log(\sigma(r_\phi(x,y_r)-r_\phi(x,y_c))
\end{aligned}$$  

如果偏好程度比较高，那么p_l的值应该更大。实践中，p_l使用了0.95、0.85、0.75、0.65这四个值。  

2、Single-sided grading as regularization  

regularization loss如下：  

$$\begin{aligned}L_{\mathrm{regu}}(\phi)&=\sum_{\text{grade}\in\text{if,verb,truth,harm}}\left(\text{cross}_\text{entropy}(u_\phi^\mathrm{grade}(x,y_c),z_c^\mathrm{grade})\right.\\&+\text{cross}_\text{entropy}(u_\phi^\mathrm{grade}(x,y_r),z_r^\mathrm{grade})\Big)\end{aligned}$$  

其中z是各个细粒度维度的打分。  

最终RM的训练loss为：  

$$L_\text{ranking}(\phi)+\lambda L_\text{regu}(\phi)$$  

### Iterative teaching committee（iTeC）  

苹果提出一个iterative RLHF框架来优化模型。  

苹果在AFM的RLHF中，学到的最重要的事情之一就是“refresh online human preference data collection using a diverse set of the best performing models”。  

具体来说，构建一个由SFT、拒绝采样、DPO/IPO和RL训练出来的最佳模型，以及前几轮的最佳模型组成的集合，称之为“model committee”，并从这个model committee收集最新的偏好数据。  

在获取最新的偏好数据之后，会更新RM，让后训练一组新的最佳模型，这些新的模型会加入model comittee，继续下一轮迭代。  

不同的优化算法训练出来的模型有不同的特点，比如使用负例的算法，online RLHF、DPO、IPO等，在数学推理方面的能力较好，而rejection sampling在指令遵循和写作方面更有效。通过在model comittee进行采样，并用最新的RM进行排序，可以结合多个模型的强项。  

### Online RLHF algorithm: MDLOO  

从经典的RLHF优化目标出发：  

$$\max_\theta\mathbb{E}_{x\sim\mathcal{D},y\sim\pi_\theta(\cdot|x)}\left[r_\phi(x,y)-\beta D_{\mathrm{KL}}\left(\pi_\theta(\cdot|x)\|\pi_{\mathrm{ref}}(\cdot|x)\right)\right]$$  

苹果选用的reward function是

$$R(x,y)=r_\phi(x,y)-\beta\log\frac{\pi_\theta(y|x)}{\pi_\text{ref}(y|x)}$$  

这个reward function的前面的expectation是等价的。  

和经典RLHF有所不同的是，这里把一整个response视为一个action，因此就不再需要critic模型来对每个token的reward进行打分了。  

此外，苹果还做了几个改动。  

1、Leave-One-Out (LOO) estimator of the advantage  

对于底k个iteration的policy model，每次输入n条prompt，每个prompt采样K条数据。  

那么按照定义，advantage就是：  

$$A_k(x,y_i)=R(x,y_i)-\mathbb{E}_{y\sim\pi_{\theta_k}(\cdot|x)}[R(x,y)]$$  

苹果使用leave-one-out (LOO)方法来估计 $A_k(x,y_i)$，即通过prompt x和其他K-1个response：  

$$\widehat{A}_k(x,y_i)=R(x,y_i)-\frac{1}{K-1}\sum_{j\neq i}R(x,y_j)$$  

同《Back to basics: Revisiting reinforce style optimization for learning from human feedback in LLMs》发现的一样，这样的advantage估计在RLHF有一些收益。另外，实践上发现这样做能让训练更加稳定。  

2、Mirror descent policy optimization (MDPO)  

和常用的clipping-based PPO不同，使用KL divergence作为regularization。  

# 赋能Apple Intelligence  

AFM是给Apple Intelligence使用的，而Apple Intelligence主要是支持iPhone、iPad和Mac等设备的，因此「计算效率」和针对这些设备场景下的「专用能力」是重点。  

虽然经过post-training之后，模型的通用能力已经不错，但是针对设别上的任务进行专门的微调，还能获得进一步的提升。苹果通过使用多个任务相关的adapter，在提升多个任务效果的同时，保持了参数和计算的高效。这些adapter很小，运行时可以在内存中随意切换。  

整体的框架如下图所示  

{% asset_img intelligence.png afm %}  

## accuracy-recovery adapter  

1、效果恢复  

端侧设备的空间比较小，所以量化是必须要做的。首先，post-training后的模型会用4-bit的精度进行量化。  

但是由于量化模型会带来一定的效果损失，所以这个量化模型并不是直接使用，而是会在固定量化模型的基础上，用16-bit的LoRA进行训练，以尽量恢复因为量化带来的效果损失。这个LoRA就叫accuracy-recovery adapter。  

accuracy-recovery adapter的训练过程和主干模型的训练保持一致，也进行了pre-training和post-training的训练。不过由于参数量很小（只有几十MB），所以整个预训练大概只用了10B的数据，并且基本可以恢复大部分由于量化带来的效果损失。  

实践上，rank 16基本上可以获得比较好的效果，不过出于灵活性的考虑，还是提供了不同rank的LoRA参数给下游使用：8、16、32。  

模型量化前后，以及使用accuracy-recovery adapter之后的效果对比如下：  

{% asset_img recover.png afm %}  

rank 16的adapter基本可以恢复大部分量化带来的效果损失，并且量化的损失越多，adapter恢复的比例越大。也就是使用了accuracy-recovery adapter之后，基本可以不用太在意量化的损失，可以进一步提高模型压缩的程度。  

2、Quantization schemes  

以往量化的时候，因为要兼顾效率和效果损失，一般把block size设成32或者64这样比较小的规模。现在有了accuracy-recovery adapter，反正损失掉的基本都可以恢复，那block size就可以设得更大了，甚至可以达到100k。  

另外，由于AFM的输入输出embedding是shared的，为了有更好的效率，embedding部分使用8-bit的per-channel quantization。  

3、混合精度量化  

模型中明显每层对效果的影响是不同的，对于对最终效果影响较小的层，苹果进一步用2-bit的量化精度，最终整体可以达到3.5~3.7的bpw。  

## task-specific adapter  

针对不同的下游任务，可以在accuracy-recovery adapter的基础上再进一步微调。这样在保持主干网络为4-bit模型的情况下，下游任务就能有很好的效果。  

以summarization为例，具体任务是对设备上的email、message和notification进行摘要。使用AFM-server模型，用设备上真实信息的格式构造训练数据，然后用这些训练数据训练adapter。  

# 小结  

- 模型设计、pre-training和post-training大部分使用的都是比较常规有效的做法，但是在训练上使用了RMSProp with momentum，和其他大部分模型的做法不太一样。  
- accuracy-recovery adapter看起来是比较合理有效的，看后面其他手机厂商怎么follow。  

***  

读到这了，来一发点赞收藏关注吧~

博客：[http://www.linsight.cn/](http://www.linsight.cn/)  
知乎：[Linsight](https://www.zhihu.com/people/us4ever)  
微信公众号：Linsight  
![](/images/qrcode.jpg)  

***  

【推荐文章】  

- MoE：  
<p style="line-height: 1.2;"><small>
[MoE模型的前世今生](http://www.linsight.cn/44e38c1b.html)  
[DeepSeek-V2和MLA](https://www.linsight.cn/83c49df0.html)  
[昆仑万维-SkyworkMoE](https://www.linsight.cn/1d5bcd45.html)  
[成本10w刀的JetMoE](https://www.linsight.cn/f3acf042.html)  
[MoE的top-p routing](https://www.linsight.cn/224c42da.html)  
[对MoE模型的一些观察](https://www.linsight.cn/5e1d14b3.html)  
[从dense到MoE -- sparse upcycling](https://www.linsight.cn/a0824e29.html)  
[MoE路由--expert choice routing](https://www.linsight.cn/2c8bbc7.html)  
</small></p>

- 预训练：  
<p style="line-height: 1.2;"><small>
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
</small></p>

- 数据：  
<p style="line-height: 1.2;"><small>
[预训练数据处理--长度分解](https://www.linsight.cn/210dbccd.html)  
</small></p>

- 长上下文：  
<p style="line-height: 1.2;"><small>
[LLM长上下文的问题](http://www.linsight.cn/c4da56c0.html)  
[解锁大模型长上下文能力](http://www.linsight.cn/cc852861.html)  
[大模型推理窗口-从有限到无限大](http://www.linsight.cn/45ee1a6d.html)  
</small></p>

- 推理加速：  
<p style="line-height: 1.2;"><small>
[大模型推理加速-投机解码](http://www.linsight.cn/f5c015c.html)  
[大模型推理加速-MEDUSA](https://www.linsight.cn/7bbe2df6.html)  
</small></p>

- 对齐：  
<p style="line-height: 1.2;"><small>
[Llama3.1--post-training要点一览](https://www.linsight.cn/93328a2a.html)  
[模型平均 -- model soup](https://www.linsight.cn/bb8fcf21.html)  
[大模型偏好对齐-DPO](http://www.linsight.cn/473f2b43.html)  
[大模型偏好对齐-ODPO](http://www.linsight.cn/da871ebe.html)  
[大模型偏好对齐-simPO](http://www.linsight.cn/280fa97a.html)  
[大模型偏好对齐-IPO](http://www.linsight.cn/4fe7b810.html)  
</small></p>

- Transformer：  
<p style="line-height: 1.2;"><small>
[理解Attention:从起源到MHA,MQA和GQA](http://www.linsight.cn/3dc22f96.html)  
[LLM的重复生成和ICL](https://www.linsight.cn/7381cae3.html)  
[transformer中normalization的二三事](http://www.linsight.cn/6a40bfa5.html)  
[从代码实现看normalization-到底做了什么](http://www.linsight.cn/b70b4a2d.html)  
[稀疏注意力计算:sliding window attention](http://www.linsight.cn/c61d17e3.html)  
[理解LLM位置编码:RoPE](http://www.linsight.cn/a051710f.html)  
[RoPE的远距离衰减](https://www.linsight.cn/f0902f1a.html)  
</small></p>

- 大模型算法题：  
<p style="line-height: 1.2;"><small>
[(1)](http://www.linsight.cn/3345028a.html)、
[(2)](http://www.linsight.cn/ad0bba9d.html)、
[(3)](http://www.linsight.cn/1736008.html)、
[(4)](http://www.linsight.cn/1736008.html)、
[(5)](http://www.linsight.cn/336f2f3e.html)、
[(6)](http://www.linsight.cn/7c04944d.html)、
[(7)](https://www.linsight.cn/dd614e12.html)、
[(8)](https://www.linsight.cn/e287b9c3.html)  
</small></p>

# Reference  

【1】Apple Intelligence Foundation Language Models https://machinelearning.apple.com/papers/apple_intelligence_foundation_language_models.pdf  

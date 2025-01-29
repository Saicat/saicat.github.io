---
title: 深度求索DeepSeek-R1详解
tags:
  - NLP
  - LLM
  - transformer
  - RL
categories:
  - CS
  - NLP
  - LLM
abbrlink: 9e4b4e6d
date: 2025-01-23 21:47:34
---

【本文已在同名 微信公众号 / 知乎 / [个人博客linsight.cn](http://www.linsight.cn/) 上线】  

***  

过年前这段时间好几个重磅工作相继发布，有深度求索的DeepSeek-V3、DeepSeek-R1、月之暗面的Kimi-K1.5，还有MiniMax的MiniMax-01、面壁智能的MiniCPM-o 2.6和智谱的GLM-Realtime，以及阶跃的Step-1o等，2025年才过了不到一个月，真·卷起来了。百花齐放的场景让人对AI充满期待，同时作为开发者也感到压力山大。  

还好不少工作都有给出技术报告，让我们有机会站在它们的肩膀上，今天就来学习一下DeepSeek-R1。  

{% asset_img perf.png r1 %}  

# overview  

先概括说说DeepSeek-R1是什么，大致干了什么：  
- ①：以671B参数的DeepSeek-V3-Base为起点，直接对预训练模型进行reasoning任务的强化学习，获得DeepSeek-R1-Zero，发现效果比单纯SFT更好，说明强化学习的self-evolution很有效果  
- ②：DeepSeek-R1-Zero虽然效果好，但是有一些小问题比如语言问题和格式问题，那么在强化学习RL之前，先做SFT，就可以缓解这些问题了  
- ③：②的方法得到的模型的reasoning效果很好，那就用它来搞reasoning数据；再加上DeepSeek-V3的SFT数据中的non-reasoning部分，合在一起获得高质量的SFT数据  
- ④：③中的数据用来对DeepSeek-V3-Base做微调，之后再进行RL，得到效果最好的模型DeepSeek-R1  
- ⑤：用③中的数据对Qwen/Llama模型进行SFT，可以视作是数据蒸馏；得到的模型效果也比非蒸馏的SFT要好  

# 训练pipeline  

DeepSeek-R1技术报告给出了几个模型的训练流程，DeepMind的大佬给训练流程画了图，原图在[https://x.com/SirrahChan/status/1881488738473357753?s=19&mx=2](https://x.com/SirrahChan/status/1881488738473357753?s=19&mx=2)。  

在这个基础上，我稍稍加了一点点修改，理清一点细节。DeepSeek-R1中提到的模型训练流程如下：  

{% asset_img pipeline.png r1 %}  

左路就是DeepSeek-R1-Zero的训练（上面的①），中路是基于SFT和RL搞数据（上面的③），获得800k Combined SFT data之后，左边是对Qwen和Llama进行蒸馏（上面的⑤），右边就是训练DeepSeek-R1（上面的④）。  

下面就一个个来看下细节。  

# DeepSeek-R1-Zero  

DeepSeek-R1-Zero以DeepSeek-V3-Base做初始化，在没有SFT阶段的情况下直接做RL，就获得了比较好的效果。  

强化学习方法用的是Deepseekmath中提出的Group Relative Policy Optimization(GRPO)，而训练的reward信号没有使用模型产生，而是仅使用规则来决定。主要包含两类reward：  
- accuracy rewards：对于数学问题这类有确定答案的问题，要求模型给出特定格式下的结果，方便进行正确性校验；而对于LeetCode问题，则是给出test case的执行结果作为反馈；  
- format rewards：格式奖励，强制模型将思考过程放在输出的\<think\>和\</think\>这两个特殊tag中间；  

那么为什么不使用模型来获取reward呢？  

因为使用reward model就有可能出现reward hacking，也就是actor有可能会找到捷径来获取高reward值，从而偏离了我们想要的优化目标。比如一个常见的hacking就是模型经常会发现“输出越长得分越高”，从而给出又臭又长的回答。因此在一些RL框架中就有专门对长度进行惩罚来避免这个捷径被滥用。  

reward hacking没法预测，不知道在什么时候就会出现一个奇怪的reward漏洞，这么一来就可能需要多次迭代reward模型，堵上这些捷径和漏洞。但是对于大规模的模型，每次迭代的更新数据和训练都要耗费比较大的成本，从而导致整个训练流程变得困难。  

用规则reward就不会有这样的问题，而且对于数学和代码类问题规则的效率也更高。  

训练DeepSeek-R1-Zero的目的是希望模型能够通过长思考自我反馈来解决复杂问题，那么就需要它按我们想要的格式输出，起码能清晰给出哪些是思考过程（隐藏的输出），哪些是最终结果（用来呈现给用户）。因此给RL训练的输出设计了模板，引导模型按照这个格式输出，方法就是前面提到的format reward：  

{% asset_img template.png r1 %}  

随着RL训练的进行，DeepSeek-R1-Zero的长思考能力持续提升。  

下图是训练期间DeepSeek-R1-Zero在AIME 2024 benchmark上的得分变化情况，每400步进行一次评测：  

{% asset_img aime.png r1 %}  

可以看到相比Base模型，得分确实有显著的提升，从15.6%提升到了71.0%，达到了与OpenAI-o1-0912相当的水平。在其他benchmark上，DeepSeek-R1-Zero也有类似的提升：  

{% asset_img reasoning_benchmark.png r1 %}  

另外还可以观察到，随着RL的进行，DeepSeek-R1-Zero的思考内容在持续变多。这表现在输出结果的CoT中，即\<think\>和\</think\>中间的内容长度在持续增加，最后达到了接近10k，而且增长的趋势完全没有减弱。可以预想继续训练的话还会变得更长（当然更长并不一定是更好）：  

{% asset_img length.png r1 %}  

这些效果提升说明不使用监督数据，而仅使用强化学习，模型可以自发探索和环境进行交互的方式，并且对复杂问题可以自发学会进行复杂的思考，从而提升处理困难问题的能力。  

## Aha moment  

DeepSeek-R1-Zero的Aha moment是在训练过程中观察到的一个有趣现象。在一些中间版本，模型在思考过程中对前面的方法进行重新审视，并为其中的问题重新分配了思考时间：  

{% asset_img aha.png r1 %}  

这是一个拟人化的心理活动，而且是在没有人类监督学习的情况下出现的。（这会不会说明RL真的可以通往更高层级的智能，真正到达人类水平甚至更高的水平呢？）  

## Drawback of DeepSeek-R1-Zero  

DeepSeek-R1-Zero虽然在reasoning任务上有了明显的提升，不过也有一些缺点：比如模型的输出可读性较差（会不会这就是模型的思考方式呢，所以人类看着费劲），对于部分语言会出现混用乱用的情况。  

# DeepSeek-R1  

接下来就是DeepSeek-R1的出场了。在DeepSeek-R1-Zero的结果和分析之下，就有两个自然的问题：  
- RL已经这么好，那么先SFT再RL不得起飞？  
- 光有reasoning CoT的能力不够，能不能搞一个通用能力也很强的版本？  

## Reasoning版本R1  

首先来看第一个问题，SFT+RL。  

为了防止Base模型在RL初期出现不稳定的情况，先收集几千个long CoT data，用来对Base模型做了SFT。这些long CoT data怎么收集的呢？就是用包含few-shot example的prompt，让DeepSeek-R1-Zero输出可读性较好的、带有reflection和verification的结果，再经过人工校验获取的。  

为了提升SFT后生成结果的可读性，专门给这些SFT数据设计readable pattern：在response后面加上一个summary，格式如下：  

|special_token|<reasoning_process>|special_token|\<summary\>

其中reasoning_process是CoT的内容，而summary是reasoning结果的总结。  

SFT之后就是进行和DeepSeek-R1-Zero一样的RL了。前面在训练DeepSeek-R1-Zero的时候，就发现模型输出会出现语言混用的情况，特别是当输入prompt涉及多种语言时。那么这次RL就为此专门设计了一个language consistency reward，具体来说就是CoT中target language word的比例。虽然在消融实验中发现加入这个语言一致性reward会带来一点效果损失，不过这样的输出结果对人类更友好。  

有了这些SFT数据做冷启动之后，再进行RL，模型主要有两点变化：  
- readability：有了SFT作为冷启动，模型的可读性更好了  
- potential：增加SFT之后，整体的效果也更好了  

## 新一轮的数据收集  

1、reasoning数据  

上一步通过少量人工参与的SFT数据+RL，获得了比DeepSeek-R1-Zero更好一点的模型。那这个模型是不是又可以用来收集更好的SFT数据了呢？答案当然是yes，不要忘了前面这些SFT数据就是从更早的版本DeepSeek-R1-Zero来的。  

为了收集更好的数据，这里使用rejection sampling来采样reasoning trajectory。之前的数据基本上至包含了可以使用规则来评估reward的样例，但是这次我们把范围扩大，增加了一些没法直接用规则判定的reasoning数据。这些新增的reasoning数据就需要用到模型来判别，而DeepSeek-V3就可以作为这个判别模型，通过输入的ground truth和prediction来judge结果的好坏。  

此外，还有一些清洗规则：  
- 语言混合  
- 长段落  
- 包含代码块（毕竟大脑不能跑代码？）  

最终采样了600k条reasoning data。  

2、non-reasoning data  

回顾前面的两个问题，第一个已经验证了，再看看第二个：光有reasoning CoT的能力不够，能不能搞一个通用能力也很强的版本？  

想要提升通用能力，那就需要包含一些non-reasoning data：比如writing、factual QA、self-cognition和translation等。  

这些数据来自于DeepSeek-V3的SFT数据。对于某些任务，会调用DeepSeek-V3在回答问题之前先生成一个CoT；而对于某些比较简单的query，比如“hello”这样的打招呼，则不增加CoT。  

最终整合大约200k的non-reasoning data。  

## SFT + RL  

上面得到了600k + 200k = 800k的SFT数据，首先用这些数据在DeepSeek-V3-Base上训了2个epoch。接下来就要进行RL了。  

RL的reward设置和前面又有点不同。对于数学、代码和logical reasoning的任务，这里还是使用和DeepSeek-R1-Zero一样的规则reward。而对于general数据，就用上了reward model。

reward model依然是基于DeepSeek-V3的。对于helpfulness，主要关注在final summary，确保给到用户的response的实用性。而对于harmlessness，则会关注整个模型数据，包括过程和结果，识别和减轻在生成过程中任何可能出现风险的地方。  

这样一套下来，就得到了最终DeepSeek-R1。  

## 评测  

评测中，所有模型的设置都是：  
- 长度32768 token  
- 对于需要采样的，使用temperature = 0.6，top-p = 0.5，每条query生成64条response  

DeepSeek-R1的评测结果如下：  

{% asset_img eval.png r1 %}  

# 蒸馏  

在前面的流程中，SFT数据的产生来自DeepSeek-V3（或进一步训练的变体），reward来自DeepSeek-V3，所有流程都是基于DeepSeek-V3来做的。最后产生的这800k数据可以说是DeepSeek-V3这个模型能给出的精华内容。  

用这800k数据训练其他更小的模型，也可以视为是一种数据蒸馏（就像大家都去拉取GPT-4/o1/o3的数据用来训练一样）。  

具体选择的小模型有：Qwen2.5-Math-1.5B、Qwen2.5-Math-7B、Qwen2.5-14B、Qwen2.5-32B、Llama-3.1-8B 和 Llama-3.3-70B-Instruct。  

这些蒸馏模型相比原模型也有很大的提升：  

{% asset_img distill_eval.png r1 %}  

DeepSeek-R1-Distill-Qwen-32B甚至超过了QwQ-32B-Preview（这是不是有点打Qwen脸了）。  

如果这些模型不蒸馏，而是进行和DeepSeek-R1-Zero类似的强化学习，能不能比数据蒸馏强呢？为了回答这个问题，在Qwen-32B-Base上进行了10k+步的强化学习，得到DeepSeek-R1-Zero-Qwen-32B，效果和QwQ-32B-Preview差不多，但是还是不如数据蒸馏的SFT模型：  

{% asset_img distill_and_rl.png r1 %}  

# Unsuccessful Attempts  

文中也提到一些失败的尝试。  

1、Process Reward Model (PRM) 

之前的PRM工作，比如：  
- Let’s verify step by step  
- Solving math word problems with process-and outcome-based feedback  
- Math-shepherd: A labelfree step-by-step verifier for llms in mathematical reasoning  

都有一些局限性。首先，在推理中明确区分各个step就不是容易的事；此外确定各个step是否正确也缺乏有效手段；另外，reward model的引入也会导致前面反复提到的reward hacking问题。  

2、search algorithms such as Monte Carlo Tree Search and Beam Search  
- Alphazero-like tree-search can guide large language model decoding and training  
- Solving olympiad geometry without human demonstrations  
- Deepseek-prover-v1.5: Harnessing proof assistant feedback for reinforcement learning and monte-carlo tree search  

搜索是另外一个方法。但是语言的搜索空间比象棋更大，因此难度更高。如果在每个节点扩大搜索范围，则可能会陷入局部最优。此外，训练好的value model也不是易事，这就到时模型的迭代比较困难。  

# 其他内容  

其他一些相关内容，可以看看的。  

1、reinforcement learning：  
- Training language models to self-correct via reinforcement learning  

2、强化学习在reasoning task的有效性：  
- Deepseekmath:Pushing the limits of mathematical reasoning in open language models  
- Math-shepherd: A labelfree step-by-step verifier for llms in mathematical reasoning  

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
- 多模态：  
[多模态入门(一)--CLIP](https://www.linsight.cn/3069051d.html)  
[多模态入门(二)--Flamingo,LLaVA系列和BLIP系列](https://www.linsight.cn/569d722c.html)  
[多模态入门(三)--MiniGPT4,DeepSeekVL,InternVL系列和QwenVL系列](https://www.linsight.cn/f16505b3.html)  
[多模态入门(四)--CogVLM,VILA,MM1,MM1.5和Pixtral-12B](https://www.linsight.cn/e00debee.html)  
[多模态入门(五)--InternVL系列](https://www.linsight.cn/52c8a4f9.html)  
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

【1】DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning, https://arxiv.org/abs/2501.12948  
【2】DeepSeek-R1训练流程图，https://x.com/SirrahChan/status/1881488738473357753?s=19  

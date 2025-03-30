---
title: 关于The_Bitter_Lesson
abbrlink: d253d7b3
date: 2025-03-30 11:41:11
tags:
  - NLP
  - LLM
  - 创业
  - scaling law
categories:
  - CS
  - NLP
  - LLM
---

【本文已在同名 微信公众号 / 知乎 / [个人博客linsight.cn](http://www.linsight.cn/) 上线】  

***  

The Bitter Lesson的原文和译文放在后面了。可以先看看。  

强化学习之父Rich Sutton这篇2019年的短文最近出场率很高。这篇文章其实就讲一个事：基于人类认知的逻辑雕花，在拉长的时间线上来看，都终将被摩尔定律加持的search和learning所碾压。  

说白了还是scaling law。最近新版的GPT-4o令人震惊的效果就再次验证了这个道理。  

{% asset_img 1.jpg The_Bitter_Lesson %}  

在GPT-4o吉卜力风格爆火之前，ComfyUI串联stable diffusion model + 各种人工设计的workflow是少有的能出"商用级"图片的方法。但是不得不说ComfyUI的使用还是有些门槛的，显卡也好模型也好流程设计也好都是有成本的。因此也有很多人做起了售卖lora和workflow或者图片定制的生意，都挺好赚的。现在GPT-4o一下子把这些都碾压了。workflow？不存在的，老夫就是一把梭，一句话修图。无论是风格还是一致性，都已经达到比较高的水准。  

{% asset_img 2.png The_Bitter_Lesson %}  

其实时间再往前一点点，Gemini 2.0 Flash的效果基本上也完全可以当傻瓜式PS使用了。  

说说自己。本科的时候，我用各种角点检测算子、边缘检测算子 + 统计计算图像的特征，再加上SVM和各种人工规则做了个机器人视觉寻路系统。中间各种人工逻辑给识别结果做处理和打补丁。那时觉得这就很厉害了，只要再继续雕花必定大有前途。然后过了没多久VGG-16和ResNet出来了，再往后就是CNN横扫天下的几年。无论什么任务，只要能转换成模型输入，就是一把梭训就完事了。所以后来有段时间变成了一个每天的任务就是造训练数据的算法工程师。这就是算力加持下learning的碾压。  

这几年搞的人机聊天也是一样的。以前需要QA知识库+匹配模型+对话管理+指代消解+意图理解...，每个部分都需要投入人力时间开发调试。现在呢，啪，扔上去一个DeepSeek-V3/R1完事了。这也是算力加持下search和learning的无情碾压。  

无论是从业界的发展，还是个人的经历来看，这确实是a bitter lesson。有另外一句话跟这个道理很相关："人们总是倾向于高估技术的短期收益，而低估技术的长期影响。"22年的时候当我还在玩Bert玩得不亦乐乎的时候，ChatGPT就这样自然又悄无声息地降临了。回头去看，其实从GPT2到GPT3就已经有一些端倪了，但是我们当时都没有很重视它。  

不过，虽然大道理是这样没错，但是一个问题是"算力加持下的search和learning"多久能够赶上现有技术+人类逻辑雕花。一个自然的想法是，如果暂时没法大幅提升模型本身的能力，那么用现有的材料给它安上一些工具总是有收益的。在不开上帝视角的情况下，没有人能够准确预知什么时候LLM的能力能够再上一层楼。现在没法解决的问题在哪个版本就能够解决，是三个月、六个月或者是一年？但是无论多久，在快速变化的市场下我们都不可能不做任何动作。就像我们总是能知道下个月一定会有更强的模型发布，但是我们也不可能把解决现在难题的希望完全寄托在未来的模型上。  

流浪地球里，几十支队伍都去拯救同一个发动机，只要有一支队伍成功了那救援就成功了，这叫饱和式救援。从整个业界来说，这些基于人类逻辑的雕花也可以叫做饱和式发展。我们没有办法知道search和learning什么时候能够解决我们的问题，那就先做能做的，只要有一个成功了，起码这个台阶就算上去了。虽然search和learning作为最强力的一支部队，只要时间拉得够长就会赶上大部分的进展，但是万一哪天它卡住了（摩尔定律也在放缓），我们至少还有plan B。  

再从成本角度来看，目前强大的通用型模型成本肯定比只完成单一任务的垂域小模型要高。在生产力场景，垂域能力的需求量并不低，因此这里的成本不可不考虑。对于大厂如此，对于创业公司更是如此。只是目前大家都还只关注在效果上，等到效果提升的进展放缓，成本一定会再次成为重点问题之一。  

而从个人角度来说，我相信即使过往的工作被碾压，也并非什么都没有留下：经验和眼界依然是很重要的财富。以前做预训练的时候热衷于研究各种结构变化，参数调整。现在来看这些都变成了屠龙之技。虽然现在没有龙了，但是屠龙技的学习还是锻炼了人的认知和思考，相信这种经历依然有它发挥价值的场景。就像三体中，丁仪在水滴到达地球前醒来，即使被冷冻了多年，但是他还是地球物理学中最顶峰的人之一，就是因为在那个基础物理被锁大多数人转向应用物理的年代，他是少数做过基础研究的人。这种经历让他拥有其他新时代没有的视角。  

最后，从AI创业者的角度来看。一方面，AI创业者正在重蹈AI研究者的覆辙，不断有"走错"赛道被新模型碾压出局的人。另一方面，风浪越大鱼越贵，winner的回报将是丰厚的。  

# The Bitter Lesson  

Rich Sutton  

March 13, 2019  

The biggest lesson that can be read from 70 years of AI research is that general methods that leverage computation are ultimately the most effective, and by a large margin. The ultimate reason for this is Moore's law, or rather its generalization of continued exponentially falling cost per unit of computation. Most AI research has been conducted as if the computation available to the agent were constant (in which case leveraging human knowledge would be one of the only ways to improve performance) but, over a slightly longer time than a typical research project, massively more computation inevitably becomes available. Seeking an improvement that makes a difference in the shorter term, researchers seek to leverage their human knowledge of the domain, but the only thing that matters in the long run is the leveraging of computation. These two need not run counter to each other, but in practice they tend to. Time spent on one is time not spent on the other. There are psychological commitments to investment in one approach or the other. And the human-knowledge approach tends to complicate methods in ways that make them less suited to taking advantage of general methods leveraging computation.  There were many examples of AI researchers' belated learning of this bitter lesson, and it is instructive to review some of the most prominent.

In computer chess, the methods that defeated the world champion, Kasparov, in 1997, were based on massive, deep search. At the time, this was looked upon with dismay by the majority of computer-chess researchers who had pursued methods that leveraged human understanding of the special structure of chess. When a simpler, search-based approach with special hardware and software proved vastly more effective, these human-knowledge-based chess researchers were not good losers. They said that "brute force" search may have won this time, but it was not a general strategy, and anyway it was not how people played chess. These researchers wanted methods based on human input to win and were disappointed when they did not.

A similar pattern of research progress was seen in computer Go, only delayed by a further 20 years. Enormous initial efforts went into avoiding search by taking advantage of human knowledge, or of the special features of the game, but all those efforts proved irrelevant, or worse, once search was applied effectively at scale. Also important was the use of learning by self play to learn a value function (as it was in many other games and even in chess, although learning did not play a big role in the 1997 program that first beat a world champion). Learning by self play, and learning in general, is like search in that it enables massive computation to be brought to bear. Search and learning are the two most important classes of techniques for utilizing massive amounts of computation in AI research. In computer Go, as in computer chess, researchers' initial effort was directed towards utilizing human understanding (so that less search was needed) and only much later was much greater success had by embracing search and learning.

In speech recognition, there was an early competition, sponsored by DARPA, in the 1970s. Entrants included a host of special methods that took advantage of human knowledge---knowledge of words, of phonemes, of the human vocal tract, etc. On the other side were newer methods that were more statistical in nature and did much more computation, based on hidden Markov models (HMMs). Again, the statistical methods won out over the human-knowledge-based methods. This led to a major change in all of natural language processing, gradually over decades, where statistics and computation came to dominate the field. The recent rise of deep learning in speech recognition is the most recent step in this consistent direction. Deep learning methods rely even less on human knowledge, and use even more computation, together with learning on huge training sets, to produce dramatically better speech recognition systems. As in the games, researchers always tried to make systems that worked the way the researchers thought their own minds worked---they tried to put that knowledge in their systems---but it proved ultimately counterproductive, and a colossal waste of researcher's time, when, through Moore's law, massive computation became available and a means was found to put it to good use.

In computer vision, there has been a similar pattern. Early methods conceived of vision as searching for edges, or generalized cylinders, or in terms of SIFT features. But today all this is discarded. Modern deep-learning neural networks use only the notions of convolution and certain kinds of invariances, and perform much better.

This is a big lesson. As a field, we still have not thoroughly learned it, as we are continuing to make the same kind of mistakes. To see this, and to effectively resist it, we have to understand the appeal of these mistakes. We have to learn the bitter lesson that building in how we think we think does not work in the long run. The bitter lesson is based on the historical observations that 1) AI researchers have often tried to build knowledge into their agents, 2) this always helps in the short term, and is personally satisfying to the researcher, but 3) in the long run it plateaus and even inhibits further progress, and 4) breakthrough progress eventually arrives by an opposing approach based on scaling computation by search and learning. The eventual success is tinged with bitterness, and often incompletely digested, because it is success over a favored, human-centric approach.

One thing that should be learned from the bitter lesson is the great power of general purpose methods, of methods that continue to scale with increased computation even as the available computation becomes very great. The two methods that seem to scale arbitrarily in this way are search and learning.

The second general point to be learned from the bitter lesson is that the actual contents of minds are tremendously, irredeemably complex; we should stop trying to find simple ways to think about the contents of minds, such as simple ways to think about space, objects, multiple agents, or symmetries. All these are part of the arbitrary, intrinsically-complex, outside world. They are not what should be built in, as their complexity is endless; instead we should build in only the meta-methods that can find and capture this arbitrary complexity. Essential to these methods is that they can find good approximations, but the search for them should be by our methods, not by us. We want AI agents that can discover like we can, not which contain what we have discovered. Building in our discoveries only makes it harder to see how the discovering process can be done.


从70年人工智能研究中可以汲取的最大教训是：利用计算的通用方法最终总是最有效的，且优势巨大。其根本原因在于摩尔定律，或者说计算单位成本持续指数级下降的普遍规律。多数AI研究都假设agent的计算能力恒定（这种情况下利用人类知识就成了提升性能的唯一途径），但只需将时间线拉长到典型研究周期之外，海量计算资源终将唾手可得。研究者们为寻求短期突破，往往试图注入特定领域的人类知识，但长远来看真正重要的只有对计算资源的驾驭。这两种路径本不必相互冲突，但现实中往往背道而驰——投入其中一方的时间就无法用于另一方，研究者心理上也会对某种方法产生路径依赖。更关键的是，依赖人类知识的方法常会使系统复杂化，反而阻碍其发挥通用计算方法的优势。AI学界对此惨痛教训的领悟往往姗姗来迟，回顾几个典型案例极具启示意义。  

在国际象棋领域，1997年击败世界冠军卡斯帕罗夫的制胜法宝正是大规模深度搜索。当时主流计算机象棋研究者对此深感沮丧，他们长期致力于利用人类对棋局特殊结构的理解。当配备专用软硬件的简单搜索方法展现出碾压性优势时，这些依赖人类知识的学者难以坦然认输，辩称"暴力搜索"只是侥幸获胜，既非通用策略，更不符合人类下棋方式。他们渴望基于人类智慧的方法获胜，失败令其倍感失落。  

计算机围棋领域重现了相似的发展轨迹，只是迟到了二十年。初期研究大量投入在利用人类棋谱知识和围棋特性来规避搜索，但当大规模搜索配合自我对弈学习（这种价值函数学习方法在其他游戏乃至象棋中均有应用，尽管在1997年的冠军程序中未起主要作用）实现突破时，先前所有努力都被证明是徒劳甚至适得其反。自我对弈与广义的学习机制，本质上都是调动海量计算资源的途径。搜索与学习正是AI研究中驾驭巨量计算的两大核心技术。与象棋如出一辙，围棋研究者初期执着于人类经验（以减少搜索需求），直到后期全面拥抱搜索与学习才取得重大突破。  

语音识别领域早在1970年代DARPA举办的竞赛中就显现端倪。参赛者既有利用人类知识（词汇、音素、声道等）的专门方法，也有基于隐马尔可夫模型(HMM)的统计方法——后者计算量更大但最终胜出。这逐渐引领自然语言处理领域长达数十年的范式转变，统计与计算最终占据主导。近期深度学习在语音识别的崛起，正是这一趋势的最新体现。深度学习方法更彻底地摆脱对人类知识的依赖，通过超大规模训练集上的学习与更强算力，实现了质的飞跃。与棋类研究相似，开发者总试图模仿自身思维模式构建系统，但当摩尔定律带来充足算力且找到有效利用途径时，这些预设反而成为阻碍，造成科研资源的巨大浪费。  

计算机视觉领域同样经历了这种范式迁移。早期方法致力于边缘检测、广义柱体或SIFT特征提取，如今这些均被抛弃。现代深度学习神经网络仅依靠卷积与特定不变性概念，性能却远超从前。  

这一深刻教训至今未被充分吸收——我们仍在重复同类错误。要认清并抵制这种倾向，必须理解其诱惑所在：我们必须咽下这颗苦果，即长期来看，将人类思维模式植入系统终将失败。历史经验表明：1) AI研究者惯于将知识hard code进系统；2) 短期确实见效且令研究者欣慰；3) 长期却会遭遇瓶颈甚至阻碍发展；4) 突破性进展最终来自相反路径——通过搜索与学习实现计算规模扩张。这种成功往往伴随着苦涩，因其颠覆了以人类为中心的传统范式。  

惨痛教训首先揭示了通用方法的强大威力：那些能随算力增长持续扩展的方法。搜索与学习正是两类具备无限扩展潜力的方法。其次，我们必须承认心智内容的极端复杂性：空间、物体、多智能体、对称性等概念本质上都是外部世界任意复杂的组成部分，不应被hard code进系统。我们真正需要构建的是能发现并捕捉这种复杂性的元方法，其核心在于能自主寻找优质近似解——但探索过程应由方法本身完成，而非依赖人类预设。我们需要的是具备自主发现能力的AI智能体，而非装载人类已有知识的容器。预先植入我们的发现，只会遮蔽发现过程的本质规律。  

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
[prompt压缩(二)](https://www.linsight.cn/ea2871bf.html)  
[reasoning压缩(一)](https://www.linsight.cn/bfa4f144.html)  
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

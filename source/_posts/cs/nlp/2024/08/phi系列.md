---
title: phi系列模型
tags:
  - NLP
  - LLM
  - transformer
  - 微软
  - 端侧模型
categories:
  - CS
  - NLP
  - LLM
abbrlink: fe13b56f
date: 2024-08-13 20:41:06
---

【本文已在同名 微信公众号 / 知乎 / [个人博客linsight.cn](http://www.linsight.cn/) 上线】  

***  

最近在做端侧模型和数据合成的工作，微软的phi系列是受到关注比较多的一个小规模模型，整理一下细节，看看有什么可以借鉴使用的。  

# phi-1  

phi-1包括两个模型：350M参数的phi-1-small和1.3B参数的phi-1-base。  

相比其他SLM/LLM，phi-1的特点是所用数据极少，预训练只有7B：  
- 6B从web数据筛选而来  
- 1B使用GPT-3.5生成  

训练资源也只用到了A100*8，共训练了4天。  

由于模型规模较小，并且为了快速验证方法的有效性，phi-1把关注点放在模型的code能力上（python语言）。phi-1-base和其他通用LM/代码LM在参数量、训练数据量，以及在HumanEval和MBPP上的效果对比如下表：  

{% asset_img phi_1_result.png phi系列 %}  

## 模型  

phi-1关注点在数据上，因此模型结构上没有特别设计，使用的标准的decoder-only，phi-1两个模型的参数如下：  

|   | phi-1-base | phi-1-small |
| ---- | ---- | ---- |
| 参数量 | 1.3B | 350M |
| 层数 | 24 | 20 |
| hidden size | 2048 | 1024 |
| intermediate size | 8192 | 4096 |
| attention head num | 32 | 16 |

两个模型都使用了MHA，位置编码RoPE的dimension为32（每个注意头的大小为64，即只有一半的维度加上了位置编码的信息）。  

而tokenizer则是复用了codegen-350M-mono的。  

## 数据  

以往的scaling law关注在模型的参数量和训练的数据量。而phi-1的工作则主要关注在另一个维度：数据质量。  

参考《Tinystories: How small can language models be and still speak coherent english?》的工作，数据质量是能够极大地改变scaling law的形状的。  

如文章标题《Textbooks Are All You Need》，phi-1最主要的工作就是提高训练数据质量，达到“textbook”的水平。  

1、现有数据集的问题  

对于代码领域，公开的数据集比如The Stack、StackOverflow和CodeContest等，都存在指导性不足的问题，具体来说有以下问题：  
- not self-contained：这些代码片段或者文件依赖外部的库或者文件，因此无法单纯从当前的文件理解代码在干什么  
- not meaningful：一些代码片段可能主要内容是大量的定义、参数设置或者GUI配置相关的内容，而没有计算和逻辑推理内容，这样的内容显然并不有足够的教育意义，下图就是一个例子  
- 部分文件或者代码过于复杂，且缺乏注释或者文档说明，这也让模型的学习变得困难  
- topic分布不均匀  

{% asset_img phi_1_code_case.png phi系列 %}  

设想一下，如果让一个人类初学者用这些资料进行代码学习，效果一定也是很差的，因为里面包含了很多噪音、不完整的内容以及概念的模糊不清。  

从这个角度出发，打造textbook级别质量的数据就是很自然的想法。  

2、数据过滤  

首先就是从已有的数据中，过滤提取高质量数据。  

The Stack和StackOverflow两个数据集的python子集，在去重之后有35M个文件，约35B的数据。微软从中抽取了100k个文件，让GPT-4进行打分，“determine its educational value for a student whose goal is to learn basic coding concepts”。  

（可惜这里没有给出具体的prompt）  

由此获得100k的训练数据，之后用codegen模型的output embedding作为feature，训练random forest classifier，再回头对35M的数据进行打分、筛选。  

单独使用这些过滤后的数据进行训练，效果已经比不过滤的好（且训练的step数更少，96k vs 36k），如下图所示（橙色和浅蓝色对比）  

{% asset_img phi_1_compare.png phi系列 %}  

3、synthetic textbook dataset  

另一个获取高质量预训练数据的方法是数据合成。  

数据合成主要问题之一就是「多样性」：
- 训练数据的多样性很重要，因为多样性的内容可以让模型学到不同topic的内容、同一问题的不同表达和不同解法，以此提升模型泛化性，以及对未见过case的稳定性  
- 生成数据的多样性不容易获得，因为LLM训练的时候就是学习输出最大概率的内容，因此模型天然倾向于给出少量几个最可能的结果  

这里phi-1参考了《Tinystories》的做法，通过prompt给GPT-3.5的输出结果注入了一些随机性，限制了topic和（模型输出文本的）目标观众，获取了约1B token的数据。  

这些数据包含自然语言和相关的代码片段，下图是一个示例：  

{% asset_img phi_1_example_1.png phi系列 %}  

4、CodeExercises dataset  

数据合成也用于生成高质量的SFT数据。  

微软用GPT-3.5生成了约180M token的微调数据，要求模型根据自然语言指令写出代码。这里提升多样性的方法是限定了function name。下图是一个示例：  

{% asset_img phi_1_example_2.png phi系列 %}  

## 训练  

phi-1的预训练和微调都使用以下配置：  
- AdamW optimizer，weight decay = 0.1  
- linear-warmup-linear-decay learning rate schedule  
- dropout = 0.1  
- 窗口大小 = 2048  
- 数据精度 = fp16  

预训练共训练了36000 step，最终选择了24000 step的checkpoint，相当于7B的预训练数据共训了8个epoch，约50B；训练参数如下：  
- batch size = 1024  
- max lr = 1e-3  
- warmup step = 750  

微调共进行了6000个step，参数如下：  
- batch size = 256  
- max lr = 1e-4  
- warmup step = 50  

论文还指出，微调对模型代码能力的提升很大，只要体现在指令遵循能力，和使用外部代码库的能力上。  

## 小结  

- phi-1使用极小的数据量，和较小规模的模型，在代码能力验证了高质量数据的影响，可惜没有给出更具体的prompt等  
- 合成数据会是一条通往更强只能的重要道路，苹果和Meta都已经做了很多工作  
- 这样的方案是否能scaling up？数据有没有可能存在污染？后面工作继续探索这些问题  

# phi-1.5  

phi-1.5延续phi-1的思路，使用和phi-1完全一样的模型结构，把目标领域扩展到了代码 + common sense reasoning，探索“how small can a LLM be to achieve certain capabilities”这个问题的答案。  

{% asset_img phi_15_result.png phi系列 %}  

## 数据  

phi-1.5的预训练数据在phi-1预训练数据集（7B token）的基础上，加入了约20B的高质量合成数据，用于让模型学习common sense reasoning和general knowledge。  

这20B数据来自于精心挑选的20k topics，并通过在prompt中加入来自web dataset的sample提升模型生成数据的多样性。  

文中指出，数据的生成不仅需要算力，“It requires intricate iterations, strategic topic selection, and a deep understanding of knowledge gaps to ensure quality and diversity of the data.”  

## 训练  

phi-1.5的预训练设置：  
- 模型随机初始化  
- max lr = 2e-4  
- no warmup  
- AdamW，beta_1 = 0.9, beta_2 = 0.98, epsilon = 1e-7  
- DeepSpeed ZeRO stage 2  
- fp16数据格式  
- batch size = 2048  

共训练了150B（多个epoch），其中20%来自phi-1的数据集，80%来自新合成的数据。  

## filtered web data  

为了探索traditional web data的效果，研究人员还搞了phi-1.5-web-only模型（只使用web data训练），和phi-1.5-web模型（混合了phi-1数据、合成数据和web data，比例为2:4:4）。所用的filtered web data共有95B，其中88B来自Falcon refined web dataset，7B来自The Stack和StackOverflow。  

## 效果  

phi-1.5、phi-1.5-web和phi-1.5-web-only与其他模型，在几个common sense的benchmark的效果对比如下  

{% asset_img phi_15_bench_1.png phi系列 %}  

- phi-1.5-web-only已经比很多其他模型效果好，微软把这个提升归功于数据过滤  
- phi-1.5-web相比phi-1.5提升不大，说明合成数据已经够好  

在language understanding task上，phi-1.5的效果如下  

{% asset_img phi_15_bench_2.png phi系列 %}  

最后，通过数学能力和代码能力来评估模型的reasoning ability，结果如下：  

{% asset_img phi_15_bench_3.png phi系列 %}  

- phi-1.5在reasoning上相比其他模型优势很大  
- phi-1.5-web则在phi-1.5的基础上，有明显提升，说明web data对reasoning能力有帮助  
- phi-1.5的代码能力和phi-1差不多，这也说明高质量数据的训练更高效（加入更多非代码数据没有太多帮助）  

# phi-2  

phi-2（2.7B）是基于phi-1.5模型参数进行scale up的工作。  

## scale up  

以phi-1-small和phi-1-base为例，直接train from scratch，结果是这样的：  

{% asset_img phi_2_0.png phi系列 %}  

而另外一个做法，就是复用小模型训练好的参数，用于初始化更大的模型。大模型和小模型的数和hidden size不同，因此需要做一些处理。  

1、scaling number of layers  

参考《Scaling language models: Methods, analysis & insights from training gopher》，通过以下映射，把phi-1.5的每层的参数复制到更大的模型（20层-->24层）：  

round_int(range(num_layers_new)/num_layers_new * num_layers_old)  

2、Scaling attention layer dimensions  

大小模型的QKV投影矩阵维度不同，最简单的复用方法就是大的矩阵部分直接使用小矩阵参数，其余多出来的参数直接随机初始化（weight reuse，WR），如下图  

{% asset_img phi_2_1.png phi系列 %}  

更进一步，还可以使用tiling，把大矩阵多出来的维度用小矩阵的参数填满，如下图  

{% asset_img phi_2_2.png phi系列 %}  

直接训练大模型、WR和WR + tiling的效果如下  

{% asset_img phi_2_3.png phi系列 %}  

## phi-2  

用WR + tiling，从phi-1.5初始化phi-2（2.7B）的效果如下  

{% asset_img phi_2.png phi系列 %}  

# phi-3  

phi-3包括3个模型：  
- phi-3-mini，3.8B参数，适用于移动设备  
- phi-3-small，7B参数  
- phi-3-medium，14B参数  

## 模型  

- phi-3使用和Llama-2相似的模型结构  
- 不同规模的模型(mini、small & medium)词表大小不同  
- 通过LongRoPE的方法把窗口扩展到了128k  
- phi-3-small使用了GQA  

推理时，使用了blocksparse attention对KV cache进行压缩：每个头仅保留部分不同的KV block，这样在减少缓存用量的同时，可以保障模型的一定程度的正常推理和长文本能力，示意图如下：  

{% asset_img phi_3_sparse.png phi系列 %}  

## 数据 & 训练  

按照《Textbooks Are All You Need》的路径，phi-3使用了”heavily filtered publicly web data“进行训练，这些数据通过”educational level“进行清洗和分类。  

预训练包括两个phase：  
- phase1：大量的web sources，让模型学习通用知识 & 语言理解  
- phase2：使用更多heavily filtered webdata（phase-1数据的子集），以及一些可以提升模型reasoning能力和其他技能的合成数据  

phi-3-mini的训练数据总共有3.3T。  

和”compute optimal“相似，微软认为给定规模下的小模型存在一个”data optimal“的状态，即把数据调到最优状态。  

有些数据是不适合给小规模的模型训练的，但是可以给更大的模型使用。例如，某一天英超联赛中一场比赛的结果可能是大模型的良好训练数据，但对于小型模型，需要去除这类信息，以便为“推理”留出更多的模型容量。  

训练中发现在部分benchmark上，phi-3-medium（14B）和phi-3-small（7B）的差距远小于phi-3-small和phi-3-mini的差距，这可能说明目前的这份数据目前并不是phi-3-medium这个规模下的”data optimal“，而需要进一步的调试。  

## 效果  

在MMLU上，phi系列和Llama-2系列模型的对比如下  

{% asset_img phi_3_result.png phi系列 %}  

phi系列的效率看起来更高，处于”data optimal regime“。  

# 数据污染？  

phi系列，以及其他一些模型，在模型参数量较小/训练数据量较小的情况下获得了媲美模型规模数倍于这些模型的效果。其中是否存在过拟合的情况？  

《A Careful Examination of Large Language Model Performance on Grade School Arithmetic》针对这个问题做了实验。研究人员参考GSM8k，构造了GSM1k数据集。GSM1k数据在长度、难度、覆盖范围等方面都和GSM8k吻合。  

如果一个模型没有过拟合到GSM8k，那么用GSM1k进行测试，应该获得和GSM8k相近的结果，反之则在GSM1k上的效果会比较差。  

文章选取了Mixtral系列、phi系列、Llama系列等模型，测试结果如下：  

{% asset_img overfit.png phi系列 %}  

phi-3在GSM1k上的效果和GSM8k的gap排在前列，某种程度上说明phi-3是有过拟合到测试数据集上的。  

不过这也未必说明phi毫无可取之处，如果在过拟合的情况下，能够保证目标领域内的效果，那从业务上来说完全是可以接受的。在ChatGPT之前，模型训练基本就是在单个任务上做数据工程。我们进入到”通用智能“的时代才不到两年，大部分的业务逻辑和形态并没有转换过来。  

当然，通用的能力依然是我们追求的目标。只是目前来看，除了Claude、GPT-4和Llama，其他模型都还有一定差距。  

***  

读到这了，来一发点赞收藏关注吧~

博客：[http://www.linsight.cn/](http://www.linsight.cn/)  
知乎：[Linsight](https://www.zhihu.com/people/us4ever)  
微信公众号：Linsight  
![](/images/qrcode.jpg)  

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
[适合移动设备的语言模型--MobileLLM](https://www.linsight.cn/5ac36d34.html)  
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
- 数据：  
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

【1】Textbooks Are All You Need https://arxiv.org/abs/2306.11644  
【2】Textbooks Are All You Need II: phi-1.5 technical report https://arxiv.org/abs/2309.05463  
【3】Phi-2: The Surprising Power of Small Language Models https://nips.cc/media/neurips-2023/Slides/83968_5GxuY2z.pdf  
【4】Phi-3 Technical Report: A Highly Capable Language Model Locally on Your Phone https://arxiv.org/abs/2404.14219  
【5】A Careful Examination of Large Language Model Performance on Grade School Arithmetic https://arxiv.org/abs/2405.00332  

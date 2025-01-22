---
title: '多模态入门(二)--Flamingo,LLaVA系列和BLIP系列'
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
abbrlink: 569d722c
date: 2024-11-28 12:29:52
---

【本文已在同名 微信公众号 / 知乎 / [个人博客linsight.cn](http://www.linsight.cn/) 上线】  

***  

这篇主要包括BLIP系列、LLaVA系列和Flamingo，大致上按时间顺序排列。  

# BLIP  

论文：《BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation》  

时间：2022年1月  

机构：Salesforce  

在这个时间点，大多数多模态模型只在理解任务或者生成任务中的一个能做得比较好。BLIP的目的是训练一个同时具备理解和生成能力的模型，提出了一个unified VLP（Vision-Language Pre-Training）framework。BLIP主要的工作在于模型架构的设计，以及多模态数据集的优化。  

## MED模型  

BLIP提出了一个MED模型，MED = Multimodal mixture of Encoder-Decoder。MED模型的设计如下：  

{% asset_img blip_archi.png 多模态入门 %}  

可以看到MED包含好几个模块，这是一个multi-task模型。左边两个模块分别是image encoder和text encoder，和CLIP中使用的类似。这里使用的image encoder是在ImageNet预训练好的ViT，而text encoder是预训练好的Bert-base模型。右边两个是混合图文模态的encoder和decoder。  

MED有三种功能：  

- Unimodal encoder：分别对图像和文本进行编码  
- Image-grounded text encoder：用一个位于text-encoder的FFN层与注意力层之间的cross-attention模块，注入视觉信息  
- Image-grounded text decoder：和Image-grounded text encoder类似，只是decoder里的注意力层是causal attention而不是双向注意力  

预训练的时候，针对不同的功能，有不同的loss：  

- Image-Text Contrastive Loss (ITC)：和Unimodal encoder相关，类似CLIP，使用对比学习，目标是对齐vision encoder和text encoder；这里参考《Align before fuse: Vision and language representation learning with momentum distillation》，使用了ITC损失，其中应用了momentum encoder来获得feature  
- Image-Text Matching Loss (ITM)：ITM是一个二分类任务，任务是让Image-grounded text encoder判断图像和文本是不是匹配的  
- Language Modeling Loss (LM)：要求Image-grounded text decoder在给定图像特征的情况下，给出文字描述的预测  

在实际训练中，为了提升训练效率，text encoder和text decoder共享除了注意力层以外的参数。这里做了消融实验，共享参数比不共享参数效果更好，而且效率更高：  

{% asset_img blip_share_sa.png 多模态入门 %}  

## CapFilt (Captioning and Filtering)  

多模态的标注数据成本是比较高的，因此很多工作都会使用从网上自动获取的大规模图文数据对作为训练数据。但是这样又引入了一个问题，那就是网络图文数据本身是带有很多噪音的，比如解析的时候图文错配，或者图文数据本身无关等。  

BLIP用来提升图文数据质量的方法称为CapFilt，大致流程如下：  

{% asset_img blip_train_framework.png 多模态入门 %}  

CapFilt包含两个模块：captioner和filter。这两个模块都由同一个预训练好的MED模型初始化而来。  

captioner是image-grounded text decoder，用来给图片生成caption。而filter是image-grounded text encoder，基于ITC和ITM目标进行了学习，用来判别数据集中的图文是否匹配，不匹配的数据将会被删除。  

文中强调了captioner生成的时候多样性很重要，这点和其他数据合成方法一样。  

消融实验对比了captioner和filter共享参数与否的效果，结果上看不共享参数更好：  

{% asset_img blip_share_filter.png 多模态入门 %}  

# Flamingo  

论文：《Flamingo: a Visual Language Model for Few-Shot Learning》  

时间：2022年4月  

机构：DeepMind  

之前我们讲了CLIP。CLIP的训练方式结合了图文对，使得图像特征能够在一定程度上和文本特征对齐。不过CLIP也有限制，一方面在形式上使用的是固定的图文对，另一方面训练出来的模型主要是image encoder，应用场景主要是表征、分类、检索等，比较有限。  

而Flamingo的目的是能够处理sequences of arbitrarily interleaved visual and textual data，这样就能完成更多复杂场景下的任务了。  

Flamingo-80B处理的一些example如下：  

{% asset_img flam_example.png 多模态入门 %}  

这种few-shot / in-context learning的能力在NLP已经显现出重要性。  

LLaVA论文中提到，Flamingo的工作可以被称之为多模态领域的GPT-3 moment，可见这个工作的重要性。  

## architecture  

Flamingo整体的架构如下：  

{% asset_img flam_archi.png 多模态入门 %}  

Flamingo的输入是任意图文交织的序列，输出是text序列。  

1、Vision Encoder  

首先是模型中的Vision Encoder。使用的是预训练好的Normalizer-Free ResNet (NFNet)，具体使用的是F6 model；其中的模型参数都是冻结的，不会再训练。这里的Vision Encoder采用了CLIP的训练方式进行了预训练。  

对于视频数据，以1FPS进行采样并独立编码。  

2、Perceiver Resampler  

Perceiver Resampler是Flamingo结构的一个重要部件。Vision Encoder把像素数据变成feature，而Perceiver Resampler把不同size的feature map映射到固定的少量visual token（64个），从而降低vision-text cross-attention的计算量。  

那Perceiver Resampler具体是怎么做的？  

> Similar to《Perceiver: General perception with iterative attention》and《End-to-end object detection with transformers》, we learn a predefined number of latent input queries which are fed to a Transformer and cross-attend to the visual features.  

结构上，Perceiver Resampler如下图：  

{% asset_img flam_resampler.png 多模态入门 %}  

Vision Encoder获得的图像token序列，再加上时间（类似Bert中的位置编码，比如对于视频就是第1帧、第2帧...）embedding，得到一个变长的序列，然后加上固定个数的learned latent queries，一起放进模型中。模型中的每一层包含attention和FFN模块，attention中把图像输入特征作为K和V，而把learned latent queries作为Q进行计算。  

最后获取learned latent queries的输出表征，就作为这些输入图像的表示。  

3、GATED XATTN-DENSE layers  

下一步是利用固定长度的图像特征，让语言模型进行文本生成。把图像特征注入到语言模型就用到GATED XATTN-DENSE的结构：  

{% asset_img flam_xattn.png 多模态入门 %}  

如上图，这些层其实就是cross-attention加上一个FFN层，再加上一个门控。在最开始的时候，这些门控会让GATED XATTN-DENSE layers的输出为0，从而保持在初始化的时候语言模型的效果。  

GATED XATTN-DENSE layers插入到预训练好的语言模型的层中，用vision feature作为K和V，而用语言模型的输入作为Q。  

## 训练  

数据上，共使用3种数据训练Flamingo：  

- 专门收集MultiModal MassiveWeb (M3W) 数据集，这里面都是交错的图像和文本数据，来自于4300万个网页；在这些网页中，会随机采样256个token长度的数据，每条最多包含5个图片  
- 成对的图像/视频和文本数据，大约有1.8B对数据  

这里有个细节，训练的时候，一段文本在注意力上只能看到前一张图片，而看不到更早的图片，更早的图片信息都被mask掉了。  

{% asset_img flam_inter_data.png 多模态入门 %}  

论文对很多设计都做了消融实验，包括所用的数据、门控机制、attention机制、xattn的frequency等：  

{% asset_img flam_ablation.png 多模态入门 %}  

# BLIP-2  

论文：《BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models》  

时间：2023年1月  

机构：Salesforce  

目前多模态模型的训练需要同时用到语言和图像模型，这两部分参数的训练成本随着模型和数据规模的提升也越来越高。BLIP-2一个出发点就是用比较低的训练成本获得好的效果。这里只使用了Flamingo的1/54的训练参数，就获得了超过Flamingo的效果。  

一个方法就是使用已经训练好的单模态模型，为了避免灾难性遗忘，在训练过程中冻结单模态模型的参数，而把主要工作放在两个模态的对齐上。  

在模态对齐这个方向，已经有一些工作，比如Frozen（《Multimodal few-shot learning with frozen language models》）和Flamingo（《Flamingo: a visual language model for fewshot learning》）。这两个工作主要使用image-to-text generation loss来对齐，但是这并不足够。  

因此BLIP-2提出一个两阶段的方案，在第一个阶段从冻结的image encoder学习vision-language representation，在第二个阶段从冻结的LLM学习image-to-text generation：  

{% asset_img blip2_intro.png 多模态入门 %}  

BLIP-2的模型包含三个部分：  

- image encoder：预训练好的ViT  
- LLM：预训练好的语言模型  
- Q-Former：用于对齐图像和文本特征的模型  

## Querying Transformer (Q-Former)  

Q-Former的结构和一阶段的使用如下图：  

{% asset_img blip2_qformer.png 多模态入门 %}  

Q-Former包含两个部分，image transformer和text transformer，二者共享self-attention部分的参数。  

在image transformer中：  

- 使用了一组learnable query embeddings作为输入，这组query的长度固定为32个，这组query的参数可以认为属于模型参数；这组query的目的是学习抽取图像特征中最重要的部分  
- 冻结的image encoder抽取的图像特征，会在image transformer里和这组query通过cross-attention进行交互  
- 每两个层transformer层插入一个cross-attention  
- corss-attention的参数随机初始化，而其他参数则是使用Bert-base初始化的  

而text-transformer：  

- 根据训练任务，选择不同的attention mask以及训练目标  
- 即是encoder也是decoder  

## 训练  

1、阶段一  

阶段一是representation learning stage。这个阶段的目标是训练Q-Former，使得那组输入query可以提取能表达最多信息的visual representation。  

这里集成BLIP的做法，联合使用3个优化目标：  

- Image-Text Contrastive Learning (ITC) ：image transformer给出图像的representation，记为Z，text transformer则给出文本的representation，记为t；这里Z有多个，因此使用Z中和t相似度最大的作为图文对representation的相似度；为了避免信息泄露，这里使用unimodal self-attention mask（见Q-Former图）  
- Image-grounded Text Generation (ITG) ：以输入图像为条件，生成文本；对于LLM，使用[DEC] token替换了[CLS] token，以指示解码任务  
- Image-Text Matching (ITM)：二分类任务，这时使用bi-directional self-attention mask  

2、阶段二  

阶段二是generative pre-training stage。这一阶段的训练如下图：  

{% asset_img blip2_stage2.png 多模态入门 %}  

先把Q-Former的输出query embedding转化为和LLM相同大小的维度，然后LLM进行解码。  

这里尝试了decoder-based LLM和encoder-decoder-based LLM。对于纯decoder的模型，直接进行生成就行；而对于encoder-decoder模型，使用prefix language modeling loss，即把文本切成两段，前一段和visual representation拼在一起输入encoder，后一段就是生成的目标。  

## 使用  

BLIP-2首先可以做Zero-shot VQA，只需使用简单的prompt即可让LLM生成答案，比如“Question: {} Answer:”。  

另外论文中微调了BLIP-2来做image captioning（期间保持LLM的参数固定，训练Q-Former和image encoder），效果也比较好。  

而在图像检索任务上，因为不涉及LLM，可以直接使用第一阶段的模型进行微调。  

# LLaVA  

论文：《Visual Instruction Tuning》  

时间：2023年4月  

机构：微软  

LLaVA = Large Language and Vision Assistant。  

如论文标题，LLaVA做的事情主要就是多模态空间的指令微调。微调后的模型能够进行多模态问答：  

{% asset_img llava_example.png 多模态入门 %}  

## 数据  

要做微调，首先就要有数据。multimodal instruction-following data在这个时间点还是很缺的，受到NLP领域的启发，这里也使用ChatGPT/GPT-4，基于已有的图文数据来构建多模态指令数据。  

怎么做呢？对于图像Xv和它的caption Xc，很自然地想到可以创建一系列的问题Xq，目的是instruct the assistant to describe the image content。而GPT-4就可以用来生成这一系列的问题。  

那么，一个原有的图文数据对就可以这样来扩展成它的instruction-following version：  

```python
Human:XqXv<STOP>Assistant:Xc<STOP>  
```  

不过这种方式虽然简便，但是获得的数据在多样性和难度上都比较欠缺。为了缓解这些问题，需要用ChatGPT/GPT-4创建包含视觉内容的指令数据。但是ChatGPT只能接收文字信息，因此需要一个把图像信息装换成文字信息的工具，而object detection给出的bounding box就是这样一个工具。  

有了一张图片的caption，以及其中物体的bounding box信息之后，就可以要求ChatGPT生成数据，LLaVA设计了三类数据：  

- Conversation：提问题提问关于图片的问题，而assistant“看着”图片回答问题；问题包括图片中的物体类型、计数、动作、位置、位置关系等  
- Detailed description：人工构造了一系列prompt（下图），每次随机抽取一条生成包含详细图片信息的描述  
- Complex reasoning：前面两种都属于直接的视觉信息，而推理任务则包含一些遵循严谨和复杂逻辑推理的过程  

{% asset_img llava_instr_list.png 多模态入门 %}  

下图是数据合成的一个例子，上半部分是图片的caption以及物体bounding
box的信息，下半部分是根据三种任务合成的指令数据：  

{% asset_img llava_syn_example.png 多模态入门 %}  

基于COCO数据集，共获得了158k条多模态指令数据，其中对话数据58k，详细描述23k，复杂推理77k。  

## 模型  

LLaVA的多模态模型架构如下：  

{% asset_img llava_archi.png 多模态入门 %}  

主模型是一个LLM，这里选用的Vicuna。  

对于输入的图像X，会使用CLIP训练的ViT-L/14作为vision encoder获得图像的representation Z，之后通过一个投影矩阵W把Z转换到LLM空间。  

## 训练  

在训练的时候，会随机地把第一轮的输入中，图像和文本问题的顺序进行交换，以提升模型在多模态生成能力中的泛化性。  

LLaVA的多模态微调使用两阶段的训练。  

1、Pre-training for Feature Alignment  

使用CC3M的595k条数据。这一阶段的训练过程中，vision encoder和LLM的参数保持冻结不训练，只训练对齐的投影矩阵W。  

2、Fine-tuning End-to-End  

这一阶段，依然保持vision encoder的冻结，而训练W和LLM的参数。微调模型主要针对两个场景：  

- 多模态聊天机器人：使用前面从COCO中获得的158k数据  
- science QA：在ScienceQA上开发，这里把数据组织成一个单轮对话  

# InstructBLIP  

论文：《InstructBLIP: Towards General-purpose Vision-Language Models with Instruction Tuning》  

时间：2023年5月  

机构：Salesforce，HKUST，NTU  

在BLIP-2上进行instruction tuning，就获得InstructBLIP了。为什么要进行指令微调？因为指令微调之后模型在没见的任务上的泛化能力能够大大提升。  

论文中收集了11个任务，共26个数据集，训练的时候使用其中的13个数据集，保留另外13个作为评测数据。  

{% asset_img instructblip_data.png 多模态入门 %}  

在BLIP-2的基础上，InstructBLIP使用instruction-aware Q-former。instruction-aware Q-former将指令文本token作为额外的输入，以提取和任务更加相关的图像特征。  

{% asset_img instructblip_model.png 多模态入门 %}  

在指令微调的时候，由于每个数据集的大小不同，差异很大，直接混合随机采样有可能导致较小的数据集学习不足，而大数据集的任务则容易过拟合。因此文中提出根据数据集大小的平方根进行采样，再加上一些手动调整，防止过拟合。  

对于每个任务，设计了多个prompt：  

{% asset_img instructblip_template.png 多模态入门 %}  

这里有个细节：对于回复都比较短的数据集，在prompt里加入了“short”或者“briefly”这样的指令，“假装”回复短时应为prompt要求的，从而避免在使用模型的时候总是生成太短的结果。  

# LLaVA-1.5  

论文：《Improved Baselines with Visual Instruction Tuning》  

时间：2023年10月  

机构：微软  

LLaVA-1.5是LLaVA的升级版，包括数据和模型。  

{% asset_img llava1.5_intro.png 多模态入门 %}  

## 模型  

- 首先是LLM的升级，增加了Vicuna-13B的版本，获得更强的理解和生成能力。  

> Results on MM-Vet shows the most significant improvement when scaling the LLM to 13B, suggesting the importance of the base LLM’s capability for visual conversations.  

- 另外就是对齐文本和图像的模块由简单的dense层升级为两层的MLP层。  

- 此外就是把vision encoder从ViT-L/14-224px升级为ViT-L/14-336px。  

虽然使用更强的ViT模型可以提供更高的分辨率，获得更好的效果，但是这种替换模型的方法总有个头，因为预训练模型就那么大，从base到large，顶多就到xxl。以往的一些提升分辨率的做法是使用positional embedding interpolation，并且在微调的时候让模型适应到新的分辨率，这往往需要很大量的数据，效率不高。  

为了解决分辨率的问题，LLaVA-1.5的做法如下：  

{% asset_img llava1.5_hd.png 多模态入门 %}  

把大图像分割成适配vision encoder的大小（比如224/336）的块，然后独立对这些小块进行编码，获得各个块的representation，最后再合并成target resolution 的feature map。这种分割难免会引入一些artifact，因此在这之外还加了一路操作，对原图做resize，以提供全局的信息。  

这种做法让LLaVA-1.5理论上可以处理resolution的数据，同时也保证了数据效率（不用重训）。  

## 数据  

1、增加Academic task oriented data  

数据上，首先纳入了更多的数据集，包括VQA、OCR，region-level perception的数据：OKVQA、A-OKVQA、OCRVQA、TextCaps等。  

实验上，发现加入region-level perception数据能提升模型处理fine-grained视觉信息的能力。  

2、数据格式 & prompt  

LLaVA在回答短问题的时候，效果没那么好，因为训练的过程中缺乏相关的数据。而InstructBLIP引入了相关数据，但是却没法在长回答和短回答的VQA任务上很好地平和，即在需要长回答的情况下，可能也会给出短的答案，如下图：  

{% asset_img llava1.5_example.png 多模态入门 %}  

造成InstructBLIP这个问题的原因：（1）没有微调LLM（2）prompt不够明确。  

因此LLaVA-1.5优化了prompt的设计，使用一个统一的能够指示输出格式的prompt。当需要简短答案是，会在prompt加上“Answer the question using a single word or phrase.”，如上图。在这样的prompt下，  

## 实验  

各种改进的效果如下表，蓝色是数据的部分，粉色是模型部分，而黄色是分辨率的变化：  

{% asset_img llava1.5_scaling.png 多模态入门 %}  

而和其他模型在各个指标上的对比如下：  

{% asset_img llava1.5_compare.png 多模态入门 %}  

LLaVA-1.5整体的训练成本大约是20小时 × 8卡A100机器，相比LLM来说并不大。  

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

【1】BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation https://arxiv.org/abs/2201.12086  
【2】BLIP-2: Bootstrapping Language-Image Pre-training with Frozen Image Encoders and Large Language Models https://arxiv.org/abs/2301.12597  
【3】InstructBLIP: Towards General-purpose Vision-Language Models with Instruction Tuning https://arxiv.org/abs/2305.06500  
【4】https://zhuanlan.zhihu.com/p/685233706  
【5】Flamingo: a Visual Language Model for Few-Shot Learning https://arxiv.org/abs/2204.14198  
【6】Visual Instruction Tuning https://arxiv.org/abs/2304.08485  
【7】Improved Baselines with Visual Instruction Tuning https://arxiv.org/abs/2310.03744  
【8】https://zhuanlan.zhihu.com/p/721428034  

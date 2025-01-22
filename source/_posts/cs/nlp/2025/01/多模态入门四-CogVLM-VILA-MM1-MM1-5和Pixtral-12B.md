---
title: '多模态入门(四)--CogVLM,VILA,MM1,MM1.5和Pixtral-12B'
abbrlink: e00debee
date: 2025-01-15 22:01:04
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
---

【本文已在同名 微信公众号 / 知乎 / [个人博客linsight.cn](http://www.linsight.cn/) 上线】  

***  

这篇主要包括CogVLM，VILA，MM1，MM1.5和Pixtral-12B。  

# CogVLM  

论文：《CogVLM: Visual Expert for Pretrained Language Models》  

时间：2023年11月  

在这之前的VLM工作大致有两个做法：  

- 浅对齐 shallow alignment：即只训练对齐部分的参数，比如InstructBLIP和MiniGPT-4；这样带来的问题是，LLM中缺乏很多图像领域的概念，因此会出现很多无法对齐的情况，导致效果不好  
- 训练LLM：比如Qwen-VL，在预训练或者SFT阶段训练LLM，这样做的话多模态的效果明显会比浅对齐好些，但是带来的问题是在语言能力上却会受到较大的损害，如下图：  

{% asset_img cogvlm_loss_and_nlp.png 多模态入门 %}  

CogVLM想要做的是，既能更好对齐图文空间，又能不损害LLM的语言能力。  

## 模型结构  

为次CogVLM提出了这样的结构：  

{% asset_img cogvlm_archi.png 多模态入门 %}  

注意只有紫色部分的参数才会参与训练，其他参数是冻结的。  

包含4个部分：  

- ViT encoder：这个比较常规，使用EVA2-CLIP-E的ViT模型，去掉了最后一层，因为最后一层在原来的训练里是用于对比学习的  
- MLP adapter：这个也是常规做法，两层的MLP（SwiGLU）  
- LLM：使用Vicuna1.5-7B  
- Visual expert module：这个是CogVLM设计里的核心部分，在LLM中的每一层加入用于处理图像部分信息的attention和MLP层，参数大小和LLM中的对应模块一致，参数初始化也是从LLM中来的；visual expert module让模型在处理文本信息和图像信息的时候可以分开来做，处理完后再合并，这样既能保持LLM的语言能力，又能加强图文模态的交互  

这里一个细节是，图像token全部共享同一个位置编码。  

## 训练  

1、预训练  

使用的数据是LAION-2B和COYO-700M，在经过NSFW、政治敏感和图像比例等规则的清洗之后保留了1.5B条数据。CogVLM还另外搞了40M的图像grounding数据加入这一阶段。  

预训练分为两个阶段，第一阶段是caption的训练，而第二阶段是image captioning 和 Referring Expression Comprehension(REC)两个训练目标混合。  

REC是在给定对象的文本描述的情况下预测图像中边界框的任务，以 VQA 的形式进行训练，BP的时候只使用answer部分的loss。在最后的3w个step中，将输入分辨率从224×224提升为490×490。  

2、对齐  

对齐阶段分别训练CogVLM-Chat和CogVLM-Grounding两个模型。  

CogVLM-Chat注重通用性，整合多种任务的多个数据集。而CogVLM-Grounding则是转为模型grounding能力而训练，包含Grounded Captioning (GC)、Referring Expression Generation (REG)、Referring Expression Comprehension (REC)和Grounded Visual Question An-swering (GroundedVQA)四类任务。  

针对不同的改进，消融实验的结果如下：  

{% asset_img cogvlm_ablation.png 多模态入门 %}  

# VILA  

论文：《VILA: On Pre-training for Visual Language Models》  

时间：2023年12月  

机构：英伟达，MIT  

VILA = VIsual LAnguage  

VILA的模型架构比较常规：  

{% asset_img vila_archi.png 多模态入门 %}  

和CogVLM有些类似，VILA主要有几点发现：  

- LLM的训练对效果提升很重要，全程冻结不行  
- 交错的图文数据更好，单纯的图文对数据不够  
- 在instruction tuning的时候混合纯文本数据不仅能恢复LLM的语言能力，对VLM任务的效果也是有益的  

针对这几点做了实验。  

1、LLM训练  

（1）Fine-tuning vs. prompt tuning  

prompt tuning在训练的时候冻结LLM，可以防止LLM的语言能力退化，但是也阻碍了图文空间的深度对齐。对此做了一系列实验：  

{% asset_img vila_llm_freeze.png 多模态入门 %}  

结论是：  

- SFT时仅训练projector效果不好  
- 预训练时冻结LLM不会影响zero-shot能力，但会影响in-context learning的能力  
- 使用参数更少更简单的projector效果更好，猜测原因是这样会迫使LLM更多学习关于图像的知识  

（2）The deep embedding alignment hypothesis  

为什么fine-tune LLM效果好？一个假设是对齐图像和文本的分布很重要。为了验证这个猜想，计算了每层中图像和文本embedding的相似度：  

{% asset_img vila_align_simi.png 多模态入门 %}  

可以看到使用linear层 + LLM微调的相似度更高，特别是在更深的层中。  

2、图文交错的预训练  

找了两个数据集，一个是图文交错的数据集，一个是图文对数据集：  

{% asset_img vila_interleave_data.png 多模态入门 %}  

为了避免是因为文本分布带来的影响，还从MMC4数据集了只保留了图文对信息构造出了MMC4-pair数据集。这几个数据集在相同的预训练+SFT流程下，效果如下：  

{% asset_img vila_interleava_data_2.png 多模态入门 %}  

可以看到图文交错的属性确实带来了比较大的提升。  

3、Recover LLM Degradation with Joint SFT  

虽然图文交错数据比单纯图文对数据好，但是LLM的语言能力还是有5%左右的损失，为了挽回这部分损失，一个方式就是SFT时添加纯文本数据：  

{% asset_img vila_joint_train.png 多模态入门 %}  

使用了来自FLAN的1M文本数据后，不仅能保持LLM的语言能力不下降，在图文能力上也略有提升。  

# MM1  

论文：《MM1: Methods, Analysis & Insights from Multimodal LLM Pre-training》  

时间：2024年3月  

机构：苹果  

苹果先在较小的模型上进行了实验，得出了一些能够知道模型设计和数据选择的结论，并把这些结论应用到更大的（30B、64B的MoE模型）的MLLM，获得预训练指标SOTA的模型MM1：  

{% asset_img mm1_intro.png 多模态入门 %}  

{% asset_img mm1_intro2.png 多模态入门 %}  

## 消融实验  

为了构建强大的MLLM，需要在各个维度找到最佳的方案。训练中，三个最主要的维度分别是：  
- Architecture：包括不同的image encoder，以及把image encoder和LLM连接的方式  
- Data：不同的数据，和它们的权重  
- Training Procedure：包括多阶段的训练（多少个阶段，每个阶段干什么），以及超参  

预训练是比较消耗资源的，因此搜索所有可能的排列组合显然是不合理的，因此这里的消融实验使用更简单的设置：从base configuration出发，每次只改动一个维度。基础的配置如下：  
- Image Encoder: A ViT-L/14 model trained with a CLIP loss on DFN-5B and VeCap-300M; images of size 336×336.  
- Vision-Language Connector: C-Abstractor with 144 image tokens.  
- Pre-training Data: A mix of captioned images (45%), interleaved imagetext documents (45%), and text-only (10%) data.  
- Language Model: A 1.2B transformer decoder-only language model.  

模型和数据的消融设置如下：  

{% asset_img mm1_ablations.png 多模态入门 %}  

每次改动之后就需要评测改动的效果，这里使用的是一系列的zero-shot和few-shot(4-shot/8-shot)的captioning和VQA任务。  

### Architecture  

Architecture主要关注：  
- 怎么预训练image encoder  
- 如何将image encoder和LLM对齐: VL connector的设置  

1、关于image encoder  

关于image encoder的预训练，各种实验和结果如下：  

{% asset_img mm1_exp.png 多模态入门 %}  

基于这个表格，有一些观察发现：  
- 提升图像分辨率对效果提升显著，从224提升到336，各指标提升~3%  
- 提升模型大小通常也有效果提升，从ViT-L到ViT-H基本能带来1%的zero-shot提升  
- 而对比contrastive loss和reconstructive loss，差异则没那么显著；从表上看contrastive loss略好一些，但是由于数据不完全相同，所以可能有部分性能差异是来自数据的  

2、关于VL connector  

对比使用不同image size（224、336）和token数（64、144），以及VL connector类型（average pooling、attention pooling、convolutional mapping）下的效果：  

{% asset_img mm1_vl_connector.png 多模态入门 %}  

从结果上来看，图像size和token数影响比较大，而connector类型则相对影响较小。  

### Pre-training Data  

多模态预训练数据主要有2类：  
- captioning data：成对的图文数据，通常较短，数据中的图片和文本信息有比较高的关联性  
- 图文交错的web数据：通常较长，相比captioning数据，其中的图文信息相关性较低  

在上面两种主要数据之外，实验中还加入了纯文本数据，用于保留LLM的语言能力。实验所用的数据如下：  

{% asset_img mm1_ptm_data.png 多模态入门 %}  

因为要关注MLLM的语言能力，因此这个实验的评测里还加入了纯语言的任务，包括ARC/PIQA/LAMBADA等一些常用语言任务。

基于这些数据所做的实验和结果如下：  

{% asset_img mm1_ptmdata_ablation.png 多模态入门 %}  

- 从图（a）可以看到，captioning数据对zero-shot效果更重要，而图文交错数据对few-shot和纯语言能力更重要  
- 从图（b）可以看到，captioning数据结合纯文本数据有助于提升few-shot任务和文本任务的效果；而纯文本数据与图文交错数据结合虽会使得分略降，但能提升纯文本任务指标  
- 从图（c）可以看到，captioning / 图文交错数据 / 纯文本数据比例为 5:5:1 时（橙色柱），能在保持较强纯文本理解能力的同时，实现比较好的多模态效果  
- 从图（d）可以看到，合成captioning数据 VeCap 质量较高，虽占比较小（所有captioning数据的 7%），但在few-shot中可带来 2.4% 至 4% 的提升  

## 最终预训练  

基于上面的消融实验，选择了这样的设置进行最终训练：  
- Image Encoder: Motivated by the importance of image resolution, we use a ViT-H  model with 378×378 resolution, pre-trained with a CLIP objective on DFN-5B.  
- Vision-Language Connector: As the number of visual tokens is of highest importance, we use a VL connector with 144 tokens. The actual architecture seems to matter less, we opt for C-Abstractor.  
- Data: In order to maintain both zero- and few-shot performance, we use the following careful mix of 45% interleaved image-text documents, 45% imagetext pair documents, and 10% text-only documents.  

为了获得更好的效果，模型规模也提升了，扩展到了3B、7B和30B。用已有的LLM和image encoder初始化之后，进行200k步，约400B token数据的多模态预训练：  
- seq len = 4096  
- resolution = 378 * 378  
- batch size = 512  
- 所有参数unfrozen  

在百亿参数规模进行超参搜索成本太高，因此这里在小尺度上（9M、85M、302M和1.2B）进行了网格搜索。  

各个规模的MM1模型预训练评测：  

{% asset_img mm1_ptm_eval.png 多模态入门 %}  

## SFT  

更高的分辨率效果更好，为了支持更高的分辨率，SFT阶段使用了两种方法：  
- Positional embedding interpolation：ViT在SFT过程中适应新的分辨率，让模型支持448 * 448、560 * 560和672 * 672的分辨率。这种情况下每个图像所用的token数提升很多。  
- Sub-image decomposition：如下图（a），比如输入图像的大小为1344 * 1344，那么会把它切分成4张672 * 672的图像，再加上downsample到672 * 672的原始图像，这样处理这5张图像比一次处理几千个图像token的attention就要高效。  

{% asset_img mm1_sft.png 多模态入门 %}  

另外，从上图（b）可以看到，输入图像分辨率对效果的影响很大，1344是比较好的输入分辨率。  

上图（c）则表明，随着预训练数据的增加，SFT的效果也在提升，很符合直觉的结果。  

# MM1.5  

论文：《MM1.5: Methods, Analysis & Insights from Multimodal LLM Fine-tuning》  

时间：2024年9月  

机构：苹果  

MM1.5是MM1的升级版，做了一些更加细致的实验和改进，使得模型能处理的任务更丰富一些，效果也更好：  

{% asset_img mm1.5_intro.png 多模态入门 %}  

另外苹果还在通用MM1.5的基础上开发了专门针对视频的MM1.5-Video，以及专门针对mobile UI的MM1.5-UI。这MM1.5-UI应该就是为iPhone以后的AI系统所做的一次尝试。  

通用的MM1.5训练流程如下：  

{% asset_img mm1.5_train.png 多模态入门 %}  

主要的优化点集中在：  
- Continual Pre-training  
- SFT  
- Dynamic High-resolution  

## SFT  

从论文篇幅和实验数来看，MM1.5有相当部分的成本是用在优化SFT上的。  

首先，苹果收集了各种高质量的SFT数据，并把它们分成六类：  

{% asset_img mm1.5_sft_data.png 多模态入门 %}  

1、Impact of Different Data Categories  

为什么要分类，因为这些类型的数据在内容和形式上都有所不同，搞清楚各种数据之间的训练和效果是怎么影响的很重要。比如我们知道LLM的数学数据和代码数据能提升Agent能力，那么在面向Agent任务场景的开发中，就可以提升数学数据和代码数据的比例，以此获得更好的效果。  

SFT数据类型的实验和效果如下：  

{% asset_img mm1.5_sft_exp.png 多模态入门 %}  

从上图来看有几个结论：  
- 使用text-rich数据对text-rich和knowledge benchmark的分数有提升；数学数据也有类似的趋势，只是程度比较小  
- 使用science data对knowledge benchmark也有提升，另外也对text-rich任务有略微作用  
- 使用code data则只对text-rich任务有效  
- 添加refer&ground数据对referring和grounding数据有提升，但是对其他类型的数据则略有损害  

2、Data Mixture Ratio Study  

了解了各种类型的数据的大致影响之后，下一步就是要找出具体的最佳配比。  

首先是single-image的类型。这里用的指标是MMBase score，即general、text-rich和knowledge三者的平均得分。以general类的数据量为基准，其他数据和它的比例设为 $\alpha$；science、math、code以及refer&ground数据在不同 $\alpha$ 下的表现如下图：  

{% asset_img mm1.5_single_image_alpha.png 多模态入门 %}  

最终选择了图上红色x的比例作为最佳比例使用。  

获得了single-image数据内部的比例之后，下一步就是研究single-image、multi-image和text-only这几个数据大类之间的比例了。  

遍历三者的所有可能比例显然不可能，因此分开观察分别加入不同比例的text-only和multi-image数据对模型最终效果的影响，获得 $w_{text}$ 和 $w_{multi}$，最终计算 $1-w_{text}-w_{multi}$ 获得single-image数据的比例。  

实验的结果如下图：  

{% asset_img mm1.5_all_alpha.png 多模态入门 %}  

观察到几个现象：  
- text-only数据对MMBase分数的影响不大  
- 增加multi-image数据会提升图像处理多图数据的能力，但是也会损害MMBase的分数，即对模型的基本多模态能力有影响  

基于上面这些实验，文中给出三种mixture方案：  
- base mixture：包括general、text-rich、science（α=0.1），math（α=0.5）和code（α=0.2）数据  
- single-image mixture：在base的基础上，加上refer&ground数据（α=2）  
- all mixture：包括所有single-image（w=0.8）、multi-image（w=0.1）和text-only（w=0.1）数据  

三种mixture方案的效果如下：  

{% asset_img mm1.5_mixture_exp.png 多模态入门 %}  

## Continual Pre-training  

MM1.5比MM1多了一个高分辨率的继续预训练，这里的实验就研究了这个阶段一些设置的影响。  

1、分辨率的影响  

使用三种分辨率进行继续预训练，而保持其他所有设置相同，然后进行相同的SFT，效果随着分辨率提升而提升：  

{% asset_img mm1.5_resolution.png 多模态入门 %}  

值得注意的是，这里并不是直接观察预训练模型的效果，而是SFT之后再进行评测。个人觉得这样更能贴近真实效果，毕竟实验中经常发现有一些预训练阶段的优势或者劣势并不会直接带到SFT模型里。当然这样成本和对SFT的要求都更高了。  

2、OCR数据和合成caption数据的影响  

使用了两个合成的caption数据集LLaVA-Recap-3M和ShareGPT4V-PT，以及OCR数据进行继续预训练实验，结果如下：  

{% asset_img mm1.5_syndata.png 多模态入门 %}  

- 所有的继续预训练效果都比不继续预训练好  
- 加入合成数据并不能比简单的OCR数据带来更多的提升  

## Pre-training  

一些知识密集型的任务，比如MMMU，对模型的文本理解能力要求比较高。一般来说，LLM在进行多模态预训练之后，都会有一定程度的文本理解能力损失。因此苹果整合了一批高质量的文本数据HQ-text，包含高质量的常识、数学和代码数据，旨在提高模型在语言方面的推理能力。  

MM1中，caption、interleaved image-text和text-only数据的比例为45:45:10，把文本数据替换成HQ-text，并把文本数据的比例提升之后，能获得更好的效果：  

{% asset_img mm1.5_text_data.png 多模态入门 %}  

## Dynamic Image Splitting  

动态数据分割在很多工作里其实都有使用了。  

基本的考虑就是静态的图像分割固定把图片切分成4个子图，这样的效率并不太高：低分辨率的图片是可以不进行切分的，而比例比较异常的数据在切分之后可能会出现比较多的padding。  

动态图像分割则是设置了子图数量的最小n_min和n_max，并对候选图像考虑所有候选的网格，只要有一个网格可以覆盖图像，就把网格的长边resize和和图像match，再对需要的区域进行padding：  

{% asset_img mm1.5_dynamic_split.png 多模态入门 %}  

从实验上看，动态图像分割比静态的效果更好：  

{% asset_img mm1.5_ablation_1.png 多模态入门 %}  

使用图像分割还有一个问题：resize后的原始图像时放在子图前面还是后面，如果放在前面，那么处理子图细节的时候，模型可以参考全局信息，而如果放在子图后面，那么模型在处理全局信息的时候就可以参考子图细节。  

另外，子图的位置信息可以通过index或者prompt格式（sep）的方式输入给模型。  

实验上看，子图位置信息的输入方式影响不大（前三行），而全局图摆放的位置则是放在后面略好一点点：  

{% asset_img mm1.5_after.png 多模态入门 %}  

## Final Model and Training Recipe  

最终MM1.5的训练设置：
- 架构：采用与 MM1 相同的模型架构  
- 预训练：数据包括 2B 图像 - 文本对、600M 交错图像 - 文本文档（共 1B 图像）和 2T tokens 的纯文本数据。与 MM1 相比，除更新纯文本数据外，数据比例从 45:45:10 调整为 50:10:40，大幅降低交错数据比例（从 45% 到 10%），增加纯文本数据比例（从 10% 到 40%）  
- 继续预训练：使用 45M OCR 数据增强对文本丰富图像的理解，基于实验结果未包含额外合成图像字幕。
- 监督微调（SFT）：最终mixture包含 80% 单图像数据、10% 多图像数据和 10% 纯文本 SFT 数据。单图像数据可进一步分为 37.2% text-rich数据、22.5% refer&ground数据、11.3% 通用数据、5.6% 数学数据、2.3% 代码数据和 1.1% science数据  
- 动态高分辨率：设置n_min=4，n_max=9，只有输入的图像数量少于3张才会使用动态分割的策略，分辨率最大可以支持到4M pixel，相当于2016 * 2016的图像大小    

MM1.5和其他模型效果对比如下：  

{% asset_img mm1.5_eval.png 多模态入门 %}  

## MM1.5-UI  

大模型可以和使用者一样，看到并操作手机UI界面是一个能给开发者和使用者带来极大想象空间的能力：  

{% asset_img mm1.5_ui.png 多模态入门 %}  

苹果通过在MM1.5的通用版本上进行further finetune得到MM1.5-UI，使用的数据主要的Ferret-UI的SFT训练数据。  

评测任务除了Ferret-UI中的评测集外，还有一些public benchmark。评测结果如下：  

{% asset_img mm1.5_ui_eval.png 多模态入门 %}  

MM1.5-UI规模较小的版本就有不错的效果。上表最后两行是做的一个消融实验：为了验证MM1.5通用模型SFT对UI任务的影响，这里用3B预训练模型直接进行UI相关任务的SFT，效果确实比先进行通用SFT要差一些，说明MM1.5的通用SFT对UI任务是有正面影响的。  

# Pixtral 12B  

论文：《Pixtral 12B》  

时间：2024年10月  

机构：Mistral  

Pixtral使用了Mistral Nemo 12B作为LLM，并且专门从头训练了一个名为PixtralViT的vision encoder。  

{% asset_img pixtral_vit.png 多模态入门 %}  

PixtralViT的参数大小为400M，它相比其他ViT主要有几点改进：  
- Break tokens：在一些情况下，不同长宽比的图像可以会有相同数量的patch，为了帮助模型区分这种情况，在每行patch后面添加一个特殊的token [IMAGE BREAK]，而在最后一个patch后面添加[IMAGE END] token。可以认为Break token的本质是辅助模型定位patch位置的一个标识符。  
- Gating in FFN：论文中的说法是不在attention block里使用gating，而在hidden layer使用。  
- Sequence packing：图像的patch串起来，并用一个block-diagonal mask保证不同图像之间的patch不会有leakage。  
- RoPE-2D：原始的RoPE是给文本场景设计的，因此只有一个“距离”的概念，但是图像是二维的，单纯patch间的距离并不足以建模完整的patch位置信息，因此这里使用RoPE-2D。对于一个位置为i，j的图像patch x，它的旋转矩阵是M：

$$RoPE-2D \left(x^{(i, j)}, \Theta\right)=M_{\Theta}^{(i, j)} x^{(i, j)}$$  

$$M_{\Theta}^{(i, j)}=\left(\begin{array}{ccccccc}
\cos i \theta_{1} & -\sin i \theta_{1} & 0 & 0 & \cdots & 0 & 0 \\
\sin i \theta_{1} & \cos i \theta_{1} & 0 & 0 & \cdots & 0 & 0 \\
0 & 0 & \cos j \theta_{2} & -\sin j \theta_{2} & \cdots & 0 & 0 \\
0 & 0 & \sin j \theta_{2} & \cos j \theta_{2} & \cdots & 0 & 0 \\
\vdots & \vdots & \vdots & \vdots & \ddots & \vdots & \vdots \\
0 & 0 & 0 & 0 & \cdots & \cos j \theta_{\frac{d}{2}} & -\sin j \theta_{\frac{d}{2}} \\
0 & 0 & 0 & 0 & \cdots & \sin j \theta_{\frac{d}{2}} & \cos j \theta_{\frac{d}{2}}
\end{array}\right).$$  

和RoPE-1D的情况相比，其实就把index为奇数的小旋转矩阵用于捕捉patch的高度距离，而把index为偶数的小旋转矩阵用于捕捉patch的宽度距离。  

为了验证PixtralViT的有效性，把PixtralViT替换成其他ViT来观察效果的变化：  

{% asset_img pixtral_vit_abaltion.png 多模态入门 %}  

从实验结果上看，PixtralViT确实有一些优势。  

Pixtral 12B整体的架构如下：  

{% asset_img pixtral_archi.png 多模态入门 %}  

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

【1】MM1: Methods, Analysis & Insights from Multimodal LLM Pre-training https://arxiv.org/abs/2403.09611  
【2】MM1.5: Methods, Analysis & Insights from Multimodal LLM Fine-tuning https://arxiv.org/abs/2409.20566  
【3】VILA: On Pre-training for Visual Language Models https://arxiv.org/abs/2312.07533  
【4】Pixtral 12B https://arxiv.org/abs/2410.07073  
【5】CogVLM: Visual Expert for Pretrained Language Models https://arxiv.org/abs/2311.03079  

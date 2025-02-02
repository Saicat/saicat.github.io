---
title: '多模态入门(三)--MiniGPT4,DeepSeekVL,InternVL系列和QwenVL系列'
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
abbrlink: f16505b3
date: 2024-11-28 22:22:24
---

【本文已在同名 微信公众号 / 知乎 / [个人博客linsight.cn](http://www.linsight.cn/) 上线】  

***  

这篇主要包括Qwen-VL系列，MiniGPT-4，InternVL和DeepSeek-VL，大致上按时间顺序排列。  

# MiniGPT-4  

论文：《MiniGPT-4: Enhancing Vision-Language Understanding with Advanced Large Language Models》  

时间：2023年4月  

MiniGPT-4是对GPT-4的多模态能力的一次探索和尝试。  

先说结论：  

- 固定vision encoder和LLM，仅训练对齐部分，就能获得不错的效果  
- 仅使用image caption进行对齐并不能获得很好的效果，而用小部分的detailed image description pairs进行further finetuning就能打破这个限制  
- MiniGPT-4表现出了和GPT-4类似的能力，比如生成复杂的image descriptions，从手写文本生成网页，还有GPT-4没有的能力，比如从食物照片生成详细的烹饪食谱，根据图像编写故事或诗歌，为图像中的产品编写广告等  

## 模型结构  

MiniGPT-4的模型结构如下：  

{% asset_img minigpt4_intro.png 多模态入门 %}  

LLM使用的是Vicuna，而图像部分和BLIP-2相同，采用ViT + Q-Former的方式，对齐部分使用的是简单的linear layer。  

其中image encoder（和Q-Former）和LLM的参数都保持固定，只训练linear layer，因此训练很快。  

## 训练  

MiniGPT-4采用两阶段的训练方式。  

1、First pretraining stage  

在这一阶段，模型的目的是从大量对齐的image-text pair获取vision-language knowledge。使用的数据集有Conceptual Caption、SBU和LAION，batch size = 256，训了2w步左右。  

第一阶段完成后，MiniGPT-4有一定的图文响应能力，但是生成并不流畅。  

2、Second-stage finetuning  

为了优化效果，需要做第二阶段的对齐。为此MiniGPT-4专门构建了一个数据集。  

首先，让第一阶段获得的模型，基于下面的prompt生成图像描述：  

```python
###Human: <Img><ImageFeature></Img>Describe this image in detail. Give as many details as possible. Say everything you see. ###Assistant:
```  

生成的结果里包含了噪声或者不连贯的内容，因此使用ChatGPT来修复这些问题。

给ChatGPT的prompt如下：  

```python
Fix the error in the given paragraph. Remove any repeating sentences, meaningless characters, not English sentences, and so on. Remove unnecessary repetition. Rewrite any incomplete sentences.
Return directly the results without explanation. Return directly the input paragraph if it is already correct without explanation.
```  

获得结果之后，再利用一些人工制定的规则进行筛选，比如包含“I’m sorry I made a mistake…”, 或者 “I apologize for that …”的回复。  

最后得到了3500条高质量的数据，用于第二阶段的训练。  

第二阶段训练使用的模板如下：  

```python
###Human: <Img><ImageFeature></Img><Instruction>###Assistant:
```  

其中Instruction是从预定义的指令集中随机采样的指令。  

一个细节是，这个prompt的损失不算入训练中。  

第二阶段由于数据非常少，使用单个A100只要7分钟就训练完了。  

下面是最终模型的生成样本：  

{% asset_img minigpt4_example.png 多模态入门 %}  

# Qwen-VL  

论文：《Qwen-VL: A Versatile Vision-Language Model for Understanding, Localization, Text Reading, and Beyond》  

时间：2023年8月  

机构：阿里  

Qwen-VL有几个特点：  

- 支持随意交错的图文输入  
- 支持多语言，多种图文任务  
- 支持图片细粒度的理解

{% asset_img qwenvl_intro1.png 多模态入门 %}  

{% asset_img qwenvl_intro2.png 多模态入门 %}  

## 模型结构  

Qwen-VL的模型包含3个部分：  

- LLM：用Qwen-7B模型初始化的  
- Vision Encoder：使用来自Openclip的ViT-bigG（patch size = 14）初始化；训练和推理的时候图像会resize为固定大小  
- Position-aware Vision-Language Adapter：adapter一方面用于把图像representation压缩到固定的大小（和Flamingo中的类似），一方面用于对齐图文特征；adapter是一个随机初始化的单层cross-attention；  

3个部分的大小：  

{% asset_img qwenvl_model.png 多模态入门 %}  

adapter要使用多长的token压缩图像特征是个超参，文中实验了不同的设置（64/144/256/400），结果如下图：  

{% asset_img qwenvl_feature_len.png 多模态入门 %}  

## 输入输出  

为了区分图像和文本输入，会使用两个特殊的token，\<img\> 和 \</img\>， 放在图像特征序列的开头结尾。  

为了增强模型 fine-grained visual understanding 和 grounding的能力，训练数据中还会涉及region description和detection等，这些任务需要模型准确理解并生成对应的描述。那么对于输入中的bounding box（直接以文本的形式输入），会添加下面两个特殊符号：\<box\> 和 \</box\>。  

另外，还会把bounding box对应的文本段也用特殊token框起来  

> Additionally, to appropriately associate bounding boxes with their corresponding descriptive words or sentences, another set of special tokens (\<ref\> and \</ref\>) is introduced, marking the content referred to by the bounding box.  

## 训练  

Qwen-VL的训练分为三个阶段：两个预训练阶段和指令微调：  

{% asset_img qwenvl_train.png 多模态入门 %}  

1、stage1：预训练  

这一阶段使用从网上爬取的大规模图文对数据。原始数据包含5B条图文对，在经过清理后保留了其中的1.4B，其中77%是英文数据，23%是中文数据。具体的来源和比例如下：  

{% asset_img qwenvl_pt_data.png 多模态入门 %}  

- 这一阶段的训练冻结了LLM的参数，训练vision encoder和VL adapter  
- 输入图像都被resize到224×224  
- 训练的batch size = 30720，step = 50000  

2、stage2：多任务预训练  

这一阶段引入了高质量和细粒度的 fine-grained VL annotation data，包含交错的图文数据，使用的分辨率也更大（224×224 -> 448×448）。所用的任务数据如下表：  

{% asset_img qwenvl_mtpt_data.png 多模态入门 %}  

这一阶段模型所有参数都参与训练。  

3、stage3：SFT  

这一阶段进行微调获得Qwen-VL-Chat。数据上，除了使用来自LLM self-instruction的数据，还通过人工标注、模型生成获取一批额外的数据，以提升模型 localization 和 multi-image comprehension的能力。  

训练的时候，除了多模态数据，还混合一些纯文本的对话数据，以确保模型对话能力的泛化性能。  

最后总共使用了350k的数据。这一阶段vision encoder的参数不参与训练，只训练LLM和VL adapter。  

# InternVL  

论文：《InternVL: Scaling up Vision Foundation Models and Aligning for Generic Visual-Linguistic Tasks》  

时间：2023年12月  

InternVL认为之前的VLM有几个问题：  

- LLM和vision encoder在规模上有比较大的差异，vision encoder比较小，可能对LLM的容量使用不足  
- 用于对齐部分也比较小，且一般是随机初始化，因此对齐效果不够好  

针对这些问题，InternVL给出改进的版本：  

- 增大vision encoder  
- 增大对齐层，并用预训练模型初始化  
- progressive image-text alignment，融合对比学习和生成学习  

{% asset_img internvl_model.png 多模态入门 %}  

## 模型  

InternVL模型架构上的主要工作在Vision Encoder和用于对齐的Language Middleware。  

1、Vision Encoder  

InternVL通过超参搜索的实验，设计出了InternViT-6B。这个模型在100M的LAION-en上训练。  

{% asset_img internvl_vit.png 多模态入门 %}  

2、Language Middleware  

之前用于对齐图文特征的模型有Q-Former，或者简单的MLP层。这里使用基于LLaMA的模型，QLLaMA：在预训练好的LLaMA基础上，增加了96个随机初始化的query（用于压缩图像representation），以及一个1B参数的随机初始化的cross-attention层。  

这样的中间件由于规模足够大，在对齐的时候效果更好，而且具有一定的zero-shot能力。  

## 训练  

InternVL的对齐训练包括三个阶段：  

{% asset_img internvl_train.png 多模态入门 %}  

1、stage1：Vision-Language Contrastive Training  

这一阶段把InternViT-6B和LLaMA-7B进行对齐，这一阶段所有参数都是可训练的。  

所用的数据包括LAION-en、LAIONmulti、LAION-COCO、COYO、Wukong等，总共有6B个图文对数据，清洗后留下5B左右。  

2、stage2：Vision-Language Generative Training  

QLLaMA集成第一阶段训练得到的LLaMA-7B参数，保持InternViT-6B和QLLaMA的参数冻结，使用进一步过滤后剩下的1B高质量图文数据进行训练。  

这一阶段的损失函数由三部分组成：  

- image-text contrastive (ITC) loss  
- mage-text matching (ITM) loss  
- mage-grounded text generation (ITG) loss  

stage1和stage2所用数据如下：  

{% asset_img internvl_train_data_12.png 多模态入门 %}  

3、stage3：SFT  

这一阶段把vision encoder和QLLaMA和现成的LLM，Vicuna用MLP连接在一起。训练的时候使用了4M条数据：  

{% asset_img internvl_train_data_3.png 多模态入门 %}  

对于数据中不是对话的部分，参考LLaVA-1.5的方式进行转换。  

训练的时候可以仅训练MLP层，由于LLM和QLLaMA的feature space比较相似，因此这里即使冻结LLM，效果也很好。  

## 使用  

通过灵活使用InternVL的不同组件，可以完成各种任务：  

{% asset_img internvl_use.png 多模态入门 %}  

- InternViT-6B可以作为视觉任务的backbone，进行图像分类或者dense prediction task  
- 对于contrastive task，有两种推理模式，InternVL-C和InternVL-G  
- 对于生成任务，QLLaMA天然具有比较强的caption能力  
- 而对于多模态对话，也有InternVL-Chat(w/o QLLaMA)和InternVL-Chat(w/ QLLaMA)两种模式  

# DeepSeek-VL  

论文：《DeepSeek-VL: Towards Real-World Vision-Language Understanding》  

时间：2024年3月  

DeepSeek-VL认为这之前的VLM普遍有几个问题：  

- 将比较多的资源放在SFT阶段，不够重视通用知识学习的预训练阶段  
- 微调的时候用了精挑细选的数据集，在评测上效果不错，但是和实际体验不符  
- 使用预训练的ViT模型，分辨率不足  
- 在图文训练后，模型的语言能力下降  

## 数据  

数据依然是重中之重，DeepSeek-VL做了比较精细的工作。预训练和微调阶段的数据详细分布如下：  

{% asset_img dsvl_ptdata.png 多模态入门 %}  

{% asset_img dsvl_sftdata.png 多模态入门 %}  

首先可以看到有一个特点，那就是text-only的数据集比重相对比较高，就是为了保持VLM的语言能力而增加的文本数据。  

另外就是任务和类型的多样性，有交错的图文数据，图表数据，OCR数据等。其中Web Code是DeepSeek-VL专门构建的，为了让模型能从图片构建代码，利用Websight和MATCHA等工具获取了1.1M的数据。  

SFT数据的三级分类体系如下：  

{% asset_img dsvl_class.png 多模态入门 %}  

SFT数据分布上，多样化的任务使得模型既能获得较好的评测效果，在使用时又能有较好的体验。  

## 模型  

模型包括三个部分。  

1、Hybrid Vision Encoder  

vision encoder使用SigLIP-L，但是仅使用SigLIP存在一些问题。SigLIP和CLIP这样的训练方式主要是为图像语义服务的，会把不同的图像encoder成相近的内容，原因是“CLIP-blind pairs”（《Eyes wide shut? exploring the visual shortcomings of multimodal llms》）。再加上这些模型的输入分辨率都有限（大部分224×224、336×336等），这些vision encoder就难以捕捉细粒度的low-level的特征，干不了比较精细的活。  

因此有些工作开始使用 additional vision-only self-supervised encoders，增强视觉编码的能力。受这个做法启发，DeepSeek-VL也使用额外的SAM-B模型来处理low-level的视觉信息，输入的分辨率为1024×1024。接收384×384输入的SigLIP模型，和接收1024×1024输入的SAM-B就构成了hybrid vision encoder，能同时保留语义信息和细节特征。  

2、Vision-Language Adaptor  

使用two-layer hybrid MLP连接vision encoder和LLM。高分辨率的特征和低分辨率的特征分别由单层的MLP处理，之后concat在一起。  

3、Language Model  

使用DeepSeek LLM。  

## 训练  

DeepSeek-VL的训练分为三个阶段：  

{% asset_img dsvl_train.png 多模态入门 %}  

Stage 1: Training Vision-Language Adaptor  

这一阶段vision encoder和LLM保持冻结，训练MLP层。数据包括从ShareGPT4V获取的1.25M数据，和2.5M文档OCR数据。  

一个问题是，在只训练MLP的情况下，是否存在scaling law？即增大训练数据量有没有收益？这里做了实验，结果表明增大训练量没有明显收益，甚至效果会变差：  

{% asset_img dsvl_ptdata_scale.png 多模态入门 %}  

Stage 2: Joint Vision-Language pretraining  

这一阶段只冻结vision encoder。

这里DeepSeek-VL验证了不同多模态和纯文本数据下模型效果的变化：  

{% asset_img dsvl_ratio.png 多模态入门 %}  

在只使用多模态数据的情况下，模型的语言性能严重下降，这是因为多模态数据中的语料比较简单，复杂性不足，并且多模态能力和语言能力之间也存在竞争关系，导致语言能力出现灾难性遗忘。  

上面的实验还能看到，增加文本数据后，多模态能力不会出现明显的损失。结合一系列观察，选择了7（语言）:3（多模态）的数据比例。  

这一阶段的训练成本比第一阶段高很多，不过好消息是在小模型（1.3B）上的实验结论很大部分都能迁移到更大的模型上（7B）。  

Stage 3: Supervised Fine-tuning  

这一阶段，由于内存有限，SAM-B保持了冻结。  

各个阶段训练的参数设置如下：  

{% asset_img dsvl_hp.png 多模态入门 %}  

# InternVL 1.5  

论文：《How Far Are We to GPT-4V? Closing the Gap to Commercial Multimodal Models with Open-Source Suites》  

时间：2024年4月  

InternVL 1.5一个特点是最大能够支持到4k分辨率的输入。  

{% asset_img internvl1.5_intro.png 多模态入门 %}  

## architecture  

InternVL 1.5整体的架构如下，和之前的很多工作一样，采用的是ViT-MLP-LLM配置，使用随机初始化的MLP层将预训练过的InternViT-6B和预训练过的InternLM2-20B连接在一起：  

{% asset_img internvl1.5_archi.png 多模态入门 %}  

首先，InternVL 1.5在InternVL基础上升级了InternViT-6B。  

（1）InternViT-6B-448px-V1.2  

首先，研究人员发现倒数第四层的feature在多模态任务上效果最好，因此直接丢弃掉最后三层，InternViT-6B从48层降为45层。然后把InternViT-6B的分辨率提升到448×448，并把它和Nous-Hermes-2-Yi-34B通过MLP连接，使用image captioning和OCR数据进行训练，获得InternViT-6B-448px-V1.2。  

（2）InternViT-6B-448px-V1.5  

在InternViT-6B-448px-V1.2基础上，把分辨率从固定的448×448，扩展为动态的448×448。怎么扩展的呢？这里面有一个dynamic resolution的策略，在训练的时候，会根据输入图像大小切分成1~12个448×448的块，而在推理的时候则把块的范围扩大到1~40个，这样使得模型最大可以处理4k分辨率的输入。  

对不同图像的大小，会从一组（35个）预定义的长宽比中找到最佳匹配来切分，使得图像不会被过分变形，比如一个800×1300的图像将被调整为896×1344。然后，调整后的图像被分成448×448像素的块：  

{% asset_img internvl1.5_reso.png 多模态入门 %}  

不过这样一来输入token的数量可能就有点多，因此会使用一个叫pixel shuffle的操作来降低token的数量。pixel shuffle是图像超分里常用的操作，用于提高分辨率，不过这里改了factor，相当于down-sample了。  

除了图像本身切分出来的块，还会增加一个缩略图放在最后面，以提供全局信息。  

## 数据  

InternVL 1.5在预训练和微调阶段所用数据如下：  

{% asset_img internvl1.5_data.png 多模态入门 %}  

在微调阶段也加入了纯文本数据。  

## 训练  

在预训练阶段，InternViT-6B和MLP进行优化，而LLM冻结；在微调阶段则所有参数参与训练。  

# Qwen2-VL  

论文：《Qwen2-VL: Enhancing Vision-Language Model’s Perception of theWorld at Any Resolution》  

时间：2024年9月  

机构：阿里  

Qwen2-VL的内容还是比较多的，这里先简单看下关键点。Qwen2-VL有2B、8B、72B三种规模，支持动态分辨率，支持视频（20分钟以上）图文模态以及多语言的多种任务。  

{% asset_img qwen2vl_ability.png 多模态入门 %}  

## 模型  

三个规模的模型的描述如下：  

{% asset_img qwen2vl_model.png 多模态入门 %}  

可以看到各种规模的LLM下，都是用相同大小的vision encoder，这样保证ViT的computation load能够恒定。（估计是因为处理长视频的情况会对ViT的速度造成挑战）  

Qwen2-VL具体的模型架构和第一代Qwen-VL保持一致：  

{% asset_img qwen2vl_archi.png 多模态入门 %}  

在这个大框架基础下，有几个关键改进点。  

1、Naive Dynamic Resolution  

采用Navit（《Patch n’pack: Navit, a vision transformer for any aspect ratio and resolution》）动态分辨率的做法，来支持各种分辨率的输入。为了支持动态分辨率，这里把ViT的绝对位置编码换成了苏神在https://www.spaces.ac.cn/archives/8397提出的2D-RoPE。  

为了减少图像的token，在ViT后会用一个MLP把相邻的2×2的token压缩成单个token，并在前后加上两个特殊token： <|vision_start|> 和 <|vision_end|> 。  

2、Multimodal Rotary Position Embedding (M-RoPE)  

M-RoPE把原始的rotaty embedding拆解成三个分量：temporal、height和width。处理图像时，每个token的时间embedding保持不变；而对于视频每帧的temporal id则会递增。  

{% asset_img qwen2vl_mrope.png 多模态入门 %}  

3、Unified Image and Video Understanding  

Qwen2-VL对视频输入以每秒两帧的频率进行采样，并使用3D卷积来处理视频输入。为了在处理视频和图像时保持一致，图像在输入时会被视为两个一样的视频帧。  

为了保持效率，会动态调整视频帧的分辨率，使得每个输入视频的token数限制在16384以内。  

## 训练  

Qwen2-VL的训练和一代Qwen-VL保持一致：  

- 第一阶段训练ViT  
- 第二阶段全部训练  
- 第三阶段冻结ViT  

整个训练过程包含1.4T token，训练过程中只BP文本的loss。  

数据上，会使用<|vision_start|>和<|vision_end|>标识视觉内容，帮助模型区分视觉信息和文本信息。  

对话数据使用ChatML的格式：  

{% asset_img qwen2vl_chatml.png 多模态入门 %}  

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

【1】MiniGPT-4: Enhancing Vision-Language Understanding with Advanced Large Language Models https://arxiv.org/abs/2304.10592  
【2】Qwen-VL: A Versatile Vision-Language Model for Understanding, Localization, Text Reading, and Beyond https://arxiv.org/abs/2308.12966  
【3】Qwen2-VL: Enhancing Vision-Language Model’s Perception
of theWorld at Any Resolution https://arxiv.org/abs/2409.12191  
【4】InternVL: Scaling up Vision Foundation Models and Aligning
for Generic Visual-Linguistic Tasks https://arxiv.org/abs/2312.14238  
【5】DeepSeek-VL: Towards Real-World Vision-Language Understanding https://arxiv.org/abs/2403.05525  
【6】Patch n' Pack: NaViT, a Vision Transformer for any Aspect Ratio and Resolution https://arxiv.org/abs/2307.06304  
【7】How Far Are We to GPT-4V? Closing the Gap to Commercial Multimodal Models with Open-Source Suites https://arxiv.org/abs/2404.16821  
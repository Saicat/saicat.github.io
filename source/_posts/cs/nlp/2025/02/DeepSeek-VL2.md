---
title: DeepSeek-VL2
tags:
  - 多模态
  - CV
  - NLP
  - transformer
  - 预训练
  - SFT
  - DeepSeek
categories:
  - CS
  - 多模态
abbrlink: b4d047c1
date: 2025-02-25 21:57:29
---

【本文已在同名 微信公众号 / 知乎 / [个人博客linsight.cn](http://www.linsight.cn/) 上线】  

***  

看DeepSeek-VL2细节之前，先简单介绍下DeepSeek-VL2提到的recaption方案和visual prompt数据。  

# recaption：PixelProse  

## why recaption  

PixelProse是《From Pixels to Prose: A Large Dataset of Dense Image Captions》中提供的一个（合成）caption数据集，共有16M条样本；论文同时也介绍了他们构造PixelProse的recaption方案。  

为什么要做recaption？因为caption数据由来已久，业界有许多开源的caption数据集，这些数据集的收集、处理方式各不相同，数据内容和质量参差不齐。直接用这些数据训练会带入很多我们不想要的噪声，效果也不太好。  

通过具体case的分析，主要有这么些问题：  

- 数据里存在一些NSFW和Child Sexual Abuse Material (CSAM)的内容，这在很多场景都不合适甚至不合法  
- 很多样本的图片和caption关联性太差，比如过于简短，或者缺乏准确的描述，这导致VL模型没法很好学习语言和图像之间的细节对齐关系  
- 文字是VL模型和SD模型要学习的一个重点，但是现有的caption数据很多都没有给出图中文字的详细内容，使得模型很难学习文字  

## 方案  

针对分析出来的这些问题，合成高质量的caption数据的流程设计成这样：  

{% asset_img prose_pipeline.png DeepSeek-VL2 %}  

一步一步来看。  

1、过滤  

source data有三个，CommonPool、CC12M 和 RedCaps。当然如果现在我们要再多，那可以多加点数据集进去。首先，这些数据集通过NSFW分类模型和commercial Google APIs进行内容过滤，仅保留图片内容合适合法的数据。  

2、选择prompt  

接下来，会从下面5个预定义的prompt中随机选择一个，用于让Gemini生成新的prompt。  

{% asset_img prose_prompt.png DeepSeek-VL2 %}  

3、加入alt-text  

在生成的时候，会随机加入图片的alt-text到prompt中。参考《CapsFusion: Rethinking Image-Text Data at Scale》的发现，加入alt-text有机会提升生成结果细节的准确性。  

4、加入Negative Descriptions  

无论是VLM还是diffusion模型，对于negative描述的指令遵循能力总是比较差。比如跟diffusion模型说“画一幅没有大象的画”，最终画出来的画就有大象。  

为了增强模型对negative instruction的遵循能力，随机让Gemini增加一些途中不存在的物体的描述。比如“途中有5个红色的苹果”，就会再加上negative description “但是没有任何梨子出现在途中”。  

5、优化Text Recognition  

文字能力是VLM和diffusion模型都很重要的一个能力，如果没有文字识别能力，多模态模型无法识别图片上的路标、广告牌、标签等信息，而diffusion模型在生成包含文字的图像时也会是一团乱码。  

为了增强模型的文字理解能力，可以看到前面的5个prompt里都包含一个要求：

> If any text is present in the image, mention where it is, and the font.Describe the text in detail with quotation marks.  

不过生成caption的模型识别文字的准确率不是100%，甚至可能只有七八成的准确率，所以后面还要check一下。  

在校验之前，先用watermark model分类一下，对于不包含watermark，且出现文字的图片，再用OCR模型进行识别。小于15个pixel的text region会被抛弃。  

最终check的结果表明大概有76%的文字可以被caption模型正确识别：  

{% asset_img prose_ocr_acc.png DeepSeek-VL2 %}  

{% asset_img prose_ocr_case.png DeepSeek-VL2 %}  

当然OCR模型本身也不是100%正确的，对于样式复杂的情况，OCR模型也识别不准，不过整体上这个准确率校验还是可以参考的。  

## PixelProse  

新合成的PixelProse文本长度相比原caption更长，包含更多细节信息：  

{% asset_img prose_length.png DeepSeek-VL2 %}  

从文本的分词结果上看，PixelProse所包含的名词多样性也更丰富：  

{% asset_img prose_noun.png DeepSeek-VL2 %}  

# Visual Prompt  

这一part主要是讲一下visual prompt。  

在纯文本的场景，prompt的使用大家都很熟悉的。而在多模态场景，一般来说用户指令也是以文本的形式给出，比如“图上这个人多少岁了”，“这只狗是什么品种”这样。  

假设现在有一张图，上面有很多人，你想要针对其中某个人对模型进行询问。如果用文本进行描述的话，就有点困难：这些人可能没有很整齐地排列，衣着也没有鲜明特点；哪怕能够通过位置或者特征进行文字描述，这也会给模型的理解和识别造成困难。  

回想一下，如果是在和人交流，那么要准确定位图上的一个人，最简单的方法就是用手指一下，或者拿笔在对应位置画个圈/箭头。那跟模型交流的时候也可以这么干：  

{% asset_img vp_example.png DeepSeek-VL2 %}  

这个圈/箭头就是visual prompt。  

如果模型具备和这些圈/箭头进行交互的能力，那么用户在交互的时候就会更加自然。  

## 数据  

要训练这样的能力，首先就要有数据。《ViP-LLaVA: Making Large Multimodal Models Understand Arbitrary Visual Prompts》就搞了一批数据。  

1、source data  

visual prompt的数据还是通过数据合成获得。源数据就是现有各种物体识别/实体分割的数据，这些数据包含物体的位置和类型/名称信息，很方便改造成visual prompt数据。  

2、visual prompt type  

研究人员定义了一下巴中visual prompt类型，用于标识图像中的物体，总共有8种：  

{% asset_img vp_prompt.png DeepSeek-VL2 %}  

个人认为，这8种其实可以分成3个大类：  

（1）外框  

椭圆、长方形、三角形、物体的mask都属于把物体框起来的方式，只是有的框比较粗糙，有的比较精细。  

在构造这类visual prompt的时候，为了引入一定的随机性，会对外框的ratio、颜色和大小进行一定的随机变化，只要保证主要物体还在框里就行。  

（2）箭头  

箭头和把物体圈起来的做法不同，箭头一般画在物体附近，而且有方向性。  

（3）涂鸦  

scribble，contour和point其实都是类似涂鸦的方式，只是涂鸦的精细程度不同，point是最简陋的，contour是最精细的，而scribble介于两者之间。scribble是用贝塞尔曲线工具模拟人类轨迹画的。  

# DeepSeek-VL2  

DeepSeek-VL2开源了三个规模的模型，都是MoE：  

- DeepSeek-VL2-Tiny：总参数3B，激活参数0.57B  
- DeepSeek-VL2-Small：总参数16B，激活参数2.4B  
- DeepSeek-VL2：总参数27B，激活参数4.1B  

原文给出的效果对比：  

{% asset_img ds_perf.png DeepSeek-VL2 %}  

不过这张图比的是激活参数。其实直接看总参数，DeepSeek-VL2的效果也是很不错的，只是没有看激活参数的优势那么大。从另一个角度想，如果DeepSeek通过模型架构和计算框架优化，可以把MoE+MLA结构做到和同样激活参数的dense模型相同效率的话，这么对比也不是不行。  

DeepSeek-VL2相比前一代，主要有3个优化点：  

- 动态高分辨率vision encoding  
- LLM架构优化  
- 数据构建pipeline优化  

LLM架构优化其实就是MoE + MLA，带来的语言模型效率和效果提升，这部分在[《DeepSeek-V3细节探索》](https://mp.weixin.qq.com/s/alKnPog2LYSRQdm9wy1_QA)中有细说，此处就不展开。三个开源模型的具体结构参数：  

{% asset_img ds_model.png DeepSeek-VL2 %}  

最小的Tiny模型没有使用MLA，而是使用MHA，这和我们之前对MLA的认知是一致的：模型每个头的大小并不需要很多，模型增大更多是增加头数，而MLA需要在头数更多的场景下才能发挥效率和效果的优势，因此模型越大MLA优势越大，而在小模型上MLA则不容易发挥优势。  

另外，只有最大的DeepSeek-VL2使用了expert correction bias和sigmoid routing function，这俩都跟expert parallelism有关。  

另外有点奇怪的是只有small版本的vocab是102400，其他两个都是129280

DeepSeek-VL2整体框架还是标准的三件套：  

{% asset_img ds_overview.png DeepSeek-VL2 %}  

## 动态分辨率：Dynamic Tiling Strategy  

使用高分辨率 + 动态分辨率基本上已经是现在的标准做法。  

DeepSeek-VL2三个规模的模型使用的vision encoder都是SigLIP-SO400M-384，这是一个基础分辨率为384 × 384的模型。基于这个分辨率，定义了一批候选分辨率，这些候选分辨率的width和height都是384的倍数：  

$$
C_R = \{(m\cdot 384, n\cdot 384) \mid m \in \mathbb{N}, n \in \mathbb{N}, 1 \leq m, n, mn \leq 9\}
$$  

对于每一个原始图像，会保持ratio进行resize到每个候选分辨率，并选择使用所需padding最少的候选resolution。  

最后还会加上一个原图的缩略图，因此总用有（1 + m × n）个tile，每个tile都是384 × 384的大小，由vision encoder来单独处理。  

以上是vision encoder的输出。接下来是VL Adaptor的处理。  

SigLIP-SO400M-384使用的patch size = 14，每个tile会产生27 × 27个visual embedding，会通过pixel unshuffle，把visual embedding的数量减少到14 × 14个。  

另外，为了帮助模型识别visual embedding的位置关系，在缩略图和子图的每行visual embedding最后都会加一个 \n token，标识一下这一行embedding的结束。  

这么一来总的token数就变成：  

14 × (14 + 1) + 14m × (14n + 1)  

最终得到的图像feature按这样排布：  

{% asset_img ds_tiling.png DeepSeek-VL2 %}  

动态分辨率的方案到这里就结束了。不知道有没有细心的同学发现，上面的基础分辨率384并不是patch size 14的整数倍数（384 / 14 ≈ 27.4），我也有点奇怪，搜索之下发现确实有问题：原来SigLIP-SO400M-384的真实分辨率并不是384，而是14 × 27 = 378，384只是由于历史遗留问题一直保持这么个写法。原链接在 [https://huggingface.co/google/siglip-so400m-patch14-384/discussions/4](https://huggingface.co/google/siglip-so400m-patch14-384/discussions/4)。（这简直和“2020年东京奥运会在2021举办”有异曲同工之妙）。  

## 多阶段训练  

DeepSeek-VL2的训练分三个阶段：  

- 对齐：训练adaptor和vision encoder，冻结LLM  
- 预训练：全参训练  
- SFT：全参训练  

{% asset_img ds_train.png DeepSeek-VL2 %}  

## 数据  

1、对齐  

在对齐阶段，DeepSeek-VL2只用ShareGPT4v数据：包含1.2M条caption和conversation样本。  

2、预训练  

预训练阶段使用了70%的VL数据和30%纯文本数据。  

（1）Interleaved image-text data  

主要来自WIT、WikiHo和OBELICS，它们的混合比例通过在eepSeek-VL2-Tiny上实验确定；还有一个in-house数据集来增强真实世界知识的覆盖。  

（2）Image captioning data  

对现有的caption数据进行recaption处理，参考PixelProse的做法，在生成新caption的时候加入：  

- OCR hints  
- meta information (e.g., location, camera settings)  
- original captions  

recaption之后还是存在一些质量问题，因此用DeepSeek Chat对文本质量再进行打分和过滤，这样一来caption效果得到了有效提升  

（3）OCR数据  

包括LaTeX OCR和12M RenderedText数据集，和一些in-house数据集，主要是中英文的。  

（4）VQA数据  

包括：  
- General VQA  
- Table, chart and document understanding  
- Web-to-code and plot-to-Python generation  
- QA with visual prompt  

（5）Visual grounding data  

数据样式：  

```python
Prompt: Locate <|ref|><query><|/ref|> in the given image.
Response: <|ref|><query><|/ref|><|det|>[[x1, y1, x2, y2],...]<|/det|>
```  

> <|ref|>, <|/ref|>, <|det|>, <|/det|> are special tokens. \<query\> is a place-holder for either the category name (e.g., “car”) or description of the object (e.g., “the leftmost person”). \[\[x1, y1, x2, y2\], ...\] is a list of bounding boxes, where each bounding box corresponds to an object’s position. The coordinates x1, y1 and x2, y2 specify the top-left and bottom-right corners respectively, normalized to values between 0 and 999 according to the resolution of the image.  

还另外构建了负样本，把一些object从原图上消去，以增加模型robustness。  

（6）Grounded conversation data  

数据样式：  

```python
Prompt: <|grounding|>Can you describe the content of the image?
Response: Two <|ref|>dogs<|/ref|><|det|>[[x1, y1, x2, y2],...]<|/det|> are running on the grass.
```  

3、SFT  

（1）General visual question-answering  

现有的VQA数据集有一些问题，包括：  

- response太短  
- OCR质量差  
- 有幻觉  

因此把original question、image和OCR信息放在一起，重生成response，以提升数据质量。  

（2）OCR and document understanding  

预训练后模型的OCR能力已经很强了，因此sft阶段专注选出低质量样本，提升数据质量。  

（3）Table and chart understanding  

同OCR类似  

（4）Reasoning, logic, and mathematics  

发现detailed response在小模型上的训练效果并不好，小模型对简洁的response的学习能力更好。  

（5）Textbook and academic questions  

使用了包含跨学科、大学水平的教科书内容的内部数据集。  

（6）Web-to-code and plot-to-Python generation  

对于开源数据也重新生成response提高质量。  

（7）Visual grounding  

把query翻译成了中文，还加了一个negative sample。  

（8）Grounded conversation  

使用《Groma: Localized visual tokenization for grounding multimodal large language models》和《Flickr30k entities: Collecting region-to-phrase correspondences for richer image-to-sentence models》数据集构建对话数据。  

（9）Text-Only datasets

使用了很多数据，但是没有给出比例。  

一个总结，在数据这块DeepSeek-VL2在强调多样性的同时，也用现有的模型构建更强的pipeline重新生成response以提高数据质量。  

## cases  

{% asset_img ds_case1.png DeepSeek-VL2 %}  

{% asset_img ds_case2.png DeepSeek-VL2 %}  

# 小结  

- 现有的多模态数据质量有高有低，直接使用可能有很好的效果  
- 数据多样性能够有效提升模型训练效果  
- 按这个趋势MoE有可能再次火起来？如果硬件的优化跟上，MoE说不定能成为attention一样的标准方案。MLA也同样有这个可能。  

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

【1】DeepSeek-VL2: Mixture-of-Experts Vision-Language Models for Advanced Multimodal Understanding，https://www.arxiv.org/abs/2412.10302  
【2】From Pixels to Prose: A Large Dataset of Dense Image Captions, https://arxiv.org/abs/2406.10328  
【3】ViP-LLaVA: Making Large Multimodal Models Understand Arbitrary Visual Prompts, https://arxiv.org/abs/2312.00784  
【4】关于SigLIP-SO400M-384的输入分辨率：https://huggingface.co/google/siglip-so400m-patch14-384/discussions/4  
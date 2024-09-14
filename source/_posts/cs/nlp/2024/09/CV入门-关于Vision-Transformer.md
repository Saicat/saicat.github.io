---
title: CV入门--关于Vision Transformer
tags:
  - CV
  - transformer
  - 预训练
  - CNN
categories:
  - CS
  - CV
abbrlink: a11e2633
date: 2024-09-13 20:19:45
---

【本文已在同名 微信公众号 / 知乎 / [个人博客linsight.cn](http://www.linsight.cn/) 上线】  

***  

Transformer在自然语言的应用很成功，而在CV领域的崛起相对来说就比较慢，毕竟Transformer最初是为自然语言设计的。  

图片相比自然语言，多了2D的结构信息，因此从输入开始就需要对数据做一些处理。  

一个最直接的处理方法就是把每个pixel当成一个token，并把二维的图片序列化成一维的序列。比如原来的图像可以看做一个二维数组：  

```python
[[1, 2, 3],
 [4, 5, 6],
 [7, 8, 9]]
```

flatten之后就成了1D的序列：  

```python
[[1, 2, 3, 4, 5, 6, 7, 8, 9]]
```

这样224×224的图片变成一个包含50176个pixel（token）的输入序列。不过这样做输入长度明显太长了。50k的长度放在今天的LLM依然是颇有挑战性的一个长度，而这还只是一张224×224的小图片，如果是更大的图片处理起来就更困难了。  

因此要处理图像信息，还需要做一些改动。有几个工作的效果不错，包括iGPT、ViT、Swin Transformer等，下面简单梳理一下这几个方法的思路和重要内容。  

（以下内容需要读者具备基本的Transformer、CNN、GPT、Bert以及图像数据的相关知识）

# iGPT  

论文：《Generative Pretraining from Pixels》  

时间：2020年  

机构：OpenAI  

这是Ilya参与的一个工作，不仅把Transformer应用到CV上，并且还是用无监督的方式进行训练。  

## 预训练方式  

论文主要是用GPT-2模型做的实验，顺带着把Bert也对比了一下。  

下图给出了怎么用GPT和Bert进行图像预训练：  

{% asset_img igpt_intro.png vision transformer %}  

第一步是要对图像数据进行处理。如前面所说，直接用原图的一个pixel作为一个token，会导致输入长度太长，那么iGPT的做法就是对图像进行下采样，降低分辨率。  

一开始，所有数据都会被resize到224×224（对于使用data augmentation的情况，则是先resize到一个更大的size，然后random crop一个224×224的image，不过总之结果都是给出一个224×224大小的图像）。  

不要忘了RGB图像还有三个channel，所以实际的图像数据点是224×224×3。iGPT会在这个基础上进行下采样，下采样的目标分辨率称之为input resolution（IR），论文里用了三种IR做实验：32×32×3、48×48×3、64×64×3。  

关于为什么不进一步降低大小：因为参考《80 Million Tiny Images: A Large Data Set for Nonparametric Object and Scene Recognition》的研究，人类在32以下的分辨率，图像分类的准确率就会快速下降，说明再继续减小分辨率可能会损失很多重要信息。  

32×32×3虽然已经很小，这些像素点拉直成序列之后，仍然有3072的长度，计算量仍然不小。为了进一步减少计算量，iGPT用9-bit color palette（也就是有512个cluster）对（R,G,B）像素点进行聚类，这样就把3个channel的数据合成为1个数据，并且在颜色上没有明显的损失。这样一来长度就减小到三分之一，这个长度（32×32/48×48/64×64）称之为model resolution（MR）。  

之后把得到的二位图像进行序列化，就得到了GPT/Bert的输入数据了。  

而在训练目标上，和自然语言的情况一样，对于GPT模型，要求模型根据前面的输入pixel，预测下一个pixel：  

$$L_{AR}=\mathbb{E}_{x\sim X}[-\log p(x)]$$  

类似地，对于Bert模型，mask掉部分pixel，并要求模型预测被mask的pixel。  

$$L_{BERT}=\underset{x\sim X}{\mathbb{E}}\underset{M}{\operatorname*{\mathbb{E}}}\sum_{i\in M}\left[-\log p\left(x_i|x_{[1,n]\setminus M}\right)\right]$$  

## 评测方式  

验证预训练模型效果好不好，有两个方法：finetune和linear probing。  

finetune的时候会用最后一层输出的average pooling，加上一个projection得到class logits。loss的计算上，除了使用cross entropy之外，还会额外加上预训练损失L_GEN：  

$$L_{FT}=L_{GEN}+L_{CLF}$$  

$$L_{GEN}\in\{L_{AR},L_{BERT}\}$$  

linear probing则是仅用预训练模型给出的特征和projection进行图像分类，并且不会训练预训练模型的参数。还有一点不同就是linear probing用的不一定是最后一层的特征，因为最后一层的特征未必是最好的。  

## 实验  

实验用的模型如下：  

<center>

| 模型 | 层数 | 参数量 |
| :----: | :----: | :----: |
| iGPT-XL | 60 | 6.8B |
| iGPT-L | 48 | - |
| iGPT-M | 36 | 455M |
| iGPT-S | 24 | 76M |

</center>  

1、对于linear probe，哪层的效果最好  

如上面所说，实验中发现，对于linear probe，最后一层的表征往往不是最好的。对于iGPT-L，在不同任务上用不同层的特征做线性分类的效果如下  

{% asset_img igpt_layer.png vision transformer %}  

基本上都是在20层左右效果最好，往前或者往后的层效果都下降了。  

2、是不是预训练模型越好，下游任务效果越好  

答案是yes。下图给出预训练的validation loss和linear probe效果的关系：  

{% asset_img igpt_pretrain_loss.png vision transformer %}  

注意横轴的值往右是减小的。  

3、iGPT和其他模型效果对比  

用linear probe任务评测，在CIFAT-10和CIFAT-100上，iGPT-L效果比SimCLR和ResNet-152都更好一些。  

{% asset_img igpt_perf1.png vision transformer %}  

而在ImageNet上测试时，iGPT则遇到问题。由于不能在标准 ImageNet 输入分辨率上进行高效训练，在model resolution为32×32时效果并不好，只有把MR提升到64×64，并且使用iGPT-XL时，才达到还不错的效果。但是计算量远超其他方法了。  

{% asset_img igpt_perf2.png vision transformer %}  

4、微调效果  

经过微调之后，iGPT的效果有所提升，不过还达不到其他最佳模型的效果。  

{% asset_img igpt_ft.png vision transformer %}  

5、对比GPT和Bert  

在linear probe上，Bert模型的效果比GPT差很多，而微调之后Bert能够赶上GPT的效果。  

{% asset_img igpt_bert.png vision transformer %}  

# ViT  

论文：《An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale》  

时间：2020年10月  

机构：Google  

## 把transformer应用在图像  

这是来自Google的工作，也是把transformer成功应用到图像领域的经典工作。  

所用的模型主体是transformer encoder：  

{% asset_img vit_intro.png vision transformer %}  

整体上和Bert和相似，最大的不同就是输入部分的处理。  

对于文本来说，输入数据是一个1D序列，但是图像是2D的，需要把2D的图像适配到transformer的输入格式。一个简单的办法就是flatten。这样就能把图像信息输入到transformer里了。当然这样会有一个问题，就是flatten之后的图像丢失了部分pixel之间的结构化位置信息。一个折中的办法就是不以pixel为单位，而是以一个patch为单位来flatten，这样可以减少丢失的结构信息。  

具体来说，对于一张图像 $\mathbf{x}\in\mathbb{R}^{H\times W\times C}$，在H和W维度切成大小为P×P的patch，这样图像数据就变成 $\mathrm{x}_p\in\mathbb{R}^{N\times(P^2\cdot C)}$，而 $N=HW/P^2$。  

每个patch再经过一个trainable linear projection之后，就是patch embedding。patch embedding相当于文本模型里的token embedding，图像模型patch进行线性变换的操作对应文本模型里从vocab取对应token embedding的操作，而图像切分patch则相当于文本的tokenization。  

不要忘记还有位置编码。论文里对下面集中情况做了实验：  
- 不使用位置编码，输入相当于是bag of patches（就像bag of words）  
- 使用1D的位置编码，和文本transformer一样  
- 使用2D的位置编码，embedding的前一半维度加上的是X-embedding，后一半维度加上的是Y-embedding  
- 1D的相对位置编码  

各个方法的效果对比如下：  

{% asset_img vit_pos.png vision transformer %}  

基本上只要加了位置编码效果就差不多，都比不加要好，因此这里选择了传统的可学习式的1D位置编码，这样实现最简单。  

由于transformer本身理论上是可以接受任意长度的输入的，所以预训练好的ViT也可以用于higher resolution的场景，不过这个时候需要对位置编码做一下处理。对于可训练的绝对位置编码，一个方法是进行插值。  

此外，ViT参考Bert的做法，在最开始加入一个[class]token，作为整个图像的表征。  

最终ViT的输入就是：  

$$\mathrm{z}_0=[\mathrm{x}_{\mathrm{class}};\mathrm{~x}_p^1\mathrm{E};\mathrm{~x}_p^2\mathrm{E};\cdots;\mathrm{~x}_p^N\mathrm{E}]+\mathrm{E}_{pos},\quad\mathrm{E}\in\mathbb{R}^{(P^2\cdot C)\times D},\mathrm{~E}_{pos}\in\mathbb{R}^{(N+1)\times D}$$  

后面的计算就和Bert一样了。  

和自然语言以及iGPT的情况不同，ViT预训练使用的是有监督的数据。  

## 实验  

在Google之前也有不少人尝试把transformer应用到CV上，但是效果没有那么显著，原因就是预训练数据量不够大。  

实验用到的3个transformer模型如下：  

{% asset_img vit_model.png vision transformer %}  

作为对比，还有几个CNN：  
- ResNet  
- hybrids：把ResNet的中间feature map作为ViT的输入  

还有3个不同规模的预训练数据集：  
- 1.3M张图片的ILSVRC-2012 ImageNet dataset  
- 14M张图片的ImageNet-21k  
- 303M张图片的JFT  

实验发现，随着预训练的数据量增大，ViT相对ResNet的优势逐渐明显：  

{% asset_img vit_perf.png vision transformer %}  

# DeiT（Data-efficient image Transformers）  

论文：《Training data-efficient image transformers & distillation through attention》  

时间：2020年12月  

机构：Facebook AI  

## TL;DR  

ViT里提到需要比较多的预训练数据，模型的效果才能比较好。而DeiT通过使用蒸馏，仅在ImageNet上训练，获得了综合throughput & 任务效果更佳的模型。  

{% asset_img deit_intro.png vision transformer %}  

## 蒸馏  

DeiT主要内容是提出一种使用图像transformer学习的蒸馏策略。student模型和ViT一样，而teacher模型可以是任意模型。  

通常使用的蒸馏，soft distillation，是在学习true label之余，增加一个KL散度损失来学习teacher model的知识：  

$$\mathcal{L}_\mathrm{global}=(1-\lambda)\mathcal{L}_\mathrm{CE}(\psi(Z_\mathrm{s}),y)+\lambda\tau^2\mathrm{KL}(\psi(Z_\mathrm{s}/\tau),\psi(Z_\mathrm{t}/\tau))$$  

Z_t是teacher model的logits，Z_s是student model的logits，τ是蒸馏的温度，而λ控制两项损失的比重。  

相对soft distillation，DeiT提出hard-label distillation：不学习teacher model的logits了，直接学习teacher model的最终label：  

$$\mathcal{L}_{\mathrm{global}}^{\mathrm{hardDistill}}=\frac{1}{2}\mathcal{L}_{\mathrm{CE}}(\psi(Z_{s}),y)+\frac{1}{2}\mathcal{L}_{\mathrm{CE}}(\psi(Z_{s}),y_{\mathrm{t}})$$  

$$y_\mathrm{t} = \mathrm{argmax}_cZ_\mathrm{t}(c)$$  

hard-label distillation也可以通过label smoothing手动转成soft label：固定把 $1-\varepsilon $ 的概率给true label，而剩余的概率就分给其他所有类别。DeiT使用了 $\varepsilon=0.1 $。  

通常来说，来自true label和teacher model的信息都可以由class token（ + classifier）来学。而DeiT的做法是增加一个distillation token专门用来学习蒸馏的知识。  

{% asset_img deit_framework.png vision transformer %}  

实验中发现class token embedding和distillation token embedding收敛到完全不同的表达，二者之间的cos相似度只有0.06；而随着层数增加，到输出层，二者的cos相似度提高到了0.93。作为对比，实验了使用2个class token的情况，结果两个class token在输出和输出层都收敛到几乎完全一样的vector。论文认为这说明使用distillation token引入了新的信息。（感觉此处有点存疑）  

## 实验  

DeiT所用实验所用模型如下：  

{% asset_img deit_model.png vision transformer %}  

1、不同teacher model  

实验发现，使用卷积模型作为teacher model比transformer更好，原因可能是相比transformer，卷积模型提供了一些inductive bias，让student模型可以学到一些图像的结构化信息（《Transferring inductive biases through knowledge distillation》）。  

{% asset_img deit_teacher.png vision transformer %}  

2、不同的蒸馏方法  

下图是各种蒸馏方法的效果对比：  

{% asset_img deit_distill.png vision transformer %}  

- DeiT– no distillation：没有蒸馏，作为baseline对比  
- DeiT– usual distillation：使用soft label，没有distillation token  
- DeiT– hard distillation：使用hard label，没有distillation token  
- DeiT⚗: class embedding：使用hard label，仅使用class token  
- DeiT⚗: distil. embedding：使用hard label，仅使用distillation token  
- DeiT⚗: class+distillation：使用hard label，使用class token和distillation token  

结论是使用hard label和distillation token有提升。  

## 小结  

和其他模型的整体对比如下：  

{% asset_img deit_perf.png vision transformer %}  

DeiT的效果略略比EfficientNet差一点。而相比ViT，DeiT的优势是训练的量比较少。  

# Swin Transformer  

论文：《Swin Transformer: Hierarchical Vision Transformer using ShiftedWindows》  

时间：2021年3月  

机构：MSRA  

Swin来自Shifted WINdow。  

## 模型结构  

Swin Transformer主要的改造是在模型设计上。下图是Swin-T（T是Tiny）模型的结构图，后面结合这个图来一步步看看Swin Transformer是怎么做的。  

{% asset_img swin_model.png vision transformer %}  

首先是输入部分。Swin Transformer跟ViT一样，把多个pixel group在一起作为一个patch，也相当于输入给模型的一个token。只是这里使用的patch size比较小，是4×4的。再乘上RGB三个channel，那么每个patch就维度就是4×4×3=48。  

同ViT类似，这里也会用一个linear layer把每个patch的48维数据映射成一个C维的embedding。这时输入图片就变成H/4×W/4×C的大小。  

linear embedding后面跟着两个Swin Transformer Block，这两个block的attention部分和一般的transformer block有所不同，后面详说。  

linear embedding和两个Swin Transformer Block合一起成为stage 1，stage 1的输出特征维度是H/4×W/4×C。  

为了得到hierarchical representation，stage 1的输出会经过一个patch merging layer进行维度的转换。具体来说，patch merging layer会把上一阶段输出中的2×2的相邻patch concat在一起，得到H/8×W/8×4C的feature map，再经过一个linear层，把feature map维度降为H/8×W/8×2C。  

和stage 1类似，patch merging layer后会跟两个Swin Transformer Block。它们合在一起就是stage 2。  

后面是stage 3、stage 4就是重复stage 2的操作，输出的大小分别为H/16×W/16×4C和H/32×W/32×8C。  

## Shifted Window Attention  

现在回过头来看Swin Transformer Block具体是怎么做的。  

首先，我们知道使用pixel直接作为token，或者用小的patch size会使得输入给transformer模型的数据长度很长，导致计算成本不可控。那么Swin Transformer为了限制计算量，就把attention计算限制在一个local的范围内，而不是全局attention。  

限制attention计算的范围就是window size。如下图（a），每个红框就是一个window，每个window包含4×4个patch。每个patch只和同window内的patch进行attention计算，而不和其他window的内容进行交互。  

{% asset_img swin_hierachical.png vision transformer %}  

这样一来只要保持window size比较小，那么模型的计算复杂度就小了。具体来说，对于一张包含h×w个patch的图像，假设window size=M×M个patch，那么传统的attention计算量和hw的平方成正比，而window attention的计算量则是和hw成线性关系：  

$$\Omega(\text{MSA})=4hwC^2+2(hw)^2C$$  

$$\Omega(\text{W-MSA})=4hwC^2+2M^2hwC$$  

这也是Swin Transformer能够使用2×2的小patch的原因。  

Swin-T模型中每个stage的第一个Swin Transformer Block就是使用window attention计算的。  

但是window attention也有缺点，那就是window之间的信息无法交互，这样就会大大限制了模型的学习能力。为了解决这个问题，Swin-T模型中每个stage的第二个Swin Transformer Block使用的是shifted window attention。  

shifted window attention，顾名思义就是把window shift一下，如下图  

{% asset_img swin_window.png vision transformer %}  

对于window size为M×M的情况，会把所有window往左上shift M/2个patch。这样把window attention和shifted window attention交错使用，就能让信息在不同window之间进行传递。  

## 实验  

Swin Transformer的各个规模模型如下：  
- Swin-T: C = 96, layer numbers = {2; 2; 6; 2}  
- Swin-S: C = 96, layer numbers = {2; 2; 18; 2}  
- Swin-B: C = 128, layer numbers = {2; 2; 18; 2}  
- Swin-L: C = 192, layer numbers = {2; 2; 18; 2}  

window size M = 7。  

1、Image Classification  

直接在ImageNet-1K上训练，和在ImageNet-22K预训练后再在ImageNet-1K上微调，Swin Transformer和其他模型效果对比如下：  

{% asset_img swin_perf1.png vision transformer %}  

Swin Transformer的top-1 acc基本上达到了最好的一档，而计算量相比之前最好的CNN的增加也在可接受范围内。  

2、Object Detection on COCO  

最佳模型在 COCO 测试集上的性能超过了之前的最佳结果。  

{% asset_img swin_perf2.png vision transformer %}  

3、Semantic Segmentation on ADE20K  

Swin-S 比 DeiT-S 和 ResNet-101 等模型性能更优，Swin-L 模型在 ImageNet-22K 预训练后在验证集上的 mIoU 达到 53.5，超过之前的最佳模型（SETR 的 50.3 mIoU）。  

{% asset_img swin_perf3.png vision transformer %}  

## 小结  

- Swin Transformer的计算量不再随着图像大小平方增加，使得可以使用更小的patch size，这是效果提升的一个要点  
- 在效果到达/超过第一梯队CNN水平的同事，Swin Transformer没有大幅提升  
- 相比ViT，Swin Transformer的hierarchical结构使得它的特征在object detection和semantic segmentation都好用  

# 其他  

其他一些Transformer/Attention相关做法，也简单梳理下。  

## Stand-Alone Self-Attention in Vision Models  

时间：2019年6月  

机构：Google Brain  

在CNN中，原本卷积计算是这样的：  

{% asset_img att_conv.png vision transformer %}  

这里把卷积替换成一个local attention：  

{% asset_img att_attention.png vision transformer %}  

相当于把卷积的内部计算替换成self-attention计算了。跟CNN一样，每层的attention会扫描当前输入feature map的所有位置，给出所有输出值，构成输出feature map。  

此外，论文认为位置信息在attention的缺失会带来permutation equivariant，因此还引入了2D的相对位置编码：  

{% asset_img att_pos.png vision transformer %}  

这些相对位置编码加到每个attention计算内部的各个pixel的key上。  

## On the Relationship between Self-Attention and Convolutional Layers  

时间：2019年11月  

这个工作里，作者通过理论证明和实验验证来表明attention架构的前几层学会了关注每个query像素周围的网格状模式，类似于卷积层的行为。  

attention模型训练时每个层有九个注意力头，与ResNet架构中使用的3×3内核相对应。训练后，观察到注意力头在query pixel周围形成网格，如下图所示，前几层的头倾向于关注局部模式，而更深层的头关注更大的模式，这表明self-attention应用于图像时确实学会了在query pixel周围形成类似于convolution filter的模式。  

{% asset_img rel_1.png vision transformer %}  

{% asset_img rel_2.png vision transformer %}  

实际训练的时候，为了减少计算量，使用了2×2 pixel的patch。  

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
[长文详解--LLM高效预训练(一)](https://www.linsight.cn/dcb57672.html)  
[Llama3.1--预训练要点一览](https://www.linsight.cn/7d7294cb.html)  
[Qwen2技术报告](https://www.linsight.cn/a8f8b641.html)  
[Yi技术报告-划重点看细节](http://www.linsight.cn/41b6a819.html)  
[InternLM系列模型](https://www.linsight.cn/7f3d361.html)  
[GLM4报告的一些技术点](https://www.linsight.cn/a5206abd.html)  
[从Yuan2.0到Yuan2.0-M32](https://www.linsight.cn/3df0cd42.html)  
[从loss视角理解大模型涌现能力](https://www.linsight.cn/f5fb75e4.html)  
- 数据：  
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

【1】Generative Pretraining from Pixels https://cdn.openai.com/papers/Generative_Pretraining_from_Pixels_V2.pdf  
【2】An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale https://arxiv.org/abs/2010.11929  
【3】Swin Transformer: Hierarchical Vision Transformer using ShiftedWindows https://arxiv.org/abs/2103.14030  
【4】Training data-efficient image transformers & distillation through attention https://arxiv.org/abs/2012.12877  
【5】How Do Vision Transformers work? https://arxiv.org/abs/2202.06709  
【6】Stand-Alone Self-Attention in Vision Models https://arxiv.org/abs/1906.05909  
【7】On the Relationship between Self-Attention and Convolutional Layers https://arxiv.org/abs/1911.03584  

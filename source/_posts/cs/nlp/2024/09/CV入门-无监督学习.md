---
title: CV入门--无监督学习
tags:
  - CV
  - transformer
  - 预训练
  - CNN
  - 无监督学习
categories:
  - CS
  - CV
abbrlink: ae81a87b
date: 2024-09-14 21:08:07
---

【本文已在同名 微信公众号 / 知乎 / [个人博客linsight.cn](http://www.linsight.cn/) 上线】  

***  

标注数据总是缺乏的，特别是对于大规模预训练。因此无监督学习在大模型时代就更加重要了。相比有监督的训练，无监督学习不需要把最终结果都reduce到一个单一的label，因此模型也能学到更丰富的数据特征。  

本篇整理图像领域一些无监督学习的方法，涉及的论文如下：  

<center>

| 模型/方法 | 时间 | 论文 |
| :----: | :----: | :----: |
| Memory Bank | 2018年5月 | Unsupervised Feature Learning via Non-Parametric Instance-level Discrimination |
| MoCo | 2019年11月 | Momentum Contrast for Unsupervised Visual Representation Learning |
| SimCLR | 2020年2月 | A Simple Framework for Contrastive Learning of Visual Representations |
| BYOL | 2020年6月 | Bootstrap Your Own Latent：A New Approach to Self-Supervised Learning |
| DINO | 2021年4月 | Emerging Properties in Self-Supervised Vision Transformers |
| BEiT | 2021年6月 | BEiT: BERT Pre-Training of Image Transformers |
| MAE | 2021年11月 | Masked Autoencoders Are Scalable Vision Learners |

</center>  

# MoCo  

论文：《Momentum Contrast for Unsupervised Visual Representation Learning》  

时间：2019年11月  

机构：FAIR  

## TL;DR  

基于对比学习，MoCo通过动量更新的方法，改进图像的无监督学习，可以在多个下游任务获得接近/匹配有监督预训练的效果。  

## 无监督学习  

无监督学习在文本数据上效果很好，而在图像上则相对没那么突出。原因可能是二者在信号空间上的差异：文本数据天然有离散的自监督信号，而图像数据的信号在连续的高维空间中，相比之下没有那么有结构化。  

无监督学习框架的设计一般包括两个主要方面：  
- pretext task：这类任务本身的能力并不是我们想要的，只是通过这类任务让模型学到泛化性好的数据表征；“The term “pretext” implies that the task being solved is not of genuine interest, but is solved only for the true purpose of learning a good data representation.”  
- loss function：决定了模型如何学习，往往也和pretext task密切相关  

在此之前的一些无监督方法：  
- auto-encoder：任务是重建输入的图像，使用L1或者L2 loss衡量误差  
- GAN：包含生成网络和判别网络，衡量的是概率分布的差异  

## 对比学习  

除了auto-encoder和GAN，使用对比学习也是一个自监督学习的思路。对比损失衡量一对数据在表示空间中的相似度，它们在表示空间中的具体值在训练过程中是会变化的，而不是固定的。  

使用对比学习，可以看作是训练一个执行dictionary look-up任务的encoder。  

假设现在有一个输入query q，以及一系列由encoder编码过的样本 $\{k_0,k_1,k_2,...\}$，其中只有一个样本 $k_{+}$ 和q是匹配的，即正样本（正样本最常用的来源就是对同一张图片使用不同的增强方法），其他样本都是负样本，那么contrastive loss如下计算  

$$\mathcal{L}_q=-\log\frac{\exp(q\cdot k_+/\tau)}{\sum_{i=0}^K\exp(q\cdot k_i/\tau)}$$  

$$q=f_{\mathbf{q}}(x^q)$$  

$$k=f_{\mathbf{k}}(x^k)$$  

其中τ是温度（实验中设为0.07），K是负样本数量。对比损失理论上会提升q和正样本的相似度，同时降低q和其他所有负样本的相似度。  

fq和fk是query和key的encoder，通常来说这两个encoder可以一样，也可以不一样，取决于具体的pretext task。而xq和xk可以是图片，图片的patch，或者一组patch的组合。  

前面提到，训练的encoder的任务是做dictionary look-up，类比到文本数据的预训练，可以认为这里的（K + 1）个表征就是dictionary的大小，正样本就是true label，负样本则是其他的token。  

那么这些负样本从哪里来呢？最简单的做法就是在一个mini-batch里，除了正样本以外的其他数据都看作是负样本：  

{% asset_img moco_a.png CV无监督学习 %}  

正样本的表征和负样本的表征同时通过bp进行训练。

这样的做法会把负样本数量和batch size耦合了起来。而直观上来说，更大的负样本数量效果会更好。由于batch size受制于显存大小（每个负样本都要通过bp更新），没法设得特别大，因此负样本的数量也就受到了限制。并且使用超大batch size进行训练，对效果也会有影响（《Accurate, large minibatch SGD: Training ImageNet in 1 hour》）。  

为了增大负样本的数量，《Unsupervised feature learning via non-parametric instance discrimination》中使用了memory bank的方法：  

{% asset_img moco_b.png CV无监督学习 %}  

memory bank中包含了训练数据集所有样本的编码表示。每次训练会从memory bank中随机抽取一批作为负样本。这里memory bank里的负样本不会随着query更新，因此能够在相同的显存下使用更大的batch size。  

但是memory bank里的数据存在过时的数据（比如部分特征向量分别是由1000/2000/3000步前的encoder给出的结果），因此存在特征不一致的问题：q是最新的encoder编码的，k是不同时间的旧的encoder编码的。这种不一致的情况会对效果有影响。  

## Momentum Contrast  

从上面的做法我们可以看到，构建好的dictionary需要满足两个条件：（1）dictionary要大（2）负样本的表征需要尽量保持编码的一致性。  

MoCo在memory bank的基础上，为了低成本地更新key的表示，不直接用bp更新，而是通过momentum更新key的encoder。  

MoCo维护一个固定大小queue，每一步会把最新的一个mini-batch的表征放到这个queue里，并移除最旧的一个mini-batch。  

假设fk的参数是θk，而fq的参数是θq，每一步θq还是正常按bp更新，而θk更新则通过momentum增量更新：  

$$\theta_\mathbf{k}\leftarrow m\theta_\mathbf{k}+(1-m)\theta_\mathbf{q}$$  

其中m是momentum。在实际使用中，m的值比较大时（0.999）比较小值（0.9）效果更好，这也说明保持负样本的一致性的重要性。  

MoCo的方案示意图如下：  

{% asset_img moco_c.png CV无监督学习 %}  

MoCo完整算法如下：  

{% asset_img moco_algo.png CV无监督学习 %}  

和memory bank相比，MoCo对queue中的表征更新更加平缓，queue中的样本表示的一致性更好。  

# SimCLR  

论文：《A Simple Framework for Contrastive Learning of Visual Representations》  

时间：2020年2月  

机构：Google Brain  

Hinton参与的一个工作。把无监督训练的模型在ImageNet上的效果做到和监督学习一样。  

{% asset_img simclr_intro.png CV无监督学习 %}  

SimCLR依然是基于对比学习的框架。为什么选择对比学习做无监督学习？生成式的方法包括encoder和decoder，一般会把重建图片作为任务，而这样复杂精细的任务其实对于对月学习representation并不是必须的，复杂的计算很多其实对下游没有什么帮助（decoder部分的学习）。而有些discriminative的方法会借用真实label或者生成的label，使用类似有监督的方式进行预训练。这种方式某种程度上是限制了模型学习的representation的泛化性。相比之下对比学习的框架在效率和效果上有更好的平衡。  

SimCLR的训练框架如论文标题所说的，确实比较简单：  

{% asset_img simclr_framework.png CV无监督学习 %}  

其中有4个重要的组件：  
- data augmentation  
- encoder f(·)  
- projection head g(·)  
- contrastive loss function  

整体来看，SimCLR更多是精细工程带来的成功。  

{% asset_img simclr_algo.png CV无监督学习 %}  

## data augmentation  

论文对不同的图像增强方法和它们的组合进行实验：  

{% asset_img simclr_aug.png CV无监督学习 %}  

下图是在SimCLR其中一个branch不增强，而在另一个branch采用不同的数据增强方式下训练出来的模型在ImageNet上的效果。对角线的实验只用了一种增强，其他位置则是对应行和列的增强方式的组合。  

{% asset_img simclr_aug_exp.png CV无监督学习 %}  

发现crop和color distort这两个的组合效果特别好。  

另外对比数据增强强度在监督学习和无监督学习的效果，发现无监督学习需要更强的增强强度来获取更好的效果：  

{% asset_img simclr_unsup.png CV无监督学习 %}  

## encoder  

SimCLR中的encoder理论上可以是任意模型。简单起见，论文中选择了ResNet。对比有监督学习，和使用不同大小的ResNet的SimCLR，发现随着模型规模的增大，无监督学习的效果和有监督学习的gap在减小。  

{% asset_img simclr_eval.png CV无监督学习 %}  

这说明无监督训练相比有监督，benefit more from更大的模型。  

## nonlinear projection head  

使用的projection head是一个MLP层  

$$z_i=g(h_i)=W^{(2)}\sigma(W^{(1)}h_i)$$  

对比三种projection的效果：（1）identity mapping（2）linear projection（3）nonlinear projection  

发现（3）比（2）提升3%，而比（1）提升则>10%：  

{% asset_img simclr_proj.png CV无监督学习 %}  

在不同输出维度下，都有相同结果。  

为什么projection head重要？论文认为，z = g (h) 被训练为对数据增强具有不变性，会去除可能对下游任务有用的信息（而在对比学习中不需要），如对象的颜色或方向，这样这些去除的操作可以在g，也就是projection head中完成，从而使得encoder的输出h本身可以保留更多对下游任务有用的信息。  

针对这个想法，用g(h)和h作为特征，用来预测数据变换的任务，发现h的效果更好，说明h相比g(h)确实保留了更多和数据变换相关的信息。  

{% asset_img simclr_proj_head.png CV无监督学习 %}  

## loss & batch size  

SimCLR的loss：  

$$\ell_{i,j}=-\log\frac{\exp(\sin(z_i,z_j)/\tau)}{\sum_{k=1}^{2N}\mathbb{I}_{[k\neq i]}\exp(\sin(z_i,z_k)/\tau)}$$  

SimCLR不使用memory bank之类的方法，而只把同batch中的除了正样本之外的2（N-1）个样本全部作为负样本（batch size = N）。  

这么一来batch size应该就是越大越好了，实验结果也是如此：  

{% asset_img simclr_bs.png CV无监督学习 %}  

为了保证在大batch size下的训练稳定，SimCLR使用LARS optimizer。  

# BYOL（Bootstrap your own latent）  

论文：《Bootstrap Your Own Latent：A New Approach to Self-Supervised Learning》  

时间：2020年6月  

机构：DeepMind  

{% asset_img byol_intro.png CV无监督学习 %}  

之前的对比学习都需要使用正样本和负样本，并且会增大负样本的数量以提升效果。为什么要使用负样本，因为如果只使用正样本，那么模型的训练可能会collapse：只使用正样本，那么模型只需要对所有输入都输出相同的结果，比如全是0，也可以符合任务要求。  

而BYOL则是设计了一个不使用负样本也不会collapse的框架：  

{% asset_img byol_framework.png CV无监督学习 %}  

框架中包含target和online两个模型，其中online模型是正常bp训练的，而target模型则是通过momentum进行更新：  

$$\theta\leftarrow\mathrm{optimizer}(\theta,\nabla_\theta\mathcal{L}_{\theta,\xi}^{\mathbf{BYOL}},\eta)$$  

$$\xi\leftarrow\tau\xi+(1-\tau)\theta $$  

输入部分则和其他对比学习一样，是对输入图片的数据增强，具体是参考了SimCLR的实现。  

# DINO（self-DIstillation with NO labels）  

论文：《Emerging Properties in Self-Supervised Vision Transformers》  

时间：2021年4月  

机构：FAIR  

## TL;DR  

ViT的效果不错，但是也存在缺点：计算量大训练量大，监督试的预训练任务信号丰富程度不如Bert或者GPT这种自监督的信号丰富，图片信息最后都被reduce到某一个类别上。  

DINO通过蒸馏的形式实现了图像的无监督学习，并且发现无监督学习得到的ViT展现出一些在监督ViT或者CNN上没有的特性：  
- feature能给出物体的layout和边界（下图）  
- 自监督ViT的特征用来K-NN分类效果很好  

{% asset_img dino_intro.png CV无监督学习 %}  

并且强调了无监督学习过程中几个重要的组件：  
- momentum encoder  
- multi-crop training  
- ViT small patches  

## 方法  

DINO的训练方法使用了知识蒸馏的框架，但是进行了一些改动。DINO训练示意图如下：  

{% asset_img dino_framework.png CV无监督学习 %}  

1、蒸馏  

在知识蒸馏中，于student network $g_{\theta_{s}}$ 学习预测teacher network $g_{\theta_t}$ 的输出。对于输入x，两个模型的输出分别是特征向量 $P_{s}$ 和 $P_{t}$：  

$$P_s(x)^{(i)}=\frac{\exp(g_{\theta_s}(x)^{(i)}/\tau_s)}{\sum_{k=1}^K\exp(g_{\theta_s}(x)^{(k)}/\tau_s)}$$  

温度τ控制输出分布的sharpness。然后通过cross entropy让student network学习teacher network的输出：  

$$\min_{\theta_s}H(P_t(x),P_s(x))$$  

$$H(a,b)=-a\log b$$  

另外上图中，teacher的输出还会经过一个centering的操作。centering操作类似BN，对数据进行归一化，稳定训练的过程。具体的centering可以有多种实现方式：  
- 均值移动  
- Sinkhorn-Knopp  
- Softmax  

2、teacher network  

通常的蒸馏，会使用已经训练好的模型作为teacher network。但是在DINO这样的无监督学习中，并没有一个已经训练好的模型。那teacher network从哪来呢？  

DINO中，teacher network会在训练中和student network一定逐渐学习。具体来说，teacher network是student network的exponential moving average (EMA)版本：  

$$\theta_t\leftarrow\lambda\theta_t+(1-\lambda)\theta_s$$  

λ会按cosine schedule从0.996逐渐增加到1。  

这里的teacher network和MoCo中的momentum encoder很类似，不过二者在各自框架中的角色有所不同。  

3、multi-crop strategy  

由于student network和teacher network都是随机初始化的，输入给student network和teacher network并不是简单的两张相同的image，否则就student network就无法学习到图像特征，而只需要模仿随机参数模型的输出就行了。  

《Unsupervised learning of visual features by contrasting cluster assignments》中的multi-crop是一个提升学习任务难度的方法。  

具体来说，对于每个输入图片x，会通过数据增强，获得两个global view xg1和xg2，以及一系列local view。global view有较大的size，一般来说可以覆盖原图超过50%的内容（如224×224），而local view的size则小一些（如96×96）。  

所有的view都会输入给student network，而teacher network只会接收global view的版本。这样能够鼓励模型学习到“local-to-global”的依赖关系：  

$$\min_{\theta_s}\sum_{x\in\{x_1^g,x_2^g\}}\sum_{x^{\prime}\neq x}H(P_t(x),P_s(x^{\prime}))$$  

DINO的算法具体描述如下：  

{% asset_img dino_algo.png CV无监督学习 %}  

## 消融实验  

1、不同组件对效果的影响  

{% asset_img dino_module.png CV无监督学习 %}  

发现如果没有momentum更新teacher network，整个框架就无法训练了。此外影响最大就是loss function，把CE改成MSE之后效果也有极大的下降。  

2、patch size的影响  

patch size越小效果越好，不过throughtput也越低，这是一个成本和效果的tradeoff：  

{% asset_img dino_patch_size.png CV无监督学习 %}  

# BEiT  

论文：《BEiT: BERT Pre-Training of Image Transformers》  

时间 2021年6月

机构：微软  

BEiT的做法和之前在学习ViT的时候讲过的iGPT类似：  

{% asset_img beit_intro.png CV无监督学习 %}  

- 相对输入图像进行patch（16×16）切分，然后获得各个patch的linear embedding  
- 使用类似MLM的MIM任务（masked image modeling），把部分patch mask掉，要求模型从corrupted image预测这些被mask的patch  

不同的是，BEiT使用dVAE训练了image tokenizer，用于把patch变成离散的token，这样Bert模型就可以和MLM训练一样，预测离散的token了。其中image tokenizer的vocabulary size设置成了8192。  

另外在选择mask的patch的时候，采用类似whole word mask的方式，使用blockwise masking。一个block包含多个patch（最小为16个）。对整个block进行mask可以提高训练的难度。总的mask比例设置为0.4。  

{% asset_img beit_blockmask.png CV无监督学习 %}  

# MAE

论文：《Masked Autoencoders Are Scalable Vision Learners》  

时间：2021年11月

机构：FAIR  

Kaiming参与的一个工作。  

大致的框架还是follow auto-encoder的结构：  

{% asset_img mae_intro.png CV无监督学习 %}  

在这个基础上，做了一些优化。  

1、masking  

和ViT一样，图像会被分成多个patch，然后经过linear transformation，再加上位置编码，获得patch embedding。然后会随机选择一部分patch进行mask。  

不一样的地方是，这些被mask掉的patch并不会用mask token代替，而是直接被移除，因此encoder处理的时候只会见到没有被mask的部分。  

实验中发现mask的比例在75%的时候，decoder仍然具备恢复原图的能力：  

{% asset_img mae_mask_ratio.png CV无监督学习 %}  

高mask率的好处是encoder的计算量少了，因此可以使用更强大的模型。此外高mask率带来的难度提升也使得训练效率更高了。  

2、encoder  

encoder使用的模型跟ViT类似。  

3、decoder  

虽然encoder的输入之后没有mask部分的patch，但是在把encoder的输出给到decoder之前，会添加上mask token，这样decoder才能获取缺失部分的位置信息，并恢复图像。  

decoder只会在预训练的时候使用，因此可以使用一个很小的decoder，从而把有用的训练更多放在encoder上。同时较弱的decoder会迫使encoder学习到更好的图像表征。  

4、reconstruction target  

decoder的训练目标是直接给出被mask掉的patch的pixel value，loss function直接是预测的pixel value和真实pixel value之间的MSE误差。  

针对各个组件，MAE做了消融实验：  

{% asset_img mae_ablation.png CV无监督学习 %}  

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
- CV：  
[CV入门--关于Vision Transformer](https://www.linsight.cn/a11e2633.html)  
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

【1】Momentum Contrast for Unsupervised Visual Representation Learning https://arxiv.org/abs/1911.05722  
【2】A Simple Framework for Contrastive Learning of Visual Representations https://arxiv.org/abs/2002.05709  
【3】Bootstrap Your Own Latent：A New Approach to Self-Supervised Learning https://arxiv.org/abs/2006.07733  
【4】Emerging Properties in Self-Supervised Vision Transformers https://arxiv.org/abs/2104.14294  
【5】BEiT: BERT Pre-Training of Image Transformers https://arxiv.org/abs/2106.08254  
【6】Masked Autoencoders Are Scalable Vision Learners https://arxiv.org/abs/2111.06377  
【7】Unsupervised Feature Learning via Non-Parametric Instance-level Discrimination https://arxiv.org/abs/1805.01978  

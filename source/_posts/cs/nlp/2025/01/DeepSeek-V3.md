---
title: DeepSeek-V3细节探索
tags:
  - NLP
  - LLM
  - transformer
  - DeepSeek
  - SFT
  - pretrain
categories:
  - CS
  - NLP
  - LLM
abbrlink: a9c496e3
date: 2025-01-29 23:12:34
---

【本文已在同名 微信公众号 / 知乎 / [个人博客linsight.cn](http://www.linsight.cn/) 上线】  

***  

DeepSeek-R1以一己之力正面刚OpenAI和Anthropic。DeepSeek-R1能有这么强力的表现和DeepSeek-V3这个基模型的强大是分不开的。  

{% asset_img perf.png dsv3 %}  

现在就来盘一下DeepSeek-V3的一些细节。（不包括infra部分）  

相关文章链接：  

[DeepSeekMoE](http://www.linsight.cn/44e38c1b.html)  

[DeepSeek-V2](https://www.linsight.cn/83c49df0.html)  

[DeepSeek-R1详解](https://www.linsight.cn/9e4b4e6d.html)  

# MLA  

DeepSeek-V3模型的基础架构和V2一样：  

{% asset_img ds3_archi.png dsv3 %}  

先来看下MLA是怎么做的。（很熟悉MLA的朋友可以跳过这部分）  

## 从MHA出发  

先回顾下标准的MHA。假设 $n_h$ 是注意力头的数量，$d_h$ 是每个注意力头的大小，$\mathbf{h}_{t}\in\mathbb{R}^{d}$ 是第t个输入token。  

MHA首先通过三个投影矩阵 
$W^{Q},W^{K},W^{V}\in\mathbb{R}^{d_{h}n_{h}\times d}$ 获得$\mathbf{q}_t,\mathbf{k}_t,\mathbf{v}_t\in\mathbb{R}^{d_hn_h}$：  

$$\mathbf{q}_t=W^Q\mathbf{h}_t$$  

$$\mathbf{k}_t=W^K\mathbf{h}_t$$  

$$\mathbf{v}_t=W^V\mathbf{h}_t$$  

之后 $\mathbf{q}_t,\mathbf{k}_t,\mathbf{v}_t$ 就会被切成 $n_h$ 份，分别进行注意力计算：  

$$[\mathbf{q}_{t,1};\mathbf{q}_{t,2};...;\mathbf{q}_{t,n_{h}}]=\mathbf{q}_{t}$$  

$$[\mathbf{k}_{t,1};\mathbf{k}_{t,2};...;\mathbf{k}_{t,n_{h}}]=\mathbf{k}_{t}$$  

$$[\mathbf{v}_{t,1};\mathbf{v}_{t,2};...;\mathbf{v}_{t,n_{h}}]=\mathbf{v}_{t}$$  

$$\mathbf{o}_{t,i}=\sum_{j=1}^t\mathrm{Softmax}_j(\frac{\mathbf{q}_{t,i}^T\mathbf{k}_{j,i}}{\sqrt{d_h}})\mathbf{v}_{j,i}$$  

$$\mathbf{u}_t=W^O[\mathbf{o}_{t,1};\mathbf{o}_{t,2};...;\mathbf{o}_{t,n_h}]$$  

其中 $\mathbf{q}_{t,i},\mathbf{k}_{t,i},\mathbf{v}_{t,i}\in\mathbb{R}^{d_{h}}$，$W^O\in\mathbb{R}^{d\times d_hn_h}$。  

在推理的时候，为了加速，会对前面已经计算过的K、V值进行缓存，那么每个token在模型每层就要保存 $2{n}_{h}{d}_{h}$ 个数值。  

那么要减少缓存的量，一个方法就是减少使用的K/V。GQA/MQA就是通过共享参数减少K、V头的数量并重复使用，从而减少了需要缓存的KV的量。  

## MLA  

MLA通过对K和V做low-rank joint compression来压缩KV cache，理论上可以更有效地压缩KV缓存值。  

{% asset_img ds3_MLA.png MLA %}  

下面看下MLA具体是怎么做的。  

在MHA中，K和V是对 $h_t$ 分别用投影矩阵进行变化得到的，而MLA把KV的变换改成使用一个共用的down-projection matrix和两个up-projection matrices进行操作：  

$$\mathbf{c}_t^{KV}=W^{DKV}\mathbf{h}_t$$  

$$\mathbf{k}_t^C=W^{UK}\mathbf{c}_t^{KV}$$  

$$\mathbf{v}_t^C=W^{UV}\mathbf{c}_t^{KV}$$  

$\mathfrak{c}_t^{KV}\in\mathbb{R}^{d_c}$ 就是K和V的compressed latent vector，这也是推理时要缓存的部分。  

这里相当于把MHA中的 $W^{K},W^{V}$ 拆成两个矩阵：  

$$\mathbf{k}_t=W^K\mathbf{h}_t\rightarrow W^{UK}W^{DKV}\mathbf{h}_t$$  

$$\mathbf{v}_t=W^V\mathbf{h}_t\rightarrow W^{UV}W^{DKV}\mathbf{h}_t$$  

$d_c$ 是KV的压缩维度，让 $d_c\ll d_hn_h$，就可以大大减少需要推理时需要缓存的数据量。  

再看回attention计算，在得到q、k、v之后，会计算权重矩阵并获得最终注意力输出结果：  

$$\operatorname{Attention}(Q,K,V)=\operatorname{softmax}(\frac{Q^TK}{\sqrt{d}})V$$  

而 $Q^TK=H^T(W^Q)^TW^{UK}C$，因此 $W^{UK}$ 可以被吸收进 $W^{Q}$ 中，而不用在计算时显式算出K，只需调整 $W^Q$ 的shape后直接输入C即可。同理 $W^{UV}$ 可以被吸收进 $W^{O}$。实操上，这样的矩阵合并可能会带来一些精度损失，这是一个值得注意的问题。  

此外，MLA还对Q也做了low-rank compression，跟对K、V的操作类似：  

$$\mathbf{c}_t^Q=W^{DQ}\mathbf{h}_t,\\\mathbf{q}_t^C=W^{UQ}\mathbf{c}_t^Q,$$  

关于对Q进行压缩的原因，虽然V2原文说的是为了减少训练时的activation，但是两个矩阵所得的activation按道理应该比直接使用单个投影矩阵还要多一些。这里Q的压缩更可能是为了减少参数量和梯度，而非激活值。  

## 兼容RoPE  

到这里似乎MLA已经完成了，即减少了缓存的量，也不用引入其他overhead（两个up-projection matrices都不用算了）。  

但是实际上还有一个问题没有解决：位置编码使用的是RoPE，而RoPE是通过在Q、K上乘一个旋转矩阵来编码位置的。  

而在上面MLA的设计中，已经没有显式计算K了，而RoPE也不能加在latent vector上。一个方法是重新把K和V显式计算出来，但是这样计算量就会增加，MLA的推理加速效果就会打折扣。  

针对这个问题，解决方案是使用decoupled RoPE：使用额外的multi-head queries $\mathbf{q}_{t,i}^R\in\mathbb{R}^{d_h^R}$ 和一个shared key $\mathbf{k}_t^R\in\mathbb{R}^{d_h^R}$ 来携带RoPE的位置信息，$d_h^R$ 是decoupled queries的维度。  

新增的q和k维度使用常规的RoPE计算，用于携带位置信息；而原来的维度依然使用低秩分解的方式计算，最后再计算attention的时候两个部分拼接起来。  

最终完整的MLA计算如下  

{% asset_img MLA_formula.png MLA公式 %}  

蓝框中的部分就是推理时需要缓存的内容。  

MLA所需的缓存量约等于组数为2.5的GQA：  

{% asset_img MLA_cache.png MLA缓存量 %}  

# MoE  

## 基础结构

DeepSeek-V3的MoE结构设计和DeepSeekMoE/DeepSeek-V2基本一致。和V2相比，有一些设置是一样的：  

- 初始化 standard deviation = 0.006  
- 128个attention head，head size = 128  
- KV的compression dimension dc = 512  
- Q的compression dimension dc' = 1536  
- decoupled queries and key per head dimension = 64  

此外，也有一些具体设置和V2相比有变化：  
- layers = 61（比V2多1层）  
- hidden dimension = 7168（比V2的5120更大）  
- 前3层不使用MoE  
- 1个共享专家 + 8/256个路由专家，专家大小为2048（更多专家，专家维度更大）  
- 每个token最多只会被分发到4个节点  
- 总参数671B，激活参数37B  
- gating在计算affinity score的时候先用sigmoid函数，再在选定的分数上进行归一化，而V2是直接使用softmax  

V2的总参数为236B，激活参数为21B；而V3的总参数为671B，激活参数为37B。可以看到相比V2，V3多的参数主要在模型宽度和专家数量，而且MoE的激活更为稀疏。  

## 负载平衡  

1、Auxiliary-Loss-Free Load Balancing  

先看下V3的MoE计算：  

$$\mathbf{h}_t^{\prime}=\mathbf{u}_t+\sum_{i=1}^{N_s}\mathrm{FFN}_i^{(s)}\left(\mathbf{u}_t\right)+\sum_{i=1}^{N_r}g_{i,t}\mathrm{FFN}_i^{(r)}\left(\mathbf{u}_t\right)$$  

第一项来自残差连接，第二项是共享专家的输出，第三项是路由专家的输出；Ns是shared expert的数量，Nr是routed expert的数量，DeepSeek-V3中Ns=1，Nr=128。  

$$g_{i,t}=\frac{g_{i,t}^\prime}{\sum_{j=1}^{N_r}g_{j,t}^\prime}$$  

g'只保留top Nr个（DeepSeek-V3中Nr=8），其他都置零了。  

$$g_{i,t}^{\prime}=\begin{cases}s_{i,t},&s_{i,t}\in\mathrm{Topk}(\{s_{j,t}|1\leqslant j\leqslant N_r\},K_r)\\0,&\text{otherwise}&&\end{cases}$$  

$$s_{i,t}=\mathrm{Sigmoid}\left(\mathbf{u}_t{}^T\mathbf{e}_i\right)$$  

Kr是activated routed expert的数量。  

之前的版本使用auxiliary loss来对top affinity score的分配不平衡进行惩罚，以此来缓解专家分配不平衡的问题。由于auxiliary loss的设计并不关注模型的效果，因此过大的权重会对模型的训练效果产生损害。  

为了避免模型效果的损失，DeepSeek-V3不使用auxiliary loss来平衡负载，而是在affinity score上加了一个bias term，这个bias term和expert是一一对应的：  

$$g_{i,t}^{\prime}=\begin{cases}s_{i,t},&s_{i,t}+b_i\in\mathrm{Topk}(\{s_{j,t}+b_j|1\leqslant j\leqslant N_r\},K_r)\\0,&\text{otherwise}&\end{cases}$$  

这个bias term只用于routing，不用于和FFN的结果相乘输出专家的feature vector。在每个训练step后，如果一个expert的负载过大了，就会把对应的bias term减小𝛾，反之则把bias term的数值增大𝛾。𝛾是个超参，控制负载平衡系统的变化速度。  

2、Complementary Sequence-Wise Auxiliary Loss  

虽然加了bias term控制负载均衡，但是为了防止极端不平衡状况的出现，还是额外加了一个Auxiliary Loss。  

complementary sequence-wise balance loss是这么算的：  

$$\mathcal{L}_\mathrm{Bal}=\alpha\sum_{i=1}^{N_r}f_iP_i$$  

其中  

$$P_i=\frac{1}{T}\sum_{t=1}^Ts_{i,t}^{\prime}$$  

s'其实就是归一化的affinity score  

$$s_{i,t}^\prime=\frac{s_{i,t}}{\sum_{j=1}^{N_r}s_{j,t}}$$  

另外  

$$f_i=\frac{N_r}{K_rT}\sum_{t=1}^T\mathbb{1}\left(s_{i,t}\in\mathrm{Topk}(\{s_{j,t}|1\leqslant j\leqslant N_r\},K_r)\right)$$  

求和部分其实就是某个token是否选择了expert i。训练中𝛼 = 0.0001。  

fi是不可导的，Pi是可导的。  

在完美负载平衡的情况下，affinity score均匀分配，每个expert的得分相同，那么有  

$$P_i=\frac{1}{T}\times T\times \frac{1}{N_r}=\frac{1}{N_r}$$  

$$f_i=\frac{N_r}{K_rT}\sum_{t=1}^T\frac{K_r}{N_r}=1$$  

那么  

$$\mathcal{L}_\mathrm{Bal}=\alpha\sum_{i=1}^{N_r}\frac{1}{N_r}=\alpha$$  

complementary sequence-wise balance loss其实就是DeepSeekMoE中的expert-level balance loss。  

而在极端不平衡的情况下，比如所有token都选择了前Kr个expert激活，那么对于激活的expert i，有  

$$P_i=\frac{1}{T}\times T\times 1=1$$  

$$f_i=\frac{N_r}{K_rT}\sum_{t=1}^T1=\frac{N_r}{K_r}$$  

那么就有  

$$\mathcal{L}_\mathrm{Bal}=\alpha\sum_{i=1}^{K_r}\frac{N_r}{K_r}=\alpha N_r$$  

3、Node-Limited Routing  

在前面的基础上，最后还加了一个机制，限制每个token最多只能分发到M个节点上，而节点的选择是基于每个节点上的affinity score的总和的。  

举个例子，在Kr=8，M=4的情况下：  

- 如果8个得分最高的专家都分布在不同的node，那么只有top4个专家会被激活，其余的专家虽然得分排在top Nr，但是由于激活节点的限制，不会被使用；  
- top8个专家分配在5个节点上：  
  - 节点1：0.1,0.1
  - 节点2：0.1,0.1
  - 节点3：0.1,0.1
  - 节点4：0.25
  - 节点5：0.15  
  在这样的情况下，虽然节点5上的专家得分是第二高的，但是由于它所在的节点的得分总和不高，因此不会被激活  

## No Token-Dropping  

由于前面的几个负载平衡策略基本上已经可以保持完全的负载平衡，因此DeepSeek-V3就不再使用token dropping的策略了。  

# Multi-Token Prediction  

Multi-Token Prediction（MTP），顾名思义，在前向计算的时候一步可以预测 >1 个token。  

这样的多token预测策略可以在训练中使用，提升模型的远距离的理解能力；也可以用在推理中，加速inference输出，不过推理加速算是副产品了。  

## 原始的MTP  

DeepSeek-V3中使用的MTP参考了24年4月的《Better & Faster Large Language Models via Multi-token Prediction》，因此先来了解下这个工作。  

1、MTP方案  

标准的语言建模使用next-token prediction，基于第1~t个token预测第t+1个token，loss是这样的：  

$$\begin{aligned}L_1=-\sum_t\log P_\theta(x_{t+1}\mid x_{t:1})\end{aligned}$$  

和NTP不同，MTP要求模型在每一步要预测n个token，即第t+1~t+n个token，loss就写作这样：  

$$\begin{aligned}L_n=-\sum_t\log P_\theta(x_{t+n:t+1}\mid x_{t:1})\end{aligned}$$  

那么怎么在一步内预测多个token呢？论文里的做法是利用多个output head，每个head负责预测一个token。下面这个图就是当n=4的例子：  

{% asset_img mtp_example.png dsv3 %}  

head1根据token 1~t预测token t+1，这和标准的NTP任务是一样的。而head2则是根据token 1~t预测token t+2，head3和head4也是类似的，分别预测token t+3和token t+4。  

所有的这些head共享同一个主干transformer fs的输出，单独的output head参数fh，另外还共享着unembedding matrix fu。第i个head的输出可以写作：  

$$P_\theta(x_{t+i}\mid x_{t:1})=\operatorname{softmax}(f_u(f_{h_i}(f_s(x_{t:1}))))$$  

由于使用了多个输出头，计算的时候就多了额外的参数和激活值，因此相比NTP，MTP会使用更多的memory。为了缓解这个问题，文中给出串行计算（而不是并行）这些output head的forward和backward的方法：  

{% asset_img mtp_order.png dsv3 %}  

多个head回传的梯度可以在共享的transformer主干处积累，这样就把增加的memory量从O(nV+d)降到了O(V+d)。  

正常推理的时候就只使用head1，其他的head就可以不用了，这和标准的推理形式是一致的。但是如果在推理时使用类似[投机解码](https://mp.weixin.qq.com/s/wOIGg9pJCXQxz3GgXApUQw?token=1318369845&lang=zh_CN)或者[MEDUSA](https://mp.weixin.qq.com/s/e3Cn_zbPlbRUUd4-ngSLTg?token=1318369845&lang=zh_CN)这样的推理加速方案，其他的head2、head3、head4都可以直接派上用场，作为draft model使用。  

2、MTP的效果  

MTP的效果怎么样呢？论文在>=91B的代码数据上训练了从0.3B到13B参数量的模型，对比NTP和MTP的效果。在各个模型上，MTP相比NTP，在两个经典代码评测集MBPP和human-eval的效果对比如下：  

{% asset_img mtp_code_result.png dsv3 %}  

随着模型规模的提升，MTP的效果逐步提升，相比NTP的收益越来越大。  

文中更多的预训练实验结果还有一些发现：  

- 随着训练的epoch数的提升，MTP的收益有所收窄，不过还是有一些的；不过现在通用预训练数据量基本够大，不太可能出现超过1个epoch的情况  
- MBPP和human-eval最佳的n为4，不过在APPS/Intro上n=6效果更好，n的设置可能和数据相关  

{% asset_img mtp_exps.png dsv3 %}  

另一个需要了解的问题是，MTP在预训练上有效，那么对于在MTP上预训练的模型，微调时n应该设置为多少。下图中，n为预训练中每步预测的token数，n'为SFT训练中每步预测的token数，红线就是预训练和SFT都是NTP，黑色虚线预训练用MTP，SFT用NTP，而浅蓝色虚线是与预训练和SFT都用MTP：  

{% asset_img mtp_sft.png dsv3 %}  

结果上看，对MTP预训练模型使用NTP微调的效果是最好的。  

前面的评测都是在code相关的任务上进行的，而在一些NLP benchmark上，MTP的效果就不如NTP：  

{% asset_img mtp_nlp_benchmark.png dsv3 %}  

这里有几个可能的原因：  

- 可能需要更大的模型让MTP发挥效果  
- 概率类或者选择题类的评测并不能很好地评估MTP学到的更远距离依赖的能力  

针对第二个猜测，另外使用了8个评测指标为ROUGH-L的任务，这些任务要求模型输出较长的文本（比如摘要）。在这类任务上，MTP模型的效果就比较好了  

{% asset_img mtp_summary.png dsv3 %}  

3、结构上的变体  

上面的MTP设计是使用n个output head，每个head「独立」地进行token预测，逻辑上这些输出头是并行的。实际上这些output head的设计可以有多种变化，比如他们之间是并行还是串行，每个头的层数和类型。针对这些变体，研究人员也做了实验，各种变体的效果如下：  

{% asset_img mtp_archi.png dsv3 %}  

其中parallel就是前面的独立方式。causal就是head2的输出是以head1的输出为基础的，而anticausal则是先预测n个token中最后一个，然后第n-1个output head再根据它的结果输出，以此类推，第1个token反而是最后输出的，并且参考后面的所有token。  

除此之后，还有一种变体，那就是每个output head维护自己的unembedding matrix，单独训练，不过这么一来参数量和训练的内存需求就会增大不少。  

## DeepSeek-V3中的MTP  

说回DeepSeek-V3。  

DeepSeek-V3中的MTP在原实现的基础上做了一些细化和改进。  

1、MTP module  

首先是MTP module的设计。DeepSeek-V3中，多个预测的token是有causal关系的，也就是output2会根据output1的特征进行输出。  

前一个MTP module的输出向量经过RMSNorm之后和embedding layer的feature拼接在了一起，再经过transformer block进行输出：  

{% asset_img ds3_mtp_module.png dsv3 %}  

这个图值得细细看，有几个要注意的地方：  

- MTP module原始的输入来自embedding layer，而不是主干transformer model的最后一层输出  
- main model的output head是预测第t+1个token的，第一个MTP是预测第t+2个token的，第二个MTP是预测第t+3个token的  
- output head的参数是共享的，也就是每个MTP module中，预测不同token的能力主要是由linear projection和transformer block部分的参数习得；使用参数共享的考虑和《EAGLE: speculative sampling requires rethinking feature uncertainty》有些相近，不过目的不同，EAGLE是为了加速推理，而DeepSeek-V3是为了优化MTP的训练效果  

在推理的时候MTP module可以完全不使用，回到正常的NTP的方式来生成结果。当然如果要考虑推理加速，这些module也可以用上。  

2、训练的损失函数  

MTP的损失是作为附加损失和main model一起训练的。几个MTP module的损失就是取平均，再通过权重λ加入到总loss里：  

$$\mathcal{L}_{MTP }^{k}=CrossEntropy\left(P_{2+k: T+1}^{k}, t_{2+k: T+1}\right)=-\frac{1}{T} \sum_{i=2+k}^{T+1} \log P_{i}^{k}\left[t_{i}\right]$$

$$\mathcal{L}_{MTP}=\frac{\lambda}{D} \sum_{k=1}^{D} \mathcal{L}_{MTP}^{k}$$  

实际使用中，MTP深度D=1，也就是除了主模型的output head，只有一个MTP module。  

# 数据构建  

数据建设上，DeepSeek-V3没有给出特别详细的内容。相比V2，V3强调了几点变化：  

- 增加了数学和代码数据的比例  
- 增加中英文之外其他语言的覆盖  
- 强调了去重了保留多样性  

最终获得了14.8T的训练数据。  

此外，文中还透露了以下几点。  

## document packing  

目前大部分的模型都是采用concat-then-split的方式，把文档分割成训练样本。这样的方式可以避免padding，从而提高训练效率。但是频繁的文档切分也会带来问题：训练数据的实际有效上下文缩短；被分割的文档缺失上下文信息，让模型在生成时需要靠想象补充缺失的部分，从而导致幻觉的产生。  

DeepSeek-V3就参考《Better & Faster Large Language Models via Multi-token Prediction》的做法Best-fit Packing，优化document packing。  

那么简单介绍一下best-fit packing。  

首先，假设模型的训练窗口长度时L，那么对于长度大于L的文档，首先就要切成长度为L的小块。这一步是无论什么训练策略都要做的，即使不进行任何拼接而对每个文档单独进行padding，也需要切分过长的文档。  

那么接下来的任务就是把这些切分出来的文档chunk拼接成长度<=L的训练样本，并且样本数量越少越好。样本数量越少，意味着数据密度越高，padding越少。

到这里，其实就转化成了一个背包问题。但是背包问题是NP-hard的，没法直接得到最优解，因此可以借用已有的高效近似解法，First-Fit-Decreasing (FFD) 和Best-Fit-Decreasing (BFD) 来获得近似解。  

算法如下：  

{% asset_img BFD_FFD.png dsv3 %}  

C就是文档集合，l(c)是文档的长度。每一步拼接中，FFD是对文档长度降序排序，然后选择第一个fit的文档加入；BFD是对文档长度降序排序，然后选择让bin的剩余空间最小的文档。实践中，使用segment tree实现BFD上的快速搜索。  

直观看下best-fit packing和concat-the-split的对比：  

{% asset_img best_fit_packing.png dsv3 %}  

那么best-fit packing的会带来多少的额外padding呢？由于实际训练数据大部分其实不是很长，所以更容易pack得很紧密。在2k和8k的训练窗口下，best-fit packing和concat-then-split相比基本没有可感知的训练样本增加，小于万分之一，并且随着训练窗口增大，这个差距还在减小：  

{% asset_img packing_padding.png dsv3 %}  

最终训练效果上，相比concat-then-split，best-fit packing在阅读理解、NLI、Context Following上有明显的提升：  

{% asset_img bfp_perf1.png dsv3 %}  

{% asset_img bfp_perf2.png dsv3 %}  

## Fill-in-Middle（FIM）  

为什么需要FIM的训练方式。我们知道GPT模型相比Bert类模型，有更高的训练效率；而从左到右自由生成的方式也使得GPT模型能够应用在更多场景，上限更高。但是传统的left-to-right的训练方式也有限制：如在代码补全的场景，需要模型同时兼顾上文和下文，对中间部分的内容进行补全，这种情况下left-to-right的训练方式就无法提供有效的信息，因为看不见下文。  

为了解决这个问题，可以对模型的输入数据做一个transformation：把原本顺序正常的文档，切分成三部分，即prefix、middle和suffix，并把middle部分放到最后面。  

document -> (prefix; middle; suffix)  -> (prefix; suffix; middle)  

训练的时候，模型需要根据给定的上文prefix和下文suffix，来生成中间的部分。  

DeepSeek-V3中有10%的数据采用了FIM的格式变换，使用PSM的顺序。  

## tokenizer和token boundary  

DeepSeek-V3的tokenizer除了加入其他语言的token之外，还增加了包含标点符号和line break的token。这些新加的token可能会引入prompt boundary的问题。  

什么是prompt boundary？先来看一个例子。用stabilityai的stablelm-base-alpha-3b模型来给这句话进行补全：  

```python  
'The link is <a href="http:'
```  

正常来说，我们希望补全的结果是一个格式正确的链接。实际生成的结果是  

```python  
'The link is <a href="http: //www.google.com/search?q'
```  

注意"http:"后面多了个空格，这显然是无效的。这就有点奇怪了，按道理这样的格式在训练数据里是足够多的，模型没有道理学习不到有效的格式。  

重新试一下生成，这次把输入prompt最后的冒号去掉

```python  
'The link is <a href="http'
```  

再让模型补全：  

```python  
'The link is <a href="http://www.youtube.com/v/s'
```  

这下就可以正常生成了。  

看来问题就出在 : 这里。把第一个prompt的token打印出来看看：  

{% asset_img token_boundary_1.png dsv3 %}  

再看看一个正常链接的token：  

{% asset_img token_boundary_2.png dsv3 %}  

发现 :// 是被当成一个token处理的。  

大多数的tokenizer都是greedy tokenization的策略。训练时可以看到完整的文本，因此所有链接中，:// 都被当做一个token处理，也就是模型在训练时几乎没有见过 : token后面跟诊 // token的情况，这就导致如果prompt中给了 : ，模型就会输出错误的结果。  

词表中有很多以 : 开头的token，它们在训练时都被当做一个token处理了：  

{% asset_img token_boundary_3.png dsv3 %}  

也就是说，对于这34个token，模型几乎没有训练过它们的冒号被拆分出来的情况，那在推理时自然也就无法正常生成。  

这个情况不仅存在于和 : 相关的token中，而是广泛存在于整个词表。  

这个现象可以称之为token boundary bias。缓解token boundary bias大致有两个方法。  

第一个方法叫做token healing。既然输入prompt中最后一个token有可能是训练数据的token中的一部分，那么就先把最后的一个token删去，然后再在后续的生成结果中，选择包含被删去字符的token作为生成结果。  

比如前面的链接生成，输入的prompt是

```python  
'The link is <a href="http:'
```  

tokenization之后 : 是最后一个token，那么就把它去掉。假设后续模型生成的top k个结果是  

```python  
s
:\\
google
```  

那么就选择包含 : 的第二个结果。  

token healing方法在guidance-ai中有实现。  

另外一个方法是subword regularization，就是在训练时，随机拆分已经分好的token，获得sub-optimal tokenization的结果。这些结果不是最好的切分结果，但是可以帮助模型缓解token boundary bias。  

DeepSeek-V3用的就是第二种方法。  

# 训练设置  

## 预训练  

DeepSeek-V3有多阶段的预训练。  

第一阶段（基础通用预训练）：  

- 长度4k  
- gradient clipping norm = 1.0  
- 前2k步中，lr从0整张到2.2e-4，然后保持constant lr训练10T token  
- 在之后的4.3T token，lr用cosine schedule下降到2.2e-5  
- 在之后的333B，保持lr=2.2e-5  
- 在最后的167B，lr切换到7.3e-6  
- batch size在最初的469B数据，逐渐从3072提升到15360，之后保持15360  
- expert分配在8个节点64个GPU上  
- 负载平衡速度𝛾在最初的14.3T设为1，之后设为0  
- MTP loss weight在前10T token 𝜆 = 0.3，后4.8T设为0.1  

第二阶段（长窗口预训练）：  

- 窗口长度从4k提升到32k，lr=7.3e-6，batch size = 1920，训练1000步  
- 窗口长度从32k提升到128k，lr=7.3e-6，batch size = 1920，训练1000步  

## 对齐  

SFT：  

- SFT数据共有1.5M条，训练2个epoch  
- lr从5e-6降到1e-6，cosine schedule  
- 使用sample masking strategy，各个sample不互相看见  
- reasoning data  
  - 部分来自DeepSeek-R1  
  - 对于每个领域，比如代码，数学，通过SFT + RL训练领域专家模型，用于生成对应领域的数据  
  - 主要有两类格式：\<problem, original response\>，\<system prompt, problem, R1 response\>  
- non-reasoning data  
  - 用DeepSeek-V2.5来生成response  
  - 人力来对数据进行检查和更正  

RL:  

- 使用Group Relative Policy Optimization  

# 小结  

- 细节部分有不少优化，包括MTP，tokenizer，document packing等  
- MoE还是延续之前的做法，所谓的新的负载平衡应该没有特别大的影响  
- MLA + MTP在降低推理成本上有应该有比较重要的地位  
- 实际上infra做了大量的工作，用于提升训练效率，这块有机会再盘  
- 总的来说，DeepSeek-V3是算法和工程的优秀实践；踏实把每个细节做好最重要  

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

【1】DeepSeek-V3 Technical Report https://arxiv.org/abs/2412.19437v1  
【2】Better & Faster Large Language Models via Multi-token Prediction https://arxiv.org/abs/2404.19737  
【3】大模型推理加速-投机解码，https://zhuanlan.zhihu.com/p/699670010  
【4】大模型推理加速-MEDUSA，https://zhuanlan.zhihu.com/p/703461293  
【5】DeepSeek-V2和MLA，https://zhuanlan.zhihu.com/p/708622695  
【6】DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model，https://arxiv.org/abs/2405.04434  
【7】理解Attention:从起源到MHA,MQA和GQA，https://zhuanlan.zhihu.com/p/686149289  
【8】MoE模型的前世今生，http://www.linsight.cn/44e38c1b.html  
【9】Fewer Truncations Improve Language Modeling，https://arxiv.org/abs/2404.10830  
【10】代码大模型(一)--业界现状，https://www.linsight.cn/a0b50049.html#fim  
【11】The Art of Prompt Design: Prompt Boundaries and Token Healing，https://medium.com/towards-data-science/the-art-of-prompt-design-prompt-boundaries-and-token-healing-3b2448b0be38  

---
title: DeepSeek-V2和MLA
abbrlink: 83c49df0
date: 2024-07-12 20:54:22
tags:
  - NLP 
  - LLM 
  - transformer 
  - 技术报告
  - DeepSeek
  - MLA
  - GQA
  - MoE 
categories:
  - CS
  - NLP
  - LLM
---

【本文已在同名 微信公众号 / 知乎 / [个人博客linsight.cn](http://www.linsight.cn/) 上线】  

***  

DeepSeek-V2发布之后，其低价策略在国产大模型界掀起一阵降价风。  

DeepSeek-V2能做到低成本推理的一个原因就是使用了MLA，使得推理时缓存量大大减小。  

本篇来看下MLA以及DeepSeek-V2一些其他细节。  

DeepSeek-V2除了一个总参数量为236B的主模型外，还有一个方便开源研究的DeepSeek-V2-Lite，总参数量为15.7B，这个在最后介绍。  

# 模型  

DeepSeek-V2介绍：  
- 总参数量为236B参数，激活21B  
- 支持128k长度  
- 相比DeepSeek-67B，DeepSeek-V2节省42.5%的训练成本和93.3%的推理KV cache需求，而最大throughput则是前者的5.76倍  

DeepSeek-V2和其他一些大模型在MMLU上的效果以及激活参数量的对比如下图  

{% asset_img intro.png DeepSeek-V2 %}  

可以看到DeepSeek-V2以更少的激活参数量达到了接近70B dense模型水平的效果。  

DeepSeek-V2模型结构如下图

{% asset_img model.png 模型 %}  

同V1版本一样，V2在MoE层使用了fine-grained expert和shared expert（或者叫DeepSeekMoE结构）（可参考《[MoE模型的前世今生](http://www.linsight.cn/44e38c1b.html)》）。而V2在结构上最重要的变动就是在注意力层使用了Multi-Head Latent Attention（MLA）。  

## MLA  

MLA是DeepSeek-V2提升推理效率，减低KV cache需求的关键。  

（关于KV cache和MHA/GQA/MQA的对比，可参考[《理解Attention:从起源到MHA,MQA和GQA》](https://zhuanlan.zhihu.com/p/686149289)）  

1、从MHA出发  

先回顾下标准的MHA。假设 $n_h$ 是注意力头的数量，$d_h$ 是每个注意力头的大小，$\mathbf{h}_{t}\in\mathbb{R}^{d}$ 是第t个输入token。  

MHA首先通过三个投影矩阵 $W^{Q},W^{K},W^{V}\in\mathbb{R}^{d_{h}n_{h}\times d}$ 获得$\mathbf{q}_t,\mathbf{k}_t,\mathbf{v}_t\in\mathbb{R}^{d_hn_h}$：  

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

在推理的时候，为了加速会对已经计算过的K、V值进行缓存，那么每个token每层就要保存 $2{n}_{h}{d}_{h}$ 个数值。  

而GQA/MQA通过减少K、V头的数量并重复使用，减少了需要缓存的KV的量。  

{% asset_img GQA.png GQA %}  

MQA相当于组数为1的GQA，它在推理时，每层每个token所需要缓存的量为 $2{d}_{h}$，相比MHA有了1~2两个数量级的减少。可以说这是这种减少KV组数的思路的极限了。但是GQA/MQA毕竟相当于减少了注意力头的数量，在效果上就会有一定的损失。  

DeepSeek-V2报告里也对此进行了验证：用1.33T token的数据分别训练了MHA、GQA、MQA的7B模型，在4个benchmark的对比如下  

{% asset_img GQA_compare_MHA.png MHA/GQA/MQA效果对比 %}  

相比MHA，MQA效果损失最大，GQA次之。  

2、MLA  

MLA通过对K和V做low-rank joint compression来压缩KV cache，理论上可以更有效地压缩KV缓存值。  

{% asset_img MLA.png MLA %}  

下面看下MLA具体是怎么做的。  

在MHA中，K和V是对 $h_t$ 分别用投影矩阵进行变化得到的，而MLA把KV的变换改成使用一个共用的down-projection matrix和两个up-projection matrices进行操作：  

$$\mathbf{c}_t^{KV}=W^{DKV}\mathbf{h}_t$$  

$$\mathbf{k}_t^C=W^{UK}\mathbf{c}_t^{KV}$$  

$$\mathbf{v}_t^C=W^{UV}\mathbf{c}_t^{KV}$$  

$\mathfrak{c}_t^{KV}\in\mathbb{R}^{d_c}$ 就是K和V的compressed latent vector，这也是推理时要缓存的部分。

这里相当于把MHA中的 $W^{K},W^{V}$ 拆成两个矩阵：  

$$\mathbf{k}_t=W^K\mathbf{h}_t\rightarrow\mathbf{k}_tW^{UK}W^{DKV}\mathbf{h}_t$$  

$$\mathbf{v}_t=W^V\mathbf{h}_t\rightarrow\mathbf{k}_tW^{UV}W^{DKV}\mathbf{h}_t$$  

$d_c$ 是KV的压缩维度，让 $d_c\ll d_hn_h$，就可以大大减少需要推理时需要缓存的数据量。  

看回attention计算，在得到q、k、v之后，会计算权重矩阵并获得最终注意力输出结果：  

$$\operatorname{Attention}(Q,K,V)=\operatorname{softmax}(\frac{Q^TK}{\sqrt{d}})V$$  

而 $Q^TK=H^T(W^Q)^TW^{UK}C$ 中，因此 $W^{UK}$ 可以被吸收进 $W^{Q}$ 中，而不用在计算时显式算出K，只需调整 $W^Q$ 的shape后直接输入C即可。同理 $W^{UV}$ 可以被吸收进 $W^{O}$。实操上，这样的矩阵合并可能会带来一些精度损失，这是一个值得注意的问题。  

此外，DeepSeek-V2还对Q也做了low-rank compression，跟对K、V的操作类似：  

$$\mathbf{c}_t^Q=W^{DQ}\mathbf{h}_t,\\\mathbf{q}_t^C=W^{UQ}\mathbf{c}_t^Q,$$  

关于对Q进行压缩的原因，这里原文说的是为了减少训练时的activation。但是两个矩阵所得的activation按道理应该比直接使用单个投影矩阵还要多一些，因此此处有点疑问。苏神在[《缓存与效果的极限拉扯：从MHA、MQA、GQA到MLA》](https://kexue.fm/archives/10091)中也认为Q的压缩更多是减少了参数量和梯度，而非激活值。  

3、兼容RoPE  

到这里似乎MLA已经完成了，即减少了缓存的量，也不用引入其他overhead（两个up-projection matrices都不用算了）。  

但是实际上还有一个问题没有解决。同大部分其他大模型一样，DeepSeek-V2使用的位置编码是RoPE，而RoPE是通过在Q、K上乘一个旋转矩阵来编码位置的。相关内容可参考[《理解LLM位置编码:RoPE》](https://zhuanlan.zhihu.com/p/684072868)。  

而在上面MLA的设计中，已经没有显式计算K了，而RoPE也不能加在latent vector上。一个方法是重新把K和V显式计算出来，但是这样计算量就会增加，MLA的推理加速效果就会打折扣。  

针对这个问题，DeepSeek-V2提出decoupled RoPE的解决方案，使用额外的multi-head queries $\mathbf{q}_{t,i}^R\in\mathbb{R}^{d_h^R}$ 和一个shared key $\mathbf{k}_t^R\in\mathbb{R}^{d_h^R}$ 来携带RoPE的位置信息，$d_h^R$ 是decoupled queries的维度。  

新增的q和k维度使用常规的RoPE计算，用于携带位置信息；而原来的维度依然使用低秩分解的方式计算，最后再计算attention的时候两个部分拼接起来。  

最终完整的MLA计算如下  

{% asset_img MLA_formula.png MLA公式 %}  

蓝框中的部分就是推理时需要缓存的内容。  

MLA所需的缓存量约等于组数为2.5的GQA：  

{% asset_img MLA_cache.png MLA缓存量 %}  

在效果上，DeepSeek-V2分别对比了MLA和MHA的16B模型（训练1.33T token）和250B模型（训练420B token）：  

{% asset_img MLA_perf.png MLA效果 %}  

在4个benchmark上看，MLA基本都比要比MHA要好。这个结果还是有些出乎意料的，这妥妥就是免费的午餐，在节省KV cache的同时还能获得效果提升。感觉MLA效果还有待进一步验证。  

## 负载均衡  

负载均衡策略是MoE永远要考虑的问题，对效果和效率都有很大的影响。  

1、Device-Limited Routing  

在使用专家并行的情况下，每个token所需的通讯量取决于它的target expert所在的device数。而由于使用了fine-grained expert，这个device数量可能会比较大，就会导致通讯成为瓶颈。  

因此DeepSeek-V2会基于target expert的得分，限制最多所能发送的device数量M。实践中，发现M≥3就能达到和不限制相同的效果了。  

2、Expert-Level Balance Loss  

和DeepSeekMoE V1一样，专家级的负载均衡如下：  

$$\begin{aligned}
\mathcal{L}_{\mathrm{ExpBal}}& =\alpha_1\sum_{i=1}^{N_r}f_iP_i
\end{aligned}$$  

$$\begin{aligned}
f_{i}& =\frac{N_{r}}{K_{r}T}\sum_{t=1}^T\mathbb{1}(\text{Token }t\text{ selects Expert }i)
\end{aligned}$$  

$$\begin{aligned}
P_{i}& =\frac1T\sum_{t=1}^Ts_{i,t} 
\end{aligned}$$  

$\alpha_1$ 是expert-level balance factor，T为token数。  

3、Device-Level Balance Loss  

在使用专家并行的情况下，专家被分成D个组$\{\mathcal{E}_1,\mathcal{E}_2,...,\mathcal{E}_D\}$，各个组之间的负载均衡损失：  

$$\mathcal{L}_\mathrm{DevBal}=\alpha_2\sum_{i=1}^Df_i^{\prime}P_i^{\prime}$$  

$$f_i'=\frac1{|\mathcal{E}_i|}\sum_{j\in\mathcal{E}_i}f_j$$  

$$P_i'=\sum_{j\in\mathcal{E}_i}P_j$$  

$\alpha_2$ 是device-level balance factor。  

4、Communication Balance Loss  

前面对token发送target专家的总device数做了限制，但是依然有可能出现某些device【接收】的token数量不平衡的情况，这同样会影响通讯效率。  

因此这里还加了一个communication balance loss：  

$$\mathcal{L}_{\mathrm{CommBal}}=\alpha_3\sum_{i=1}^Df_i^{\prime\prime}P_i^{\prime\prime}$$  

$$f_i^{\prime\prime}=\frac D{MT}\sum_{t=1}^T1(\text{Token t is sent to Device i})$$  

$$P_i''=\sum_{j\in\mathcal{E}_i}P_j$$  

$\alpha_3$ 是communication balance factor。  

5、Token-Dropping Strategy  

前面虽然加了各种负载均衡loss，但是实际上还是没有办法保证能够得到严格的负载均衡，因此在训练时还引入了一个device-level token-dropping strategy，对每个device设定一个capacity，如果在一个batch中，某个device所处理的token达到了容量，那么后面再分配到这个device的token就都会被drop。  

另外为了保证模型能够处理到完整的sequence，训练时有10%的sequence保证永远不drop任何token。  

注意这个策略只在训练时时候，推理时不会给device设置容量限制。  

# 训练  

DeepSeek-V2使用和DeepSeek 67B一样的tokenizer，BBPE训练出来的100k词表。  

模型的所有预训练数据约有8.1T，其中12%是中文。  

## 超参  

1、模型超参  

- layer num = 60  
- hidden size = 5120  
- initialization standard deviation = 0.006  
- attention head数量 = 128，每个attention head size = 128  
- KV压缩维度 $d_c=512$  
- Q压缩维度 $d_c'=1536$  
- decoupled queries and key per head dimension = 64  
- 2个共享专家 + 6/160路由专家  
- 专家大小 = 1536  
- 总参数236B，激活参数21B  

2、预训练超参  

- AdamW：beta_1 = 0.9，beta_2 = 0.95，weight_decay = 0.1  
- lr scheduler：warmup-and-step-decay，warmup = 2k step，最大lr = 2.4E-4；在训练进度60%和90%的时候lr乘以0.316  
- gradient clipping norm = 1.0  
- batch size scheduling strategy：在训练的前225B，batch size逐渐从2304增大到9216，之后保持不变  
- maximum sequence length = 4k  
- 负载均衡权重：$\alpha_1=0.003$，$\alpha_2=0.05$，$\alpha_3=0.02$  

## 长窗口  

在完成基础预训练后，通过在 $k_t^R$ 上使用YaRN把模型窗口从4k推广到128k。YaRN的参数设置如下：  
- s = 40  
- α = 1  
- β = 32  
- target maximum context length = 160k  

和原始的YaRN有所不同，由于注意力机制有所改动，所以把length scaling factor改成 $\sqrt{t}=0.0707\ln s+1$，以更好调控注意力熵。  

整个长文本训练在32k长度，batch size = 576的数据上训练了1000步，最终在大海捞针评测上的结果如下  

{% asset_img needle.png 大海捞针 %}  

## 评测  

DeepSeek-V2的base模型和其他较大规模模型的效果对比如下  

{% asset_img pt_eval.png 评测 %}  

DeepSeek-V2看起来基本达到了和70B规模dense模型竞争的水平。  

## 对齐  

SFT共使用了1.5M条数据，其中1.2M条以helpfulness为主，0.3M条以safety为主。  

训练设置：  
- epoch = 2  
- lr = 5e-6  
  
在SFT基础上，DeepSeek-V2通过GRPO进行了强化学习训练。  

最终对齐模型的评测如下  

{% asset_img align_eval.png 评测 %}  

# DeepSeek-V2-Lite

为方便开源研究，研究人员还提供一个稍小一点规模的DeepSeek-V2-Lite。  

模型超参：  
- layer num = 27  
- hidden size = 2048  
- initialization standard deviation = 0.006  
- attention head数量 = 16，每个attention head size = 128  
- KV压缩维度 $d_c=512$  
- Q不进行压缩  
- decoupled queries and key per head dimension = 64  
- 2个共享专家 + 6/64路由专家  
- 第一层不使用MoE  
- 专家大小 = 1408  
- 总参数15.7B，激活参数2.4B  

预训练超参：  
- AdamW：beta_1 = 0.9，beta_2 = 0.95，weight_decay = 0.1  
- lr scheduler：warmup-and-step-decay，warmup = 2k step，最大lr = 4.2E-4；在训练进度60%和90%的时候lr乘以0.316  
- gradient clipping norm = 1.0  
- constant batch size = 4608  
- maximum sequence length = 4k  
- 负载均衡权重：$\alpha_1=0.003$，没有使用其他负载均衡loss  
- 总训练量 = 5.7T  

{% asset_img lite_eval_1.png 评测 %}  

{% asset_img lite_eval_2.png 评测 %}  

# 小结  

- MLA是DeepSeek-V2很重要一个模块，在提升推理效率上有很大帮助，这个方向后续应该会有更多工作。  
- MoE受到越来越多的关注，几乎有一半的popular的模型是MoE结构了。  

***  

读到这了，来一发点赞收藏关注吧~

博客：[http://www.linsight.cn/](http://www.linsight.cn/)  
知乎：[Linsight](https://www.zhihu.com/people/us4ever)  
微信公众号：Linsight  
![](/images/qrcode.jpg)  

***  

【往期文章】  
[MoE模型的前世今生](http://www.linsight.cn/44e38c1b.html)  
[昆仑万维-SkyworkMoE](https://www.linsight.cn/1d5bcd45.html)  
[成本10w刀的JetMoE](https://www.linsight.cn/f3acf042.html)  
[从loss视角理解大模型涌现能力](https://www.linsight.cn/f5fb75e4.html)  
[LLM长上下文的问题](http://www.linsight.cn/c4da56c0.html)  
[解锁大模型长上下文能力](http://www.linsight.cn/cc852861.html)  
[大模型推理窗口-从有限到无限大](http://www.linsight.cn/45ee1a6d.html)  
[理解Attention:从起源到MHA,MQA和GQA](http://www.linsight.cn/3dc22f96.html)  
[大模型推理加速-投机解码](http://www.linsight.cn/f5c015c.html)  
[大模型推理加速-MEDUSA](https://www.linsight.cn/7bbe2df6.html)  
[LLM的重复生成和ICL](https://www.linsight.cn/7381cae3.html)  
[大模型偏好对齐-DPO](http://www.linsight.cn/473f2b43.html)  
[大模型偏好对齐-ODPO](http://www.linsight.cn/da871ebe.html)  
[大模型偏好对齐-simPO](http://www.linsight.cn/280fa97a.html)  
[大模型偏好对齐-IPO](http://www.linsight.cn/4fe7b810.html)  
[Yi技术报告-划重点看细节](http://www.linsight.cn/41b6a819.html)  
[MiniCPM](https://www.linsight.cn/376db710.html)  
[GLM4报告的一些技术点](https://www.linsight.cn/a5206abd.html)  
[Gemma2](https://www.linsight.cn/cf3f1f81.html)  
[苹果的OpenELM](https://www.linsight.cn/f845f3e4.html)  
[从Yuan2.0到Yuan2.0-M32](https://www.linsight.cn/3df0cd42.html)  
[bilibili的index-1.9B](https://www.linsight.cn/770b63e1.html)  
[transformer中normalization的二三事](http://www.linsight.cn/6a40bfa5.html)  
[从代码实现看normalization-到底做了什么](http://www.linsight.cn/b70b4a2d.html)  
[稀疏注意力计算:sliding window attention](http://www.linsight.cn/c61d17e3.html)  
[理解LLM位置编码:RoPE](http://www.linsight.cn/a051710f.html)  
[RoPE的远距离衰减](https://www.linsight.cn/f0902f1a.html)  
[大模型算法题(1)](http://www.linsight.cn/3345028a.html)  
[大模型算法题(2)](http://www.linsight.cn/ad0bba9d.html)  
[大模型算法题(3)](http://www.linsight.cn/1736008.html)  
[大模型算法题(4)](http://www.linsight.cn/1736008.html)  
[大模型算法题(5)](http://www.linsight.cn/336f2f3e.html)  
[大模型算法题(6)](http://www.linsight.cn/7c04944d.html)  
[大模型算法题(7)](https://www.linsight.cn/dd614e12.html)  

***  

# Reference  

【1】DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model https://arxiv.org/abs/2405.04434  
【2】缓存与效果的极限拉扯：从MHA、MQA、GQA到MLA https://kexue.fm/archives/10091  

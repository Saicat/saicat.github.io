---
title: LLM预训练数据策略(一)
tags:
  - NLP
  - LLM
  - transformer
  - 预训练
  - 数据
categories:
  - CS
  - NLP
  - LLM
abbrlink: 2c2cdc34
date: 2024-09-04 21:36:27
---

【本文已在同名 微信公众号 / 知乎 / [个人博客linsight.cn](http://www.linsight.cn/) 上线】  

***  

# Code-Based English Models Surprising Performance on Chinese QA Pair Extraction Task  

时间：2024年1月  

## TL;DR  

在“基于中文文档生成QA”的生成任务上，使用多个规模相同/相近的LLM进行实验。结果发现，代码模型效果比通用模型更好，并且英文模型表现出了优异性能。  

## 任务 & 数据  

1、任务  

基于中文的文档，给出中文QA数据。下面是一个样例：  

{% asset_img codebased_intro.png LLM数据策略 %}  

2、数据  

（1）训练数据  

从wiki和新闻文章中获取的 143,846 个文档，每个文档有相应的问答对。  

这些数据来自开放的人工标注数据集。  

（2）评测数据  

测试集由 300 个internet technology相关的private文档组成。来自于实际业务中收集的hard case。  

训练数据和评测数据的domain很不同，这就要求模型有比较强的泛化能力。  

## 指标 & 实验  

1、指标  

文中使用了5个细粒度的指标：  
- Coverage Analysis：使用 ROUGE - L 来评估summary在多大程度上涵盖了源文本中呈现的信息。ROUGE - L 值越高，表示模型输出的覆盖率越好。  
- Hallucination Analysis of Answers：定义了content creation rate（CCR）（不过文中没有给出具体定义），代表模型产生幻觉的倾向。CCR 分数越高，表明幻觉越严重。  
- Completeness Analysis of Answers：定义了average text coverage ratio（COV）（同样没有给出具体定义）。COV 指的是在所有结果中，输出文本与输入文本的最长公共子串的平均比值。这个比值反映了输出文本对输入文本内容的平均使用程度。COV 分数越高，表示输出文本对输入文本的复制保真度越高。  
- Reject Capability Analysis：定义了模型的reject capability（REJ）（同样没有给出具体定义），它表明模型是否知道何时拒绝完成任务（训练和测试数据中包含一些无法用于生成有效QA的文档，需要模型拒绝）。REJ 分数越高，表明模型更好地辨别哪些材料更值得进行知识生成任务。  
- Summarize Capability Analysis：定义了text extraction aggregation count（TEAC）（同样没有给出具体定义），用来代表模型的总结能力。TEAC分数越高表示模型从内容整体中多个地方提取信息的能力越强，即提取的信息更复杂。同时还定义了longest increasing subsequence ratio（LISR），表示模型保持一致性的能力，LISR 分数越高，表明知识提取行为更有序。  

还有一个EXPERTS指标，是用其他LLM对生成结果进行打分。  

2、实验  

论文具体做了以下四个实验。实验用的基模型都是预训练模型，没有经过对齐。  

（1）Code-based LLMs better than other LLMs  

比较了code model与非code model在QA数据微调后各种指标上的性能差异，发现code model在 EXPERTS 等方面通常优于其他LLM，而在幻觉、泛化和拒绝等方面也有显著增强，如下表所示：  

{% asset_img codebased_perf.png LLM数据策略 %}  

（2）Less Domain Knowledge, Better Performance  

把 DeepSeek-code-6.7B 和 Code Llama-7B 拉出来看，发现Code Llama-7B整体上更好一些。而根据二者透露的信息，DeepSeek-code-6.7B所用的中文数据比例应该是远高于Code Llama-7B的。  

{% asset_img codebased_codemodel.png LLM数据策略 %}  

DeepSeek-code-6.7B 的主要问题是容易出现幻觉。这可能说明，在预训练数据中包含大量的中文通用数据，更容易导致模型出现幻觉问题。  

（3）A Moderate Amount of Chinese is Better  

Code Llama的词表中，中文的token很少。如果加入8076个中文token，并使用词表中其他相关token的平均值初始化这些新加入的token，那么在一些维度上有一些提升，比如TEAC和CCR。如果是对这些新加入的token进行随机初始化，效果则差很多。  

{% asset_img codebased_emb_figure.png LLM数据策略 %}  

{% asset_img codebased_emb_result.png LLM数据策略 %}  

文中认为加入一定量的中文知识能够提升效果。不过个人认为这里也有可能是训练量太少，随机初始化没有能够收敛。利用平均值初始化新token可以加速收敛。  

（4）QLoRA fails to replicate the effects  

使用 QLoRA 对 Code Llama-7B 模型进行两种方法（scaling 和 noscaling）的训练，发现无论哪种 QLoRA 方法，在该任务下都无法获得与全参数微调相同的效果。  

{% asset_img codebased_scaling.png LLM数据策略 %}  

EXPERTS得分的差距非常大。  

## 小结  

感觉整个实验应该说indicate了一个事情：训练语言（中文、英文）之间的差距，比训练任务的的差距更小。如果一个模型在英文上具备了很好的摘要能力，那么这个摘要能力比较容易迁移到中文上；而如果一个中文模型在预训练中没有得到任何摘要相关能力的训练，那么想要这个模型获得摘要能力，那么难度会比前面语言上的迁移更大。  

因此，想要在复杂任务上获得好效果，就应该拆解复杂任务所需的基本能力，然后在训练数据中加入支持相应能力的数据。比如agent能力，需要调用接口，总结多方信息，那么上游的代码数据、长文总结、阅读理解应该都对agent能力有帮助。  

# RHO-1: Not All Tokens Are What You Need  

时间：2024年4月  

## TL;DR  

论文的主要思想是“Not all tokens in a corpus are equally important for language model training”。文中提出Selective Language Modeling (SLM)的方法，通过仅选择部分有效的token参与继续预训练，能够在提升收敛速度（5x-10x）的同时，获得更好的效果。  

{% asset_img rho_intro.png LLM数据策略 %}  

左边是常规的Causal Language Modeling (CLM)，右边是这里改进的Selective Language Modeling (SLM)。  

## 分析  

1、token粒度脏数据  

预训练数据中存在着像下图这种，包含部分低质量token的数据。这些数据难以通过文档级别的过滤来处理，因为如果把过滤阈值卡得太严的话，可能会误伤很多正常数据。而如果直接移除这些token，又可能影响正常的训练。  

{% asset_img rho_noise.png LLM数据策略 %}  

2、数据分布不一致  

预训练的数据分布通常和下游的任务的数据分布不一致，直接把所有token一视同仁进行预训练，可能会有很多对下游无效的训练。  

## Selective Language Modeling (SLM)  

首先看下训练过程中token的loss变化情况。从OpenWebMath数据集中抽15B的数据，训练Tinyllama-1B，然后每训练1B token保存一次checkpoint。然后用一个包含约320,000个token的validation set，跑出所有checkpoint的token-level loss。发现这些token根据训练过程loss的变化可以分成4类：  
- persistent high loss (H→H)，约11%  
- increasing loss (L→H)，12%  
- decreasing loss (H→L)，26%  
- consistent low loss (L→L)，51%  

各个类型的loss变化如下图（a）  

{% asset_img rho_types.png LLM数据策略 %}  

和我们设想的不同，随着训练进行，loss减小的token占比只占全量数据的1/4。L→L和H→H类型的token在训练过程中都呈现强烈的震荡状态。实际上，部分H→L类型的token的loss下降过程也并不平滑。  

下图把这些震荡的token用橙色标记出来  

{% asset_img rho_text_1.png LLM数据策略 %}  

发现有相当部分其实可以算是噪音，这和前面分析的一致：存在token粒度脏数据，会影响训练。  

那么基于上面的这些分析和实验：  

> If we can select the appropriate tokens for the model to focus on during training, we may be able to stabilize the trajectory of the model’s training and enhance its efficiency.  

受document-level filtering中使用reference mode的启发，文中提出 token-level data selection的pipeline，也就是Selective Language Modeling (SLM)。  

SLM有3个step：  

{% asset_img rho_slm.png LLM数据策略 %}  

- step 1：在高质量数据集上训练reference model；这个高质量数据集的分布反映我们想要的下游任务分布  
- step 2：用reference model获得将要使用的预训练数据集上的token-level loss  
- step 3：正式的训练中，只使用LM和RM间excess loss较大的token进行学习  

RM模型的训练使用标准的cross-entropy loss。训练好的RM对预训练数据进行打分，按如下计算：  

$$\mathcal{L}_{\mathrm{RM}}(x_i)=-\log P(x_i|x_{<i})$$  
excess loss则是当前训练的模型，和RM之间的打分差：  

$$\mathcal{L}_\Delta(x_i)=\mathcal{L}_\theta(x_i)-\mathcal{L}_\mathrm{RM}(x_i)$$  

正常的causal模型训练是对所有token的loss取平均训练的：  

$$\mathcal{L}_{\mathrm{CLM}}(\theta)=-\frac1N\sum_{i=1}^N\log P(x_i|x_{<i};\theta)$$  

SLM的训练则是基于excess loss，只选择每个训练batch中，excess loss处于top k%（token selection ratio）的token进行训练：  

$$\mathcal{L}_{\mathrm{SLM}}(\theta)=-\frac1{N*k\%}\sum_{i=1}^NI_{k\%}(x_i)\cdot\log P(x_i|x_{<i};\theta)$$  

$$\begin{aligned}&I_{k\%}(x_i)=\begin{cases}1&\text{if }x_i\text{ ranks in the top }k\%\text{ by }\mathcal{L}_\Delta(x_i)\\0&\text{otherwise}\end{cases}\end{aligned}$$  

## 实验  

### 设置  

在数学领域和通用领域进行继续预训练，实验SLM的效果。  

1、RM Training  

- mathematical RM：使用0.5B的高质量数学数据，这些数据包括来自GPT-4的生成数据和人工处理获得的数据（《Metamath: Bootstrap your own mathematical questions for large language models》，《Key-pointdriven data synthesis with its enhancement on mathematical reasoning》；《Mammoth: Building math generalist models through hybrid instruction tuning》，《Exploring the mystery of influential data for mathematical reasoning》）  
- general RM：使用1.9B开源训练数据  
- 训练3个epoch  
- 1B模型和7B模型的lr分别是5e-5和1e-5  
- cosine decay schedule  
- 1B模型和7B模型的max length分别为2048和4096  
- RM模型和继续预训练的模型使用相同的初始化  

2、Pretraining Corpus  

在数学领域，使用包含14B token的OpenWebMath (OWM) 数据集；而在通用领域，则是把SlimPajama、StarCoderData和OpenWebMath按6:3:1混合得到80B的训练数据集。  

3、Pretraining Setting  

数学领域：  
- 基于Tinyllama-1.1B和Mistral-7B继续预训练  
- lr分别为8e-5和2e-5  
- batch size = 1M  

通用领域：  
- 基于Tinyllama-1.1B继续预训练  
- lr = 1e-4  
- batch size = 1M  

token selection ratio：  
- Tinyllama-1.1B：60%  
- Mistral-7B：70%  

4、Baseline Setting  

和SLM作为对比，分别用同样的数据进行常规的继续预训练，作为baseline：  
- Tinyllama-CT  
- Mistral-CT  

还有其他一些效果比较好的同规模预训练模型，包括Gemma、Qwen1.5、Phi-1.5 、DeepSeekLLM、DeepSeekMath、CodeLlama、Mistral、Minerva、Tinyllama、 InternLM2-Math和LLemma等。还有几个SFT模型：MAmmoTH和ToRA。  

### 效果对比  

各个预训练模型在Few-shot CoT reasoning的效果如下：  

{% asset_img rho_math_perf.png LLM数据策略 %}  

RHO-1-Math-1B相比Tinyllama-CT有16.5%的提升，而7B模型则有10.4%的提升。1B模型在训练多几个epoch之后还能进一步提升。  

而相比其他业界知名的模型，RHO-1基本上也能达到同一水准，但是训练所用的数据量相比这些模型则减少许多。  

Tool-Integrated Reasoning的效果也做了对比：  

{% asset_img rho_tool_perf.png LLM数据策略 %}  

相比预训练，SFT后RHO-1的收益没有那么大，但还是有提升。  

而在通用数据上训练的模型，在各个benchmark的对比如下：  

{% asset_img rho_general_perf.png LLM数据策略 %}  

也都有稳定的提升，其中代码和数学类提升最大。  

### Self-Reference  

前面的实验有一个前提，就是RM模型是用和下游任务比较分布相关的数据训练的。那么如果没有和下游任务分布相关的数据，是否还可以使用SLM来获得提升呢？  

论文认为RM模型打分的关键并不是和下游任务的分布对齐，而是要把noisy数据过滤掉，因此尝试用两个不同的scoring function来打分：一个就是上面用的reference loss score function（$\mathcal{L}_\Delta(x_i)$），另一个是information entropy score function：  

$$\mathcal{H}_{\mathbf{RM}}(x_i)=-\sum_{k=1}^VP(t_k|x_{<i})\log P(t_k|x_{<i})$$  

V是词表大小。  

如果一个token的information entropy score越大，那么说明在这个context下它的不确定性越大。  

不使用下游任务分布相关的数据，而仅用预训练数据，采用不同的score function，用SLM方法训练，结果如下：  

{% asset_img rho_self_ref.png LLM数据策略 %}  

无论用哪个score function，仅用预训练数据训练RM也可以获得提升。如果选择使用reference loss score function和information entropy score function的交集token进行训练，效果更好。  

### 其他  

1、Token Select Ratio  

在1B模型上实验不同的token select ratio：  

{% asset_img rho_ratio.png LLM数据策略 %}  

结果表明在60%~70%的效果最好。  

2、训练了什么token  

RM模型在一段文本上选择的token如下：  

{% asset_img rho_text_2.png LLM数据策略 %}  

而在训练过程中选择的token有什么变化呢？4个checkpoint，分别是训练0%、33%、66%和100%时，这4个阶段在同一段文本下，选择用于训练的token如下，token选择的preference从高到低分别用深蓝、浅蓝、黑色橙色和深橙色标记出来：

{% asset_img rho_text_3.png LLM数据策略 %}  

## 小结  

仅选择重要度高的token进行训练，在成本上来说比较高，可以尝试用于预训练的第二甚至第三阶段。  

# Reuse, Don’t Retrain: A Recipe for Continued Pretraining of Language Models  

时间：2024年7月  

## TL;DR  

一套针对general abilities的LLM继续预训练的指南，包括数据分布的设计，和lr schedule的设置。在一个15B（先经过8T数据的预训练）验证，在下游benchmark平均有9%的提升。实验结果适用于100B~1T数据量的继续训练。    

以往的一些工作主要是垂域的：  
- 《Simple and scalable strategies to continually pre-train large language models》  
- 《 Investigating continual pretraining in large language models: Insights and implications》  
- 《Continual pre-training of language models》  
- 《Towards continual knowledge learning of language models》  

## 实验设置  

### 预训练  

现在的模型很多都在万亿（1T）/十万亿（10T）级数据量上训练。为了得到一个使用可靠的效果，论文先用一个15B参数模型，在8T的数据上进行实验。8T的预训练数据分布如下：  

{% asset_img reuse_phase1_data.png LLM数据策略 %}  

### 继续预训练  

在实际场景中，我们通常没有太多的新数据用于继续预训练，因此继续预训练绝大部分所用数据就是来自上面的8T数据集。在这个基础上，加入了2.8B的QA数据集。  

QA数据有助于模型更好地抽取已经学到的知识（《Physics of language models: Part 3.1, knowledge storage and extraction》）。  

QA数据集的分布如下：  

{% asset_img reuse_qa_data.png LLM数据策略 %}  

### 模型  

- 15B参数：3.2B embedding参数，12.5B非 embedding参数  
- 32层  
- hidden size = 6144  
- 48个注意力头  
- RoPE  
- quared ReLU activations  
- vocabulary size = 256k  
- no bias terms  
- tied embedding weight  
- GQA，8个KV头  
- pretrain sequence length = 4096  
- 在训练的前5%token，batch size逐渐从384增大到1152  
- cosine learning rate schedule  
- warmup = 16B token  
- lr = 4.5e-4 --> 4.5e-5  
- AdamW，beta_1 = 0.9，beta_2 = 0.95，weight decay = 0.1  
在继续预训练中，只有lr会改变，其他都不变。  

### 评测  

评测数据集包括：  
- MMLU  
- Hellaswag  
- MGSM（Multilingual Grade School Mathematics）：用于多语言评估，具体报告西班牙语、日语和泰语这三种分别代表高、中、低资源语言的子集的平均准确率。  
- HumanEval的 Python 代码生成任务：在 pass@1 设置下评估模型的编码能力。  

## 继续预训练Recipe  

1、先在原预训练数据分布的基础上，提升高质量数据的比重进行训练，之后再切换到包含QA数据的数据集  

2、lr从预训练阶段的min_lr开始，按cosine annealing decay到min_lr / 100  

3、数据分布的切换要在min_lr / 5的时候进行  

## 实验  

这里提到一点：发现在继续预训练时，是否加载预训练阶段的optimizer参数影响并不大。  

预训练模型在评测数据集上的平均accuracy为48.9。  

### 数据分布  

1、QA数据  

如果把想要学习的新知识放在继续预训练的前面，模型会更容易学到（lr更大），但是同时也可能因此带来训练的不稳定性。为此使用了三种数据进行实验：  
- 纯使用预训练数据  
- 纯使用QA数据  
- 先使用预训练数据，然后混入10%的QA数据  

三种策略的效果如下：  

{% asset_img reuse_blend.png LLM数据策略 %}  

使用第三种策略效果有比较大的提升。  

2、general blend实验  

把initial blend称之为general blend（GB），把后面包含QA的数据分布成为QA blend（QB）。那么GB用什么分布最好呢？  

假设最优的GB是注重高质量数据和模型薄弱的领域的数据，且不会大幅偏离预训练分布。实验不同GB分布下的效果：  
- Reweight Domains：对预训练数据中的不同领域进行重新加权。  
- Pretraining w/ High Quality Web：在预训练数据中使用高质量的 Web 数据。  
- No Web：去除预训练数据中的 Web 数据。  
- Upweight Non Web w/ High Quality Web：提高非 Web 数据的权重，并使用高质量的 Web 数据。  

每种分布都训练300B数据，然后对比效果：  

{% asset_img reuse_datamix.png LLM数据策略 %}  

Upweight Non Web with High Quality虽然没有达到最高平均准确率，但在所有任务中最稳定地取得了较高分数，因此被选为后续实验的 GB。  

{% asset_img reuse_task_perf.png LLM数据策略 %}  

3、QA blend实验  

使用三种QA分布：  

{% asset_img reuse_qa_version.png LLM数据策略 %}  

在训练250B的GB数据后，进行QB的训练：  

{% asset_img reuse_qa_perf.png LLM数据策略 %}  

同时强调STEM和Chat数据的分布效果最好。  

最终的训练方案是，先训练权重修改后的GB数据，然后把QB数据混合到GB数据中。  

### lr schedule  

对比3个lr的decay设置，结果是decay到1%的继续预训练的lr时是最好的：  

{% asset_img reuse_qa_lr.png LLM数据策略 %}  

### 切换分布的时间  

从GB切换到QB分布，应该在什么时候进行？结果如下：  

{% asset_img reuse_time.png LLM数据策略 %}  

可以看到切换的时间点选择对结果的影响还是比较大的。  

## 小结  

算是一次实验记录，不过不是很够完善，解释得也不够清楚。可以作为实验参考吧。  

***  

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

【1】Code-Based English Models Surprising Performance on Chinese QA Pair Extraction Task https://arxiv.org/abs/2401.10286  
【2】RHO-1: Not All Tokens Are What You Need https://arxiv.org/abs/2404.07965  
【3】Reuse, Don’t Retrain: A Recipe for Continued Pretraining of Language Models https://arxiv.org/abs/2407.07263  

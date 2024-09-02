---
title: InternLM系列模型
tags:
  - NLP
  - LLM
  - transformer
categories:
  - CS
  - NLP
  - LLM
abbrlink: 7f3d361
date: 2024-08-20 21:32:53
---

【本文已在同名 微信公众号 / 知乎 / [个人博客linsight.cn](http://www.linsight.cn/) 上线】  

***  

InternLM系列模型的参与方有上海AI实验室、商汤、香港中文大学，以及复旦和上交。主力应该是前两个，InternLM中的Intern这个名字也是继承自它们之前的视觉模型项目的名字。  

最近InternLM2.5发布，在HuggingFace的大模型榜单上有不错的成绩，因此梳理一下InternLM相关的资料，学习一下。  

# InternLM一代  

首先是最早发布的一代模型。  

InternLM第一代是2023年6月发布的，时间比较早，技术报告中透露的内容也不多，把关键信息简单整理一下：  
- 104B参数的模型  
- 1.6T的多语言数据，以英语为主，部分中文，少量其他语言  
- 窗口长度为2k  
- 使用多阶段的预训练策略，在每个阶段分别进行lr退火；不同的阶段在数据比例上有所不同；使用多阶段的训练策略好处是方便调整效果，并且如果需要回退不用全部重新训练  
- 对齐阶段包括SFT和RLHF；5M条SFT数据，部分来自self-instruct  
- 效果上，和Llama一代可比，和ChatGPT还有差距  

后来发布了新版，模型规模提升到了123B。在2023年8、9月又发布了7B和20B规模的base模型和chat模型。  

# InternLM2  

2024年初，InternLM2发布。  

## InternLM2概览  

InternLM2包括1.8B、7B和20B三个规模的模型，最大支持200k的窗口长度（Needle-in-a-Haystack评测），不同规模的模型训练数据量从2.0~2.6T不等。  

对齐阶段包括SFT和COnditional OnLine Reinforcement Learning
from Human Feedback (COOL RLHF)，细节后面说。  

## 模型  

开源模型比如Llama、Qwen、Baichuan和Mistral等基本上把当前LLM的标准结构定了下来，而很多训练和推理相关的配套优化也是支持这样的标准化设置的。  

出于对目前业界优化复用的考虑，在模型结构上InternLM并没有做什么改变，三个规模的模型结构超参如下：  

{% asset_img model.png InternLM系列模型 %}  

## 预训练  

### 数据  

预训练数据主要包括三类：通用的文本数据、代码相关的数据，以及长文本数据。  

1、通用文本数据  

通用数据来源主要包含web数据、书籍和technical literature（techlit）等，这几类数据在预训练数据中的占比如下：  

{% asset_img pt_data_dist.png InternLM系列模型 %}  

通用数据的处理流程基本上也是标准做法：  

{% asset_img pt_data_pipeline.png InternLM系列模型 %}  

highlight几个细节：  
- 去重的时候，对文档的5-gram使用128个hash function，threshold = 0.7  
- 出现重复的时候保留most recent的数据  
- 安全过滤使用了kaggle的“Toxic Comment Classification Challenge”数据集  

2、代码数据  

代码数据主要来自github爬取、公开数据，以及一些和代码/编程相关的论坛、网页等。  

不同编程语言在代码数据的分布如下  

{% asset_img code_lang_dist.png InternLM系列模型 %}  

代码数据会经历format cleaning、deduplication、quality filtering和dependency sorting几个阶段。  

（1）format cleaning  

所有代码数据都要转成unified markdown format。会有少量数据由于原始数据的问题，没有直接转换格式成功，因此会再用一些规则进行处理。  

使用markdown格式是因为它比较简单，并且可以兼容自然语言和代码。  

预训练中实际使用的格式会更加复杂一点，包括多段代码和dependecy文件的拼接处理。  

> The main idea is to utilize the interleaving
data, which is pivotal for teaching the model about programming.  

这和《DeepSeek-Coder》提到的类似。  

（2）deduplication  

去重的过程和通用的文本数据大致一样，除了一个地方 -- tokenizer。  

比如python中，有的代码indent会用2个空格，有些是4个token，还有写是tab。这就对去重效果造成了影响。因此需要用一个能把这些情况合并的tokenizer。  

实际上目前有很多数据去重方法已经做到了段落或者line的粒度，这样当然去重效果更好。不过这里还是用了file level的去重。  

（3）quality filtering  

几个要点：  
- 基于规则的打分可以对各种维度进行打分，不过代码风格可能不是一个好的评分维度，会很容易把数据错分为低质量  
- 模型评分和人类评分的一致性并不很高，因此只在“模型评分和人类评分一致性足够高”的语言上，应用了模型分数进行过滤  
- 为了提高模型打分的准确率，使用了三轮的迭代，每轮包括人类标注和重新训练  

（4）dependency sorting  

InternLM2的最大训练窗口达到了32k。而在之前的数据处理流程中，来自同一个代码仓库的文件已经被打乱，因此需要把这些文件重新按顺序排列好。  

最终代码数据质量被分成高中低三类，其中高质量数据会训练多轮，中等质量数据训练一轮，而低质量数据则不会用于训练。  

{% asset_img code_quality.png InternLM系列模型 %}  

3、长文本数据  

长文本数据的处理参考《Longwanjuan: Towards systematic measurement for long text quality》的做法。  

长文本数据过滤包括：  
- 长度选择，选择至少32k byte长度的数据  
- statistical filters  
- perplexity filters  

（1）statistical filters  

用于过滤掉无意义的数据，而不是用于筛选高质量数据。统计过滤器在长文本特别有效，因为长文本的统计特征相比短文本更加稳定和可靠。  

（2）perplexity filters  

这里ppl的用法和以往的有所不同。  

对于一段文本S，计算其ppl可能会受到模型和tokenizer的影响。InternLM2这里的做法是把文本切分成S1和S2，对于正常长文本，P(S2|S1)应该比P(S2)要低，因为前文提供了更多的信息。而如果P(S2|S1)>P(S2)，那么就说明S1对S2造成了干扰，那么这段文本上下文关联就很弱甚至是负向的。这样的文本就会被过滤掉。  

（3）threshold selection  

关于阈值的选择，有两个要点：  
- 对不同的domain使用不同的阈值能够比使用统一阈值效果更好  
- 更多关注在得分在阈值周围的数据，因为统计过滤器和ppl相比model-based的打分更加平滑，这些边界case可以很大程度反映过滤器的打分标准  

### tokenizer  

基于cl100k，抽取了top 60,004个token，加上32,397个中文token获得新词表。为了让词表的大小是128的倍数，方便分布式训练，再加上147个spare token。  

### 预训练设置  

各个模型的预训练设置如下：  

{% asset_img model.png InternLM系列模型 %}  

- AdamW optimizer，beta_1 = 0.9, beta_2 = 0.95  
- epsilon = 1e-8  
- weight decay = 0.1  
- final lr = 10% * max lr  

### 多阶段预训练  

不同规模的模型总训练量为2.0T~2.6T不等。预训练过程分为3个阶段。  

1、阶段1：4k  

窗口长度为4096，消耗90%的token。  

2、阶段2：长窗口  

窗口长度为32,768，消耗9%的token。  

虽然窗口长度为32,768，不过超过50%的数据本身长度还是 < 4096的。  

这一阶段把RoPE的base frequency从50,000提升到100,000。得益于flash attention和可扩展的训练框架，32k训练窗口下的训练速度相比4k只降低了40%。  

3、阶段3：能力增强  

这一阶段是针对reasoning、数学能力、知识学习的提升的，共消耗约24B的数据。  

所用数据分布如下  

{% asset_img specific_data.png InternLM系列模型 %}  

## 对齐  

对齐阶段包括SFT和RLHF。  

### SFT  

共有10M的SFT数据，分布如下：  

{% asset_img sft_data.png InternLM系列模型 %}  

数据格式使用ChatML（《chat markup language》https://github.com/MicrosoftDocs/azure-docs/blob/main/ articles/ai-services/openai/includes/chat-markup-language.md），7B和20B模型都进行了一个epoch的微调，lr = 4e-5。  

### COOL RLHF  

RLHF有两个问题：  
- preference conflict：比如满足helpful的response更容易在安全性上出现问题，目前的做法通常是使用多个preference模型，导致训练慢了  
- reward hacking：随着训练进行，actor模型学到一些获得高reward的捷径，但实际上response的质量并没有提升  

为了解决这两个问题，InternLM把RLHF框架修改为Conditional OnLine RLHF。  

#### Conditional Reward Model  

简单地说，conditional reward model就是利用不同的system prompt，达到一个模型对多个维度进行打分的目的，如下图：  

{% asset_img crm.png InternLM系列模型 %}  

reward模型的训练用了2.4M对偏好数据，覆盖了不同的能力。  

1、loss function  

reward模型训练的loss function做了一些改动。  

首先，参考focal loss的思路，为了让难样本的loss更大，而简单样本的loss更小，loss函数修改如下：  

$$L_{ranking}=-(1-2\times\max(0,P_{i,j}-\frac12))^\gamma\log(P_{i,j}))$$  

图像画出来是这样的  

{% asset_img l_rank.png InternLM系列模型 %}  

附上代码  

```python
import numpy as np
import matplotlib.pyplot as plt


# Define the function again for plotting
def g(P):
    return -(1 - 2 * np.maximum(0, P - 0.5))**2 * np.log(P)

# Generate P values
P = np.linspace(0.01, 1, 400)  # P values from a small number above 0 to 1

# Generate y values
y = g(P)

# Plotting the corrected graph
plt.figure(figsize=(10, 6))
plt.plot(P, y, label=r'$-(1-2 \times \max(0, P-\frac{1}{2}))^{2} \log(P)$')
plt.title(r'Graph of $-(1-2 \times \max(0, P-\frac{1}{2}))^{2} \log(P)$')
plt.xlabel('P')
plt.ylabel('Function Value')
plt.legend()
plt.grid(True)
plt.show()
```

模型输出的win vs lose response的概率越大，说明case越简单，loss就趋向于0，反之则loss会增大。  

另外就是如果不对reward的分数进行限制，那么就有可能出现绝对值特别大的数，这样可能导致训练不稳定。因此再加上一个对reward score的logarithmic barrier penalty：  

$$L_{penalty}=-(\log(x+5)+\log(5-x))$$  

这个函数只在(-5, 5)之间有定义，画一下图像如下：  

```python
import numpy as np
import matplotlib.pyplot as plt

# Define the function
def f(x):
    return -np.log(x + 5) - np.log(5 - x)
    
x = np.linspace(-10, 10, 1000)  # x values from -10 to 10
x = x[(x + 5) > 0]  # Filter out values where the function is undefined for the first log
x = x[(5 - x) > 0]  # Filter out values where the function is undefined for the second log

# Generate y values for the filtered x values
y = f(x)

# Plotting the updated range
plt.figure(figsize=(10, 6))
plt.plot(x, y, label=r'$-(\log(x+5)+\log(5-x))$')
plt.title(r'Graph of $-(\log(x+5)+\log(5-x))$')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()
```

{% asset_img u_shape.png InternLM系列模型 %}  

对于绝对值比较大的reward score（靠近-5或者5），这个loss值就会增大。  

最终reward模型的训练loss如下：  

$$L=L_{ranking}+\lambda L_{penalty}$$  

$L_{penalty}$ 的系数 $\lambda=0.02$。  

2、训练  

一些训练细节：  
- RM训练的时候保持每个batch的token数为16,384，不管数据条数为多少  
- cosine lr schedule，max lr = 1e-5, final lr = 5e-6  
- weight decay = 0.01  
- 为了防止过拟合，只训练一个epoch  

#### Online RLHF  

Online RLHF说白了就是建立feedback机制，多次迭代。  

InternLM2的online RLHF建立了两条反馈机制的path： fast path和slow path。  

1、Fast Path  

fast path用于修正在训练中出现的reward hacking的问题。  

随着PPO训练进行，actor模型很可能会找到一些获得高reward的捷径，相当于是作弊了。这些情况大部分都源自RM训练时的漏洞，或者是覆盖不全，或者是过拟合了。  

总之fast path就是在每次进行完RLHF的训练之后，找到这些容易触发很高reward值的case，并根据这些case构建对应的训练样例，修补漏洞。20~100条样本就足够修补对应的漏洞了。  

2、Slow Path  

相对于fast path，slow path可以说是regular的、更全面的反馈修复。其中会用到更多的数据和标注。  

由于大量数据的标注需要时间，所以slow path的训练数据可能是滞后的，比如在第二次RLHF完成之后，第一轮的slow path标注数据才完成，那么这些数据就会加到第三轮的训练中去。  

整个online RLHF一共进行了3个round。  

PPO训练用了200k左右的query，模型更新了约400次。  

文中还提到了一些训练的实用细节。  

1、关于初始化  

在PPO训练开始的时候，会先固定actor模型，单独对critic模型训练50步。  

因为critic模型的目标和sft模型或者reward模型都有所不同，所以从这个两个模型初始化之后，critic模型在训练前期其实是出于一性能不佳的状态的。在这个阶段如果参考critic模型的反馈对actor模型进行更新，可能会带来不稳定。  

{% asset_img critic_loss.png InternLM系列模型 %}  

2、conditional reward  

如前面提到的，RM是使用不同的system prompt获取对应维度的reward score的，而在PPO训练时也需要根据输入的prompt调整给RM的system prompt。  

{% asset_img condition_ppo.png InternLM系列模型 %}  

3、pretrain gradient  

为了缓解灾难性遗忘的问题，PPO训练中加入了预训练数据，并计算pretrain loss，加入到模型的整体损失中。pretrain loss的系数为0.5，预训练数据量大约相当于PPO训练数据的50%。  

4、超参  

- KL divergence coefficent = 0.01  
- actor model lr = 1e-6  
- critic model lr = 5e-6  
- actor model解码top_p = 0.9  
- larger λ value for PPO leads to higher rewards in our case, so we set it to 0.99  

### 长文本finetune  

在SFT和RLHF中都使用了长文本数据。数据主要来源有两个：  
- 书籍  
- github仓库  

代码数据使用了DS-1000（《Ds-1000: A natural and reliable benchmark
for data science code generation》）中超过10k star的仓库，并按如下流程进行拼接，以获取32k长度的数据：  

{% asset_img long_code_data.png InternLM系列模型 %}  

### Tool-Augmented LLMs  

为了让模型具备一定的工具调用能力，修改了ChatML格式，加入了新的角色 -- “environment”，让模型可以从外部接口获取反馈。  

下面是一个模型调用工具的例子：  

{% asset_img tool_case.png InternLM系列模型 %}  

工具训练的方式是按照《Agent-flan: Designing data and methods of effective agent tuning for large language models》的做法进行的。  

# InternLM2.5  

InternLM2.5的模型结构和InternLM2一样。  

InternLM2.5主要有几个提升：  

> 卓越的推理性能：在数学推理方面取得了同量级模型最优精度，超越了 Llama3 和 Gemma2-9B。  

> 有效支持百万字超长上下文：模型在 1 百万字长输入中几乎完美地实现长文“大海捞针”，而且在 LongBench 等长文任务中的表现也达到开源模型中的领先水平。  

> 工具调用能力整体升级：InternLM2.5 支持从上百个网页收集有效信息进行分析推理，相关实现将于近期开源到 Lagent。InternLM2.5 具有更强和更具有泛化性的指令理解、工具筛选与结果反思等能力，新版模型可以更可靠地支持复杂智能体的搭建，支持对工具进行有效的多轮调用，完成较复杂的任务。  

评测结果如下：  

{% asset_img internlm25.png InternLM系列模型 %}  

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
[phi系列模型](https://www.linsight.cn/fe13b56f.html)  
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

【1】InternLM: A Multilingual Language Model with Progressively Enhanced Capabilities https://github.com/InternLM/InternLM-techreport/blob/main/InternLM.pdf  
【2】InternLM2 Technical Report https://arxiv.org/abs/2403.17297  
【3】书生·浦语 https://www.baike.com/wikiid/7382383761788551219?anchor=lxmlbsze188r  
【4】https://github.com/InternLM/InternLM/blob/main/README_zh-CN.md  

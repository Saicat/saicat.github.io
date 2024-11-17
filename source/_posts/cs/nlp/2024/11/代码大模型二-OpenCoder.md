---
title: 代码大模型(二)--OpenCoder
tags:
  - NLP
  - LLM
  - transformer
  - 预训练
  - 对齐
  - 数据
  - 代码能力
categories:
  - CS
  - NLP
  - LLM
abbrlink: 7856bcc1
date: 2024-11-12 21:59:12
---

【本文已在同名 微信公众号 / 知乎 / [个人博客linsight.cn](http://www.linsight.cn/) 上线】  

***  

最近由M-A-P、无限光年、墨尔本大学、复旦大学等机构共同开发的OpenCoder开源了模型和部分数据，并且后续还会有更多资料放出。先来学习下技术报告的内容。  

目前各个规模和阶段的模型在 https://huggingface.co/OpenCoder-LLM 可下载。  

OpenCoder有1.5B和8B两个规模的模型，分别有base模型和instruct模型放出。  

{% asset_img opencoder_model.png 代码模型 %}  

base模型和instruct模型的效果如下表，从表上数据来看还是不错的，基本上达到Qwen2.5-Coder的水平。  

{% asset_img opencoder_perf1.png 代码模型 %}  

{% asset_img opencoder_perf2.png 代码模型 %}  

# 预训练数据  

OpenCoder构建了RefineCode数据集用于预训练，RefineCode主要包含两部分数据：raw code和code-related web data。raw code主要来自github（截至2023年11月），并从The Stack V2数据集中补充非github数据；而code-related web data则从web语料库抽取。  

RefineCode和The Stack数据集的对比如下，RefineCode包含了更多的code-related data，更多的rules，还有language specific rules。  

{% asset_img opencoder_refinecode.png 代码模型 %}  

raw code和code-related web data的处理流程示意图如下：  

{% asset_img opencoder_dataprocess.png 代码模型 %}  

## raw code  

1、preprocessing  

首先是去除了8M个非文本文件，此外根据文件扩展名，仅保留与编程语言相关的文件，包括代码、数据和文本文件（具体类型参考https://github.com/github-linguist/linguist/blob/main/lib/linguist/languages.yml），low capacity（这里low capacity大概是指比较小的文件？）或者low quality的类型也会被筛选掉。  

2、去重  

因为包含有大量的分支和版本，github中源码数据重复率比较高（大约75%的文件完全重复），因此去重的影响很大。  

目前对于代码数据，MinHash+LSH去重是比较成熟的方案了（StarCoder2、DeepSeekCoder）。在这个基础上，DeepSeekCoder提出了repository level的去重，OpenCoder这里做了实验对比file level和repository level去重的效果。具体来说，在485M个python文件上分别使用file level和repository level去重，结果上，repository level保留的token数大约是file level是三倍：  

{% asset_img opencoder_deduplication_level.png 代码模型 %}  

用这两份数据分别训练1.5B的模型，结果是file level的效果更好：  

{% asset_img opencoder_deduplication_perf.png 代码模型 %}  

去重细节上，先基于SHA256进行了精确去重，之后进行模糊去重。模糊去重使用5-gram，计算2048个MinHash函数；而LSH设置为band=16，row=128。遇到重复文件时，会保留star数更多，commit时间更晚的那一份文件。这一步大约去除了6%的文件。  

3、transformation  

有一些问题本身在每个文件中占比不多，但是在各个文件重都普遍存在，比如源码文件开头的版权声明：“Copyright Intel Corporation (C) 2014-2016”。对于这种情况，直接删除整个文件显然是不合适的，因此在过滤模块之前需要针对这些问题进行识别和转换改写。  

4、Filtering  

在《Textbooks Are All You Need》中，关于代码数据质量有一些评价和准则：  
- Many samples are not self-contained, meaning that they depend on other modules or files that are external to the snippet, making them hard to understand without additional context.  
- Typical examples do not involve any meaningful computation, but rather consist of trivial or boilerplate code, such as defining constants, setting parameters, or configuring GUI elements.  
- Samples that do contain algorithmic logic are often buried inside complex or poorly documented functions, making them difficult to follow or learn from.  
- The examples are skewed towards certain topics or use cases, resulting in an unbalanced distribution of coding concepts and skills across the dataset.  

参考这些原则，OpenCoder过滤数据时考虑这几个做法：  
- 将self-containment差的文件过滤掉  
- 将logical structure差的文件过滤掉  
- 将和standard format差很多的文件过滤掉  

基于这些guildline，OpenCoder开发了一个启发式过滤框架。在RedPajama的基础上，完善了StarCoder的规则。过滤规则分为三类：  
- Natural Language Filtering Rules：通用的rules，比如文件大小、行数等所有数据通用的指标，代码数据和文本数据都共享  
- General Code Filtering Rules：适用于所有代码文件，比如变量数量、平均函数长度等  
- Language-Specific Filtering Rules：语言定制的规则，比如python语言中pass的频率，或者C语言中goto语句的使用  

启发式过滤会计算很多指标，这就涉及到很多阈值的调整。开发过程中，会先按经验设置一个大概阈值，然后根据运行结果再进行精细调整。调整阈值的原则就是在尽量过滤掉低质量数据的情况下，保持数据的整体分布不受明显的影响。  

在检查一个阈值的有效性时，可以引入入PPL等评估手段，看被过滤掉的数据是否是PPL极高或者极低的数据。  

OpenCoder实践中一些阈值设置的example：  

{% asset_img opencoder_rules.png 代码模型 %}  

{% asset_img opencoder_python_rule.png 代码模型 %}  

4、sampling  

在尽量保持数据分布的情况下，对资源过多的语言类型进行下采样。比如Java数据从409GB减少到200GB，HTML从213GB减少到64GB，最终得到了730B token的数据。  

用PCA对从CodeBert获得的The Stack V2和RefineCode的embedding进行可视化，如下图：  

{% asset_img opencoder_data_dist.png 代码模型 %}  

可以看到The Stack V2有更多的异常数据，这些数据包括纯文本注释、16进制的数据文件以及过短代码等。这些数据都是对训练有害的。相比之下RefineCode的分布更为紧密，有更少的异常数据。  

## code-related web data  

受DeepSeekMath启发，OpenCoder也从web数据中收集代码相关的数据。  

首先参考《Automathtext: Autonomous data selection with language models for mathematical texts》的方法，AutoDS，从CommonCrawl中选取50w high-quality code-like data。  

这些数据会用于训练fasttext。为了在fasttext中保持vocab不要太大，会用BPE tokenizer处理预料库，再进行fasttext训练。  

训练好的fasttext会从大量的web数据中筛选出代码相关数据。对召回的数据进行分析，把来自相同base url（如stackoverflow）的页面定义为同一个domain。这一步大概把召回的数据中的10%判断为code-related。  

code-relate的域名比如stackoverflow会被手动标注出来，那些域名为code-related，但是又没有被fasttext分对的数据会被手动加入到正类中。跑了3个iteration之后（重复标注&打标），总共获得了220G的code-related web data。  

这个pipeline也应用到FineWeb、Skypile和AutoMathText中，recall了330GB的code-related web data。另外，发现github数据里也有部分类似的数据，从中也抽了178GB。  

手动标注属于代码和数学的url如下：  

{% asset_img opencoder_url.png 代码模型 %}  

## RefineCode数据集  

最终得到用于预训练的RefineCode数据集包含960B数据，组成如下：  

{% asset_img opencoder_ptm_data.png 代码模型 %}  

其中代码数据语言的分布如下：  

{% asset_img opencoder_lang_dist.png 代码模型 %}  

## annealing data  

现在我们已经知道，预训练的退火阶段对模型的效果影响很大。  

首先这期间的数据分布不能有太显著的变化，否则会导致模型的灾难性遗忘。因此在退火阶段，84%的数据和来自RefineCode的原始分布。  

在这个基础上，加入高质量的数据来提升模型的最终效果。  

1、algorithmic corpus  

从包含"leetcode"、"def solution"、"class solution"等关键字的原始预训练数据中采样了部分算法数据，这些数据有很强的逻辑性，和self-contain的特性。  

2、合成数据  

合成数据有两种形式：High Quality Code Snippet和Code Textbooks。  

（1）High Quality Code Snippet  

参照Phi系列合成CodeExercises数据集的做法，用algorithmic corpus作为种子，让LLM合成self-contained independent functions，以及对应的测试csae，并且通过执行反馈保留通过测试case的函数。这个方法在多种编程语言都使用了。  

（2）Code Textbooks  

使用Qwen2-72B-Instruct在hqcode上生成educational text snippets。  

hqcode是由gpt-4o-mini合成的代码数据，每条数据包含一段自然语言描述的问题，以及对应的solution。（https://huggingface.co/datasets/yuxiang630/hqcode）  

合成数据时要求Qwen2-72B-Instruct在hqcode上对代码进行分析（interactive analysis），并解释相关的代码知识。  

# 预训练  

- 使用WSD lr schedule  
- warmup = 2000  
- seqlen = 8192  
- global bs = 1024 sample -> 8M token  
- lr = 3e-4  

前13w步使用的maxseq=4096，bs=2048。  

整个训练在512个H100跑了187.5小时。  

# post training  

## 数据  

包括四个部分。  

1、Open-source Training Data  

收集了Evol-Instruct、Infinity-Instruct、McEval数据。在这个基础上训了个二分类模型用于从Infinity-Instruct中抽取代码相关数据。  

另外还从WildChat和Code-290k-ShareGPT中抽取真实的用户query以及和代码相关的对话历史。对于质量较低的部分还用LLM来重新生成内容。  

2、Educational Instruction Synthesis  

和前面合成python数据类似，不过在这个基础上使用了一个评分模型，用于筛选出高质量的种子数据，以进一步提高合成的数据的质量。  

所用prompt：  

```python
You are a teaching assistant helping to create a Python programming task from a given code snippet. You must provide the best response to the Python programming task, including reasoning thought, reference solutions, explanation of test cases, and test code.
[Code Snippet]
{Code}
Your response must have these parts:
[Task]
{Create an independent and detailed Python programming task}
[Analysis]
{Analyze the task and reason about the given task step by step}
[Solution]
{Write a high-quality reference solution in a self-contained script that solves the task}
[Test]
{Provide ten assert statements to check the correctness of your solution}
```  

3、Package-related Instruction Synthesis  

package经常会更新，而LLM有可能在训练数据集中学了一些过时的用法和接口。因此搞了一个最新的代码集，用来微调模型，让模型能尽量给出最新版本的答案。  

所用prompt：  

```python
You are exceptionally skilled at crafting high-educational level problems and offering precise solutions. Please gain inspiration from the following code snippet to create a highquality programming problem, which is beneficial for learning the use of corresponding
libraries. Present your output in two distinct sections: [Problem Description] and [Solution].
[Code Snippet]
{Code}
[Library Api Requirements]
{Api Requirements}
[Library Api Doc]
{Api Doc}
Guidelines for each section:
1. [Problem Description]: This should be **completely self-contained**, providing all the contextual information one needs to understand and solve the problem. Assume common programming knowledge, but ensure that any specific context, variables, or code snippets pertinent to this problem are explicitly included. This problem should be **educational for learning the provided Library api, and please explicitly request the use of the relevant package in the question. This question should only concern the writing of **one function**, and you need to be clear about the function name and role of this function.
2. [Solution]: Offer a comprehensive, **correct** solution that addresses the [Problem Description] you provided. This solution should follow the standard of corresponding Library Api doc. Please ensure that the Solution only involves answering the Problem, **without addressing the requirements I provided!** Please provide essential explanation abouth this solution, especially the use of requiremed Library Api.
```  

4、Large-scale Diverse Instruction Synthesis  

参考《Mammoth2: Scaling instructions from the web》的做法来提升instruction的多样性。  

（1）首先，使用web数据中有用的句子作为生成问题的种子。（2）用一个task specification module随机选择语言、难度、任务类型，基于这些设置生成prompt。（3）更大参数量的LLM生成问题和对应的答案，并结合执行反馈来筛选正确样本。（4）用大模型给代码添加注释和解释。  

所用prompt：  

```python
You are an expert in designing high-quality programming questions based on the given text.
[Guidelines]
- You can draw inspiration from the given text to create the programming questions.
- The created question should be a self-contained question, which does not depend on any external context.
- The created response must contain the complete code snippet.
[Given Text]
{Given Text}
[Created Question]
{Created Question}
```  

## 训练  

fine-tuning分成两个阶段。  

第一阶段注重理论知识，让模型学习计算机原理、算法、数据结构等。  

第二阶段把重点从理论转到实际任务。  

两个阶段的数据组成如下：  

{% asset_img opencoder_sft_data.png 代码模型 %}  

# Autonomous Data Selection（AutoDS）  

前面提到从web数据中收集代码相关数据的方法的时候，参考了AutoDS的做法。  

比如我们想要用LLM给一段文本在某些维度上打分，比如代码风格、质量等，一般来说可能需要给一些训练数据线微调一下。这些微调数据的打分基本上也都是离散的，因为人类标注结果就是离散的，1分2分5分这样。这种traditional的做法一方面需要引入人类的打分，不能完全自动化，另一方面人类本身的打分也是有偏，并且打分的结果也只能是离散的，无法连续。  

AutoDS提出的做法，不需要引入人类标注数据，也不需要微调，完全依赖模型自身学到的内容，并且可以得到更加公正、连续的打分结果。  

具体来说，AutoDS就是在打分prompt中要求打分模型回答关于输入样本的问题，而这些问题只能用"YES"和"NO"来回答。比如问题是“这条数据是否和数学相关”，那么模型的打分结果就是：  

$$\mathrm{LM-Score}(\cdot)=\frac{\exp(\mathrm{logit}(`\text{YES'}))}{\exp(\mathrm{logit}(`\text{YES'}))+\exp(\mathrm{logit}(`\text{NO'}))}$$  

一个prompt的例子如下：  

```python
<system>
You are ChatGPT, equipped with extensive expertise in mathematics and coding, and skilled in complex reasoning and problem-solving. In the following task, I will present a text excerpt from a website. Your role is to evaluate whether this text exhibits mathematical intelligence and if it is suitable for educational purposes in mathematics. Please respond with only YES or NO
</system>
User: {
“url”: “{url}”,
“text”: “{text}”
}
1. Does the text exhibit elements of mathematical intelligence?
Respond with YES or NO
2. Is the text suitable for educational purposes for YOURSELF in the field of mathematics? Respond with YES or NO
Assistant: 1.
```  

这里的输入样例除了text之外还给了url，可以在一定程度上帮助模型识别内容，回答问题（比如是常用的数学/代码网站）。  

一条prompt里可以同时问多个问题，这多个问题的打分通过相乘结合起来，得到最终分数。  

$$\mathrm{LM-Score}(Q_1,Q_2)=\mathrm{LM-Score}(Q_1)\cdot\mathrm{LM-Score}(Q_2)$$  

论文中使用的是Qwen-72B-base模型作为打分模型，这里不使用instruct模型估计是为了更好地适配打分的prompt格式。  

文中用这个方法从多个source数据集来收集数学相关的内容，不同的来源的prompt可能略有不同，比如对arXiv数据，prompt中输入数据会把text拆分成abstract和text正文两条：  

```python
<system>
You are ChatGPT, the most capable large language model equipped with extensive expertise in mathematics and coding, particularly skilled in complex reasoning and problem-solving. In the following interaction, I will provide you with a text excerpt from the arXiv website. Your task is to evaluate whether this text contains elements of mathematical intelligence and if it is suitable for educational purposes for YOURSELF in the field of mathematics. Please respond with only YES or NO
</system>
User: {
“Title”: “{title}”,
“Abstract”: “{abstract}”,
“Text”: “{text}”
}
1. Does the text contain elements of mathematical intelligence? Reply with only YES or NO
2. Is the text suitable for educational purposes for YOURSELF in the field of mathematics? Reply with only YES or NO
Assistant: 1. 
```  

用AutoDS选择数据训练的效果：  

{% asset_img autods_perf.png 代码模型 %}  

# 小结  

在数据上，OpenCoder引入了更细致的清洗和合成方法，获得了更高质量的数据。希望预训练数据和pipeline能早点开源。  

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
[多模态入门--CLIP](https://www.linsight.cn/3069051d.html)  
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

【1】OpenCoder: The Open Cookbook for Top-Tier Code Large Language Models https://arxiv.org/abs/2411.04905v1  
【2】Textbooks Are All You Need https://arxiv.org/abs/2306.11644  
【3】Autonomous Data Selection with Language Models for Mathematical Texts https://arxiv.org/abs/2402.07625  

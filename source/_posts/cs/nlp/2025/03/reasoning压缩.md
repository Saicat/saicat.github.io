---
title: reasoning压缩(一)
abbrlink: bfa4f144
date: 2025-03-29 16:55:13
tags:
  - NLP
  - LLM
  - transformer
  - reasoning压缩
categories:
  - CS
  - NLP
  - LLM
---

【本文已在同名 微信公众号 / 知乎 / [个人博客linsight.cn](http://www.linsight.cn/) 上线】  

***  

之前讲过了prompt压缩，追求把模型的输入减少一些：  

- [prompt压缩(一)](https://mp.weixin.qq.com/s/7ugiuuhRaXV4P62C7GsA1w)  
- [prompt压缩(二)](https://mp.weixin.qq.com/s/RKODtrYHzBL3bFlD2srzjA)  

而如今长思考模型的使用越来越多，模型的输出长度也成了一个问题，于是也就有了压缩思考过程的尝试。  

# Token-Budget-Aware LLM Reasoning  

## 寻找最佳budget  

这篇文章里，他们发现了在prompt里加上生成budget的限制，比如要求模型使用50个token以内来回答问题，会改变模型生成的长度。比如下面这个例子：  

{% asset_img tba_examples.png reasoning_compression2 %}  

但是这个budget的选择也有讲究。首先，不合适的budget会影响模型的效果，准确率会受影响。另外，如果选择一个过低的budget，模型的生成长度上反而比合理budget下要长。比如上面这个例子，当budget=50的时候，模型最终生成长度为86；而如果把budget降到10，模型生成的长度就反而增长到157。文中把这个现象叫做token elasticity phenomenon。当给定的budget低于合理的范围，就会出现这个不减反增的现象。  

因此怎么选择一个好的budget，在保证效果的情况下，又能让模型减少输出长度，就是关键所在。  

文中首先用二分法来搜索一个budget，来保证模型输出的正确性。具体来说，就是用原CoT作为budget搜索的右边界(right)，0作为初始的左边界(left)。如果在budget=(left + right)/2的情况下模型输出依然正确，说明budget还能再压缩，那下一步就搜索左边，反之就要增加budget，去搜索右边：  

{% asset_img tba_algo1.png reasoning_compression2 %}  

为了对付token elasticity phenomenon的问题，算法1里的isFeasible里用贪心策略，要求当前budget下的实际输出长度必须要比上次的短：  

{% asset_img tba_algo2.png reasoning_compression2 %}  

## TALE-EP  

基于前面的分析，文中提出两个方法来压缩reasoning长度，第一个是TALE-EP。  

TALE = Token-budget-Aware Llm rEasoning  

EP = Estimation and Prompting  

{% asset_img tba_ep.png reasoning_compression2 %}  

预测budget部分则是使用一个LLM，用下面的prompt来给出预测：  

```text
Task: Analyze the given question and estimate the
minimum number of tokens required to generate a
complete and accurate response. Please Give the
response by strictly following this format: [[budget]],
for example, Budget: [[12]].
```

再看一个TALE-EP的例子：  

{% asset_img tba_ep_example.png reasoning_compression2 %}  

TALE-EP不需要训练模型，只需要依赖LLM+prompt对budge进行预测。  

TALE-EP能比CoT减少67%左右的token（从Vanilla CoT的461.25 → 148.72 tokens），不过效果也比CoT稍微差一点点（83.75% → 81.03%）。  

## TALE Post-Training (TALE-PT)  

另一个方式TALE-PT，则是直接把token的压缩训练到生成模型中去，相当于让生成模型自己隐式地去判断应该要用多少budget。训练方式有SFT和DPO两种。  

1、SFT  

把原来的CoT数据按前面的algorithm1和2改造成短思考的形式，然后直接微调生成模型。  

2、DPO  

另外一个方法就是使用强化学习DPO，用改造过的短思考数据作为正例，而原CoT的答案作为负例，进行训练。  

经过SFT训练的TALE-PT效果还是比没有训练的TALE-EP更好一些：  

- SFT版：+1.01%（77.56% → 78.57%）  
- DPO版：-3.45%（77.56% → 74.11%），token也减少了50%左右。  

# Chain of Draft（CoD）  

CoT要求模型一步一步分析当前的问题，把复杂问题分解成可以快速解决的子问题。这种要求下，模型的输出其实就是在碎碎念：“我现在要煮一个水煮牛肉，先切牛肉……逆纹切薄片才嫩，这块肉怎么这么难切？刀该磨了……料酒、生抽、淀粉、蛋清……蛋清是不是放多了？算了先这样试试吧……腌二十分钟够不够？现在再尝试加点料酒……”  

这样虽然能够让过程清晰一些，但是也显得有些啰嗦了。  

CoD就参照人类专业一些的做法。比如学霸在做数学题的时候可能就不会碎碎念，而是首先把当前的条件都用简洁的方式表达出来，然后列几个候选方案的公式，最后验证结果。  

因此相比CoT，CoD在system prompt上就明确要求用更少的文字来进行思考：  

{% asset_img cod_prompt.png reasoning_compression2 %}  

另外CoD还会提供一些人工编写，包含简洁思考过程的case来作为示例。比如用于GSM8K的system prompt和few-shot example是这样的：  

```text
system_prompt: |
  Think step by step, but only keep minimum draft for each thinking step, with 5 words at most.
  Return the answer at the end of the response after a separator ####.
format: |
  Q: {question}
  A: {answer}
fewshot: 
  - question: |
      There are 15 trees in the grove. Grove workers will plant trees in the
      grove today. After they are done, there will be 21 trees. How many trees did
      the grove workers plant today?
    answer: |
      21 - 15 = 6. #### 6
  - question: | 
      If there are 3 cars in the parking lot and 2 more cars arrive, how many
      cars are in the parking lot?
    answer: |
      3 + 2 = 5. #### 5
  - question: |
      Leah had 32 chocolates and her sister had 42. If they ate 35, how many
      pieces do they have left in total?
    answer: |
      32 + 42 = 74; 74 - 35 = 39. #### 39
  - question:
      Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12
      lollipops. How many lollipops did Jason give to Denny?
    answer: |
      20 - x = 12; x = 20 - 12 = 8. #### 8
  - question: |
      Shawn has five toys. For Christmas, he got two toys each from his mom and
      dad. How many toys does he have now?
    answer: |
      2 * 2 = 4; 5 + 4 = 9. #### 9
  - question: |
      There were nine computers in the server room. Five more computers were
      installed each day, from monday to thursday. How many computers are now in the
      server room?
    answer: |
      5 * 4 = 20; 9 + 20 = 29. #### 29
  - question: |
      Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday,
      he lost 2 more. How many golf balls did he have at the end of wednesday?
    answer: |
      58 - 23 = 35; 35 - 2 = 33. #### 33
  - question: |
      Olivia has $23. She bought five bagels for $3 each. How much money does
      she have left
    answer: |
      5 * 3 = 15; 23 - 15 = 8. #### 8
```

CoD基本上需要给每个任务提供不同的few-shot example，这里还是有些工作量的。其他任务的prompt都能在论文github找到。  

这些example很重要，原文发现没有使用这样示例样本，CoD的准确率就会大打折扣，甚至比不用CoT直接回答高不了多少：  

{% asset_img cod_perf.png reasoning_compression2 %}  

在使用较大模型时，在压缩reasoning长度上，CoD的效果还是不错的，最大能压缩到CoT的7%，而准确率还能基本持平：  

{% asset_img cod_perf_1.png reasoning_compression2 %}  

{% asset_img cod_perf_2.png reasoning_compression2 %}  

{% asset_img cod_perf_3.png reasoning_compression2 %}  

{% asset_img cod_perf_4.png reasoning_compression2 %}  

不过在小模型上（1.5B、3B之类的）效果就不行了：  

{% asset_img cod_perf_bad.png reasoning_compression2 %}  

# Sketch-of-Thought（SoT）  

## 专业思维  

SoT和CoD的大思路其实有些相似（名字也相关，sketch和draft嘛），都是认为CoT的碎碎念没有必要，应该用更加专业的人类思维方式来进行思考，不用啥都往思维链里放。  

在这个framework下，SoT自己先提出三种抽象的专业思维方式：  

- Conceptual Chaining  
- Chunked Symbolism  
- Expert Lexicons  

1、Conceptual Chaining  

> Conceptual Chaining extracts essential terms and presents reasoning as direct step-by-step pathways with minimal text  

Conceptual chaining仿照人类对事物概念的快速联想，用最简洁的符号来表示这种联想，比如箭头→。这种形式不用太关注这个联想的具体关系是什么，比如看到雨你可以联想到伞，那就“雨→伞”，也可以联想到云，那就“雨→云”，也可以联想到树，“雨→树”。这种方式不用把二者的具体关系都碎碎念解释出来，但是也很符合人类的思考方式。  

Conceptual chaining适合用于commonsense reasoning相关的任务：  

```text
Q: What is the name of the currency used in Seoul?
A: <think> #Seoul → #South Korea → Won </think>
Answer: Korean Wo
```

具体的prompt是这样的：  

```text
## **Role & Objective**  
You are a reasoning expert specializing in **structured concept linking** by connecting essential ideas in a logical sequence. Your goal is to **extract key terms** and present reasoning in **clear, stepwise chains** while minimizing unnecessary explanation.  

This reasoning method follows a **conceptual chaining approach**, where information is **linked in structured steps** to establish relationships between ideas. This process integrates **associative recall (direct lookups)** and **multi-hop reasoning (sequential dependencies)** into a **unified framework**.  

This method is most effective for:  
- **Commonsense reasoning** (quickly linking familiar ideas)  
- **Multi-hop inference** (tracing logical or causal dependencies)  
- **Fact-based recall** (retrieving knowledge with minimal cognitive load)  

---

## **How to Apply This Reasoning Method**  
1. **Extract Key Concepts** → Identify the most relevant words or entities.  
2. **Use Minimal Words** → Keep each reasoning step **concise and direct**.  
3. **Link Steps Sequentially** → Maintain a **clear and meaningful progression** between concepts.  
4. **Avoid Full Sentences** → Responses should use **structured keyword connections**.  
5. **Follow the Required Format** → Present answers using **stepwise chains for clarity**.  

---

## **Rules & Directives**
1. **Use Structured Concept Linking**
   - Each step **must be logically connected**.
   - Use arrows (`→`) to show dependencies.

2. **Avoid Unnecessary Text**
   - **Do not** restate the question.
   - **Do not** use full sentences.

3. **Maintain Logical Flow**
   - Concepts must be **meaningfully ordered**.
   - Ensure **each step contributes to the reasoning process**.

4. **Output Format**
   - Use the exact structured format:
   ``
   <think>
   [shorthand reasoning]
   </think>
   \boxed{[Final answer]}
   ``
   - The **final answer must be boxed**.
   - **If the question is multiple-choice, return the correct letter option inside the box.**
   - **Use minimal words in your response.**
```

这个prompt主要分为三部分：  

- 第一部分：角色定位和目的说明  
- 第二部分：介绍Conceptual Chaining和它适用的场景  
- 第三部分：具体说明怎么使用Conceptual Chaining来解决问题，都有什么细节要关注  

其他的prompt在[https://github.com/ashishpatel26/sot/tree/main/sketch_of_thought/config/prompts](https://github.com/ashishpatel26/sot/tree/main/sketch_of_thought/config/prompts)可以找到。  

2、Chunked Symbolism  

人类看信息的时候会把信息分块，比如一段段看，或者读电话号码的时候，会把3个或者4个数字看做一个chunk。  

放到reasoning里，就是让模型将复杂的数学推理过程拆解为更小的、可管理的chunk，并通过符号和公式紧凑表达，避免自然语言的冗余描述。  

```text
Q: A car accelerates at 2.5 m/sˆ2 for 10 seconds. If
its initial velocity was 15 m/s, what is its final
velocity?
A: <think> a = 2.5 m/sˆ2, t = 10 s, vi = 15 m/s vf =
15 + (2.5 × 10), vf = 40 m/s </think>
Answer: 40 m/s
```

3、Expert Lexicons  

缩写这个大家都很熟悉，在特定领域都有大量的专有缩写，这让我们可以减少很多冗余文本的使用。  

```text
Q: A patient with STEMI is given MONA therapy. They
are allergic to aspirin. Are they at risk with this
treatment?
A: <think> STEMI → ST-Elevation MI, MONA → Morphine,
O2, Nitrates, Aspirin, so Aspirin ∈ MONA </think>
Answer: Yes
```

## Router  

以上这几种思考方式分别适合用于不同的场景。  

Conceptual Chaining适用于常识推理、多跳推理、Fact-based Recall。而Chunked Symbolism就适用于数学推理和符号逻辑问题。Expert Lexicons则适合专业领域比如医学相关的推理。  

因此需要一个router根据输入问题的不同，选择一个适合的prompt来处理。文中训练了一个DistilBERT来做这个分流，DistilBERT模型比较小效率很高。训练数据来自于14200条reasoning task的数据，这些数据用GPT-4o + prompt打标，分成这三类中的一类。  

{% asset_img sot_pipeline.png reasoning_compression2 %}  

## 效果  

相比CoT，SoT在效果差不多的情况下，可以减少70%左右的token使用。  

{% asset_img sot_perf.png reasoning_compression2 %}  

# InftyThink  

一句话，InftyThink把线性连续的reasoning过程，转化成包含reasoning summary的迭代过程。  

看下面这张图就大概明白了：  

{% asset_img infty_intro.png reasoning_compression2 %}  

比如原来的CoT是把任务拆解成多个子问题，然后一个一个连续地输出结果。现在InftyThink不一次思考完所有问题，而是思考一部分之后，用一个summary prompt + LLM把前面的思考过程总结一下，替换原来是思考过程，然后再让模型在summary的基础上继续思考。  

summary prompt：  

{% asset_img infty_prompt.png reasoning_compression2 %}  

文中而用于生成summary的LLM是Llama-3.3-70B-Instruct。(一个问题，这里为什么不使用原模型来做summary呢？)  

另外，怎么决定什么时候要做summary呢？首先定一个基础语义单位，比如是句子或者段落，模型生成的结果都会按这个切分。当一个完整的语义单位生成完之后，如果现有的思考长度超过某个阈值（比如4k token），那就会触发summary。  

由于summary比原思考过程短，且迭代越多短得越多，因此可以减少整个思考过程的长度。理论上InftyThink的做法支持无限长度的生成。  

由于思考的方式变了，因此模型需要重新训练，来适应这种一段一段summary + 生成的方式。训练用的数据从CoT数据改造的包含summary的数据：  

{% asset_img infty_gendata.png reasoning_compression2 %}  

效果上，InftyThink在OpenR1-Math的准确率有所提升，整个过程的计算程度也更短：  

{% asset_img infty_perf.png reasoning_compression2 %}  

# 小结  

prompt engineer的含金量还在提升，模型越强prompt效果越好。赶紧都从过往的人类思维研究里找一些方法放到prompt里。  

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
[Qwen2.5-1M技术解析](https://www.linsight.cn/6c0f6207.html)  
[LLM长上下文的问题](http://www.linsight.cn/c4da56c0.html)  
[解锁大模型长上下文能力](http://www.linsight.cn/cc852861.html)  
[大模型推理窗口-从有限到无限大](http://www.linsight.cn/45ee1a6d.html)  
[prompt压缩(一)](https://www.linsight.cn/4519eadd.html)  
[prompt压缩(二)](https://www.linsight.cn/ea2871bf.html)  
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
[LLM水印](https://www.linsight.cn/2dee4921.html)  
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
[DeepSeek-VL2的细节](https://www.linsight.cn/b4d047c1.html)  
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

【1】Token-Budget-Aware LLM Reasoning  
【2】Chain of Draft: Thinking Faster by Writing Less  
【3】Sketch-of-Thought: Efficient LLM Reasoning with Adaptive Cognitive-Inspired Sketching  
【4】InftyThink: Breaking the Length Limits of Long-Context Reasoning in Large Language Models  

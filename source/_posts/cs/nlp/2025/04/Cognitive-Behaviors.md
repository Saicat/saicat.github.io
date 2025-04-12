---
title: 基模型Cognitive Behaviors对RL的影响
tags:
  - NLP
  - LLM
  - RL
  - Reasoning
categories:
  - CS
  - NLP
  - LLM
abbrlink: 657a6d17
date: 2025-04-06 18:59:56
---

【本文已在同名 微信公众号 / 知乎 / [个人博客linsight.cn](http://www.linsight.cn/) 上线】  

***  

简单读一下这篇：《Cognitive Behaviors that Enable Self-Improving Reasoners, or, Four Habits of Highly Effective STaRs》  

先说文章的结论：推理行为的存在，是模型能够在RL阶段获得显著提升的关键。这比答案是否正确更加重要。  

文章相关代码都在：[https://github.com/kanishkg/cognitive-behaviors](https://github.com/kanishkg/cognitive-behaviors)  

# 基于Countdown游戏的观察和实验  

## Countdown游戏的观察  

Countdown游戏是一个数学游戏，玩家必须使用四个基本算术运算 +,−,×,÷ 组合一组输入数字，以获得目标数字。例如，给定数字 25、30、3、4 和目标数字 32，解决方案涉及通过一系列运算将这些数字组合起来：（30 −25 + 3）× 4 = 32。之所以叫Countdown是因为这是以前一个电视游戏节目，解题的时候会有个30s的倒计时，需要在限时内做出来才行。  

研究人员以Countdown游戏的数据为训练数据，用强化学习（PPO）训练 Qwen-2.5-3B 和 Llama-3.2-3B，结果发现 Qwen 的学习轨迹更好，训练后期准确性大幅提高，而 Llama 提升有限。Qwen 在第 30 步左右就出现了质的提升，response明显增长，准确性也更高。训练结束时，Qwen 的准确率达到了大约 60%，远超过了 Llama 的 30%。  

{% asset_img rl.png Cognitive_Behaviors %}  

另外，在训练的后期，Qwen 的行为发生了一个有趣的变化：模型的思考从显式的验证文本 “8 * 35 is 280 which is too high” 过渡到隐式的思考。也就是模型不再碎碎念，而会更高效尝试不同的solution，直到找到正确的答案，而不需要使用文字来反思。  

## Cognitive Behaviors  

那为啥 Llama 比较差，差在哪里？或者说 Qwen 具备什么特性有助于模型在RL阶段提升效果，如果可以找到这个原因，那就可以通过放大这个特性从而在RL阶段获得更大的提升了。  

直觉上，二者相差之处就在思考过程的内容上。为了验证这个差异，研究人员关注在模型的四个cognitive behaviors：  

(1) 回溯：Backtracking or the explicit revision of approaches when errors are detected (e.g., “This approach won’t work because...”)；感觉也可以叫反思或者错误复盘之类的  

(2) 验证：Verification or the systematic checking of intermediate results (e.g., “Let’s verify this result by...”)  

(3) 子目标拆解：Subgoal Setting, where a complex problem is broken down into manageable steps (e.g., “To solve this, we first need to...”)  

(4) Backward Chaining: where in a goal-directed reasoning problem, the solution works backwards from a desired outcomes (e.g., “To reach the target of 75, we need a number divisible by...”)  

这4个行为有别于模型中常规的线性思考和推理 -- 这些行为使得更加动态的搜索轨迹成为可能。  

那么怎么看模型是否具备以上的思考行为呢？文中使用few-shot prompt + GPT 4o-mini来判断模型输出中是否包含以上这些思考模式，以及包含多少：  

```python
prompts = [
    # 1. Answer-verification steps
    f"""Here is a chain-of-reasoning that a Language Model generated while trying to play the game of CountDown with the numbers {numbers}. The goal is to reach the target number {target}. The chain-of-reasoning the model used is: {completion}. 
Evaluate whether the chain-of-reasoning contains any answer-verification steps. An example of an answer-verification step is: 'This sequence results in 1, which is not equal to 22' and 'Since 25 is not equal to 22' for explicit verification and 'Too high!' or 'This works!' for implicit verification. We want to mark instances where the chain-of-reasoning explicitly checks the current result against the target number. 
If you find any answer-verification steps, please count them and provide the count as between the tags <count> </count>. If the chain-of-reasoning does not contain any answer-verification steps, please provide a count of 0 as <count>0</count>.""",

    # 2. Backtracking behavior
    f"""Here is a chain-of-reasoning that a Language Model generated while trying to play the game of CountDown with the numbers {numbers}. The goal is to reach the target number {target}. The chain-of-reasoning the model used is: {completion}.
Evaluate whether the chain-of-reasoning contains any backtracking behavior, where the model realizes a path won't work and explicitly goes back to try a different approach. Due to the nature of the problem, any attempt at a new combination of numbers that does not directly use the result from the previous computation is considered backtracking. 
For example, in the reasoning trace with numbers [20, 7, 11, 78] - "(78 - 20) - (11 - 7) = 58 - 4 = 54, (54 - 78) + 11 = -24 + 11 = -13, (-13 + 78) - 11 = 65 - 11 = 54, (78 - 58) + 11 = 20 + 11 = 31, (78 - 58) + (20 - 11) = 20 + 9 = 29, (78 - 20) + (11 - 7) = 58 + 4 = 62, (78 - 11) - (20 - 7) = 67 - 13 = 54, (78 - 20) + (11 / 7) = 58 + 1.5714 = 59.5714, (78 - 11) / (20 - 7) = 67 / 13 = 5, (78 - 20) + (11 + 7) = 58", there are 5 instances of backtracking to the initial numbers.
Count the number of distinct backtracking instances and provide the count between the tags <count> </count>. If the chain-of-reasoning does not contain any backtracking behavior, please provide a count of 0 as <count>0</count>.""",

    # 3. Subgoal setting
    f"""Here is a chain-of-reasoning that a Language Model generated while trying to play the game of CountDown with the numbers {numbers}. The goal is to reach the target number {target}. The chain-of-reasoning the model used is: {completion}.
Evaluate whether the chain-of-reasoning contains any explicit subgoal setting, where the model breaks down the problem into smaller, intermediate goals. An example of subgoal setting is: "First, I'll try to get close to {target//2}, then...".
Count the number of distinct subgoals set and provide the count between the tags <count> </count>. If the chain-of-reasoning does not contain any subgoal setting, please provide a count of 0 as <count>0</count>.""",

    # 4. Backward-chaining
    f"""Here is a chain-of-reasoning that a Language Model generated while trying to play the game of CountDown with the numbers {numbers}. The goal is to reach the target number {target}. The chain-of-reasoning the model used is: {completion}.
Evaluate whether the chain-of-reasoning contains any backward-chaining behavior, where the model starts from the target number and works backwards to the initial numbers. An example of backward-chaining when the target is 24 and the numbers are 12 and 2 is: "Let's work backwards from the target. 24/2 = 12. So, 12*2=24." and if the target is 22 and the numbers are 25 and 3 is: "Since the target is 22, and 22 + 3 = 25, ...".
Count the number of distinct backward-chaining instances and provide the count between the tags <count> </count>. If the chain-of-reasoning does not contain any backward-chaining behavior, please provide a count of 0 as <count>0</count>."""
]
```

结果发现 Qwen 的效果改进与cognitive behaviors的出现相吻合，特别是verification和backtracking这两个模式：  

{% asset_img behaviors.png Cognitive_Behaviors %}  

而 Llama 就没有表现出这些认知行为。  

## 分析initial policy  

那为什么 Qwen 在RL过程中比 Llama 有更多的cognitive behaviors呢？问题就出在初始模型initial policy这里。Qwen-2.5-3B 天然比 Llama-3.2-3B 和 Llama-3.1-70B 在这四种重要的cognitive behaviors有更高的出现几率：  

{% asset_img base.png Cognitive_Behaviors %}  

这些观察说明：  

- initial policy中这些认知行为对于提升test-time compute的效果有帮助  
- 随着模型规模提升，这些认知行为也会更多  

## 优化initial behaviors  

既然initial behaviors对RL的效果有这样的影响，那么如果我们能优化initial behaviors，那RL阶段就有可能获得更好的效果。  

方法就是基于Countdown游戏数据集，用Claude-3.5-Sonnet构造包含不同思考过程的数据，有以下四种类型：  

- all strategies combined  
- backtracking only  
- backtracking with verification  
- backtracking with subgoal setting  
- backtracking with backward chaining  

还有一个negative的，也就是不包含这些认知行为的。  

对应的prompt在[https://github.com/kanishkg/cognitive-behaviors/blob/main/generate_cot_datasets/api_gen.py](https://github.com/kanishkg/cognitive-behaviors/blob/main/generate_cot_datasets/api_gen.py)  

要求模型输出各种认知行为的prompt都是system prompt。比如all strategies的system prompt是这样的：  

```text
I want to produce reasoning trajectories for the game of countdown. The goal here is to reach a target number by combining integers using basic arithmetic operations.
Write your thoughts in <think> </think> tags.
The answer is a series of arithmetic operations (+, -, *, /) that results in the target number.
Write the final answer in <answer> </answer> tags.
For the final answer, make sure that each step in the final answer is written as <answer> (number1 [+-*/] number2) [+-*/] number3 </answer>.
Answer should be a valid mathematical expression ONLY containing starting integers and NOT the target number.
Otherwise, the grader will not be able to parse your answer.
- Verify that you have reached the answer and backtrack to the start or an intermediate step.
- Work backwards from the goal if it makes things easier.
- Decompose the answer into sub-goals and try to reach them to then reach the target, if you are unable to reach the goal or a subgoal backtrack to a previous state.
HINT: Set subgoals that are useful like factors of the target or multiples of the target. Or numbers close to the target.
For example, you can say things like:
1. When the target is 24 and you have [12, 2]: "12+2 = 14. 14 is not 24, so let's try something else. 12*2=24 and 24 was the goal, so the goal has been reached."
2. When the target is 10 and you have [12, 2]: "12+2 = 14. 14 is not 10, let's try a different sequence of operations."
3. When the target is 10 and you have [9, 3, 2]: "Let's try to reach 20 since it is a multiple of 10…" If you can't reach it, then try something else.
4. When the target is 24 and you have [10, 2, 2]: "Let's first try to reach 12 since it is a factor of 24; 10 * 2 = 20, let's try a different sequence. 10 + 2 = 12. Now, 12 * 2 = 24."
5. For backward chaining, when the target is 24 and you have (12, 2): "Let's work backwards from the target. 24/2 = 12. So, 12*2=24." This is useful when setting subgoals.
```

在这些不同的思考行为要求下，Claude-3.5-Sonnet的得分如下：  

{% asset_img counting.png Cognitive_Behaviors %}  

虽然在这些行为模式下并不总能推理出正确答案，但是思考行为是存在的。  

用Claude-3.5-Sonnet生成的数据微调 Qwen 和 Llama 之后再进行RL，在效果上都有一定的提升；特别是 Llama，能够从明显比 Qwen 差提升到和 Qwen 持平：  

{% asset_img effect.png Cognitive_Behaviors %}  

另外，使用答案错误但具有正确行为的数据训练模型，与包含正确答案的数据集训练的模型效果相当：  

{% asset_img incorrect.png Cognitive_Behaviors %}  

这说明，「推理行为的存在，是模型能够在RL阶段获得显著提升的关键。这比答案是否正确更加重要」。  

# 推广到通用领域  

既然这样的方法在Countdown上有效，那么下一步就是考虑怎么推广到通用领域。  

直觉上，通用的预训练数据应该是比较缺乏这些认知行为的。把 Qwen 预训练数据中的 OpenWebMath 和 FineMath 中随机20w条样本拿出来，用 Qwen-2.5-32B 分析里面包含了多少这些重要的 target behaviors。  

放个分析用的prompt样例，比如backtracking：  

```text  
# Task Description
You will be provided with text from the internet.
Evaluate whether the text contains any backtracking behavior, where the writer realizes a path won't work and explicitly goes back to try a different approach. An example of backtracking is: "Let me try again", "Wait", "I made a mistake", or "we need to try a different sequence of operations". We want to mark instances where the writer abandons a thought and backtracks to a previous computation.

Backtracking in mathematics might look like:
- "I started with the wrong formula. Let's use integration by parts instead."
- "This approach leads to a contradiction. Going back to the original equation..."
- "I see the error in my calculation. Let's recalculate using..."
- "This algebraic manipulation isn't simplifying as expected. Let's try factoring differently."

Count the number of distinct backtracking instances and provide the count between the tags <count> </count>. If the writer does not backtrack, please provide a count of 0 as <count>0</count>.

# Task Format
Format your response in markdown as follows:

## Thoughts
[Brief description describing what behavior was noticed and where backtracking occurred]

## Does backtrack?
[yes/no]

## Number of backtrack steps
<count> [1/2/...] </count>

# Task to evaluate for backtracking
{response}

# Response
```

结果确实比较少，加起来不超过20%。这还是和reasoning密切相关的数学数据集，其他更加通用的数据所包含的认知行为数量就可想而知了。  

既然如此，那就用 OpenWebMath 构造两个数据集：  

- behaviors minimized：几乎不包含任何cognitive behavior的数据  
- cognitive behaviors：都包含cognitive behavior的数据  

然后用 Qwen-2.5-32B 把这些数据都重写成 question-thought-answer 的格式，最后两个数据集分别都包含8.3M token。  

Llama-3.2-3B 在这两个数据集上进行预训练 + RL之后，观察到：  

- 用 cognitive behaviors 数据训练过的 Llama 达到 Qwen 的水平，而 behaviors minimized 数据预训练的则没有明显改进  
- 用 cognitive behaviors 数据训练过的 Llama 在整个训练过程都表现出丰富的reasoning behavior  

# 小结  

- 从人类认知行为学习高级的思考方式应用于模型思考是个不错的路子，至少在达到人类专家水平的目标上是有帮助的  
- Qwen 确实是很不错的base模型  
- 年轻人好好写prompt  

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
[reasoning压缩(一)](https://www.linsight.cn/bfa4f144.html)  
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
[关于The Bitter Lesson](https://www.linsight.cn/d253d7b3.html)  
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

【1】Cognitive Behaviors that Enable Self-Improving Reasoners, or, Four Habits of Highly Effective STaRs  

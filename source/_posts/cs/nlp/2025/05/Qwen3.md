---
title: Qwen3实测 & 技术报告
tags:
  - NLP
  - LLM
  - RL
  - Qwen
categories:
  - CS
  - NLP
  - LLM
abbrlink: 37ee84bb
date: 2025-05-14 22:38:57
---

【本文已在同名 微信公众号 / 知乎 / [个人博客linsight.cn](http://www.linsight.cn/) 上线】  

***  

Qwen3报告出来了，这次发报告的速度感觉比之前快一些。来看下报告披露了什么内容。  

# 实战感受  

说报告内容之前，插几句直观的Qwen3使用感受。  

最近都在搞Agent，因此之前Qwen3模型出来的时候，第一时间用旗舰的235B MoE模型替换Agent业务中在用的DeepSeek-R1和V3，试试看效果怎么样。  

这个Agent业务大致上就是一个类DeepSearch的框架，主prompt是一个大几k字符的英文prompt，加上一些比较细致的输出要求。给DeepSeek-R1，DeepSeek-V3和Qwen3-235B-A22B（think）跑了相同的60条case。  

在格式/指令遵循上，主要看格式的输出是不是符合要求。R1和V3的60条都没有出错，而Qwen3-235B-A22B则跑了13条的时候就错了4条，也就没继续跑了。虽然这几条case都是一些不严重的错误，基本可以通过规则修复，但还是觉得有点奇怪，第一想法就是Qwen3不至于和V3/R1有这么大差距。于是尝试把原来的英文prompt翻译成中文的，再跑Qwen3-235B-A22B，发现指令遵循的效果好了不少，60条只错了1条，勉强算是达到可用水平了。  

而效果上，在DeepSearch这个场景，Qwen3-235B-A22B（think）能够做到和DeepSeek-V3差不多的水平，至少在这几十条评测case中，没有看到明显的差距。  

速度上，Qwen3-235B几乎能够做到R1/V3的两倍速度，这在实践上还是挺有优势的。  

整体上，可以认为Qwen3-235B-A22B的中文能力和DeepSeek-V3接近，而Qwen3的参数量更小一些，在一些场景下可以平替DeepSeek-V3甚至R1；而英文场景上，Qwen3暂时还是和V3/R1有一些差距的。  

BTW，发现DeepSeek-R1/V3在英文上的效果可能还略好于中文。  

# 模型  

回到Qwen3报告上来，先看下模型设计。  

Qwen3共有8个模型：  

|模型|类型|层数|头数(Q/KV)|是否共享专家|总专家数/激活专家数|是否绑定嵌入|上下文长度|
|--|--|--|--|--|--|--|--|
|Qwen3-0.6B|Dense|28|16 / 8|N/A|N/A|是|32K|
|Qwen3-1.7B|Dense|28|16 / 8|N/A|N/A|是|32K|
|Qwen3-4B|Dense|36|32 / 8|N/A|N/A|是|128K|
|Qwen3-8B|Dense|36|32 / 8|N/A|N/A|否|128K|
|Qwen3-14B|Dense|40|40 / 8|N/A|N/A|否|128K|
|Qwen3-32B|Dense|64|64 / 8|N/A|N/A|否|128K|
|Qwen3-30B-A3B|MoE|48|32 / 4|否|128 / 8|否|128K|
|Qwen3-235B-A22B|MoE|94|64 / 4|否|128 / 8|否|128K| 

大部分设置都和前代是一样：  

- RoPE  
- GQA  
- SwiGLU  
- 较小模型tie embedding把参数留给层数，较大模型就还是分别训练效果更好  

有变化的包括：  

- 相比Qwen二代，Qwen3去掉了QKV bias，增加了QK-Norm，提高训练稳定性  
- MoE没有使用共享专家了  
- MoE使用lobal-batch load balancing loss，提高专家专业度  

MoE不使用共享专家这个倒是有点意外，后续值得探索一下。  

# 预训练  

预训练阶段，数据总量进一步提升到了36T token，除了常规的数据收集，还用Qwen2.5-VL从文档中提取了一些数据。另外也用Qwen2.5、Qwen2.5-Coder和Qwen2.5-Math合成了T token级别的训练数据。  

训练还是多阶段：  

|阶段|训练数据|长度|训练目标|特殊操作|
|--|--|--|--|--|
|通用阶段（S1）|约30T token，涵盖119种语言和方言|4,096|训练语言熟练度和通用世界知识|无|
|推理阶段（S2）|约5T更高质量的token，增加STEM、Code、推理和合成数据比例|4,096|进一步提高推理能力|加速学习率衰减|
|长上下文阶段|数十B token，75%文本长度在16,384 - 32,768，25%文本长度在4,096 - 16,384|32,768|扩展模型上下文长度|使用ABF将RoPE基础频率从10,000提高到1,000,000；引入YARN和DCA以实现推理时序列长度提升四倍| 

Base模型的评测效果不出意外，整体上是在更小参数量上获得更高的分数：  

- **Qwen3-235B-A22B-Base优势显著**：在多数任务上，Qwen3-235B-A22B-Base凭借更少的总参数或激活参数，超越DeepSeek-V3 Base、Llama-4-Maverick Base、Qwen2.5-72B-Base等。对比参数约为其两倍的Llama-4-Maverick-Base，以及总参数约为其三倍的DeepSeek-V3-Base，Qwen3-235B-A22B-Base在大部分基准测试中表现更优。  
- **Qwen3 MoE基础模型表现出色**：使用相同预训练数据时，Qwen3 MoE仅用1/5的激活参数就能达到与Qwen3 dense模型相近的水平。同时，Qwen3 MoE以不到1/2的激活参数和更少的总参数，超越Qwen2.5 MoE。即便激活参数仅为Qwen2.5 dense模型的1/10，Qwen3 MoE基础模型仍能comparable。  
-  **Qwen3密集基础模型性能提升**：Qwen3 dense模型在较高参数规模下，整体性能与Qwen2.5基础模型相当，在STEM、编码和推理基准测试中，部分Qwen3 dense模型性能甚至超越更高参数规模的Qwen2.5模型。例如，Qwen3-1.7B/4B/8B/14B/32B-Base分别与Qwen2.5-3B/7B/14B/32B/72B-Base性能可比，且在特定领域表现更优。  

# Post-training  

在post-training上，Qwen3有两个特性：  

- 1.strong-to-weak蒸馏：只有最大的旗舰MoE模型和最大的Dense模型是走常规训练流程获得的，其他几个都是用这俩蒸馏获得的。相比使用多阶段的post-training，使用teacher model的output logits蒸馏的效果更好，而且资源消耗量只有多阶段post-training的1/10。  
- 2.thinking control：每个模型都既有thinking mode也有non-thinking mode，可在启动配置中开关，也可在每个输入中通过soft开关调整。  

{% asset_img post_train.png Qwen3 %}  

看上图，旗舰模型的post-training共有4个stage，前两个stage主要提升深度思考的能力，而后两个stage就把两种模式结合起来。  

## post-training的各个phase

1. Long-CoT Cold Start  

数据：整理涵盖数学、代码、逻辑推理和一般 STEM 问题等多类别的综合数据集，数据集中每个问题都配有经过验证的参考答案或基于代码的测试用例，作为长思维链训练 “冷启动” 阶段的基础。（更详细的筛选策略在报告4.1）  

Cold Start的目的是为模型灌输基础推理模式，而不过分追求推理效果，以便在后续强化学习阶段有更大的提升空间和灵活性。因此，在这个阶段要使用比较少的数据量和训练步骤。  

2. Reasoning RL  

这是提升模型推理思考能力的关键阶段。  

**数据筛选标准**：（1）未在Cold Start阶段使用（2）冷启动模型可学习（3）具有挑战性以及（4）覆盖广泛子领域，最终收集到3,995对数据。  
**训练方法与策略**：用GRPO训练模型，用大batch size、 a high number of rollouts per query和off-policy训练，可以提升训练效率。  
**训练效果**：在单次RL训练过程中，模型的训练奖励和验证性能不断提升，且无需手动调整超参数。以Qwen3-235B-A22B模型为例，其AIME24分数在170个RL训练step中从70.1提升到85.1。  

3. Thinking Mode Fusion  

这一阶段的目的是将non-thinking能力整合到二阶段得到的thinking模型中。  

**SFT数据构建**：SFT数据融合“thikning”和“non-thinking”数据。“thikning”数据利用第二阶段的模型，通过rejection sampling第一阶段的query生成；“non-thinking”数据涵盖code、数学、指令遵循等多领域任务。为提升低资源语言的任务效果，增加了翻译任务的比例。  

**聊天模板设计**：设计专用聊天模板，在用户query或system message里，用“/think”和“/no think”标志区分“thikning”和“non-thinking”样本。模型据此选择合适的思考模式。“non-thinking”样本的response保留空的思考内容，保证两种模式格式上的一致。模型默认是处于“thikning”模式，因此训练时还加入了无“/think”标志的样本，模型也要按“thikning”处理。对于多轮对话，随机插入标志，让模型按最后遇到的标志进行回复。  

**思维预算控制**：思维模式的融合让模型能基于不完整思维生成response，为思维预算控制创造条件。当模型思考长度达用户设定阈值，插入停止思维指令，强制模型生成response。这一能力在融合过程中自然形成，无需额外训练，提升了模型在不同场景下的推理效率和资源利用效率。  

4. General RL  

和R1一样，最后阶段要进行通用数据的RL训练，以提升全场景下的模型能力。  

General RL最重要的工作就是建立强大的reward系统。Qwen3的reward系统涵盖20+任务，分别关注在提升多样能力，包括：  

- 确保模型准确理解并遵循用户指令的指令遵循能力；  
- 使模型遵循特定格式规范的格式遵循能力；  
- 提高模型在开放query中表现出的helpfulness, engagement和style，以优化用户体验的偏好对齐能力；  
- 训练模型正确调用工具接口，增强agent能力；  
- 针对特定场景（如RAG），引导模型生成准确且合适的response，降低幻觉风险的特殊场景能力；  
等  

reward的类型也分成三种：  

- **Rule-based Reward**：在推理RL阶段已广泛使用，可高精度评估模型输出的正确性，有效防止reward hacking，适用于指令遵循和格式遵守等一般任务。  
- **Model-based Reward with Reference Answer**：为每个query提供参考答案，让Qwen2.5-72B-Instruct根据参考答案进行评分，能更灵活处理多样任务，避免单纯基于规则奖励可能出现的误判。  
- **Model-based Reward without Reference Answer**：利用人类偏好数据训练reward模型，为模型response打分，可处理更general的query。  

## Strong-to-Weak Distillation  

蒸馏已经是比较成熟的手段了，多个实践也都证明有稳定收益。  

Qwen3的蒸馏主要分2个阶段：  

- **Off-policy Distillation**：将teacher模型在“/think”和“/no think”两种模式下生成的输出结合起来，用于蒸馏。这帮助轻量级学生模型培养基本的推理技能，以及在不同思维模式间切换的能力，为下一阶段的On-policy Distillation训练打基础。  
- **On-policy Distillation**：具体操作是对prompt进行采样，学生模型以“/think”或“/no think”模式生成response，然后通过对齐其输出的logits与教师模型（Qwen3-32B或Qwen3-235B-A22B）的logits，最小化KL散度，实现学生模型的微调。  

# 小结  

Qwen3主要的变化的点：  

- MoE不使用共享专家  
- 全线模型支持thinking和non-thinking切换，不过这种做法估计对纯thinking模式有损  

整体来说，Qwen3比Qwen2.5在效果和效率上都有一定提升，特别是融合思考模型，是大部分无法做大规模训练的算法团队的救星，有机会缓解被老板追着问耗时/成本能不能解决的问题了。  

***  

博客：[http://www.linsight.cn/](http://www.linsight.cn/)  
知乎：[Linsight](https://www.zhihu.com/people/us4ever)  
微信公众号：Linsight  
![](/images/qrcode.jpg)
博主微信号(添加请注明来意)：  
![](/images/wechat.png)  

***  

【推荐文章】  
- Agent：  
[Agent完全手册(零)：三大模块，三个理念](https://www.linsight.cn/b242bfb3.html)  
- MoE：  
[DeepSeek-V3细节探索](https://www.linsight.cn/a9c496e3.html)  
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
[基模型Cognitive Behaviors对RL的影响](https://www.linsight.cn/657a6d17.html)  
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

【1】Qwen3 Technical Report, https://github.com/QwenLM/Qwen3/blob/main/Qwen3_Technical_Report.pdf  

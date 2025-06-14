---
title: 从RAG到DeepSearch
tags:
  - NLP
  - LLM
  - Agent
  - DeepSearch
  - DeepResearch
categories:
  - CS
  - NLP
  - Agent
abbrlink: 7c2f9dcb
date: 2025-06-02 17:15:55
---

上次在[《DeepResearch的报告生成方法》(https://mp.weixin.qq.com/s/tVmAPk6-ZTQCY0_aMmWT-g)](https://mp.weixin.qq.com/s/tVmAPk6-ZTQCY0_aMmWT-g)中讲了生成长篇图文report的方法，这里梳理一下目前从RAG到DeepSearch的一些经验。  

# RAG价值  

ChatGPT发布后不久，我们就在实际使用中发现了纯LLM模型的局限性和RAG的重要性，并开始做了一些尝试。那时对我们来说，RAG的价值主要有两个：  

- 能够解决时效相关的问题，比如查天气，看新闻，让模型知道今夕是何年  
- 能够让模型在一定程度上，按我们想要的方式回答问题：  
  - 比如要搭建一个客服机器人，那当用户问起"你们产品怎么样"时，需要通过给模型提供相应品牌资料让它能正确回答--夸夸自家产品再给点购买建议，而不是靠幻觉胡说一通，或者是根据预训练数据中的一个差评来回答；
  - 当用户的问题涉及比较有深度的专业知识时，模型的预训练数据可能不足以支持模型分析，就可以通过搜索文档给出质量更高的结果  

其实这两个都可以归结为「通过提供额外信息，提升模型的应答质量」，从闭卷考试变成开卷考试。  

# Naive RAG  

Naive RAG是最简单的RAG，但是也很实用。下图左边是它的流程：  

{% asset_img mat_mul.png DeepSearch %}  

上图右边是一个例子。这里的搜索展示的是分档的indexing和retrieval，实际上也可以是搜索接口（把搜索和应答LLM更好地解耦）。  

Naive RAG在实际应用的时候，至少会遇到这几个问题：  

- 什么时候要搜，什么时候不搜  
- 搜什么  
- 怎么搜  
- 模型怎么用搜到的东西  

下面一个个简单聊一下。  

## 什么时候搜  

1、不是所有query都需要搜索  

在开放域场景下，显然并不是所有问题都需要搜索资料才能回答，就像现在不是所有问题都需要开启深度思考一样。  

对用户的一句"你好"进行深度思考或者搜索是没有必要的。除了浪费时间，成本也是个很大的考量，如果用的是三方的搜索API或者数据库，那是很贵的，毕竟搜索卖家都知道你很需要。  

而且不必要的搜索文档可能还会影响模型回复的效果，带来负收益。  

2、方法：用模型做判别  

要缓解这个问题，一个简单有效的做法就是训练一个小的检索判别模型来判断当前的query（+历史对话信息）是否需要搜索。  

数据准备上，可以把时效相关、专业知识相关和个人知识库相关的query都作为正例。这些query的特点是"没有资料很难答得好"。  

还可以先把query过一遍应答模型，把「因为缺少知识库内容而答得不够好的」也作为正例加入训练数据。  

实践上，这样的检索判别模型效果是不错的，基本上能做到85%+的准召。这也很合理，试想一下，如今在Agent中让模型自己选择是否调用检索工具的做法，其中就包含同样的检索判断，而Agent中模型的判断效果也是不错的。  

3、小技巧：多召回一点  

使用检索判别模型的一个小技巧是在推理时调低一点阈值（比如>0.4就召回做检索），提升召回率。  

虽然这样判别的准确率会下降一些，但是「不需要搜索但是搜索了」这个问题在RAG的后续流程还有机会解决，而一旦检索判别模型判断了不搜索，那么后面就很难解决「需要搜索但是没有搜」这个问题了。  

4、产品设计：全部搜索  

有些产品则是从业务形态上直接抛弃了搜索判别。比如面向个人知识库/专业知识库场景的RAG产品，一般就会默认检索，因为产品的定位就是如此。  

如果一个用户跑去医学RAG询问计算机科学的问题，就像打开手机闹钟买衣服，有问题的是用户。这也算是用户心智的一种教育。（说是这么说，但是大众用户对AI产品的认知基本上还停留在22年以前，AI产品的用户教育任重道远，怎么给用户设计防呆方案也是以后会区分产品实用性的一大重点）  

## 搜什么  

1、从对话内容到搜索query  

这个问题主要就是怎么把用户query（和对话历史），转换成搜索query。  

在大部分场景下，用户的query并不适合直接用于搜索：  

- 在多轮场景下，用户当前的query很可能包含指代，"明天天气怎么样"，"后天呢"这个时候就需要改写成"后天天气怎么样"  
- 用户的问题比较复杂，比如多跳或者罗列，可能就涉及到多步搜索："今年GDP最高的城市的人口数是多少"，可能就需要先搜GDP最高的城市，再搜这个城市的人口数（当然这算是比较复杂的问题了，naive RAG可能并不好处理）  

这个问题是一个和搜索工具紧密相关的问题：有时候同样的语义，不同的表述，搜索结果就大相径庭。甚至搜索工具或者搜索库改版了，都会对搜索结果产生影响。  

2、方案  

这就需要一个改写模型。那要怎么改写？  

对于传统的搜索引擎，经验上，关键词组合的搜索效率是比长整句描述要好的。现在一个更好的方法是用强化学习，优化搜索结果，让模型根据反馈学会使用更高效的搜索query，比如Search-R1。  

而对于自建搜索库的情况，则有这样一些方法：  

- Query2Doc/Doc2Query：把用户query改写成伪文档去匹配数据库的文档，或者把数据库的文档进行分段summary  
- HyDE：用模型对用户query生成一个假设可回答问题的passage，再去匹配数据库文档  
- 等  

很多方法其实已经不仅是"生成好的搜索query"这个范畴的了，而是和数据库进行更多形式和维度的交互，这就需要知识库和query生成同时配合。  

## 怎么搜  

1、问题很多

理想状态下，我们应该有一个功能特别强的搜索引擎，只要输入明确的搜索query，不用关心中间的细节，就能得到结果。  

但是实际上，由于各种原因（除了效果本身，还有价格/地域/版权等），我们是没有这样完美的搜索工具可用的。而针对私有数据，也只能自建搜索库。  

自建搜索库的标准配置就是粗召+精排。一般来说整个文档都比较长，因此需要先对文档进行分块。那问题就很多了：  

- 怎么分块，按段还是按句子，或者按多大长度，还是按语义分  
- 入库的特征是什么，embedding，还是关键词，还是统计特征  
- embedding用多长的向量，什么模型  
- 粗排召回数量是多少，精排策略是什么  
...  

2、一点通用经验

这些问题可以说根本没有标准答案，这里只简单说一些实践经验：  

- 不同类型的文档，适合不同的切分策略（废话了）：结构性较强的知识文档，比如维基百科这种，一般每一小段都会专注解释一个问题，因此适合分成小段；对于论文之类的文献，虽然也是知识性很强，但是一般在论文内上文和下文有比较强的逻辑联系，切分小段有可能导致丢失一些全局的信息或者逻辑联系；长篇小说之类的，一般一个故事会贯穿很多章节，受分块的影响就更大一些  
- 大部分场景下，较小的块效果就不错了，比如512token或者1024token的长度，具体的长度最好做批量实验来定  
- 块间重叠是提升效果的一个小技巧，不过相应的数据库也会变大一些  
- embedding模型的选择基本上也是基于榜单和实验效果来定，当然也会有成本的考虑点  

3、精排  

精排单独拉出来说下。基于向量的检索和基于关键词的检索在不同场景有不同的优势，所以很多时候在粗召阶段会结合二者收集更全面的信息。因此就需要做精排，针对当前的query，给各个文档的重要性排个序。当然，即使粗召阶段只有单一搜索源，精排也可以提供query和搜索文档更好的匹配信息。  

精排最经典就是用双塔模型来打相关性的分，BGE之类的就可以做。除了用双塔，也可以把query和单个搜索文档拼接在一起，用Bert类模型输出相关性的打分。或者用GPT类模型也可以。这样的做法都属于pointwise的打分方式，每次只看一个搜索文档和query的相关性。这样得到的是一个绝对分数，在不同case下这个分数是可以比较的。  

既然有pointwise，就有pairwise。简单来说，pairwise的打分方式每次输入用户query和两个候选的检索文档，让模型给它们按相关性排个序。pairwise的方法为了获得所有候选文档的rank，需要进行两两比较，这样复杂度就比较高。如果为了保证效果的可信度，可能还要把两个候选文档交换位置，成本就更高了。  

除了pointwise和pairwise，还有listwise的排序方式：把所有文档交给LLM，让LLM来决定排序。具体做法可以看RankGPT。  

题外话，这里把加上各种优化手段建库 & 精排的RAG仍然当做是naive RAG，因为这些手段本质上都是搜索工具的优化，最后都封装在搜索API里，模型应该是可以不感知搜索API里的事情的。概念上，有些人会把带搜索优化的叫做retrieval and rerank RAG。  

## 模型使用文档  

搜到了文档，应答模型还得用得好。  

正常来说，需要把用户问题、回复要求和搜索结果都放在prompt里，让模型生成回复。这种情况下，我们首先要面对的是长文本的问题。搜索结果凑个万八千token的并不困难。  

虽然现在模型支持的输入窗口长度基本上是64k起步了，但是实际使用中，随着任务复杂性的提升和数据长度的增加，模型还是会出现问题：lost in the middle的现象，模型容易关注在输入数据的前面和后面部分，中间的就容易忽略了；这跟人的阅读习惯也很像，想快速阅读就会看个开头看个结尾，跟老师阅卷似的；因此组织搜索文档的时候可以根据精排的结果把重要度高的放在前面和后面。  

另外，无论使用自检搜索还是使用搜索工具，都有可能返回错误或者无关的搜索结果（比如一个缩写在不同领域都存在，那么就有可能搜索文档里存在错误领域的资料）。这个时候模型需要学会判断和忽略错误的文档。  

还有一种情况是缺了文档，比如某个query本来需要搜索日本和韩国的人口，结果搜索文档里只有日本的。这时就要防止模型由于信息缺失而产生幻觉。实践上这是个很容易出问题的点：当你要求模型必须回答这个问题，而资料又缺失的时候，模型很容易就开启胡说八道模式，无论是Qwen还是DeepSeek-R1都会有这样的情况。  

搜索结果多了或者少了的情况，一种优化方式是合成这样的数据，然后做微调，让模型能够学会自己判断哪些是可用哪些是不可用的。现在有了reasoning模型，也可以直接prompt它们在思考中一一分析。  

## 进一步优化  

上面所提到的每一个步骤，每一步都可以用更强的LLM来提供更好的效果：  

- 比如更新的embedding模型提供更好的粗召效果  
- 大参数量的reasoning模型的改写效果就比小模型要好，并且还能根据prompt提供不同的改写效果，以及给出一定的推理过程，提供可解释性  
- 精排阶段给R1之类的大模型提供环境信息、对话历史、排序规则等，让它能以更贴近业务需求来排序（比如客服场景中想把推荐信息的优先级提高就可以这么做）  
- 多模态模型一定程度上也可以处理图文输入，比如在输入阶段通过caption把用户query转换成文本  

# Graph RAG  

Naive RAG（以及带上搜索优化的版本）适合处理那种「答案就存在于一个或少数几个文档」的情况，对于答案不存在一个地方的全局性问题，Naive RAG基本就没法处理了。比如数据库是一本小说，用户要求总结一下主人公经历过的几个大事件，这个时候模型除非读完整本小说，否则基本上是很难回答这个问题的。而Naive RAG的检索中，分块和检索只会返回一些离散的小段落，而且数量受到限制，因此应答模型是没有机会临时去读完整小说的。  

回想一下，RAG的检索可以分为indexing和retrieval两个阶段。Graph RAG在indexing阶段，用LLM把文档构建成包含实体和关系的graph。还是以上面的小说文档为例，它能够被构建出一个包含文中人物和事件互相关联的图谱，这样模型就能快速在图谱中找到和主角生平相关的信息，并进行总结。  

Graph RAG在资料入库的时候就需要构建图谱，并且每次更新可能会动到已建设的图谱，因此整体的indexing成本是比较高的。  

Graph RAG和naive RAG是可以结合起来使用的，这样就全局和细节信息都可以照顾到了。这也可以称为hybrid RAG。  

# RAG + Agent = Agentic RAG  

前面提到的RAG，无论是retrieval and rerank RAG，graph RAG还是hybrid RAG，基本上都是「输入->搜索->应答」这样的一个线性流程。  

这样的流程适合解决「资料库中有答案」的问题，但是对于复杂的「多跳推理」，或者「有常规搜索工具以外的工具需求」的问题，就可能没法很好地处理。  

要解决这些复杂问题，搜索只是其中一个步骤，但是只有搜索是不够的，系统还需要具备一定的自主决策能力，比如根据当前状况选择工具，复杂任务拆解成可执行的子任务等。  

那么一个自然的选择就是用Agent来优化RAG。什么是Agent？其实现在Agent的概念已经相当泛用，边界也很模糊，我个人认知上是能够感知环境、自主思考决策并采取行动和环境交互的就算是Agent。  

Agentic RAG一个很大的特点就是自主性，因为要解决更复杂的问题，而这些问题需要什么手段来处理，人类是没法在事前完全定义的。这就使得Agentic RAG必须具备一定的自主性，需要见机行事，因地制宜，利用手上的资源来制定case by case的解决问题的方案。  

## Agentic RAG类型  

那Agent和RAG有哪些具体结合方式？  

1、Tool Use & Routing Agent  

最简单的一类就是用agent做routing。  

Router需要根据环境信息、prompt要求和输入query选择合适的工具。比如我们有一个数据库是医学数据库，另一个是计算机数据库，那么不同的query就要选择不同的搜索源。  

实际上除了选择搜索源之外，也可以做其他工具的routing，比如计算器或者导航之类的。这其中就包含Tool Use的能力，选择工具，使用适合的入参。  

2、Query Planning Agent  

这个主要是把用户的复杂query分解成subquery，每个subquery都可以走单独的pipeline来解决，最后再把多个subquery的结果合在一起。比如"世界最高楼和第十一高楼分别叫什么"，就可以拆解成两个query："世界最高楼叫什么"，"世界第十一高楼叫什么"。  

多个subquery之间也可能有依赖关系，那有些subquery可以并行，有些就不行。比如"世界最高楼和第十一高楼的距离是多少"，就需要拆解成三个subquery：  

- Q1："世界最高楼叫什么" --> 获得答案A1  
- Q2："世界第十一高楼叫什么" --> 获得答案A2  
- Q3："{A1}和{A2}的距离是多少"  

Q1和Q2可以并行，Q3就需要等Q1和Q2结束后才能执行。  

Planner可以是一个LLM，也可以是一个复杂的系统，它内部甚至可能是个多agent系统，这取决于要解决的任务的复杂度。  

3、ReAct Agent  

把routing、tool use和plan这几个能力合在一起，再加上环境信息的循环流转，让系统迭代地处理问题，就是ReAct Agent。  

比如在上面这个"世界最高楼和第十一高楼的距离是多少"的case里，Q3"{A1}和{A2}的距离是多少"有可能搜索不到结果，那么第一次循环之后，plan LLM就需要重新规划方案，把问题拆解成：  

- Q4："{A1}的经纬度是多少" --> 调用地图工具，获得经纬度X  
- Q5："{A2}的经纬度是多少" --> 调用地图工具，获得经纬度Y  
- Q6："计算{X}和{Y}的距离"  

## Agentic RAG特性  

首先，Agentic RAG的prompt会更加meta一点，规则和指令的抽象等级会更高。如果prompt的规则写得太细，人工干预的规则太多，反而会限制LLM的发挥空间。  

传统RAG的检索操作相对更独立一些，缺乏上面提到的多任务拆解的能力，而Agentic RAG能够拆解任务，并且把多个子任务之间的依赖处理清楚。  

为了让LLM能够做出更合理的决策，Agentic RAG的输入一般会提供更多的信息，比如用户所在的地点、时间、历史聊天记录都可能需要。  

传统RAG一般是一次过输出结果，而Agentic RAG则经常需要「循环」地规划思考、调用工具、整合结果。  

| Feature | Traditional RAG Systems | Agentic RAG Systems |
| --- | --- | --- |
| Prompt engineering | Relies heavily on manual prompt engineering and optimization techniques. | Can dynamically adjust prompts based on context and goals, reducing reliance on manual prompt engineering. |
| Static nature | Limited contextual awareness and static retrieval decision-making. | Considers conversation history and adapts retrieval strategies based on context. |
| Overhead | Unoptimized retrievals and additional text generation can lead to unnecessary costs. | Can optimize retrievals and minimize unnecessary text generation, reducing costs and improving efficiency. |
| Multi-step complexity | Requires additional classifiers and models for multi-step reasoning and tool usage. | Handles multi-step reasoning and tool usage, eliminating the need for separate classifiers and models. |
| Decision-making | Static rules govern retrieval and response generation. | Decide when and where to retrieve information, evaluate retrieved data quality, and perform post-generation checks on responses. |
| Retrieval process | Relies solely on the initial query to retrieve relevant documents. | Perform actions in the environment to gather additional information before or during retrieval. |
| Adaptability | Limited ability to adapt to changing situations or new information. | Can adjust its approach based on feedback and real-time observations. |

        
其实我觉得Agentic RAG虽然套用了RAG的名字，但是实际上解决问题的方式和重点已经距离传统RAG比较远了：  

- 传统RAG更多关注在怎么搜索，比如前面提到的数据库建设，粗召精排和query改写等  
- Agentic RAG更多关注在逻辑推理，任务规划和工具之间的配合使用，解决的任务也从简单单跳变成复杂多跳  

# DeepSearch  

DeepSearch也算是Agentic RAG的一种。从概念上来讲，Agentic RAG可以说是一种架构，而DeepSearch则算是一种产品形态。不同的DeepSearch框架设计不同，但是目标都是提供完备的准确的信息。  

DeepSearch本身可以直接和用户交互，也可以作为DeepResearch中的搜索模块，为report生成模块提供数据。二者在设计细节上稍微会有些不同，但是大致的流程设计是相通的。（其实DeepSearch和DeepResearch在概念上的边界也有点模糊,姑且认为DeepResearch提供的是更长的报告，而DeepSearch的结果相对简短一些）  

Jina在[《DeepSearch/DeepResearch 实施实用指南》https://jina.ai/news/a-practical-guide-to-implementing-deepsearch-deepresearch/](https://jina.ai/news/a-practical-guide-to-implementing-deepsearch-deepresearch/)中总结的DeepSearch核心设计原则是比较合理的：  

> DeepSearch 的核心设计原则：搜索、阅读和推理的持续循环。

## 框架  

以一个基础的DeepSearch方案为例，看下每个模块在干什么。  

{% asset_img deepsearch.png DeepSearch %}  

③④⑤是个循环，一直到结果满意输出为止。  

(先挖个坑：这里先给一个基础的DeepSearch方案，以后再写一篇整理更多种方案的。)  

## planner 任务拆解  

这一步主要就是根据当前的输入信息，决定接下来要干什么。  

这里的输入信息不只有用户的query，而是包含之前循环中做过的任务和得到的结果（比如调用了什么工具，工具的结果是什么，也就是observation），以及环境信息（比如用户地点、时间、用户画像、长期记忆信息等）。  

另外，我对它的描述是"接下来干什么"，而不是"下一步干什么"，因为很多情况下，planner是可以给出多步的计划的。比如用户的问题是"2024年中国GDP最高的三个城市的人口分别是多少"，那么规划出来的步骤应该是：  

- step 1：查询"2024年中国GDP最高的城市" -> 获取前三个城市A、B、C  
- step 2：查询城市A的人口  
- step 3：查询城市B的人口  
- step 4：查询城市C的人口  

step2/3/4虽然是依赖step1的结果的，但是在规划阶段就可以先用占位符把任务给规划出来了。  

有一些产品比如Qwen Chat或者Suna等，规划会分为一二级，类似一个目录。比如"对比一下近三个月新上市的国产手机的价格和配置，按销量排名列成列表"这个case，规划出来的可以是：  

- 手机新上市国产手机信息  
  - 查询"华为近三月新上市手机"  
  - 查询"荣耀近三月新上市手机"  
  - 查询"小米近三月新上市手机"  
  - 查询"OPPO近三月新上市手机"  
  - 查询"VIVO近三月新上市手机"  
- 获取销量数据  
  - 查询"华为xx手机销量"  
  - 查询"荣耀xx手机销量"  
  - 查询"小米xx手机销量"  
  - 查询"OPPOxx手机销量"  
  - 查询"VIVOxx手机销量"  
- 制作表格  

planner模块需要读入比较多的信息，并给出合理的规划（不然后面的模块可能无法执行），因此要求LLM的逻辑推理能力比较强。  

## 任务澄清  

用户既然用到了DeepSearch，很可能是要解决比较复杂的问题，那么就有可能要执行很久。因此准确理解用户的意图就很重要，不然跑个半小时给出来的结果并不是用户想要的，那用户体验就太差了，也很浪费token。  

现在一些DeepSearch的产品会在初步规划完之后做一次展示和确认，在必要的情况下通过对话完善有歧义的地方，比如扣子空间和Skywork AI。  

理想情况下，在后续执行的任何过程中如果出现任务不清楚的地方，都应该可以和用户交互进行澄清，这样的效果是最好的。就像你工作中带了一个实习生，你在给他布置任务的过程中，他如果有什么不清楚的可以随时问题，这样能保证不容易出错。  

不过这样的设计在产品上可能暂时不太成立：用户不可能一直盯着你执行，他只想要最终结果，如果长期的等待没有能拿到结果，体验会很不好。所以大部分的产品设计上不会进行太多次交互澄清，或者在澄清阶段等一小会没有反馈之后，就会直接按默认设置执行。  

举个例子，比如用户说"帮我规划一个国庆节的东南亚旅游攻略"，那这时模型需要知道一个旅游攻略都需要什么输入参数：目的地、出发地、交通工具、人数、预算、天数，更高级一些有路线偏好比如人文、建筑、自然等。然后看用户的query缺失了哪些重要参数，让用户来选。如果模型本身的知识无法确定旅游攻略需要什么参数，那至少要懂得借助general search去搜"制订旅游攻略需要先确定什么"。  

实际开发上，可以有一个专门的agent来处理澄清的问题。甚至可以针对一个输入query，出一个简单的可视化界面，天工AI就是这样。比如上面这个旅游的case，让用户勾选每一个字段要的参数值：  

- 目的地：□泰国，□新加坡，□马来西亚，□越南，□老挝
- 预算：□3000-4999，□5000-6999，□7000-8999
- ...
- 路线偏好：□人文，□建筑，□宗教，□自然

用户选好之后，再传给planner完善新的规划。  

## 信息获取  

信息获取阶段，主要就是根据planner规划出来的子任务，获取对应的信息。  

设计上，可以让planner直接输出function call的json string，也可以让planner只输出搜索任务的自然语言描述，再由另外的function模型来生成各个工具的调用参数。个人经验，实操上二者的效果区别并不大。  

DeepSearch的工具集一般包含多个搜索源，比如通用搜索引擎的接口，垂域数据库（比如专门搜商品的、搜美食的、搜菜谱的、搜车型的等）。  

接口的介绍和功能用法的文档对LLM准确地使用工具很重要。  

比如搜索引擎一般有一些使用技巧：加双引号表示必须包含，用减号表示排除关键词，用site:表示站内搜索等，这些都能帮助模型提高搜索效率。  

如果有些概念不好用简短的语言介绍清楚，也可以加一个典型的例子，实践证明加example非常有效。当prompt中的指令和example出现矛盾时，LLM更愿意向example的做法靠近。  

### 信息过多  

DeepSearch的搜索-阅读-推理是循环进行的，收集到的信息会越来越多，在后面的轮次中，生成function call的模型的输入就很长，甚至有可能超出了模型窗口限制，导致循环无法继续。  

因此每次搜索之后，对有用信息进行压缩就很重要。  

1、精排  

简单有效的一个方法是对搜索得到的每个文档和query/subquery进行打分匹配，过滤掉相关性低的。这可以缓解由于function的入参不佳，或者搜索接口准确率不够带来的问题。  

2、去重  

搜索时用同一个query获取多个文档，一般都会有一些信息重复。因此可以对搜索query进行信息去重。  

具体做法是用一个小模型，对多个文档内容进行整合。要求仅删除冗余信息，尽量避免对原文的修改，以减少幻觉的引入。  

最终把所有历史搜索结果压缩成一个或者几个去重过的文档，这些文档就是在最后会输出的文档，而原始搜索结果就不输出了。这些去重文档就比原始文档的长度减少了许多。  

在DeepSearch进行下一个搜索循环中，function call的模型需要分析这些去重过的文档已经包含什么，避免进行重复搜索。  

## 结果反思  

结果反思的任务是做循环判断：当前的搜索结果（去重文档）是不是已经足够了，如果不够，那就继续拆解任务，继续搜索。  

理论上，这个判断由planner来做也可以，用额外的结果反思模块的好处是可以用不同的模型，或者用更多的模型来协助判断。和人看问题的角度会有不同一样，不同的模型判断时也会从不同维度来分析，这样集思广益，可以避免planner一家独大。  

还有一个好处是planner的任务少一点，压力也轻一些。  

这个模块的功能和RL的reward有点相似。  

## 通用工具  

这里还有一个模块是通用工具。通用工具包括计算器和Python代码解释器等。这些工具本身并不提供搜索功能，但是在DeepSearch的各个环节中都有可能要用到他们。  

举个例子，比如用户要用3人出行，预算5000块，花在美食上的比例在30%到40%之间。那么无论是planner，还是function call模块，都有可能需要计算每人每天的饮食预算，以此来设计对应的就餐地点。虽然模型本身也具备一定的计算能力，但是使用计算器可以保证结果的准确性，那这时就可以调用计算器了。  

再举个例子，比如用户要求罗列一些以科幻为题的文章，字数要求在500字到600字之间。由于模型本身没法数字数，所以就需要调用代码解释器，写个小脚本来统计字数了。  

随着模型能力的提升，这个工具池的大小一定会越来越大，能处理的问题也越来越复杂。目前来看，扩展到多模态的输入输出已经是一个必然。  

# 小结  

- RAG适合解决搜索文档有答案的问题，但是复杂问题效果不好  
- Graph RAG能处理一些普通RAG无法解决的全局问题，但是成本高  
- DeepSearch是Agentic RAG的一种，用Agent来处理routing（工具选择），分析推理（plan + reflect）的任务，这个循环可以一直跑下去，知道结果满足需求  
- DeepSearch的大流程不复杂，但是作为产品，要考虑的细节问题非常多，所需能力也很综合  
- DeepSearch的形态还在不断变化中  

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
[DeepResearch的报告生成方法](https://www.linsight.cn/44c62dc5.html)  
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
[Qwen3实测&技术报告](https://www.linsight.cn/37ee84bb.html)  
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
- 训练框架  
[LLM训练框架：从优化器和精度讲到ZeRO](https://www.linsight.cn/fe0adaa5.html)  
[LLM训练各种并行策略](https://www.linsight.cn/4cd8532f.html)  
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

【1】Retrieval-Augmented Generation for Large
Language Models: A Survey  
【2】12 RAG Pain Points and Proposed Solutions，https://archive.is/bNbZo  
【3】Seven Failure Points When Engineering a Retrieval Augmented Generation System  
【4】DeepSearch/DeepResearch 实施实用指南，https://jina.ai/news/a-practical-guide-to-implementing-deepsearch-deepresearch  
【5】https://www.zhihu.com/question/642650878/answer/1908280187600213891  
【6】RAG 切块Chunk技术总结与分块实现思路分享，https://zhuanlan.zhihu.com/p/19010809414  
【7】RAG 2.0 深入解读，https://zhuanlan.zhihu.com/p/1903437079603545114  
【8】DeepSearcher深度解读：Agentic RAG的出现，传统RAG的黄昏，https://zilliz.com.cn/blog/DeepSearcher-Insights-Agentic-RAG  
【9】Agentic RAG Explained (RAG Agent)，https://aisera.com/blog/agentic-rag/  

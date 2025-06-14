---
title: 'agent调研(1)--MetaGPT,OpenManus和OWL'
tags:
  - NLP
  - LLM
  - Agent
categories:
  - CS
  - NLP
  - Agent
abbrlink: 226b059f
date: 2025-06-14 19:43:40
---

# MetaGPT  

MetaGPT的开发团队就是搞OpenManus的团队。  

MetaGPT项目做得很早，23年就开始搞multi-agent了。MetaGPT主要是想通过多智能体的协作提升代码能力。做法上，大致上来说就是把一个软件团队所需的角色，比如产品经理、项目经理、架构师、开发工程师、测试工程师等都用LLM给角色扮演出来。不同的LLM角色和人类团队一样相互合作，共同开发项目：  

{% asset_img metagpt.png agent %}  

MetaGPT对Agent和Multi-Agent的定义：  

> 智能体 = 大语言模型（LLM） + 观察 + 思考 + 行动 + 记忆  

> 多智能体 = 智能体 + 环境 + 标准流程（SOP） + 通信 + 经济

- SOP = Standard Operating Procedures：是管理智能体行动和交互的既定程序，确保系统内部的有序和高效运作。例如，在汽车制造的SOP中，一个agent焊接汽车零件，而另一个安装电缆，保持装配线的有序运作。可以看出SOP非常依赖人为的设计，每个agent的操作空间、交互方式是被定义好的。  
- 经济：这指的是多智能体环境中的价值交换系统，决定资源分配和任务优先级。  

MetaGPT中每个角色的输入输出都是格式化的信息，这样保证来自不同Agent的信息可以方便地被其他角色使用。  

通信上，系统中设计了一个shared message pool，各个Agent可以自由地在这里发布和订阅消息。和传统人工设计的信息流通方式相比，shared message pool更方便解耦多个Agent之间的工作。比如产品经理发布 PRD → 消息池自动通知订阅该类型的架构师、工程师 → 后者按需读取，无需等待前者主动推送。  

传统模式中，工程师必须等待项目经理分配任务才能开始工作；而在 MetaGPT 中，工程师可直接从消息池获取架构设计文档，甚至在产品经理完成 PRD 前，即可通过消息池的 “草稿” 版本提前介入，实现并行工作。  

总体来说，MetaGPT是针对代码场景的定制化设计，同时各个LLM「在设计范围内具有一定自由度」的一个Multi-Agent System。  

# OpenManus  

Manus出来当天，MetaGPT的团队基于原来的方案，花了几个小时复刻了一个OpenManus。后来也一直在更新，而且更新了很多版。  

相比MetaGPT，OpenManus的大框架更为general，可以支持更多的场景。可以通过部分改动，快速定制支持不同的场景；也可以直接用通用模块，这样所有中间流程都交给LLM自己规划和处理。  

1、组件  

（1）Agent  

OpenManus中的Agent的实现是这样的：  

```text
        ┌─────────────┐
        │  BaseAgent  │
        └──────┬──────┘
               │
               ▼
        ┌─────────────┐
        │ ReActAgent  │
        └──────┬──────┘
               │
               ▼
        ┌─────────────┐
        │ToolCallAgent│
        └──────┬──────┘
               │
  ┌────────────┬────────────┐
  │            │            │
  ▼            ▼            ▼
┌─────────┐ ┌───────┐ ┌───────┐
│Planning │ │SWE    │ │Manus  │
│Agent    │ │Agent  │ │       │
└─────────┘ └───────┘ └───────┘
```

采用了多层集成的方式来实现不同的Agent，而每个Agent都是一个ReAct Agent，迭代地执行任务，直到任务完成或者超过循环次数。比如Manus Agent设置的默认最大循环次数就是20轮。  

- BaseAgent：抽象类，run是主入口，run中迭代执行step  
- ReActAgent：step中加入抽象函数reasoning和action  
- ToolCallAgent：引入工具，实装了think和action，think中决定使用什么工具  
- Manus：OpenManus中的主要Agent，相对通用  

每个Agent有自己的prompt，定义了角色和任务。  

除了system prompt，OpenManus还使用了一个NEXT_STEP_PROMPT。NEXT_STEP_PROMPT主要让模型选择下一步干什么。比如Manus Agent的NEXT_STEP_PROMPT：  

```python
NEXT_STEP_PROMPT = """
Based on user needs, proactively select the most appropriate tool or combination of tools. For complex tasks, you can break down the problem and use different tools step by step to solve it. After using each tool, clearly explain the execution results and suggest the next steps.
"""
```

使用上，NEXT_STEP_PROMPT是以user角色输入的：  

```python
[
    {
        "role": "system", 
        "content": SYSTEM_PROMPT
    },
    {
        "role": "user", 
        "content": query
    }, 
    {
        "role": "user", 
        "content": NEXT_STEP_PROMPT
    }
] 
```

我试了下，连续两个user角色输入，和把两次输入的content直接拼接在一起，response回复的差异不大。其实就可以把NEXT_STEP_PROMPT视作一个template，它和user query一起构成了user prompt，包含了任务query，以及设定好的instruction。  

（2）Tool  

OpenManus中的tool主要包括代码执行（python代码解释器、Bash等），信息搜索（Google Search和BrowserUse），文件操作等。  

所有工具都继承自 BaseTool 基类：  

```python
class BaseTool(ABC, BaseModel):
    name: str
    description: str
    parameters: Optional[dict] = None

    class Config:
        arbitrary_types_allowed = True

    async def __call__(self, **kwargs) -> Any:
        """Execute the tool with given parameters."""
        return await self.execute(**kwargs)

    @abstractmethod
    async def execute(self, **kwargs) -> Any:
        """Execute the tool with given parameters."""

    def to_param(self) -> Dict:
        """Convert tool to function call format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }
```

每个工具都通过 parameters 字典定义其接受的参数，遵循 JSON Schema 格式：

```python
parameters: dict = {
    "type": "object",
    "properties": {
        "command": {
            "description": "The command to execute...",
            "enum": ["create", "update", "list", "get", "set_active", "mark_step", "delete"],
            "type": "string",
        },
        # 其他参数...
    },
    "required": ["command"],
    "additionalProperties": False,
}
```

使用的时候，工具通过to_param转成模型可接受的格式。  

（3）Flow  

Agent理解任务，输出action，Tool进行操作，返回结果。这中间的流转就需要Flow来调度。  

个人理解上，Flow的设计可能和场景和任务比较有关联。比如一般来说就是串行的，模型 → 工具 → 模型 → 工具 → 回复这样，而在一些场景下也可能有并行的，比如DeepSearch中拆解出来的任务是可以并行搜索的。  

OpenManus中实现的PlanningFlow大致上就是串行为主的设计。  

（4）Sandbox  

Agent在执行任务过程中会对文件等进行很多操作，这其实是有风险的，因此需要一个沙箱环境来隔离，免得把主机搞崩了。  

2、流程  

OpenManus的流程是典型的：思考+行动循环。  

3、架构分层  

分为5层：  

> - 用户层：通过命令行界面与用户交互，处理输入并展示结果。入口文件为 run_flow.py。  
> - Flow层：作为框架的调度中心，负责任务编排和执行策略。核心组件是 PlanningFlow。  
> - Agent层：实现任务执行逻辑，具备思考(think)和行动(act)能力。包含 ToolCallAgent、PlanningAgent 和 Manus 等。  
> - Tool层：提供可扩展的工具集，增强 Agent 的执行能力。核心工具为 PlanningTool。  
> - LLM层：统一封装大语言模型接口，支持多种模型配置。  

```
# OpenManus分层架构示意图

┌───────────────┐
│   用户层       │
└──────┬────────┘
       ↓
┌───────────────┐
│   Flow层      │
└──────┬────────┘
       ↓
┌───────────────┐
│   Agent层     │
└──────┬────────┘
       ├──────────────┐
       ↓              ↓
┌───────────────┐ ┌───────────────┐
│   LLM层       │ │   Tool层      │
└───────────────┘ └───────────────┘
```

详细的可以看[《OpenManus 代码框架详解》](https://zhuanlan.zhihu.com/p/1897429266813125945)  

# OWL  

OWL = OPTIMIZED WORKFORCE LEARNING  

OWL最近出了技术报告，看下它提供了什么信息。  

首先是分析现有的multi-agent system的问题：  

- 目前的MAS大部分是domain-specific的，cross-domain transferability受限。  
- 推理上：一般需要针对业务场景定制，无法一个系统通用处理不同场景下的问题，比如MetaGPT，就是针对代码场景的设计，依赖SOP进行开发。  
- 训练上：大部分多Agent框架在训练的时候需要优化多个Agent，比如MALT（Multi-Agent LLM Training）。  

OWL的设计包含三个主要模块：

- Domain-Agnostic Planner：做abstract task decompositions，不和任何场景绑定  
- Coordinator：把Planner拆解出来的task分配给不同的worker  
- Domain Agent（Worker Nodes）：worker，封装工具API，负责完成某项具体任务  

下图左边是传统的框架，右边是OWL的framework，推理的WORKFORCE和训练的Optimized Workforce Learning：  

{% asset_img owl_overview.png agent %}  

WORKFORCE的一个示意图：  

{% asset_img owl_workforce.png agent %}  

其中有一个task channel，coordinator和worker的信息发布和结果发布都在这里，各个worker之间是互不相见的，仅由coordinator来调配。  

个人感觉OWL的核心设计其实就是Planner和Coordinator的拆分。通常来说，这二者的任务是可以由同一个角色来完成的。拆分开之后，在不同的任务场景下，Coordinator和worker都可以不动，仅通过特异化训练Planner来适配不同场景。  

Planner使用了Qwen+SFT+DPO，效果就有比较不错的提升。  

OWL附录里提供各个Agent包括Worker的prompt，基本上是比较标准的agent prompt。有一点可以参考的就是，prompt中把整体的角色定义放在的system prompt，而具体的信息和规则就放在user prompt。这应该和模型训练时会更多follow system prompt的大原则的特点相关。  

总体上，OWL的拆解有点像职场那样，进行了层级的拆解：有领导planner，只管拆任务，拆完任务就做完了；coordinator是基层管理者，负责worker工作的发布和管理。这有点上升到管理哲学的层面了，到底这样的架构有多大的实际收益，个人有点打问号。  

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
[从RAG到DeepSearch](https://www.linsight.cn/7c2f9dcb.html)  
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

【1】OWL: Optimized Workforce Learning for General Multi-Agent Assistance in Real-World Task Automation，https://github.com/camel-ai/owl/blob/main/assets/OWL_Technical_Report.pdf  
【2】MetaGPT，https://docs.deepwisdom.ai/v0.4/zh/guide/tutorials/concepts.html  
【3】OpenManus 代码框架详解，https://zhuanlan.zhihu.com/p/1897429266813125945  
【4】[源码学习] 通过OpenManus了解Agent系统的实现，https://zhuanlan.zhihu.com/p/1904129741365154580  
【5】从prompt看OpenManus的实现思路，https://zhuanlan.zhihu.com/p/1889982149991580008  
【6】OpenManus LLM 工具调用机制详解，https://zhuanlan.zhihu.com/p/1886362220297967012  

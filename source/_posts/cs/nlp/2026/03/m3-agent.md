---
title: 字节的M3-Agent
tags:
  - NLP
  - LLM
  - Agent
categories:
  - CS
  - NLP
  - Agent
hidden: false
abbrlink: 5f0d3c82
date: 2026-03-22 13:43:15
---

当前的多模态Agents在实时处理视觉和听觉输入方面已有一定能力，但面临一个限制：长期记忆能力的缺失。  

具体而言，现有方法在处理长视频流时面临三大挑战：  

- 身份识别混乱：基于语言描述的记忆（如"穿红衣服的男人"）在长时程视频中容易产生歧义和混淆  
- 关键细节遗忘：视觉和听觉细节在文本摘要过程中丢失，跨模态关联难以建立  
- 知识无法沉淀：智能体无法从具体事件中提炼抽象知识（如"Alice喜欢早晨喝咖啡"）  

字节的M3-Agent要解决的问题：如何让多模态Agent具备类人的长期记忆系统，使其能够在线处理实时视频/音频流，构建一致性的实体表示，并在后续推理中有效检索和利用这些记忆？  

# 整体架构设计：双过程并行

M3-Agent的架构类似于一个双脑系统：一个感觉脑（Memorization）持续不断地观察世界，将看到听到的一切都记下来；另一个思考脑（Control）只在被问到问题时才启动，查阅感觉脑记下的笔记，经过多轮思考后给出答案。  

这两个大脑通过共享笔记本（长期记忆图）协作。感觉脑不停地往笔记本上写东西，思考脑需要时翻阅。  

感觉脑是多模态专家（论文中用Qwen2.5-Omni-7B），擅长看图听声；思考脑是推理专家（论文中用Qwen3），擅长逻辑分析与决策。  

# 长期记忆的图结构

M3-Agent以graph的形式记忆信息。  

graph的储存主要又以实体为中心，比如人脸、声音、物体。  

## 记忆节点的完整数据格式

这里先说下graph的设计。  

每个记忆节点是一个JSON对象，包含以下字段：  

基础字段（所有节点通用）：  

- `id`: 字符串，唯一标识符，格式如`face_alice_001`、`event_20240801_103000`、`voice_bob_003`  
- `type`: 枚举值，`text`（文本描述）、`image`（人脸/物体截图）、`audio`（声音片段）  
- `content`: 字符串，具体内容。text类型为纯文本；image/audio为base64编码的二进制数据  
- `embedding`: 浮点数数组，长度为768或1024，用于相似度检索。不同模态可能使用不同模型生成，但映射到同一向量空间  
- `weight`: 浮点数，初始值通常为1.0，每次reactivate增加1.0，表示该记忆的激活频率与置信度  
- `extra_data`: JSON对象，包含时间戳、位置、关联ID等元数据  

extra_data的具体字段：  

```json
{
  "timestamp_start": "2024-08-01T10:30:00",
  "timestamp_end": "2024-08-01T10:30:30",
  "clip_id": "clip_042",
  "location": "kitchen",
  "face_id": "alice",
  "voice_id": "alice_01",
  "object_class": "cup",
  "color": "red",
  "associated_entities": ["face_alice_001", "voice_alice_001", "obj_cup_001"],
  "derived_from": ["epi_001", "epi_005"],
  "confidence_history": [0.8, 0.85, 0.9]
}
```

## 节点的三种具体形态

每一个输入视频都可以产生多个记忆节点：有人脸、物体的，有声音的，还有文本的，其中文本又分为情景记忆和语义记忆。情景记忆即是单次发生的事实性内容，而语义记忆是可以跨多个节点的推断内容。  

文本节点（情景记忆示例）：  

```json
{
  "id": "epi_kitchen_042",
  "type": "text",
  "content": "Alice坐在厨房餐桌旁，右手拿起红色陶瓷杯，喝了一口咖啡，随后放下杯子，看向窗外的花园",
  "embedding": [0.12, -0.05, 0.88, ...],
  "weight": 3.0,
  "extra_data": {
    "timestamp_start": "10:30:00",
    "timestamp_end": "10:30:15",
    "clip_id": "clip_042",
    "location": "kitchen",
    "associated_entities": ["face_alice_001", "obj_red_cup_001", "voice_alice_001"],
    "memory_type": "episodic"
  }
}
```

图像节点（人脸示例）：  

```json
{
  "id": "face_alice_001",
  "type": "image",
  "content": "base64 encoded image data...",
  "embedding": [0.23, 0.56, -0.12, ...],
  "weight": 15.0,
  "extra_data": {
    "face_id": "alice",
    "first_seen": "10:00:00",
    "last_seen": "11:30:00",
    "appearance_count": 15,
    "voice_associations": ["voice_alice_001", "voice_alice_002"]
  }
}
```

音频节点（说话声示例）：  

```json
{
  "id": "voice_alice_003",
  "type": "audio",
  "content": "base64 encoded audio waveform...",
  "embedding": [0.45, -0.23, 0.67, ...],
  "weight": 5.0,
  "extra_data": {
    "voice_id": "alice_01",
    "timestamp": "10:30:05",
    "transcription": "这咖啡真香，早上不喝一口总觉得少点什么",
    "speaker_id": "alice",
    "clip_id": "clip_042"
  }
}
```

## 边的数据结构

graph中包含三种边：

- 实体关联边：相同实体ID的节点相连（如某人的face_id与voice_id关联）  
- 时序边：表示事件先后顺序  
- 语义边：表示空间包含、属性从属等逻辑关系  

实体关联边（人脸与声音绑定）：  

```json
{
  "edge_id": "link_001",
  "source": "face_alice_001",
  "target": "voice_alice_001",
  "relation_type": "entity_link",
  "weight": 8.5,
  "created_at": "10:00:00",
  "last_activated": "10:30:05",
  "activation_count": 8
}
```

时序边（事件延续，时间顺序关系）：  

```json
{
  "edge_id": "temp_042_043",
  "source": "epi_kitchen_042",
  "target": "epi_kitchen_043",
  "relation_type": "temporal",
  "weight": 2.0,
  "relation_detail": "sequential_clip"
}
```

语义边（比如空间关系）：  

```json
{
  "edge_id": "sem_001",
  "source": "obj_red_cup_001",
  "target": "face_alice_001",
  "relation_type": "semantic",
  "relation_detail": "held_by",
  "weight": 3.0
}
```

# 记忆化过程

## 30秒Clip的处理流水线

M3-Agent中，简单粗暴把每个30秒的视频clip作为一个输入单位处理，处理过程包含：  

工序1：模态分离（输入：原始视频流）  

- 视频轨道：每秒采样1-2帧，得到30-60帧图像序列  
- 音频轨道：提取30秒波形，转录为文本（ASR）  

工序2：实体提取（并行处理）  

- 人脸检测：使用RetinaFace或类似工具检测每帧人脸，提取面部特征向量，通过聚类或匹配确定face_id。如果新face与库中某face相似度>0.85，判定为同一人，分配已有face_id；否则分配新ID如`face_person_007`。  
- 声纹识别：使用ERes2NetV2等声纹模型提取音频特征，确定voice_id。同样基于相似度阈值判定是新声音还是已知声音。  
- 物体检测：使用开放词汇检测器（如OWL-ViT或Qwen2.5-Omni内置检测）识别显著物体（杯子、手机、书本等），提取视觉特征，分配object_id。  

对于检测到的人脸、物体、声音等实体，可以基于embedding的快速搜索，找到这个实体是已有的实体，还是新的实体，从而实现ID的分配和节点的绑定。  

工序3：关联检测（时间对齐）  

- 检查在同一时间窗口（如10:30:00-10:30:05）内是否同时出现face_id和voice_id。如果是，建立临时实体关联边，初始weight=1.0。  
- 检查物体与人的交互：如果在10:30:02检测到`face_alice_001`的手部区域与`obj_cup_001`有重叠，建立语义边`held_by`，weight=1.0。  

工序4：记忆生成（文本描述）  

- 将30帧图像+音频转录文本输入Qwen2.5-Omni  
- 生成情景记忆：详细描述这30秒内发生的具体动作与对话，如"Alice在10:30:05拿起红色杯子说'咖啡真香'，10:30:10放下杯子"  
- 生成语义记忆：提取抽象知识，如"Alice喜欢喝黑咖啡"、"Alice有红色陶瓷杯"  

语义记忆在每个clip处理时即时生成，不是事后批量总结。  

工序5：入库与关联  

- 将生成的节点存入图数据库  
- 执行`refresh_equivalences()`：检查新节点与现有节点的相似度，如果文本描述相似度>0.9（使用文本embedding计算），判定为同一事件的延续，增加旧节点weight而不是创建新节点  

## 跨Clip事件处理的详细机制

场景：Alice从10:00:00开始喝咖啡，持续到10:02:00，跨越4个clips（clip_001到clip_004）。  

处理流程：  

Clip 001（10:00:00-10:00:30）：  

- 生成Node `epi_001`: "Alice开始喝咖啡，拿起红色杯子"  
- weight=1.0  
- 生成Node `sem_001`: "Alice喜欢喝咖啡"  
- weight=1.0  

Clip 002（10:00:30-10:01:00）：  

- 生成临时Node `epi_002_temp`: "Alice继续喝咖啡，看着窗外"  
- 调用`search_node(query="Alice喝咖啡")`，检索到`epi_001`相似度0.95  
- 判定为同一事件延续：  
  - 不创建新的独立情景节点（或创建但标记为关联）  
  - Reactivate `epi_001`，weight增加至2.0  
  - 创建时序边 `epi_001` → `epi_002`（如果保留clip级节点）或简单增加weight  
  - 更新`sem_001`，weight增加至2.0（确认"喜欢咖啡"这一知识）  

Clip 003与004：  

- 重复上述过程  
- `epi_001`的weight最终达到4.0，表示该记忆被4个clips确认  
- `sem_001`的weight同样累积，表示知识的置信度随观察次数增加  

数据结构表现：  

```json
{
  "id": "epi_coffee_event",
  "type": "text",
  "content": "Alice在厨房喝咖啡（多clip聚合描述）",
  "weight": 4.0,
  "extra_data": {
    "clips_involved": ["clip_001", "clip_002", "clip_003", "clip_004"],
    "time_range": "10:00:00-10:02:00",
    "sub_events": [
      {"clip": "clip_001", "action": "开始喝", "timestamp": "10:00:05"},
      {"clip": "clip_002", "action": "继续喝", "timestamp": "10:00:35"},
      {"clip": "clip_004", "action": "喝完", "timestamp": "10:01:55"}
    ]
  }
}
```

## 单Clip多事件处理

场景：10:30:00-10:30:30内，Alice同时做三件事：跑步机上跑步、喝水、听音乐。  

处理策略：  

- 不强制切分30秒clip，而是生成多个节点记录并行事件：  

生成的节点集合：  

1. `epi_run_042`: "Alice在跑步机上跑步，配速8km/h"，timestamp=10:30:00-10:30:30  
2. `epi_drink_042`: "Alice拿起水瓶喝了一口水"，timestamp=10:30:15-10:30:20  
3. `epi_music_042`: "背景播放爵士音乐"，timestamp=10:30:00-10:30:30  
4. `face_alice_042`: Alice面部图像（贯穿整个clip）  
5. `voice_alice_042`: Alice说话声"这音乐不错"，timestamp=10:30:25  

关联建立：  

- 所有节点`extra_data.clip_id`都指向"clip_042"  
- 所有节点的`associated_entities`都包含`face_alice_042`  
- 通过时间戳范围隐式表示并发：`epi_run_042`与`epi_drink_042`在10:30:15-10:30:20重叠  

查询时的并发检测：  

当用户问"Alice跑步时做了什么"，系统：  

1. 找到`epi_run_042`（时间范围10:30:00-10:30:30）  
2. 查找同一clip内其他节点  
3. 发现`epi_drink_042`（时间范围10:30:15-10:30:20）落在跑步时间范围内  
4. 返回"Alice边跑步边喝水"  

# 实体关联与冲突解决

## 实体关联边的建立过程

初始关联（Clip 1）：  

- 检测到`face_alice_001`（新人脸，weight=1）  
- 检测到`voice_alice_001`（新声音，weight=1）  
- 时间重叠：两者在10:00:00-10:00:05同时出现  
- 建立边：  
```json
{
  "source": "face_alice_001",
  "target": "voice_alice_001",
  "relation_type": "entity_link",
  "weight": 1.0,
  "first_seen": "10:00:00"
}
```

后续确认（Clip 5）：  

- 再次检测到`face_alice_001`和`voice_alice_001`同时出现  
- 找到已有边，执行reactivate  
- weight增加至2.0  

多次确认后（Clip 20）：  

- weight累积到8.0  
- 系统高置信度判定：这个声音确实属于这张脸  

## 错误纠正的详细例子

错误场景：  

- Clip 3（光线昏暗）：`voice_alice_001`被错误匹配到`face_bob_001`（Bob也在场，系统误识别）  
- 建立错误边：  
```json
{
  "source": "face_bob_001",
  "target": "voice_alice_001",
  "weight": 1.0
}
```

纠正过程：  

- Clip 4：系统再次检测到`voice_alice_001`，这次与`face_alice_001`同时出现（正确关联）  
- 建立/强化正确边：  
```json
{
  "source": "face_alice_001",
  "target": "voice_alice_001",
  "weight": 1.0  // 新边或从已有边增加
}
```

- Clip 5-20：在后续15个clips中，`voice_alice_001`与`face_alice_001`共同出现15次，与`face_bob_001`仅共同出现1次（初始错误）  
- Weight对比：  
  - 正确边weight = 16.0  
  - 错误边weight = 1.0  

冲突解决：  

当系统需要回答"这个声音是谁"时：  
1. 检索`voice_alice_001`关联的所有face节点  
2. 发现两个候选：`face_alice_001`（weight=16）和`face_bob_001`（weight=1）  
3. 选择weight更高的`face_alice_001`  
4. 错误边因长期未被激活，weight不增加，逐渐被系统忽略（或定期清理时删除）  

边界情况：  
如果错误关联被系统性重复使用（如用户反复问"Bob在说话吗"而系统基于错误边回答"是"），则错误weight也会累积，导致错误固化。文献假设此类情况较少，依赖正确信息的统计显著性压倒错误。  

# 物体识别的ID分配机制

## 新物体vs已有物体的判定流程

场景：检测到红色杯子。  

步骤1：特征提取  

- 提取杯子的视觉特征向量`emb_cup_new`（使用CLIP或DINOv2）  

步骤2：相似度检索  

- 调用`search_node(embedding=emb_cup_new, type="image", object_class="cup")`  
- 检索记忆库中所有object_class为"cup"的image节点  
- 计算余弦相似度：  
  - 与`obj_red_cup_001`（Alice的红杯）：相似度0.92  
  - 与`obj_blue_cup_001`（Bob的蓝杯）：相似度0.45  
  - 与`obj_red_cup_002`（办公室的红杯）：相似度0.88  

步骤3：阈值判定  

- 设定阈值（假设为0.90）：  
  - 0.92 > 0.90 → 判定为已有物体`obj_red_cup_001`  
  - 0.88 < 0.90 → 判定为不同物体（即使都是红色杯子，但纹理/形状差异足够大）  

步骤4：更新或创建  

- 已有情况：  
  - Reactivate `obj_red_cup_001`，weight += 1  
  - 添加新的时间戳到`extra_data.appearance_history`  
  - 建立与当前face节点（Alice）的语义边`used_by`  

- 新物体情况（假设相似度都<0.90）：  
  - 分配新ID：`obj_red_cup_003`  
  - 创建新节点，weight=1.0  
  - 建立与当前face节点的边  

## 跨天物体的追踪例子

Day 1：Bob送Alice红杯子  

- 生成`obj_red_cup_001`，建立语义边`gift_from`指向`face_bob_001`  
- 生成语义记忆`sem_001`: "Bob送了Alice一个红色杯子"  

Day 30：红杯子再次出现  

- 检测到红色杯子，特征匹配成功（相似度0.91）  
- Reactivate `obj_red_cup_001`，weight增加  
- 新时间戳加入`extra_data`  

Day 60：用户问"Bob送的那个杯子长什么样"  

- Control过程解析查询：  
  1. `search.node(query="Bob")` → 找到`face_bob_001`  
  2. 查找与`face_bob_001`关联的语义记忆 → 找到`sem_001`（"Bob送杯子"）  
  3. `sem_001`通过`derived_from`或语义边关联到`obj_red_cup_001`  
  4. 返回`obj_red_cup_001`的content（图像base64）和描述（红色、陶瓷、特定纹理）  

# 控制过程与检索详解

算法伪代码：  

```
输入: 用户问题 q, 长期记忆 D, 控制策略 π_θ
输出: 最终答案

对于轮次 t = 1 到 H（最大轮数）:
    1. π_θ 生成响应，包含：推理过程 reasoning、动作 action、参数 argument
    2. 如果 action == [Search]:
        - 使用 argument 查询记忆 D
        - 可调用 search.node（检索人物/物体）或 search.clip（检索事件片段）
        - 将检索结果追加到上下文
    3. 如果 action == [Answer]:
        - 返回答案内容，终止过程
```

## 算法流程的详细展开

Algorithm 1的完整执行示例：  

用户问题："认证专家检查完Led Zeppelin专辑上的墨迹后做了什么？"  

初始状态：  
- 记忆库D包含30个clips的相关记忆  
- 对话历史τ初始为空  
- 最大轮数H=5  

Round 1：  

- π_θ接收(q="认证专家检查完墨迹后做了什么", D, τ=空)  
- 生成：  
  - reasoning: "用户问的是认证专家在检查墨迹后的下一个动作。我需要先找到'检查墨迹'这个事件，然后查找紧随其后的事件。"  
  - action: [Search]  
  - argument: "认证专家 检查 墨迹 Led Zeppelin专辑"  
- 执行`search.clip(query="认证专家 检查 墨迹")`  
- 检索结果：返回clip_015，包含事件"他比较了笔迹"  
- 更新τ：加入Round 1的推理与检索结果  

Round 2：  

- π_θ接收(q, D, τ=包含clip_015的信息)  
- 生成：  
  - reasoning: "已找到检查墨迹的事件，接下来需要确认这个动作是否就是用户问的事件之后发生的，还是需要更多上下文。"  
  - action: [Search]  
  - argument: "认证专家 比较笔迹 接下来"  
- 执行`search.node`或继续`search.clip`确认时序  
- 检索结果：确认"比较笔迹"是在"检查墨迹"之后立即发生的动作  

Round 3：  

- π_θ生成：  
  - reasoning: "已确认：检查墨迹后的动作是比较笔迹。我有足够信息回答。"  
  - action: [Answer]  
  - argument: "认证专家在检查完墨迹后比较了笔迹。"  
- 终止，返回答案  

## search.node与search.clip的粒度对比

| 维度 | search.node | search.clip |
|------|-------------|-------------|
| 检索对象 | 单个记忆节点（人脸、物体、具体描述） | 30秒记忆片段（包含该时间段所有节点） |
| 返回内容 | Top-k个节点及其embedding相似度分数 | Top-k个clip的ID及其时间范围 |
| 使用场景 | 查找特定人物、物体、或精确记忆条目 | 定位时间段、获取场景上下文、浏览某时刻的所有事件 |
| 查询输入 | 文本、图像embedding、或音频embedding | 主要是文本查询 |
| 示例查询 | "Alice的脸"、"红色杯子"、"咖啡相关描述" | "10:30左右发生了什么"、"认证专家检查墨迹的时刻" |

协同使用示例：  

1. 先用`search.clip("认证专家 墨迹")`定位到clip_015  
2. 再用`search.node("认证专家", clip_filter="clip_015")`在clip内查找具体人物节点  

# 训练策略与数据详情

## 记忆化过程（Qwen2.5-Omni）的训练

训练数据：  

- 500个长视频，切分为26,943个30秒clips  
- 每个clip的人工标注包括：  
  - 情景记忆文本（详细描述）  
  - 语义记忆文本（抽象知识）  
  - 人脸与声音的ID标注（用于监督实体关联）  

训练格式（ChatML格式）：  

```json
{
  "messages": [
    {
      "role": "system",
      "content": "You are a memorization agent. Process the video and audio to generate episodic and semantic memories."
    },
    {
      "role": "user",
      "content": [
        {"type": "video", "video_path": "clip_042.mp4"},
        {"type": "audio", "audio_path": "clip_042.wav"}
      ]
    },
    {
      "role": "assistant",
      "content": {
        "episodic_memory": "Alice坐在厨房餐桌旁，右手拿起红色陶瓷杯...",
        "semantic_memory": ["Alice喜欢喝咖啡", "Alice有红色陶瓷杯"],
        "entities": [
          {"type": "face", "id": "alice", "bbox": [100, 200, 300, 400]},
          {"type": "object", "class": "cup", "color": "red", "id": "cup_001"}
        ]
      }
    }
  ]
}
```

## 控制过程（Qwen3）的RL训练

奖励函数设计：  

1. 格式奖励：检查输出是否包含reasoning、action、argument三部分，结构正确得1分，否则0分。  
2. 检索正确性奖励：评估search动作是否检索到了包含答案相关信息的节点/clip。使用GPT-4o作为评判器，判断检索结果与问题的相关性，得分0或1。  
3. 答案正确性奖励：最终答案与标准答案的匹配度。使用GPT-4o或规则匹配（完全匹配得1分，部分匹配0.5分，错误0分）。  

GRPO训练细节：  
- Group size：1024（rollouts.n=1024）  
- 对每个问题采样1024条推理轨迹  
- 计算组内相对奖励（每条轨迹的reward减去组内平均reward）  
- Actor学习率：1e-6（极小，防止破坏预训练权重）  
- PPO epochs：4（对1024条轨迹进行4轮优化）  
- KL散度惩罚：防止策略偏离太远  

训练数据（2,736个QA对）示例：  

```json
{
  "question": "认证专家检查完Led Zeppelin专辑上的墨迹后做了什么？",
  "answer": "比较了笔迹",
  "evidence_clips": ["clip_015"],
  "reasoning_steps": [
    "定位到认证专家检查墨迹的时刻",
    "查找该时刻之后的动作",
    "确认动作是'比较笔迹'"
  ],
  "type": "multi-hop"
}
```

## 分离训练的原因验证

记忆化用Qwen2.5-Omni的原因：  

- 需要同时处理视频帧（Vision Transformer）和音频（Audio Encoder）  
- 支持多模态输入的tokenization（图像patch + 音频频谱图）  
- 预训练已具备基础的多模态理解能力  

控制用Qwen3的原因：  

- 纯文本输入（检索结果以文本形式传入）  
- 需要强大的CoT（Chain-of-Thought）推理能力  
- 支持ReAct范式（Reasoning + Action）  
- RL训练更高效（不需要处理多模态梯度）  

# 边界情况与限制

## 已知的系统限制

错误固化风险：如前所述，如果初始错误关联被系统性高频查询使用，weight会持续累积，导致难以自动纠正。系统缺乏外部纠错机制（如人工标注或主动验证）。  

记忆容量瓶颈：尽管使用向量数据库，但随着视频流持续输入（如7×24小时监控），节点数量将无限增长。文献未明确说明长期的记忆清理策略，仅提及weight-based pruning的可能性。  

事件切分粒度：30秒固定窗口可能导致事件边界模糊（如"倒水"动作在第30秒被截断）。系统通过时序边连接跨clip的片段，但未明确说明如何处理clip内部的事件边界检测。  

多模态对齐误差：人脸与声音的关联依赖时间对齐（同时出现）。如果存在延迟（如回声导致声音稍晚于画面），或只闻其声不见其人（如门外说话），关联准确性下降。  

## 跨模态检索的局限性

语义鸿沟：虽然embedding映射到同一空间，但文本描述的"红色杯子"与图像中的红色杯子可能存在语义差异（如文本忽略杯子的具体纹理，而图像包含）。检索时可能因描述不匹配而漏检。  

长时序推理：对于跨越多个clips的复杂推理（如"找出Alice上周所有喝咖啡的时刻"），系统需要遍历大量时序边，计算复杂度随时间范围线性增长，可能影响实时性。  


# References

【1】Seeing, Listening, Remembering, and Reasoning: A Multimodal Agent with Long-Term Memory, https://arxiv.org/pdf/2508.09736  

***  

【推荐文章】  
- Agent：  
[DeepResearch的报告生成方法](https://www.linsight.cn/44c62dc5.html)  
[从RAG到DeepSearch](https://www.linsight.cn/7c2f9dcb.html)  
[阿里通义Lab: WebWalker,WebDancer和WebSailor](https://www.linsight.cn/f7d600f3.html)  
[Agent评测数据集](https://www.linsight.cn/72150a83.html)  
[Agent完全手册(零)：三大模块，三个理念](https://www.linsight.cn/b242bfb3.html)  
[agent调研(1)--MetaGPT,OpenManus和OWL](https://www.linsight.cn/226b059f.html)  
[Devin和Anthropic的Agent开发经验](https://www.linsight.cn/f93b3aaf.html)  
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
[VeRA，LoRA-XS和TinyLoRA](https://www.linsight.cn/cc1c31d.html)  
[腾讯的Training-Free GRPO](https://www.linsight.cn/9cb56255.html)  
[深度求索DeepSeek-R1详解](https://www.linsight.cn/9e4b4e6d.html)  
[基模型Cognitive Behaviors对RL的影响](https://www.linsight.cn/657a6d17.html)  
[Llama3.1--post-training要点一览](https://www.linsight.cn/93328a2a.html)  
[模型平均 -- model soup](https://www.linsight.cn/bb8fcf21.html)  
[大模型偏好对齐-DPO](http://www.linsight.cn/473f2b43.html)  
[大模型偏好对齐-ODPO](http://www.linsight.cn/da871ebe.html)  
[大模型偏好对齐-simPO](http://www.linsight.cn/280fa97a.html)  
[大模型偏好对齐-IPO](http://www.linsight.cn/4fe7b810.html)  
- Transformer：  
[Attention Residuals](https://www.linsight.cn/5b81d487.html)  
[理解Attention:从起源到MHA,MQA和GQA](http://www.linsight.cn/3dc22f96.html)  
[LLM的重复生成和ICL](https://www.linsight.cn/7381cae3.html)  
[transformer中normalization的二三事](http://www.linsight.cn/6a40bfa5.html)  
[从代码实现看normalization-到底做了什么](http://www.linsight.cn/b70b4a2d.html)  
[稀疏注意力计算:sliding window attention](http://www.linsight.cn/c61d17e3.html)  
[理解LLM位置编码:RoPE](http://www.linsight.cn/a051710f.html)  
[RoPE的远距离衰减](https://www.linsight.cn/f0902f1a.html)  
[LLM水印](https://www.linsight.cn/2dee4921.html)  
- 训练框架  
[Muon优化器](https://www.linsight.cn/f25d614e.html)  
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
- 论文阅读：  
[最近阅读--关于数据合成、agent、reasoning和多任务](https://www.linsight.cn/e96c7aac.html)  
[最近阅读2-关于自适应深度思考、context engineering和模型训练](https://www.linsight.cn/af7f9363.html)  
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

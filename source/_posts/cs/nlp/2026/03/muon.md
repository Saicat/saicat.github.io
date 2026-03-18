---
title: Muon优化器
tags:
  - NLP
  - LLM
categories:
  - CS
  - NLP
  - LLM
abbrlink: f25d614e
date: 2026-03-12 22:45:02
---

Muon的全称是MomentUm Orthogonalized by Newton-schulz。  

# AdamW的问题

对于Transformer模型，大部分待优化的参数都是以矩阵的形式存在。  

而Adam优化器的对模型参数的更新是element-wise的，也就是每个参数单独更新。回想一下，Adam里的每个参数有自己的一阶动量二阶动量：  

$$
\begin{aligned}
m_t &= β_1 \cdot m_{t-1} + (1-β_1) \cdot g_t \\
v_t &= β_2 \cdot v_{t-1} + (1-β_2) \cdot g_t^2 \\
\hat{m}_t &= \frac{m_t}{1-β_1^t}, \quad \hat{v}_t = \frac{v_t}{1-β_2^t} \\
θ_{t+1} &= θ_t - \frac{η}{\sqrt{\hat{v}_t} + ϵ} \hat{m}_t
\end{aligned}
$$  

但是各个参数之间的更新却没有关联。  

苏神在分析Muon的博客里从多个角度进行了分析，大致来说，就是element-wise的更新方式忽略了矩阵参数的结构特性：  

- 比如矩阵的对角线元素和非对角线元素地位不对等  
- 矩阵乘法涉及的行空间和列空间有特定几何关系  

因此简单地按元素处理，无法区分优化的"重要方向"和"次要方向"。  

再理解一下：在 AdamW 的 element-wise 视角中，'重要'仅仅由梯度绝对值大小决定。但苏神指出，从矩阵几何视角，奇异值大的方向（主导方向）和奇异值小的方向（稀有方向）都应被平等对待——因为后者往往对应着数据中的稀有模式或细粒度特征。  

从这个角度，个人想法：训练数据本身就极可能是分布不均衡的：常见模式和稀有模式的比例差异很大，而如果鼓励平等对待所有模式，那么就可能对训练数据的质量提出了更高的要求，因为这种情况下一旦引入噪音或者错误数据（错误的稀有模式），被Muon跟主流末流一视同仁，就会对训练效果影响更大。  

# 梯度矩阵的低秩

AdamW在过去几年一直是最主流的优化器，甚至是唯一的选择。但是使用AdamW训练的模型，梯度矩阵却都展现出低秩的特性。实际上（训练后期）梯度矩阵的低秩特性，正是LoRA可行的理论基础：在增量训练中，梯度矩阵的秩很低，因此干脆只更新一个增量的低秩矩阵即可。  

一些其他的文献也有提到相关的观察，比如：  

1. 《Gradients are Not All You Need》 (2021，Google Research 团队) 发现：在大型 Transformer 中，梯度矩阵的数值秩（numerical rank）通常只有名义维度的 1-10%。一个2048*2048的矩阵，有效秩可能只有200或者20甚至更少。这是实际情况的观察。  
2. 《On the Low-Rank Structure of Gradient Matrices in Deep Learning》：发现：越深层，梯度的秩越低（直观理解，深层的特征更抽象，抽象的特征就更稀疏）；训练初期秩较高，后期迅速降低（比如LLM的输出分类，大部分的概率都集中在top10或者top50，其他大量类别在数值上都是接近0）。这个也是实际观察到的情况。  
3. 《Large Batch Optimization of Deep Neural Networks》：这篇论文里提到大batch size导致梯度协方差矩阵的低秩近似更粗糙，解释了为什么大批量容易陷入 sharp minima。怎么理解呢？小 batch 的梯度噪声具有'正则化'效果：噪声掩盖了真实梯度的极端低秩结构，使得优化轨迹在各方向上相对分散，客观上起到了探索更广参数空间的作用，使得秩相对真实的梯度矩阵更高，缓解了梯度更新过于集中的问题，也就是搜索的范围更广，更不容易陷入局部最优；而大 batch 下，噪声减小，真实梯度的病态低秩结构暴露无遗，优化器被迫在极度狭窄的子空间内更新，容易陷入 sharp minima。  

## Condition Number：矩阵的"病态程度"

Condition Number（通常记作 $\kappa$）衡量的是矩阵在各个方向上的"不均衡程度"或"病态程度"。它表示"矩阵的健康程度"——condition越好（数值越小），矩阵越容易处理；条件越差（数值越大），数值计算越困难。  

直观理解：想象你在一个山谷中寻找最低点（优化损失函数）：  

- 低条件数（$\kappa \approx 1$）：山谷是圆形的，往哪个方向走都一样容易，一步就能找到谷底  
- 高条件数（$\kappa \gg 1$）：山谷是极扁的椭圆形，沿长轴走很慢，沿短轴走会来回震荡，优化困难  

数学定义：  

$$\kappa = \frac{\sigma_{\max}}{\sigma_{\min}} = \frac{\text{最大奇异值}}{\text{最小奇异值}}$$  

梯度矩阵的低秩意味着 $\sigma_{\min} \approx 0$，因此 $\kappa \to \infty$（极度病态）。AdamW 试图在这个"极度压扁"的空间里寻找方向，就容易导致某些方向更新过度，某些方向更新不足。  

# Muon 的核心思想——谱均衡化（Spectral Whitening）

面对 AdamW 无法处理梯度低秩的困境，Muon 的核心洞察是：既然梯度矩阵的病态性来源于奇异值（singular values）的极端不均衡，那么干脆丢弃奇异值的大小信息，只保留方向信息。  

想象你在一个极度扭曲的山谷中行走（loss landscape）：  

- AdamW：按照坡度最陡的方向走，但山谷在某些方向极其狭窄（陡峭），某些方向极其平坦。你很容易被困在狭窄的沟壑里，来回震荡，而看不到旁边宽阔的平原。  
- Muon：先把山谷"拉平"——不管原本多陡或多平，都统一成 45 度坡。这样你每一步在各个方向上走的距离是一样的，不会在某个方向走太多，也不会在另一个方向走太少。  

数学上，对于 SGD-momentum 累积的动量矩阵 $M \in \mathbb{R}^{n \times m}$，首先考虑其 SVD 分解：  

$$M = U \Sigma V^\top = \sum_{i=1}^r \sigma_i u_i v_i^\top$$  

其中 $\sigma_1 \geq \sigma_2 \geq \dots \geq \sigma_r$ 是奇异值，$u_i$ 和 $v_i$ 分别是左、右奇异向量。AdamW 的更新量正比于 $M$，意味着 $\sigma_1$ 对应的方向（主导方向）获得最大更新幅度，而 $\sigma_r$ 方向（稀有方向）几乎被数值忽略——这就是"富者愈富，穷者愈穷"的马太效应。  

Muon 定义的msign（matrix sign function）为：  

$$\text{msign}(M) = UV^\top = \sum_{i=1}^r u_i v_i^\top$$

这相当于将所有奇异值强制置为 1，实现各向同性更新（isotropic update）。从几何上看，如果原始动量 $M$ 将一个单位球映射为一个极度扁平的椭球（条件数 $\kappa \gg 1$），那么 $\text{msign}(M)$ 将其映射为一个正交归一的球体，在所有方向上给予同等的更新幅度。  

关键性质：  

- $\text{msign}(M)$ 是 $M$ 的最优正交近似：$\arg\min_{O^\top O=I} \|M - O\|_F$  
- 对于正定矩阵，$\text{msign}(M) = (MM^\top)^{-1/2}M = M(M^\top M)^{-1/2}$  

直观理解：如果 $M$ 是一个"压扁的椭球"，$\text{msign}(M)$ 就是"同样方向但压成球"的版本。  

# Newton-Schulz 迭代——高效实现与系数优化

直接用 SVD 计算 $UV^\top$ 的复杂度为 $O(\min(nm^2, n^2m))$，每步优化都做一次 SVD 对训练开销不小。因此 Muon 采用Newton-Schulz（NS）迭代来近似 $\text{msign}(M)$。  

## 为什么不用 SVD？

- 计算成本：SVD 需要 $O(n^3)$ 迭代，无法利用 GPU 的矩阵乘法优化（matrix multiplication 是 GPU 的强项，SVD 不是）  
- 精度要求：SVD 通常需要 float32 精度以保证数值稳定性，而 NS 迭代可以在 bfloat16 下稳定运行（这应该是一个比较重要的点）
- 内存占用：SVD 需要存储中间矩阵 $U, \Sigma, V$  

## NS 迭代的数学原理与系数来源

目标：计算 $X = (MM^\top)^{-1/2}M$，即 $M$ 的"逆平方根"形式。  

思路来源：对于标量 $x$，$\text{sign}(x) = x / \sqrt{x^2} = x \cdot (x^2)^{-1/2}$。矩阵版本就是：  

$$\text{msign}(M) = M(M^\top M)^{-1/2}$$  

标准 NS 迭代源于对 $f(t) = t^{-1/2}$ 在 $t=1$ 处的 Taylor 展开：  

$$t^{-1/2} \approx \frac{15 - 10t + 3t^2}{8}$$  

这给出理论系数 $(a, b, c) = (1.875, -1.25, 0.375)$。但 Keller Jordan 发现这个系数收敛太慢，需要数十步才能稳定。  

实际系数的优化过程（针对 Marchenko-Pastur 分布）：  

设单个奇异值的迭代函数为：  

$$\varphi(x) = x + \kappa x(x^2 - x_1^2)(x^2 - x_2^2)$$  

其中 $x_1 < 1 < x_2$ 是不动点。通过网格搜索（Grid Search）或梯度下降优化 $(\kappa, x_1, x_2)$，使得在固定 $T=5$ 步内，最小化：
$$\mathbb{E}_{\sigma \sim \text{MP分布}}[(\varphi^T(\sigma) - 1)^2]$$  

优化后的系数牺牲了对所有 $\sigma \in [0,1]$ 的均匀收敛性，换取了对常见梯度分布（Marchenko-Pastur 分布）的快速收敛。实际 5 步后，奇异值收敛到 $U[0.5, 1.5]$ 而非精确的 1，但这在实践中不影响模型性能。  

实际使用的经验优化系数 $(3.4445, -4.7750, 2.0315)$ 是针对 square matrix 和 $T=5$ 的近似最优解。  

PyTorch 实现代码（from Kimi）：  

```python
def zeropower_via_newtonschulz(G, steps=5, eps=1e-7):
    """
    计算矩阵 G 的零次幂（即正交化）
    返回近似 UV^T
    """
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    
    # 使用 bfloat16 加速，且 NS 迭代在此精度下数值稳定
    X = G.bfloat16()
    
    # 归一化，确保最大奇异值 <= 1
    X /= (X.norm() + eps)
    
    # 如果行数 > 列数，转置以减小计算量（因为计算 X @ X.T）
    transpose = G.size(0) > G.size(1)
    if transpose:
        X = X.T
    
    # NS 迭代
    for _ in range(steps):
        A = X @ X.T           # X X^T
        B = b * A + c * A @ A  # 多项式中间项
        X = a * X + B @ X      # 更新
    
    if transpose:
        X = X.T
    
    return X
```

## Marchenko-Pastur 分布：随机矩阵的"普适分布"

在讨论 Newton-Schulz 迭代系数优化时，我们提到了Marchenko-Pastur（MP）分布。这是随机矩阵理论中的一个核心结果，描述的是大随机矩阵的奇异值会呈现什么样的统计分布。  

想象你有一个巨大的矩阵，里面的每个元素都是独立随机采样的（比如从高斯分布 $\mathcal{N}(0,1)$ 中随机抽取）。当你计算这个矩阵的奇异值（SVD）时，这些奇异值不会均匀分布，而是中间高两边低，而且是有偏的（MP分布）。  

在深度学习中的体现：  

- 大的梯度矩阵（特别是训练初期或随机初始化时）的奇异值近似服从 MP 分布  
- 训练后期，由于低秩结构的出现，大奇异值会"跳出" MP 边界（outliers），形成几个很大的奇异值，其余集中在 0 附近  

这就是为什么 NS 迭代的系数要针对 MP 分布优化——针对最常见、最自然的梯度分布进行优化。  

### Preconditioning：把"扁山谷"变成"圆山谷"

在理解 Muon 和 Shampoo 之前，先理解Preconditioning。  

核心思想：通过坐标变换，把病态的优化问题（高条件数）转化为良态的问题（条件数接近 1）。  

类比：你有一张被压扁的橡皮泥（梯度矩阵），某些方向被压缩了 1000 倍，某些方向被拉伸了 1000 倍。预条件化就是把它重新捏成标准的球体，让各个方向尺度一致。  

数学操作：  

对于梯度 $G$，预条件化更新为 $\Delta W = P^{-1} G$，其中 $P$ 是Preconditioner。理想情况下 $P$ 应该接近 Hessian 矩阵，但计算太贵。Shampoo 和 Muon 都在寻找计算可行且有效的预条件子。  

# 缩放因子：从初始化到更新尺度

实际使用中，不仅要对动量进行msign操作，还有一个对msign的缩放因子要乘上。  

有4个版本的缩放选择：  

对于矩阵 $W \in \mathbb{R}^{d_{out} \times d_{in}}$（注意 PyTorch Linear 层是 $d_{out} \times d_{in}$）：  

| 版本 | 缩放公式 | 适用场景 | 学习率调整 |
|------|---------|---------|-----------|
| 朴素版（Naive） | $1$ | 理论研究 | 需大幅调参 |
| KellerJordan 版 | $\sqrt{\max(1, d_{out}/d_{in})}$ | 追求层间一致性 | 需放大 5-10 倍 |
| MuP 版 | $\sqrt{d_{out}/d_{in}}$ | 跨模型尺寸迁移 | 需配合 MuP 调参 |
| Moonlight 版 | $0.2 \times \sqrt{\max(d_{out}, d_{in})}$ | 零成本迁移 | 直接使用 AdamW lr |

关键区别示例：  

对于 $W \in \mathbb{R}^{1024 \times 4096}$（矮胖，$d_{out} < d_{in}$）：  

- KellerJordan：$\sqrt{\max(1, 0.25)} = 1$（不放大）  
- Moonlight：$0.2 \times \sqrt{4096} = 12.8$（大幅放大，补偿 msign 的收缩）  

为什么有这么个缩放呢？理解 Muon 的缩放因子需要从神经网络初始化的最基本原则开始推导。  

## 为什么初始化要缩放？（Xavier/Kaiming初始化）

想象一个简单的线性网络 $h_{l+1} = W_l h_l$。如果 $W_{ij} \sim \mathcal{N}(0, 1)$（标准正态，方差=1）：  

- 经过矩阵乘法，$h_{l+1}$ 的方差变为 $d_{in} \cdot \sigma_l^2$（$d_{in}$ 个输入累加）  
- 经过 $L$ 层，方差变为 $d_{in}^L$，指数级爆炸  

为了保持前向传播中信号的方差恒定，必须让 $\mathbb{E}[W_{ij}^2] \approx \frac{1}{d_{in}}$，即方差反比于输入维度。这就是 Xavier/Kaiming 初始化。  

## 谱范数（Spectral Norm）与矩阵形状

在 Xavier 初始化下（$W_{ij} \sim \mathcal{N}(0, 1/d_{in})$），矩阵的谱范数（最大奇异值，即 $\|W\|_2$）满足：  

$$\|W\|_2 \approx \begin{cases} 
\sqrt{\frac{d_{out}}{d_{in}}} & \text{if } d_{out} \gg d_{in} \text{（高瘦）} \\
1 & \text{if } d_{out} \ll d_{in} \text{（矮胖）}
\end{cases}$$  

直观理解：  

- 高瘦矩阵（$d_{out} > d_{in}$）：把少量输入"扩张"到大量输出，需要更大的"放大系数"来填充输出空间，因此谱范数 $\propto \sqrt{d_{out}/d_{in}}$  
- 矮胖矩阵（$d_{out} < d_{in}$）：把大量输入"压缩"到少量输出，天然具有信息瓶颈，谱范数保持 $O(1)$  

## msign 破坏了尺度关系

Muon 使用 $\text{msign}(M) = UV^\top$，这相当于将所有奇异值设为 1（或接近 1），完全抹去了原始矩阵与维度相关的尺度信息。  

问题：  

- 如果不加缩放，高瘦矩阵（需要扩张能力）和矮胖矩阵（需要压缩能力）获得完全相同的更新幅度  
- 这违背了网络架构的原始设计：高瘦层需要更强的"驱动力"来填充高维输出空间  

## KellerJordan 缩放：恢复 Jacobian 一致性

KellerJordan 的缩放因子 $\sqrt{\max(1, d_{out}/d_{in})}$ 源于 MuP（Maximal Update Parametrization） 理论：  

- 当 $d_{out} > d_{in}$（高瘦）：乘以 $\sqrt{d_{out}/d_{in}} > 1$，放大，补偿扩张需求  
- 当 $d_{out} \le d_{in}$（矮胖或方阵）：保持 1，不进行压缩（避免进一步削弱矮胖层）  

目的：确保不同形状的权重矩阵在正交化后，其"影响力"（对网络输出的作用能力）与架构设计时一致。  

## Moonlight 缩放：与 AdamW 的 Update RMS 对齐

问题：KellerJordan 版需要重新调参（通常学习率要放大 5-10 倍）。  

解决思路：通过对齐 Update RMS（Root Mean Square of Update） 来自动确定缩放因子。  

推导：  

1. AdamW 在稳定期的 Update RMS $\approx \eta \times 0.2$  
2. $\text{msign}(M)$ 的元素期望幅值为 $1/\sqrt{\max(d_{in}, d_{out})}$  
3. 为了对齐，需要缩放因子 $c = 0.2 \times \sqrt{\max(d_{in}, d_{out})}$  

优势：直接使用 AdamW 的学习率，无需重新 grid search。  

# 混合使用

实际使用中，AdamW和Muon要配合起来，一起使用。两个优化器分别优化不同的参数。  

为什么必须混合使用？  

实证发现（Keller Jordan 博客 & Moonlight 论文）：  
- Embedding 层：使用 Muon 效果差甚至不稳定（sparse lookup，不满足稠密矩阵假设）  
- LM Head / 分类头：大词表下梯度极度稀疏（只有 top-k 有值），Muon 的 full-matrix 假设不适用  
- Bias / LayerNorm：1D 向量，Muon 退化为 SignSGD，效果不如 AdamW  

推荐配置：  

- Muon 用于：所有 Hidden Layers 的 Linear / Dense 权重（2D，稠密矩阵乘法）  
- AdamW 用于：Embedding、LM Head、Bias、LayerNorm 参数  

性能基准：  
- NanoGPT（GPT-2 规模）：达到同等验证 loss 的步数减少 30-40%  
- 1.5B 参数模型：10 小时 vs AdamW 13.3 小时（8×H100）  
- 稳定性：Moonlight 版在大学习率下比原版更稳定，loss 曲线更平滑  

## 完整配置脚本

```python
# 参数自动分组（Moonlight 版推荐配置）
muon_params = []
adam_params = []

for name, p in model.named_parameters():
    # Embedding 和输出层用 AdamW（稀疏访问）
    if 'embed' in name or 'lm_head' in name or 'head' in name:
        adam_params.append(p)
    # Bias 和 Norm 用 AdamW（1D 向量）
    elif p.ndim <= 1 or 'bias' in name or 'norm' in name:
        adam_params.append(p)
    # 2D 矩阵参数用 Muon（稠密矩阵乘法）
    elif p.ndim == 2:
        muon_params.append(p)
    else:
        adam_params.append(p)

# Moonlight 版 Muon（对齐 AdamW 学习率，零成本迁移）
optimizer_muon = Muon(
    muon_params, 
    lr=3e-4,  # 直接使用 AdamW 的学习率，无需调整
    momentum=0.95,
    weight_decay=0.01,
    scale_mode='moonlight'
)

optimizer_adam = torch.optim.AdamW(
    adam_params,
    lr=3e-4,
    betas=(0.9, 0.95),
    weight_decay=0.01
)

# 分布式训练注意事项
# 张量并行（Tensor Parallelism）下，Muon 需要 all-gather 完整梯度矩阵
# 这会增加通信开销，是超大规模训练的主要工程挑战
```

# 总结

Muon 代表了优化器设计从 Element-wise 到 Matrix-wise 的范式转变。通过 Newton-Schulz 迭代高效近似 matrix sign function，强制梯度更新在各奇异方向上均衡分布，从而：  

1. 打破低秩陷阱：防止优化被少数主导方向垄断，挽救稀有特征的学习机会；而通常来说，超高质量数据占比少，就属于稀有特征，因此在困难任务的优化上有优势  
2. 尊重矩阵几何：利用谱范数（Spectral Norm）而非 F-范数，通过左右 Gram 矩阵分别处理输入/输出空间的几何结构，将病态的优化 landscape（高 Condition Number）转化为良态（$\kappa \approx 1$）  
3. preconditioning视角：通过 $L^{-1/4} G R^{-1/4}$ 的几何操作（解耦与均衡），或简化为 $\text{msign}(G)$ 的谱均衡化，实现矩阵参数的有效更新（其实不是很懂，还得多看苏神文章）  
4. 工程实用性：相比 Shampoo 的 $O(m^3+n^3)$ 复杂度，Muon 通过 NS 迭代将每步开销控制在 $O(mn \cdot \min(m,n))$，且支持 bfloat16 快速计算  

在实践中，Moonlight 版 Muon 通过对齐 AdamW 的 Update RMS（$0.2 \times \sqrt{\max(d_{in}, d_{out})}$ 缩放因子），实现了从 AdamW 的零成本迁移——保持学习率不变，只更换优化器，即可享受矩阵结构感知的优化优势。对于已有 AdamW 配置的用户，这是目前最平滑的升级路径。  

# *关于Spectral Norm

谱范数 $\|W\|_2$ 定义为矩阵的最大奇异值 $\sigma_{\max}(W)$，等价于矩阵作为线性算子的算子范数：$\|W\|_2 = \sup_{\|x\|=1} \|Wx\|$。它刻画了矩阵对输入向量的最大拉伸倍数，直接决定前向传播时信号可能达到的最大增益。  

F范数 $\|W\|_F$ 定义为矩阵所有元素平方和的平方根，即 $\|W\|_F = \sqrt{\sum_{i,j} W_{ij}^2}$。若对 $W$ 进行奇异值分解，F范数也可表示为所有奇异值的平方和开根：$\|W\|_F = \sqrt{\sum_k \sigma_k^2}$。因此，F范数反映的是矩阵的整体能量或总变化幅度，而非单一方向的极值。  

两者关系

对任意 $d_{out} \times d_{in}$ 矩阵 $W$（设秩为 $r$），两个范数满足如下不等式链：  

$$\|W\|_2 \leq \|W\|_F \leq \sqrt{r} \|W\|_2 \leq \sqrt{\min(d_{out}, d_{in})} \|W\|_2$$  

下界表明最大奇异值不会超过总能量；上界则表明当矩阵秩为1时两者相等，随着秩增加，F范数可能显著大于谱范数，但至多不超过 $\sqrt{r}$ 倍。  

Xavier初始化下的谱范数  

Xavier初始化将权重设为 $W_{ij} \sim \mathcal{N}(0, 1/d_{in})$。这等价于 $W = \frac{1}{\sqrt{d_{in}}}X$，其中 $X$ 为元素服从标准正态分布 $\mathcal{N}(0,1)$ 的随机矩阵。  

根据随机矩阵理论（Marchenko-Pastur定律），当维度较大时，高斯随机矩阵 $X$ 的最大奇异值几乎必然收敛于 $\sigma_{\max}(X) \approx \sqrt{d_{out}} + \sqrt{d_{in}}$。代入Xavier的缩放因子，得到：  

$$\|W\|_2 \approx \frac{\sqrt{d_{out}} + \sqrt{d_{in}}}{\sqrt{d_{in}}} = 1 + \sqrt{\frac{d_{out}}{d_{in}}}$$  

由此产生两种典型情况：  

1. 矮胖矩阵（$d_{out} \ll d_{in}$，如降维层）：此时 $\sqrt{d_{out}/d_{in}} \approx 0$，故 $\|W\|_2 \approx 1$。输入维度高于输出维度，矩阵主要起压缩作用，最大拉伸倍数被限制在1附近，信号传播稳定。  

2. 高瘦矩阵（$d_{out} \gg d_{in}$，如升维层）：此时常数项1可忽略，$\|W\|_2 \approx \sqrt{d_{out}/d_{in}}$。输出维度显著高于输入维度时，即使采用Xavier初始化，矩阵的最大奇异值仍会随宽高比的平方根增长。例如，当输出维度是输入的100倍时，谱范数约为10，意味着特定方向的信号可能被放大100倍（方差放大100倍）。  

对神经网络训练的影响  

这一结果揭示了Xavier初始化在不同网络结构中的局限性。对于包含大量升维层（如扩散模型中的某些架构）的网络，仅靠 $1/\sqrt{d_{in}}$ 的方差缩放无法控制谱范数，可能导致前向信号爆炸或反向梯度异常。实践中需配合LayerNorm、BatchNorm或谱归一化（Spectral Normalization）等技术，强制控制每一层的最大拉伸倍数，确保训练稳定。  

# Reference  

【1】Muon: An optimizer for hidden layers in neural networks，https://kellerjordan.github.io/posts/muon/  
【2】Muon优化器赏析：从向量到矩阵的本质跨越，https://kexue.fm/archives/10592  
【3】Muon优化器指南：快速上手与关键细节，https://www.spaces.ac.cn/archives/11416  
【4】Deriving Muon，https://jeremybernste.in/writing/deriving-muon，Muon 作者的数学推导博客，解释 MuP（Maximal Update Parametrization）与 Muon 的关系  

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

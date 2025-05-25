---
title: LLM训练框架：从优化器和精度讲到ZeRO
tags:
  - NLP
  - LLM
  - 预训练
  - 分布式
categories:
  - CS
  - NLP
  - LLM
abbrlink: fe0adaa5
date: 2025-05-17 16:33:15
---

【本文已在同名 微信公众号 / 知乎 / [个人博客linsight.cn](http://www.linsight.cn/) 上线】  

***  

这篇文章主要从一个搞数据和训练策略的LLM算法工程师角度总结一下之前用到的训练框架相关知识，包括优化器、精度和混合精度训练和DP、ZeRO的相关内容。  

# Optimizer  

现在模型的优化器就是AdamW。虽然这几年也试过Lion和Muon等一些新兴的optimizer，不过实践中最稳当的暂时还是AdamW。  

## 从SGD到AdamW  

先复习下从SGD到AdamW这些个优化器。  

1. SGD  

SGD的更新公式：  

$$
θ_{t+1} = θ_t - η \cdot g_t
$$

- $θ_t$：模型参数  
- $η$：learning rate  
- $g_t$：当前梯度（$\nabla_θ L(θ_t)$）  

SGD只依赖当前最新计算出的梯度，直接更新模型的参数值。  

2. Momentum SGD  

公式：  

$$
\begin{aligned}
v_t &= γ \cdot v_{t-1} + g_t \\
θ_{t+1} &= θ_t - η \cdot v_t
\end{aligned}
$$

- $v_t$：包含动量项的梯度（加权移动平均的累积梯度）  
- $γ$：动量系数/加权系数（比如0.9，越大表示梯度更新越慢，设为0就等于SGD了）  

模型在训练初期，轮次之间的梯度变化比较大，梯度甚至可能发生180°大调头的情况，导致震荡，所以SGD不容易收敛。Momentum SGD通过累积历史的梯度值，减少震荡，从而稳定训练，加速收敛。  

3. AdaGrad  

AdaGrad尝试让不同的参数有自己的学习率，并且可以自适应调整。  

公式：  

$$
\begin{aligned}
G_t &= G_{t-1} + g_t^2 \\
θ_{t+1} &= θ_t - \frac{η}{\sqrt{G_t} + ϵ} g_t
\end{aligned}
$$

- $G_t$：梯度平方的累积值  
- $ϵ$：防止除零（如1e-8）  

如果一个参数的更新速度比较快，那么对应的G就会比较大，那么相应的学习率也会减小；反之则学习率会相对较大。  

4. RMSProp  

AdaGrad中因为会累积所有历史梯度平方值，这样到后期每个参数的学习率都衰减到比较小，如果训练的step比较多，到后面就效率太低了。  

RMSProp比AdaGrad多使用一个加权移动平均。  

公式：  

$$
\begin{aligned}
v_t &= β \cdot v_{t-1} + (1-β) \cdot g_t^2 \\
θ_{t+1} &= θ_t - \frac{η}{\sqrt{v_t + ϵ}} g_t
\end{aligned}
$$

- $v_t$：梯度平方的指数移动平均  
- $β$：加权移动平均衰减率（通常0.9）  

5. Adam  

把RMSProp和Momentum SGD的改进结合起来就是Adam了。  

Adam = Adaptive + Momentum。  

公式：  

$$
\begin{aligned}
m_t &= β_1 \cdot m_{t-1} + (1-β_1) \cdot g_t \\
v_t &= β_2 \cdot v_{t-1} + (1-β_2) \cdot g_t^2 \\
\hat{m}_t &= \frac{m_t}{1-β_1^t}, \quad \hat{v}_t = \frac{v_t}{1-β_2^t} \\
θ_{t+1} &= θ_t - \frac{η}{\sqrt{\hat{v}_t} + ϵ} \hat{m}_t
\end{aligned}
$$

- $m_t$：一阶动量，控制当前梯度的方向，初始状态m=0  
- $v_t$：二阶动量，控制当前梯度的大小，初始状态v=0  
- $β_1,β_2$：衰减率（通常$β_1=0.9,β_2=0.999$）

本来更新的时候直接使用

$$
θ_{t+1} = θ_t - \frac{η}{\sqrt{v_t} + ϵ} m_t
$$

就可以了，为什么还要给v和m做一个缩放修正呢？因为在一开始的时候，历史的动量值都是0，这样导致训练初期更新的梯度太小，因此在前期给缩放一下，基本上跑个几百几千步，这个这个缩放基本就趋近于1了。  

6. AdamW  

为了防止过拟合，提高泛化性，模型训练的时候可以加入L2 norm。一般来说L2 norm是直接加在训练loss上的。  

L2 norm项是这样的：  

$$\frac{\lambda}{2} \|\theta\|^2$$

直接加到训练loss上：  

$$L'(\theta) = L(\theta) + \frac{\lambda}{2} \|\theta\|^2$$

那么梯度就有：  

$$g_t=\nabla_\theta L(\theta_{t-1})+\lambda\theta_{t-1}$$

Adam在对梯度进行缩放的时候，L2 norm的衰减项也会被缩放，因此就达不到本来想要的效果了。  

所以就有了AdamW的改进，让L2 norm的正则化能力可以正常实现。  

AdamW不把L2 norm加到loss项中，而是直接把对应梯度加到参数更新中。  

公式：  

$$
θ_{t+1} = θ_t - \frac{η}{\sqrt{\hat{v}_t} + ϵ} \odot \hat{m}_t - λ \cdot θ_t
$$

- $λ$：L2 norm的权重衰减系数  

## AdamW的显存需求  

AdamW训练的时候，除了模型参数，还需要维护一阶动量和二阶动量。  

在全部使用fp32的情况下，假设模型的总参数量为$\Phi$，那么模型本身参数所需的显存就是$4\Phi$。  

而AdamW维护的一阶动量和二阶动量的参数则是$4\Phi+4\Phi=8\Phi$。  

此外还有梯度值，每个模型参数有一个梯度，那么梯度所需的量也是$4\Phi$。  

那么模型参数 + 优化器参数 + 梯度总共就是$16\Phi$的显存需求。  

最后还有中间激活值，激活值的量和模型结构有关，对于transformer也和输入长度有关，再加上现在还有gradient checkpoint等做法，所以激活值就得具体情况具体分析了。  

# 精度  

说到LLM训练，就离不开训练精度的事。  

## 以FP32为例说明  

先从FP32说起。FP32的二进制结构分为三部分：  

- 符号位（S，Sign）：1位，0表示正数，1表示负数。  
- 指数位（E，Exponent）：8位，存储偏移后的指数值（为了能够表达正值和负值，加上了127的偏移量，实际指数为E - 127）。  
- 尾数位（M，Mantissa）：23位，存储规范化后的二进制小数部分（隐含前导1.）。  

十进制和FP32转换公式：  

$$
(-1)^S \times 1.M \times 2^{E-127}
$$

举几个例子看看二进制和十进制的转换。  

示例1. 十进制 → FP32（以9.625为例）  

step 1：十进制转二进制  

- **整数部分**：`9` → `1001`（二进制）。  
- **小数部分**：`0.625` → 连续乘2取整：  
  - `0.625 × 2 = 1.25` → 取`1`，剩余`0.25`  
  - `0.25 × 2 = 0.5` → 取`0`，剩余`0.5`  
  - `0.5 × 2 = 1.0` → 取`1`，剩余`0`  
  - 结果：`0.101`（二进制）。  
- **合并**：`9.625` → `1001.101`。  

step 2：规范化科学计数法  

- `1001.101` → `1.001101 × 2^3`（左移3位）。对于二进制来说，整数位一定是1。  

step 3：填充FP32三部分  

- **符号位**：`0`（正数）。  
- **指数位**：`3 + 127 = 130` → `10000010`（二进制）。  
- **尾数位**：`001101` + 补零至23位 → `00110100000000000000000`。  

最终9.625的FP32表示：  

`0 10000010 00110100000000000000000`  

（验证工具：[IEEE-754 Converter, https://www.h-schmidt.net/FloatConverter/IEEE754.html](https://www.h-schmidt.net/FloatConverter/IEEE754.html)）  

示例2：FP32 → 十进制（反向解析出十进制）  

二进制为：`0 10000010 00110100000000000000000`  

- **符号位**：`0` → 正数。  
- **指数位**：`10000010` → 十进制`130` → 实际指数`130 - 127 = 3`。  
- **尾数位**：`001101...` → 隐含`1.` → `1.001101`（二进制）。  

计算十进制值：  

step 1：将`1.001101`转为十进制：  
   - `1.001101` = $1 + 0 \times 2^{-1} + 0 \times 2^{-2} + 1 \times 2^{-3} + 1 \times 2^{-4} + 0 \times 2^{-5} + 1 \times 2^{-6}$
     = $1 + 0.125 + 0.0625 + 0.015625$ ≈ `1.203125`。  
step 2：乘以指数部分：`1.203125 × 2^3 = 9.625`。  

- **特殊值处理**：  

  - 指数全`0`且尾数非零：非规格化数（极小值）。  
  - 指数全`1`且尾数全`0`：表示无穷大（`±∞`）。  
  - 指数全`1`且尾数非零：`NaN`（非数字）。  
  
- **精度限制**：某些十进制数（如`0.3`）无法精确表示为FP32，会存在舍入误差。  

## 其他常用精度  

目前LLM训练的除了FP32，还有FP16、BF16，以及更新的FP8。  

这几个的对比：  

|格式|符号位|指数位|尾数位|总位数|数值范围(近似)|特点|
|-|-|-|-|-|-|-|
|FP32|1|8|23|32|±1.2×10⁻³⁸ ~ ±3.4×10³⁸|高精度（约7位有效数字），通用计算标准，适合训练但资源消耗大。|
|FP16|1|5|10|16|±6.1×10⁻⁵ ~ ±6.6×10⁴|内存占用减半，速度快但易溢出/下溢，需混合精度训练。|
|BF16|1|8|7|16|±1.2×10⁻³⁸ ~ ±3.4×10³⁸|指数范围同FP32，训练稳定但精度低（约2位有效数字），适合大模型。|
|FP8(E4M3)|1|4|3|8|±1.56×10⁻⁵ ~ ±448|内存占用极低，适合推理；E4M3侧重精度，范围较小。|
|FP8(E5M2)|1|5|2|8|±3.9×10⁻⁸ ~ ±57344|E5M2侧重范围，精度更低，适合大动态范围计算。|

目前最常用的还是FP16和BF16（FP8我自己还没怎么用，先挖个坑，以后用熟了FP8再来填）。这俩的对比：  

- **指数位**：BF16与FP32相同（8位），FP16仅5位，因此表示范围小，更易溢出；  
- **尾数位**：FP32（23位）> FP16（10位）> BF16（7位）> FP8（3/2位），精度依次降低。  
- **应用场景**：FP32用于高精度训练，FP16/BF16都可以用于混合精度训练，FP8用于端侧设备推理。  

直观上，BF16的精度大概是在0.01到0.001之间，而BF16的精度是在0.001到0.0001之间。也就是说，如果一次梯度更新小于这个值，那么参数很可能没法正确地变化。  

# 混合精度训练  

混合精度训练时减少显存使用，提升训练速度的方法。  

为什么用混合精度训练，不直接使用低精度的格式进行训练？从前面精度的表格可以看到，无论是FP16还是BF16，要么在精度上有损失，要么在表达范围上有限制，因此直接用低精度格式训练，可能会在需要高精度或者大范围的部分导致不稳定。因此混合精度方案在大部分计算使用半精度的同时，用FP32对关键部分进行备份，在速度、显存和稳定性间取得平衡。  

## 显存  

AdamW的单精度和半精度的混合精度训练如下（图上是FP16，也可以换成BF16）：  

{% asset_img mix_precision_fp16.png 混合精度训练 %}  

输入是FP16，前向计算激活值是FP16，loss值是FP32的，反向计算的值和梯度是FP16，AdamW的一阶和二阶动量是FP32，而AdamW更新模型参数权重用的是FP32，而在进行前后向计算的时候，模型参数用的是FP16的版本。  

算一下显存：  

模型参数：一份单精度一份半精度，总共就是$2\Phi+4\Phi=6\Phi$。  

优化器参数：每个参数有单精度的一阶动量和二阶动量，总共就是$8\Phi$。  

梯度：每个参数有半精度的梯度，$2\Phi$。  

那么总共就是$16\Phi$的显存消耗。  

从模型参数+优化器参数+梯度的显存消耗上看，单精度训练和混合精度（FP32 + FP16/BF16）的显存消耗量是一样。但是，混合精度在效率上的收益有：  

- 有硬件支持下，半精度的计算更快，因此整体的计算速度更快。  
- 激活值所需的显存减少一半，从而可以使用更大的batch。  
- 一些原来单卡放不下的，现在能放下了，不用做张量并行或者流水并行。  

## 训练  

前面说到直接用半精度进行训练会有问题，那么混合精度训练具体是怎么解决这些问题的。  

首先，半精度的精度不足，因此混合精度中，AdamW维护了一份FP32的模型权重，这个是真正用于更新模型的数据，这样可以保持较小的梯度更新也不会被舍弃。每次更新完之后，再把获得的FP32参数转成FP16，用于前后向计算。  

另外，由于半精度值的精度问题，较小的梯度值可能直接变成0了，这样就导致没法训练参数了。那么一个解决方法就是像上面的图中那样，给loss值做一个scaling，变大一些，尽量远离太小的值。由于loss值变大，会导致梯度值也变大相应倍数，因此在更新完模型参数值之后，要做一个逆scaling，把值变回去。  

{% asset_img loss_scaling.png 混合精度训练 %}  

另外，还有一招：使用FP16进行乘法和存储，只使用FP32进行加法操作，避免累加误差。因为加法的误差会一直累积，因此用单精度计算。  

# Data Parallel  

模型大，数据多，难免就需要分布式计算。其中，最常用的就是数据并行。其实我们训练百亿以下的模型，基本上都是只用数据并行。  

使用最朴素的数据并行，每个GPU会维护一套完整的模型参数 + 优化器参数 + 梯度。每次更新，每个GPU用不同的数据「单独」进行训练，获取梯度，然后所有GPU会同步各自获得的梯度，计算个平均值，然后更新参数。每轮更新过后，模型参数会统一，而优化器状态则每个GPU有各自的版本（因此保存训练checkpoint的时候会有大量的优化器状态值）。  

可以看到，每次更新时，各个GPU需要同步梯度，这就涉及到大量的卡间通讯，甚至节点间通讯。比如128卡训练模型，那么naive的数据同步方式就是两两之间都要进行数据传输和接收；那么训练一个14B模型，在用半精度的梯度的情况下，每张卡要发送127 * 28G = 3556G数据，同时要接收127 * 28G = 3556G的数据，而且随着集群的变大，这个数值还会增大。就算是A100，卡间带宽也只有2TB/s，那同步一次就是的1s多，这期间所有卡都得停下计算，等通讯完成。多节点之间的带宽更小，那GPU的利用率就更低了。  

这也太低效了，因此实际上就不是这样同步数据的，而是用到了Ring AllReduce的梯度同步算法。  

## Ring AllReduce算法  

顾名思义，Ring AllReduce把各个GPU组成一个ring，以ring的形式进行通讯，以减少通讯量。  

{% asset_img ring.jpg 混合精度训练 %}  

allreduce同步梯度数据的过程主要包含reduce-scatter和all-gather两个操作。  

1. reduce-scatter  

假设一共有5个GPU，要同步梯度。那么把梯度数据均匀划分成A、B、C、D、E五块。  

初始状态下，每个GPU持有的数据如下：  

| |A|B|C|D|E|
|-|-|-|-|-|-|
|GPU0|a0|b0|c0|d0|e0|
|GPU1|a1|b1|c1|d1|e1|
|GPU2|a2|b2|c2|d2|e2|
|GPU3|a3|b3|c3|d3|e3|
|GPU4|a4|b4|c4|d4|e4|

reduce-scatter的操作，每个GPU会发送自己持有的A、B、C、D、E中的其中一块数据，同时接收和自己发送的数据不同块的一块数据。  

比如在这个例子中，GPU0发送a0，并接收e4，GPU1发送b1，并接收a0，以此类推。  

第一次reduce-scatter操作之后：  

| |A|B|C|D|E|
|-|-|-|-|-|-|
|GPU0|a0|b0|c0|d0|e4+e0|
|GPU1|a0+a1|b1|c1|d1|e1|
|GPU2|a2|b1+b2|c2|d2|e2|
|GPU3|a3|b3|c2+c3|d3|e3|
|GPU4|a4|b4|c4|d3+d4|e4|

第二次reduce-scatter操作之后：  

| |A|B|C|D|E|
|-|-|-|-|-|-|
|GPU0|a0|b0|c0|d3+d4+d0|e4+e0|
|GPU1|a0+a1|b1|c1|d1|e4+e0+e1|
|GPU2|a0+a1+a2|b1+b2|c2|d2|e2|
|GPU3|a3|b1+b2+b3|c2+c3|d3|e3|
|GPU4|a4|b4|c2+c3+c4|d3+d4|e4|

第三次reduce-scatter操作之后：  

| |A|B|C|D|E|
|-|-|-|-|-|-|
|GPU0|a0|b0|c2+c3+c4+c0|d3+d4+d0|e4+e0|
|GPU1|a0+a1|b1|c1|d3+d4+d0+d1|e4+e0+e1|
|GPU2|a0+a1+a2|b1+b2|c2|d2|e4+e0+e1+e2|
|GPU3|a0+a1+a2+a3|b1+b2+b3|c2+c3|d3|e3|
|GPU4|a4|b1+b2+b3+b4|c2+c3+c4|d3+d4|e4|

第四次reduce-scatter操作之后：  

| |A|B|C|D|E|
|-|-|-|-|-|-|
|GPU0|a0|b1+b2+b3+b4+b0|c2+c3+c4+c0|d3+d4+d0|e4+e0|
|GPU1|a0+a1|b1|c2+c3+c4+c0+c1|d3+d4+d0+d1|e4+e0+e1|
|GPU2|a0+a1+a2|b1+b2|c2|d3+d4+d0+d1+d2|e4+e0+e1+e2|
|GPU3|a0+a1+a2+a3|b1+b2+b3|c2+c3|d3|e4+e0+e1+e2+e3|
|GPU4|a0+a1+a2+a3+a4|b1+b2+b3+b4|c2+c3+c4|d3+d4|e4|

假设共有N个GPU，经过N-1次操作之后，每个GPU上，都有1/N块数据是同步了所有GPU数据的。  

在这个例子中，GPU0的B块是包含了完整的5个GPU的数据的，而GPU1则是C块是完整的，以此类推。  

接下来，就需要用all-gather把每个GPU上这份同步了所有GPU数据的块传播给其他GPU。  

2. all-gather  

其实all-gather和reduce-scatter的操作是很类似的，只不过reduce-scatter是相加/取平均，而all-gather是直接覆盖数据。  

all-gather第一次操作后：  

| |A|B|C|D|E|
|-|-|-|-|-|-|
|GPU0|a0+a1+a2+a3+a4|b1+b2+b3+b4+b0|c2+c3+c4+c0|d3+d4+d0|e4+e0|
|GPU1|a0+a1|b1+b2+b3+b4+b0|c2+c3+c4+c0+c1|d3+d4+d0+d1|e4+e0+e1|
|GPU2|a0+a1+a2|b1+b2|c2+c3+c4+c0+c1|d3+d4+d0+d1+d2|e4+e0+e1+e2|
|GPU3|a0+a1+a2+a3|b1+b2+b3|c2+c3|d3+d4+d0+d1+d2|e4+e0+e1+e2+e3|
|GPU4|a0+a1+a2+a3+a4|b1+b2+b3+b4|c2+c3+c4|d3+d4|e4+e0+e1+e2+e3|

...

以此类推，最后得到  

| |A|B|C|D|E|
|-|-|-|-|-|-|
|GPU0|a0+a1+a2+a3+a4|b1+b2+b3+b4+b0|c2+c3+c4+c0+c1|d3+d4+d0+d1+d2|e4+e0+e1+e2+e3|
|GPU1|a0+a1+a2+a3+a4|b1+b2+b3+b4+b0|c2+c3+c4+c0+c1|d3+d4+d0+d1+d2|e4+e0+e1+e2+e3|
|GPU2|a0+a1+a2+a3+a4|b1+b2+b3+b4+b0|c2+c3+c4+c0+c1|d3+d4+d0+d1+d2|e4+e0+e1+e2+e3|
|GPU3|a0+a1+a2+a3+a4|b1+b2+b3+b4+b0|c2+c3+c4+c0+c1|d3+d4+d0+d1+d2|e4+e0+e1+e2+e3|
|GPU4|a0+a1+a2+a3+a4|b1+b2+b3+b4+b0|c2+c3+c4+c0+c1|d3+d4+d0+d1+d2|e4+e0+e1+e2+e3|

## Ring AllReduce特点  

Ring AllReduce理论上已经是同步算法的最佳，它的特点是随着GPU数量的增多，整个过程所需的时间几乎保持不变，也就是通讯时间成本和机器数量无关！  

这么一来，在使用更大集群的时候，节点间的通讯就不会成为提升线性扩展比的瓶颈。比如你原来128卡要训一天，那几乎可以认为256卡训半天就能达到相同的程度。  

当然理论是理论，实际上当设备数非常大，还会有另外的问题。  

OneFlow的这篇文章介绍得很清楚，可以一读：[手把手推导Ring All-reduce的数学性质(https://zhuanlan.zhihu.com/p/504957661)](https://zhuanlan.zhihu.com/p/504957661)。  

# ZeRO  

那么除了上面的混合精度方案，Ring AllReduce的DP之外，还有没有什么方法能简单快捷减少显存，提升训练效率？兄弟，有的，而且很强，那就是ZeRO。  

ZeRO = Zero Redundancy Optimizer  

ZeRO核心是优化显存，减少训练所需的显存占用。

ZeRO有三个stage，ZeRO-1，ZeRO-2，ZeRO-3，对显存的优化逐步变强（但是代价也逐步增加）。  

## ZeRO-1  

原来呢，在FP32 + FP16的混合精度训练下，对于包含$\Phi$个参数的模型，每个GPU都存有一份完整的模型参数、梯度和优化器状态：  

- 模型参数（FP16 + FP32）：$6\Phi$（byte）  
- 梯度（FP16）：$2\Phi$  
- 优化器状态（FP32）：$8\Phi$  

每次同步完梯度之后，各个GPU会各自更新优化器状态。  

这里面其实就有巨大的显存冗余，因为每个GPU都有一份一样的优化器状态，而AdamW的优化器状态又占了很大一部分显存（比如7B的模型就有28G的优化器状态）。  

那ZeRO-1就想办法消除这个优化器状态的冗余。核心思想就是：  

- partition：有N个GPU，就把优化器状态切分成N份，每个GPU在整个训练过程中，只保存和管理其中的一份。  
- distributed update：每个GPU只负责更新其所持有的那部分优化器状态 对应的 模型参数。  

有开ZeRO-1和没有开ZeRO-1，在流程上差别就在于梯度同步之后的操作。开ZeRO-1的情况下，同步梯度之后，由于每个GPU只有1/N的优化器状态，因此只能更新对应的1/N的模型参数。更新完1/N的参数之后，为了能在下次迭代时保持各个GPU上模型参数的一致性，就还要做一次all-gather来同步模型的参数。  

显存上，每个GPU只需维护$8\Phi/N$的优化器状态，而且GPU数越多N越大，那么每个GPU所需的显存就越少。这简直就是训练框架界的PDD：用得越多省得越多！有可能你一个模型本来8卡会CUDA OOM，那在开ZeRO的情况下，可能多加点卡，比如32卡，就不会OOM了。  

而在通讯量上，多了一次模型参数的all-gather，所以理论上是$2\Phi+\Phi=3\Phi$个参数。但是这里还有个优化点，可以把实际的通讯量降到$2\Phi$个参数。  

回想一下，同步梯度的时候，分为reduce-scatter和all-gather两步，每步的通讯量都是$\Phi$个参数。reduce-scatter让每个GPU拥有1/N块完整的梯度，all-gather把每个GPU拥有的这块完整梯度同步给其他所有GPU。但是在ZeRO-1的情况下，就可以不做梯度的all-gather。因为ZeRO-1的情况下，每个GPU只有1/N的优化器状态，也只会更新1/N的模型参数，同步整个模型的梯度没有意义。因此这个我们只需要对梯度做reduce-scatter，让每个GPU拥有需要更新的参数的那部分完整梯度就可以了！等每个GPU都更新完自己的那部分模型参数之后，再对模型参数做all-gather就可以了。这么一来ZeRO-1的通讯量完全没有增加，但是显存消耗量却减少了，这完全是免费午餐。  

## ZeRO-2 & ZeRO-3  

ZeRO-2在ZeRO-1的基础上，对梯度也进行了切分。每个GPU只有模型参数是完整的，而梯度和优化器状态都只会储存和管理1/N的小块，而不是完整的一份。ZeRO-2和ZeRO-1的通讯量一样，都是$2\Phi$个参数：一次梯度的reduce-scatter，一次新参数的all-gather。  

而ZeRO-3更进一步，每个GPU上连模型参数都是不完整的。forward计算的时候，要先做一次all-gather，计算完就把不属于自己的模型参数都释放掉。同样地，backward的时候也是类似操作。之后就是和ZeRO-1/2一样，同步新参数。因此总通讯量是$3\Phi$个参数。ZeRO-3的通讯量会增大，而显存的训练则大大减小，颇有点时间换空间的意味。  

DDP，ZeRO-1，ZeRO-2和ZeRO-3在7.5B参数的模型和64卡集群下，显存的消耗对比（没有包含激活值）：  

{% asset_img zero.png 混合精度训练 %}  

另外，模型计算过程中的激活值也是可以切割和分块维护的，这块就比较复杂了，要根据实际情况灵活设计要保存的activation和要切分的块。  

## ZeRO-Offload  

ZeRO-Offload另辟蹊径，把放不进显存的变量放到内存上了。  

ZeRO-Offload基于ZeRO-2的优化器状态和梯度分片策略，但进一步将这两类数据卸载（offload）到CPU内存中，同时利用CPU的计算能力执行部分低复杂度任务（如参数更新），将高计算复杂度的前向/反向传播保留在GPU，低复杂度的优化器更新卸载到CPU，避免CPU成为性能瓶颈。  

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

【1】深度学习分布式训练框架 Horovod --- (1) 基础知识，https://www.cnblogs.com/rossiXYZ/p/14856464.html  
【2】手把手推导Ring All-reduce的数学性质，https://zhuanlan.zhihu.com/p/504957661  
【3】大模型涉及到的精度有多少种？FP32、TF32、FP16、BF16、FP8、FP4、NF4、INT8都有什么关联，一文讲清楚，https://zhuanlan.zhihu.com/p/673708074  
【4】十分钟速通优化器原理，通俗易懂（从SGD到AdamW），https://zhuanlan.zhihu.com/p/686410423  
【5】机器学习11种优化器推导过程详解(SGD,BGD,MBGD,Momentum,NAG,Adagrad,Adadelta,RMSprop,Adam,Nadma,Adamx)，https://blog.csdn.net/yangwohenmai1/article/details/124882119  
【6】【LLM101n】7：流行的LLM优化算法 - AdamW，https://zhuanlan.zhihu.com/p/7272881104  
【7】Huge and Efficient! 一文了解大规模预训练模型高效训练技术，https://aiorang.com/article/PqmOhWF.html  
【8】大模型精度（FP16，FP32，BF16）详解与实践，https://www.53ai.com/news/qianyanjishu/2024052494875.html  
【9】LLM 时代，如何优雅地训练大模型？，https://zhuanlan.zhihu.com/p/660394604  
【10】deepspeed 滴 ZERO 介绍，https://blog.csdn.net/weixin_42253689/article/details/147568376  
【11】图解大模型训练之：数据并行下篇( DeepSpeed ZeRO，零冗余优化)，https://zhuanlan.zhihu.com/p/618865052  

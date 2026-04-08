---
title: 大模型八股
tags:
  - interview
  - 八股
categories:
  - CS
  - interview
  - 八股
abbrlink: 1177fb52
date: 2006-04-02 22:28:28
hidden: false
---

# LoRA

## 初始化

LoRA的公式是这样的，输入先过A矩阵，后过B矩阵：  

$$W' = W + \frac{\alpha}{r} \cdot BA$$  

AB = 0可以保证初始时模型保持原性能。那么可以通过A=0，也可以通过B=0来实现。原文选择的是B=0来实现。为什么呢？  

先从方差讲起。设输入 $x_j$ 满足 $\text{Var}(x_j) = 1$，且 $A_{ij} \sim \mathcal{N}(0, \sigma_A^2)$ 独立于 $x$。  

计算 $z = Ax$**：  

$z_i = \sum_{j=1}^{d_{in}} A_{ij} x_j$  

先算单个乘积项的方差（乘法）：  

$$\text{Var}(A_{ij} x_j) = \mathbb{E}[A_{ij}^2]\mathbb{E}[x_j^2] = \sigma_A^2 \cdot 1 = \sigma_A^2$$  

再算求和后的方差（加法，各项独立）：  

$$\text{Var}(z_i) = \sum_{j=1}^{d_{in}} \text{Var}(A_{ij} x_j) = d_{in} \cdot \sigma_A^2$$  

那么这个时候方法变成了 $d_{in}$ 倍。  

再计算 $y = Bz$**：  

$y_k = \sum_{i=1}^{r} B_{ki} z_i$，其中 $B_{ki} \sim \mathcal{N}(0, \sigma_B^2)$ 独立于 $A$（从而独立于 $z$）。  

同样先算乘积项方差（乘法）：  

$$\text{Var}(B_{ki} z_i) = \sigma_B^2 \cdot \text{Var}(z_i) = \sigma_B^2 \cdot d_{in}\sigma_A^2$$  

再算求和方差（加法）：  

$$\text{Var}(y_k) = \sum_{i=1}^{r} \text{Var}(B_{ki} z_i) = r \cdot \sigma_B^2 \cdot d_{in}\sigma_A^2$$  

为了使整个计算过程，各层的方差都能稳定，不容易出现梯度爆炸或者消失的情况，方差就要保持恒定。  

为使输出方差 $\text{Var}(y_k) = \Theta(1)$（不随维度爆炸或消失），需要：  

$$r \sigma_B^2 d_{in} \sigma_A^2 = 1$$  

- 如果 $\sigma_A^2 = 1/d_{in}$（标准 He/Xavier），则必须 $\sigma_B^2 = 1/r$  
- 如果反过来让 $B$ 随机且用 $1/d_{out}$ 的小方差，则 $\text{Var}(y_k) = r/d_{out} \ll 1$，信号严重衰减；而$\sigma_B^2 = 1/r$又太大（为了补偿从低维 r 到高维输出的映射，必须用大方差），导致初始梯度方差过大，对学习率极其敏感，训练不稳定。  

| 候选方案                  | 随机侧所需方差                              | 对梯度的影响                                                                 |
| --------------------- | ------------------------------------ | ---------------------------------------------------------------------- |
| **方案 A**：$A$ 随机，$B=0$ | $\sigma_A^2 = 1/n$（极小，如 $1/4096$）    | $Ax$ 方差可控（$\Theta(1)$），梯度稳定                                            |
| **方案 B**：$B$ 随机，$A=0$ | $\sigma_B^2 = 1/r$（大 256 倍，如 $1/16$） | $B$ 方差大，梯度 $\frac{\partial \mathcal{L}}{\partial A} = B^T\delta$ 初始噪声强 |

## 系数


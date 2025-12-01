---
layout: post
title: "简单理解 RL 中的 KL 散度估计器：从数值估计到梯度估计"
date: 2025-12-01
description: "在强化学习中，KL 散度的估计方式直接影响训练稳定性。本文系统剖析三种经典估计器 k1、k2、k3 的性质差异，并给出用于 reward 惩罚与用于 loss 回传时的选型指南。"
categories: reinforcement-learning
lang: zh
---

* TOC
{:toc}

> 在强化学习中，KL 散度的估计方式直接影响训练稳定性。本文系统剖析三种经典估计器 $k_1, k_2, k_3$ 的性质差异，并给出「用于 reward 惩罚」与「用于 loss 回传」时的选型指南。

[English Version](https://xihuai18.github.io/reinforcement-learning/2025/12/01/kl-estimators-en.html) \| [知乎版本 ![Zhihu](https://static.zhihu.com/heifetz/favicon.ico)](https://zhuanlan.zhihu.com/p/1978993413425763764)

## 引言：KL 散度在强化学习中的角色

在策略优化（PPO、GRPO 等）或对齐训练（RLHF/RLAIF）中，**KL 惩罚**是约束新策略不偏离参考策略的核心手段，用以防止训练不稳定或策略崩溃。

### 正向 KL 与反向 KL 的区别

设 $q_\theta$ 为当前 actor 策略，$p$ 为参考策略，两种方向的 KL 散度分别为：

**反向 KL（Reverse KL）**：
$$
D_{\mathrm{KL}}(q_\theta \| p) = \mathbb{E}_{x \sim q_\theta}\left[\log \frac{q_\theta(x)}{p(x)}\right]
$$

<figure style="text-align:center;">
  <img src="/assets/img/kl-estimator-reverse.png" style="width:95%;max-width:100%;">
  <figcaption style="font-size:0.9em;color:gray;">图片来源：<a href="https://dibyaghosh.com/blog/probability/kldivergence/">Dibya Ghosh's Blog</a></figcaption>
</figure>

**正向 KL（Forward KL）**：
$$
D_{\mathrm{KL}}(p \| q_\theta) = \mathbb{E}_{x \sim p}\left[\log \frac{p(x)}{q_\theta(x)}\right]
$$

<figure style="text-align:center;">
  <img src="/assets/img/kl-estimator-forward.png" style="width:95%;max-width:100%;">
  <figcaption style="font-size:0.9em;color:gray;">图片来源：<a href="https://dibyaghosh.com/blog/probability/kldivergence/">Dibya Ghosh's Blog</a></figcaption>
</figure>

**直觉理解**：
- **反向 KL** 倾向于「模式寻优」（mode-seeking）——策略会集中在参考分布的高概率区域，可能牺牲多样性
- **正向 KL** 倾向于「质量覆盖」（mass-covering）——策略会尽量覆盖参考分布的支撑集

在 RLHF 的主流实现中，**反向 KL** 更为常见，因为我们希望 actor 不要偏离 reference policy 太远，而非要求完全覆盖所有模式。


## 三种估计器的定义与设计原理

设比值 $r(x) = \frac{p(x)}{q_\theta(x)}$，John Schulman 提出的三种单样本估计子定义如下：

### $k_1$：最朴素的估计器

$$
k_1(x) = -\log r = \log q_\theta(x) - \log p(x)
$$

这是最直接的定义——直接取 log-ratio 的负值。它对反向 KL 无偏，但有一个致命缺陷：**可能取负值**，而 KL 散度始终非负。这导致其方差极高，因为正负样本会相互抵消。

### $k_2$：基于 f-散度的低方差估计器

$$
k_2(x) = \frac{1}{2}(\log r)^2
$$

**设计动机**：$k_1$ 的问题在于可正可负，而 $k_2$ 通过取平方保证**每个样本都是正的**，直观上每个样本都在告诉你 $p$ 和 $q$ 相差多远。

**为什么偏差很小？** $k_2$ 本质上是一个 **f-散度**（f-divergence），其中 $f(x) = \frac{1}{2}(\log x)^2$。f-散度有一个优美的性质：**所有可微的 f-散度在 $q \approx p$ 时，二阶展开都形如**

$$
D_f(p, q_\theta) = \frac{f''(1)}{2} \theta^T F \theta + O(\theta^3)
$$

其中 $F$ 是 Fisher 信息矩阵。KL 散度对应 $f(x) = -\log x$，有 $f''(1) = 1$；而 $k_2$ 对应的 $f(x) = \frac{1}{2}(\log x)^2$，同样有 $f''(1) = 1$。这意味着**当策略接近时，$k_2$ 与真实 KL 的行为几乎一致**，偏差仅体现在高阶项。

### $k_3$：控制变量法构造的「最优」估计器

$$
k_3(x) = r - 1 - \log r
$$

**设计动机**：我们想要一个**既无偏又低方差**的估计器。标准做法是给 $k_1$ 加一个**控制变量**（control variate）——一个期望为零但与 $k_1$ 负相关的量。

注意到 $\mathbb{E}_q[r - 1] = \mathbb{E}_q\left[\frac{p}{q}\right] - 1 = 1 - 1 = 0$，所以对于任意 $\lambda$，

$$
k_1 + \lambda(r - 1) = -\log r + \lambda(r - 1)
$$

仍然是无偏估计。

**为什么选 $\lambda = 1$？** 由于 $\log$ 是凹函数，有 $\log x \leq x - 1$，因此

$$
k_3 = (r - 1) - \log r \geq 0
$$

**始终非负**！这保证了每个样本都在「正向」贡献信息，消除了 $k_1$ 正负抵消的问题。

**几何直觉**：$k_3$ 实际上是一个 **Bregman 散度**。考虑凸函数 $\phi(x) = -\log x$，它在 $x=1$ 处的切线为 $y = 1 - x$。Bregman 散度定义为「函数值与切线值之差」：

$$
D_\phi(r, 1) = \phi(r) - \phi(1) - \phi'(1)(r - 1) = -\log r - 0 - (-1)(r-1) = r - 1 - \log r = k_3
$$

由于凸函数始终位于其切线上方，这个差值**天然非负**。更重要的是，在 $r \to 1$ 时，函数与切线「贴合」得越来越紧，差值以 $(r-1)^2$ 的二阶速度趋近于零——这正是 $k_3$ 在策略接近时方差小的根本原因。


### 三者对比总结

| 估计器 | 定义                    | 设计原理                   |  对数值的偏差  | 方差特性       |
| :----: | :---------------------- | :------------------------- | :------------: | :------------- |
| $k_1$  | $-\log r$               | 最朴素定义                 |      无偏      | 高（可正可负） |
| $k_2$  | $\frac{1}{2}(\log r)^2$ | f-散度，二阶行为与 KL 一致 | 有偏（但极小） | 低（恒正）     |
| $k_3$  | $r - 1 - \log r$        | 控制变量 + Bregman 散度    |      无偏      | 低（恒正）     |

从数值估计的角度看，$k_3$ 是「无偏 + 低方差」的最优选择；但正如后文将分析的，**梯度层面的故事完全不同**。


## 核心分析

### 估计 KL 数值时的偏差与方差

假设从 $q_\theta$ 采样来估计反向 KL $D_{\mathrm{KL}}(q_\theta \| p)$：

**无偏性分析**：

$$
\mathbb{E}_{q}[k_1] = \mathbb{E}_{q}\left[\log \frac{q}{p}\right] = D_{\mathrm{KL}}(q \| p) \quad \textbf{（无偏）}
$$

$$
\mathbb{E}_{q}[k_3] = \mathbb{E}_{q}[r - 1 - \log r] = 1 - 1 + D_{\mathrm{KL}}(q \| p) = D_{\mathrm{KL}}(q \| p) \quad \textbf{（无偏）}
$$

$$
\mathbb{E}_{q}[k_2] = \frac{1}{2}\mathbb{E}_{q}[(\log r)^2] \neq D_{\mathrm{KL}}(q \| p) \quad \textbf{（有偏）}
$$

**结论**：对于估计反向 KL 的**数值**，$k_1$ 和 $k_3$ 是无偏估计，而 $k_2$ 是有偏的。

**方差特性的 Trade-off**：

John Schulman 的实验（$q = \mathcal{N}(0,1)$，$p = \mathcal{N}(0.1,1)$，真实 KL = 0.005）表明：

| 估计器 | bias/true | stdev/true |
| :----: | :-------: | :--------: |
| $k_1$  |     0     |     20     |
| $k_2$  |   0.002   |    1.42    |
| $k_3$  |     0     |    1.42    |

当 KL 较大时（$p = \mathcal{N}(1,1)$，真实 KL = 0.5）：

| 估计器 | bias/true | stdev/true |
| :----: | :-------: | :--------: |
| $k_1$  |     0     |     2      |
| $k_2$  |   0.25    |    1.73    |
| $k_3$  |     0     |    1.7     |

**核心直觉**：
- $k_1 = -\log r$ 以一阶项起步，当 $r$ 接近 1 时波动较大，且可能取负值
- $k_3 = r - 1 - \log r$ 在 $r=1$ 处是二阶小量，始终非负，因此在策略接近时方差更小
- 但当覆盖严重不足（$r$ 可能爆炸）时，$k_3$ 的方差会被权重爆炸拖累；此时 $k_1$ 反而更稳定

> **注**：若要估计**正向 KL 的数值** $D_{\mathrm{KL}}(p \| q) = \mathbb{E}_p[\log r]$，而只能从 $q$ 采样，可用重要性采样 $\mathbb{E}_q[r \log r]$。


### 估计 KL 梯度时的关键区分

**这是最容易混淆、也是实践中最关键的部分。**

#### 正向与反向 KL 真梯度的推导

在分析估计器之前，我们先推导正向和反向 KL 散度对 $\theta$ 的**真梯度**作为参照。

记 score function $s_\theta(x) = \nabla_\theta \log q_\theta(x)$，它有一个重要性质：$\mathbb{E}_{q_\theta}[s_\theta] = 0$（因为 $\int \nabla_\theta q_\theta dx = \nabla_\theta \int q_\theta dx = \nabla_\theta 1 = 0$）。

**反向 KL 的梯度**：

$$
D_{\mathrm{KL}}(q_\theta \| p) = \int q_\theta(x) \log \frac{q_\theta(x)}{p(x)} dx
$$

对 $\theta$ 求梯度（使用乘积法则）：

$$
\nabla_\theta D_{\mathrm{KL}}(q_\theta \| p) = \int \nabla_\theta q_\theta \cdot \log \frac{q_\theta}{p} dx + \int q_\theta \cdot \nabla_\theta \log \frac{q_\theta}{p} dx
$$

利用 $\nabla_\theta q_\theta = q_\theta \cdot s_\theta$ 以及 $\nabla_\theta \log q_\theta = s_\theta$、$\nabla_\theta \log p = 0$：

$$
= \mathbb{E}_q\left[s_\theta \cdot \log \frac{q_\theta}{p}\right] + \mathbb{E}_q[s_\theta] = \mathbb{E}_q\left[s_\theta \cdot \log \frac{q_\theta}{p}\right]
$$

即：

$$
\boxed{\nabla_\theta D_{\mathrm{KL}}(q_\theta \| p) = \mathbb{E}_q\left[s_\theta \cdot \log \frac{q_\theta}{p}\right] = -\mathbb{E}_q[s_\theta \cdot \log r]}
$$

**正向 KL 的梯度**：

$$
D_{\mathrm{KL}}(p \| q_\theta) = \int p(x) \log \frac{p(x)}{q_\theta(x)} dx
$$

由于 $p(x)$ 不依赖于 $\theta$：

$$
\nabla_\theta D_{\mathrm{KL}}(p \| q_\theta) = \int p(x) \cdot \nabla_\theta \left(-\log q_\theta(x)\right) dx = -\mathbb{E}_p[s_\theta]
$$

为了用 $q$ 的样本估计这个量，进行重要性采样：

$$
-\mathbb{E}_p[s_\theta] = -\mathbb{E}_q\left[\frac{p}{q_\theta} \cdot s_\theta\right] = -\mathbb{E}_q[r \cdot s_\theta]
$$

利用 $\mathbb{E}_q[s_\theta] = 0$，可改写为：

$$
\boxed{\nabla_\theta D_{\mathrm{KL}}(p \| q_\theta) = \mathbb{E}_q[(1-r) \cdot s_\theta]}
$$

有了这两个结果，我们就能判断各估计器的梯度期望究竟对应哪个 KL 的真梯度。

#### 两种求导顺序

在代码实现中，存在两条路径：

1. **先梯度、后期望**：对每个样本的 $k_i(x)$ 求梯度，再对梯度求期望（Monte Carlo 估计）
2. **先期望、后梯度**：把 $\mathbb{E}_q[k_i]$ 当作损失函数，对解析表达式求梯度

**在典型的深度学习代码中，我们实际执行的是「先梯度、后期望」**——自动微分对每个样本计算梯度，然后在 batch 上取平均。

#### 三种估计器的梯度推导

现在我们计算三种估计器的梯度，看它们的期望分别对应哪个 KL 的真梯度。

**推导 $\nabla_\theta k_1$**：

$$
k_1 = -\log r = -\log \frac{p(x)}{q_\theta(x)} = \log q_\theta(x) - \log p(x)
$$

$$
\nabla_\theta k_1 = \nabla_\theta \log q_\theta(x) - \nabla_\theta \log p(x) = s_\theta - 0 = s_\theta
$$

**推导 $\nabla_\theta k_2$**：

$$
k_2 = \frac{1}{2}(\log r)^2
$$

由链式法则：

$$
\nabla_\theta k_2 = (\log r) \cdot \nabla_\theta (\log r) = (\log r) \cdot \nabla_\theta \left(\log p(x) - \log q_\theta(x)\right) = (\log r) \cdot (-s_\theta) = -(\log r) \cdot s_\theta
$$

**推导 $\nabla_\theta k_3$**：

$$
k_3 = r - 1 - \log r
$$

首先计算 $\nabla_\theta r$。由于 $r = p(x) \cdot q_\theta(x)^{-1}$：

$$
\nabla_\theta r = p(x) \cdot (-1) \cdot q_\theta(x)^{-2} \cdot \nabla_\theta q_\theta(x) = -\frac{p(x)}{q_\theta(x)} \cdot \frac{\nabla_\theta q_\theta(x)}{q_\theta(x)} = -r \cdot s_\theta
$$

再计算 $\nabla_\theta \log r$：

$$
\nabla_\theta \log r = \frac{1}{r} \nabla_\theta r = \frac{1}{r} \cdot (-r \cdot s_\theta) = -s_\theta
$$

因此：

$$
\nabla_\theta k_3 = \nabla_\theta r - 0 - \nabla_\theta \log r = -r \cdot s_\theta - (-s_\theta) = (1 - r) \cdot s_\theta
$$


对它们在 $q_\theta$ 下取期望：

| 估计器 | $\mathbb{E}_{q}[\nabla_\theta k_i]$                                                | 等价于               |
| :----: | :--------------------------------------------------------------------------------- | :------------------- |
| $k_1$  | $\mathbb{E}_{q}[s_\theta] = 0$                                                     | **无意义（恒为零）** |
| $k_2$  | $-\mathbb{E}_{q}[(\log r) \cdot s_\theta] = \nabla_\theta D_{\mathrm{KL}}(q \| p)$ | **反向 KL 的真梯度** |
| $k_3$  | $\mathbb{E}_{q}[(1-r) \cdot s_\theta] = \nabla_\theta D_{\mathrm{KL}}(p \| q)$     | **正向 KL 的真梯度** |

**关键洞察**：
- **$k_2$ 的梯度**等价于反向 KL 的真梯度——这是优化「约束策略不偏离 ref」的正确选择
- **$k_3$ 的梯度**等价于正向 KL 的真梯度——这对应「覆盖型」目标
- **$k_1$ 的梯度期望恒为零**——作为 loss 反传毫无意义！

#### 「先期望后梯度」vs「先梯度后期望」

如果从解析角度把 $\mathbb{E}_q[k_i]$ 当作一个关于 $\theta$ 的函数再求梯度（即「先期望后梯度」），那么：

$$
\nabla_\theta \mathbb{E}_q[k_1] = \nabla_\theta D_{\mathrm{KL}}(q \| p)
$$

$$
\nabla_\theta \mathbb{E}_q[k_3] = \nabla_\theta D_{\mathrm{KL}}(q \| p)
$$

两者都给出反向 KL 的梯度。但在代码中直接对 $k_3$ 的样本均值调用反传时，自动微分执行的是「先梯度后期望」，得到的是 $\mathbb{E}_q[\nabla_\theta k_3]$，即**正向 KL 的梯度**。

这个区分非常重要：**同一个估计器，两种求导顺序可能给出完全不同的结果**。


## RL 实践指南

### KL 作为 Reward 惩罚（不需要梯度）

当 KL 仅作为标量惩罚加入 reward shaping 时，我们只需要准确的**数值估计**，不需要反传梯度。

**推荐**：
- 使用 **$k_1$** 或 **$k_3$**（两者对反向 KL 数值均无偏）
- 当策略已接近参考策略时，$k_3$ 往往更低方差
- 覆盖不足或尾部错配明显时，$k_1$ 更稳健

> **注**：若想施加**正向 KL 惩罚**（偏向覆盖行为分布），数值上可用 $\mathbb{E}_q[r \log r]$ 或（若可从 $p$ 采样）$\mathbb{E}_p[\log r]$。

### KL 作为 Loss（需要梯度回传）

当 KL 作为 loss 的一部分参与反传时，必须考虑梯度的正确性。

#### 优化反向 KL（最常见场景）

目标：控制 actor 不偏离 reference policy。

**正确做法**：使用 **$k_2$** 作为 loss。

$$
\mathcal{L}_{k_2} = \frac{1}{2}(\log r)^2
$$

其梯度期望 $\mathbb{E}_q[\nabla k_2] = \nabla_\theta D_{\mathrm{KL}}(q \| p)$ 正是反向 KL 的真梯度。

#### 优化正向 KL（覆盖型场景）

目标：让策略覆盖参考分布的支撑集（如离线 RL、模仿学习等）。

**正确做法**：使用 **$k_3$** 作为 loss。

$$
\mathbb{E}_q[\nabla k_3] = \mathbb{E}_q[(1-r) \cdot s_\theta] = \nabla_\theta D_{\mathrm{KL}}(p \| q)
$$

直接对 $k_3$ 的样本均值调用反传，自动微分计算的就是 $\mathbb{E}_q[\nabla_\theta k_3]$，即正向 KL 的梯度，无需额外处理。


## 一份「拿来就用」的对照表

| 目标                            | 采样来源 | 用于**数值**            | 用于**梯度** |
| :------------------------------ | :------: | :---------------------- | :----------- |
| 反向 KL $D_{\mathrm{KL}}(q\|p)$ |   $q$    | $k_1$ 或 $k_3$（无偏）  | $k_2$        |
| 正向 KL $D_{\mathrm{KL}}(p\|q)$ |   $q$    | $\mathbb{E}_q[r\log r]$ | $k_3$        |


## 常见实现陷阱

**陷阱 1：把 $k_1$ 直接当 loss 反传**

$k_1$ 的梯度期望恒为零（$\mathbb{E}_q[\nabla k_1] = \mathbb{E}_q[s_\theta] = 0$），作为 loss 完全无效。

> **解决**：reward shaping 用 $k_1$ 或 $k_3$（不需要梯度），loss 用 $k_2$ 或 $k_3$。

**陷阱 2：混淆 $k_3$ 的「数值无偏性」与「梯度对应的目标」**

$k_3$ 对**反向 KL 的数值**是无偏估计，但它的**梯度**对应的是**正向 KL**。如果你的目标是优化反向 KL，却用 $k_3$ 作为 loss，实际上在优化正向 KL。

> **解决**：明确你的优化目标。优化反向 KL 用 $k_2$；优化正向 KL 才用 $k_3$。

**陷阱 3：$r$ 重尾导致方差爆炸**

当策略与参考分布差异过大时，$r = p/q$ 可能出现极端值，导致 $k_3$ 的方差爆炸。



## 总结

**一句话记忆**：

- **只要数值（KL 作为 reward 惩罚）**：选 $k_1$ 或 $k_3$（均对反向 KL 无偏）
- **需要梯度（KL 作为 loss）**：
  - 优化**反向 KL** → 用 $k_2$
  - 优化**正向 KL** → 用 $k_3$

把「**从谁采样**」、「**估计谁的值**」、「**对谁求梯度**」这三个问题捋清楚，三种估计器就不再让人混淆了。


## 参考文献

1. Dibya Ghosh. "KL Divergence for Machine Learning". https://dibyaghosh.com/blog/probability/kldivergence

2. John Schulman. "Approximating KL Divergence". https://joschu.net/blog/kl-approx.html

3. Verl Documentation. "Proximal Policy Optimization (PPO)". https://verl.readthedocs.io/en/latest/algo/ppo.html

4. 初七123334. RLHF/RLVR 训练中的 KL 近似方法浅析（k1 / k2 / k3）. https://zhuanlan.zhihu.com/p/1966872846212010437

5. Kezhao Liu, Jason Klein Liu, Mingtao Chen, Yiming Liu. "Rethinking KL Regularization in RLHF: From Value Estimation to Gradient Optimization". https://arxiv.org/abs/2510.01555

```bibtex
@misc{WangZhang2025KLEstimators,
  author       = {Wang, Xihuai and Zhang, Shao},
  title        = {Understanding {KL} Divergence Estimators in {RL}: From Value Approximation to Gradient Estimation},
  year         = {2025},
  month        = dec,
  day          = {01},
  url          = {https://xihuai18.github.io/reinforcement-learning/2025/12/01/kl-estimators-en.html}
}
```
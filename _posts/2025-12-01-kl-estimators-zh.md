---
layout: post
title: "RL 中的 KL 估计器选型：从数值无偏到梯度正确"
date: 2025-12-01
description: "在强化学习中，KL 估计器不能只看数值估得准不准，还要看它在 loss 或 reward 写法下究竟优化了谁。本文比较 k1、k2、k3 在 on-policy 与 off-policy 场景中的差异，并给出可直接落地的选型建议。"
og_image: /assets/img/kl-estimators/kl-estimator.png
categories: reinforcement-learning
lang: zh
en_url: /reinforcement-learning/2025/12/01/kl-estimators-en.html
zhihu_url: https://zhuanlan.zhihu.com/p/1978993413425763764
wechat_url: https://mp.weixin.qq.com/s/VD_NBty5na4PfAa7wLoGAw
---

![Mini-class](/assets/img/kl-estimators/kl-estimator.png){: style="display:block;margin:0 auto;width:95%;max-width:100%;" }

> 在强化学习中，KL 估计器不能只看数值估得准不准，还要看它在 loss 或 reward 写法下究竟优化了谁。本文比较三种经典估计器 $k_1, k_2, k_3$ 在 on-policy 和 off-policy 场景下的性质差异，并说明当 KL 作为可微损失项或 detached 的奖励塑形项时，选型结论会怎样变化。

## 1. 引言：KL 散度（Kullback-Leibler 散度）在强化学习中的角色

这篇文章只想回答一个实现层面的问题：同样都写着“加 KL 惩罚”，为什么换一个估计器、换一个采样分布，或者只多一个 `.detach()`，优化目标就可能已经变了？在策略优化算法（如 PPO、GRPO）或对齐训练框架（如 RLHF、RLAIF）中，KL 惩罚项看起来只是个稳定训练的正则项；但一旦落实到代码，实现方式就会分出好几个维度：**估计器的选择**（$k_1$、$k_2$、$k_3$）、**采样分布的选择**（on-policy 或 off-policy），以及**KL 项的用法**（作为损失项直接反传，还是作为奖励塑形项）。

要把这个问题讲清楚，得分开两件事：一是“估计某个 KL 的数值”，二是“反向传播时实际在优化哪个目标”。很多实现恰恰是在这两件事上把概念混在了一起。

### 1.1 正向 KL 与反向 KL 的区别

设 $q_\theta$ 为当前 actor 策略，$p$ 为参考策略，两种方向的 KL 散度分别为：

**反向 KL（Reverse KL）**：

$$
D_{\mathrm{KL}}(q_\theta \| p) = \mathbb{E}_{x \sim q_\theta}\left[\log \frac{q_\theta(x)}{p(x)}\right]
$$

<figure style="text-align:center;" markdown="0">
  <img src="/assets/img/kl-estimators/kl-estimator-reverse.png" style="width:80%;max-width:100%;">
  <figcaption style="font-size:0.9em;color:gray;">图片来源：<a href="https://dibyaghosh.com/blog/probability/kldivergence/">Dibya Ghosh's Blog</a></figcaption>
</figure>

**正向 KL（Forward KL）**：

$$
D_{\mathrm{KL}}(p \| q_\theta) = \mathbb{E}_{x \sim p}\left[\log \frac{p(x)}{q_\theta(x)}\right]
$$

<figure style="text-align:center;" markdown="0">
  <img src="/assets/img/kl-estimators/kl-estimator-forward.png" style="width:80%;max-width:100%;">
  <figcaption style="font-size:0.9em;color:gray;">图片来源：<a href="https://dibyaghosh.com/blog/probability/kldivergence/">Dibya Ghosh's Blog</a></figcaption>
</figure>

**直观解释**：

- **反向 KL** 是 mode-seeking 的：优化后的策略会集中到参考分布的高概率区域，多样性可能降低。
- **正向 KL** 是 mass-covering 的：策略会尽量铺满参考分布的整个支撑集。

在 RLHF 的主流实现中，**反向 KL**更为常用，其原因在于我们通常期望 actor 策略不过度偏离参考策略，而非要求其完全覆盖参考分布的所有模式。

### 1.2 三个会改变结论的选择

可以把实现 KL 惩罚时的分歧理解成三个选择：样本从谁来、约束哪个方向的 KL、KL 项是直接反传还是只做 reward 的系数。只要切换其中任意一个，推荐估计器就可能变化。

1. **采样来源**：样本应来自当前策略 $q_\theta$（同策略 on-policy），还是来自行为策略 $\mu$（异策略 off-policy）？
2. **估计目标**：需要估计的是反向 KL 散度 $D_{\mathrm{KL}}(q_\theta \| p)$，还是正向 KL 散度 $D_{\mathrm{KL}}(p \| q_\theta)$？
3. **应用方式**：KL 项应作为损失函数的一部分参与反向传播，还是作为奖励塑形项（应用 stop-gradient 操作）？

**本文范围**：本文主要讨论 token/sample 级 KL 项及其在 policy-gradient 主项中的行为。Critic、GAE、baseline 归一化，以及一般多步 MDP 的严格 off-policy 修正，只在相关处简要说明，不做系统展开。

与主要讨论 KL 数值近似的经典笔记不同，本文更关心近期 LLM-RL 文献反复提到的那个问题：同一个估计器一旦从 reward 系数改成可微损失项，梯度到底还在不在优化你以为的目标。

> **先看结论（只针对本文讨论的 token/sample 级 KL 项）**
>
> - 若目标是反向 KL，且把 KL 写成可微损失项：on-policy 的朴素写法直接用 $k_2$ 最省心；若显式构造 $\rho$，则推荐 $\rho k_3$ 或 $\mathrm{sg}(\rho)k_2$。
> - 若把 KL 写成 stop-grad reward shaping：就策略梯度主项而言，只有 $k_1$ 保持与反向 KL 正则一致的无偏梯度。
> - 其余常见配置的问题通常不是“数值偏差稍大”，而是梯度目标已经变了。

## 2. 准备工作：符号与基本概念

在进入正文前，先统一符号，并把后文反复要用的两个基础结论写清楚。

### 2.1 符号、采样分布与解析梯度

**符号约定**

- $q_\theta$：当前 actor 策略（参数为 $\theta$）
- $q$：若无歧义，后文简写 $q := q_\theta$
- $p$：参考策略（reference policy），不依赖于 $\theta$
- $\mu$：行为策略（behavior policy），用于 off-policy 采样，不依赖于 $\theta$
- $s_\theta(x) = \nabla_\theta \log q_\theta(x)$：score function（得分函数）
- $\text{sg}(\cdot)$：stop-gradient 操作（在代码中对应 `.detach()`）

#### 统一的采样策略视角：引入 $\rho$ 记号

分析 KL 估计器的梯度性质时，on-policy 和 off-policy 看起来需要分开处理，但其实可以放进同一个框架。

为此，我们引入**采样策略** $\mu$，即数据来源于分布 $x \sim \mu$，并定义**统一的重要性权重比率**：

$$
\rho(x) := \frac{q_\theta(x)}{\text{sg}(\mu(x))}
$$

此定义的关键在于：**无论同策略还是异策略场景，我们都将采样策略 $\mu$ 视为梯度常数**（即对 $\mu$ 应用stop-gradient操作）。

- **异策略（Off-policy）场景**（$\mu \neq q_\theta$）：由于 $\mu$ 本身不依赖于 $\theta$，故 $\text{sg}(\mu) = \mu$，此时 $\rho = \frac{q_\theta}{\mu}$。
- **同策略（On-policy）场景**（$\mu = q_\theta$）：令 $\mu = q_\theta$ 但对其应用stop-gradient操作，则 $\rho = \frac{q_\theta}{\text{sg}(q_\theta)} \equiv 1$（数值恒为1），但 $\nabla_\theta \rho = s_\theta \neq 0$。

在同策略情况下，尽管数值上 $\rho\equiv 1$，但仍要在计算图中显式构造 $\rho=\frac{q_\theta}{\text{sg}(q_\theta)}$（或等价写成 $\rho=\exp(\log q_\theta-\text{sg}(\log q_\theta))$）。如果直接把 $\rho$ 写成常数 1，score function 这条梯度路径就会丢掉，推导也会退化成后文说的“朴素同策略实现”。

$\rho$ 补上的，正是“采样分布对参数 $\theta$ 的依赖”这条梯度路径。On-policy 时，“先取期望后求梯度”和“先求梯度后取期望”之所以不一样，就是因为少了这条路径；显式构造 $\rho$，就是把它补回来。

有了这个记号，后面的推导就不需要再分 on-policy 和 off-policy 两套写法了。

#### 得分函数与KL散度的解析梯度

得分函数具有一个重要性质：$\mathbb{E}_{q_\theta}[s_\theta] = 0$（由 $\int \nabla_\theta q_\theta dx = \nabla_\theta \int q_\theta dx = \nabla_\theta 1 = 0$ 可得）。

基于这一性质，我们可以推导正向与反向KL散度关于参数 $\theta$ 的**解析梯度**。

**反向 KL 的梯度**：

$$
D_{\mathrm{KL}}(q_\theta \| p) = \int q_\theta(x) \log \frac{q_\theta(x)}{p(x)} dx
$$

对 $\theta$ 求梯度（应用乘积法则）：

$$
\nabla_\theta D_{\mathrm{KL}}(q_\theta \| p) = \int \nabla_\theta q_\theta \cdot \log \frac{q_\theta}{p} dx + \int q_\theta \cdot \nabla_\theta \log \frac{q_\theta}{p} dx
$$

利用 $\nabla_\theta q_\theta = q_\theta \cdot s_\theta$，以及 $\nabla_\theta \log q_\theta = s_\theta$、$\nabla_\theta \log p = 0$：

$$
= \mathbb{E}_{q_\theta}\left[s_\theta \cdot \log \frac{q_\theta}{p}\right] + \int q_\theta(x) \cdot s_\theta(x)\, dx \\
= \mathbb{E}_{q_\theta}\left[s_\theta \cdot \log \frac{q_\theta}{p}\right] + \mathbb{E}_{q_\theta}[s_\theta] \\
= \mathbb{E}_{q_\theta}\left[s_\theta \cdot \log \frac{q_\theta}{p}\right]
$$

即：

$$
\boxed{\nabla_\theta D_{\mathrm{KL}}(q_\theta \| p) = \mathbb{E}_{q_\theta}\left[s_\theta \cdot \log \frac{q_\theta}{p}\right] = -\mathbb{E}_{q_\theta}\left[s_\theta \cdot \log \frac{p}{q_\theta}\right]}
$$

> **注**：后文将定义 $k_1 := -\log\frac{p}{q_\theta}$，因此上式可简写为 $\nabla_\theta D_{\mathrm{KL}}(q_\theta \| p) = \mathbb{E}_{q_\theta}[s_\theta \cdot k_1]$，这一形式将在后续梯度分析中反复出现。

**正向KL散度的梯度**：

$$
D_{\mathrm{KL}}(p \| q_\theta) = \int p(x) \log \frac{p(x)}{q_\theta(x)} dx
$$

由于 $p(x)$ 不依赖于参数 $\theta$：

$$
\nabla_\theta D_{\mathrm{KL}}(p \| q_\theta) = \int p(x) \cdot \nabla_\theta \left(-\log q_\theta(x)\right) dx = -\mathbb{E}_p[s_\theta]
$$

为使用来自 $q$ 的样本估计该梯度，我们引入重要性采样技术：

$$
-\mathbb{E}_p[s_\theta] = -\mathbb{E}_{q_\theta}\left[\frac{p}{q_\theta} \cdot s_\theta\right]
= \mathbb{E}_{q_\theta}\left[\left(1-\frac{p}{q_\theta}\right) \cdot s_\theta\right]
$$

其中最后一步用到了 $\mathbb{E}_{q_\theta}[s_\theta]=0$。因此

$$
\boxed{\nabla_\theta D_{\mathrm{KL}}(p \| q_\theta) = \mathbb{E}_{q_\theta}\left[\left(1-\frac{p}{q_\theta}\right) \cdot s_\theta\right]}
$$

> **注**：后文将推导 $\nabla_\theta k_3 = (1-\frac{p}{q_\theta}) s_\theta$，因此 $\mathbb{E}_{q_\theta}[\nabla_\theta k_3] = \nabla_\theta D_{\mathrm{KL}}(p \| q_\theta)$（正向KL散度）——这解释了为何直接对 $k_3$ 进行反向传播会产生「错误」的梯度方向。

基于这两个结果，我们能够在后续分析中判断各估计器的梯度期望分别对应何种KL散度的解析梯度。

## 3. 三种估计器的定义与设计原理

基于概率比值 $\frac{p(x)}{q_\theta(x)}$，John Schulman 在一篇经典笔记里系统比较了三种单样本 KL 估计器。本节先把它们的定义和设计动机放在一起看。

### 3.1 三种估计器：定义与直观解释

**$k_1$：最朴素的 log-ratio 估计器**

$$
k_1(x) = -\log \frac{p(x)}{q_\theta(x)} = \log q_\theta(x) - \log p(x)
$$

这是最直观的定义——直接取对数比值的负值。它对反向 KL 是无偏的，但主要问题不是目标错了，而是**单样本值可正可负**；当真实 KL 很小时，样本仍可能大幅正负摆动，因此相对方差往往很高。

**$k_2$：基于平方 log-ratio 的估计器**

$$
k_2(x) = \frac{1}{2}\left(\log \frac{p(x)}{q_\theta(x)}\right)^2
$$

**设计动机**：$k_1$ 的估计值可正可负，而 $k_2$ 通过取平方确保**每个样本的估计值都非负**，从而每个样本都能直观地衡量 $p$ 和 $q$ 之间的差异程度。

**为什么偏差通常不大？** 更准确地说，$\mathbb{E}_{q_\theta}[k_2]$ 不是标准 KL 本身，但在 $q_\theta \approx p$ 的邻域里，它与反向 KL 共享相同的二阶局部展开，因此可以把它理解为一个局部有效的 surrogate；一旦离开小-KL 邻域，这个近似就不一定仍然可靠。

<details>
<summary>技术注记：$k_2$ 为什么与反向 KL 具有相同的二阶局部行为？</summary>

若取 $\theta_0$ 使得 $q_{\theta_0}=p$，并在标准正则性条件下（以保证积分与微分可交换）对参数做小扰动 $\Delta\theta$，则有

$$
\mathbb{E}_{q_{\theta_0+\Delta\theta}}[k_2]
= \frac{1}{2}\, \Delta\theta^T F(\theta_0)\, \Delta\theta + O(\|\Delta\theta\|^3),
$$

同时

$$
D_{\mathrm{KL}}\big(q_{\theta_0+\Delta\theta} \| p\big)
= \frac{1}{2}\, \Delta\theta^T F(\theta_0)\, \Delta\theta + O(\|\Delta\theta\|^3),
$$

其中 $F(\theta_0)$ 是 $\theta_0$ 处的 Fisher 信息矩阵。

</details>

**$k_3$：控制变量法构造的 Bregman 散度估计器**

$$
k_3(x) = \frac{p(x)}{q_\theta(x)} - 1 - \log \frac{p(x)}{q_\theta(x)}
$$

**设计动机**：我们希望得到一个**既无偏又低方差**的估计器。标准做法是为 $k_1$ 添加一个**控制变量**（control variate）——即期望为零但与 $k_1$ 负相关的量。

注意到 $\mathbb{E}_{q_\theta}\left[\frac{p}{q_\theta} - 1\right] = \mathbb{E}_{q_\theta}\left[\frac{p}{q_\theta}\right] - 1 = 1 - 1 = 0$，因此对于任意 $\lambda$，

$$
k_1 + \lambda\left(\frac{p}{q_\theta} - 1\right) = -\log \frac{p}{q_\theta} + \lambda\left(\frac{p}{q_\theta} - 1\right)
$$

仍然是无偏估计。

**为什么选择 $\lambda = 1$？** 由于 $\log$ 是凹函数，有 $\log x \leq x - 1$，因此

$$
k_3 = \left(\frac{p}{q_\theta} - 1\right) - \log \frac{p}{q_\theta} \geq 0
$$

**始终非负**！这保证了每个样本都「正向」贡献信息，避免了 $k_1$ 中正负估计值相互抵消的问题。

**几何视角**：$k_3$ 实际上是一个 **Bregman 散度**。考虑凸函数 $\phi(x) = -\log x$，它在 $x=1$ 处的切线为 $y = 1 - x$。Bregman 散度定义为函数值与切线值之差：

$$
\begin{aligned}
D_\phi\left(\frac{p}{q_\theta}, 1\right) &= \phi\left(\frac{p}{q_\theta}\right) - \phi(1) - \phi'(1)\left(\frac{p}{q_\theta} - 1\right) \\
&= -\log \frac{p}{q_\theta} - 0 - (-1)\left(\frac{p}{q_\theta} - 1\right) \\
&= \frac{p}{q_\theta} - 1 - \log \frac{p}{q_\theta} \\
&= k_3.
\end{aligned}
$$

由于凸函数始终位于其切线上方，该差值**自然非负**。更重要的是，当 $\frac{p}{q_\theta} \to 1$ 时，函数与切线「贴合」得越来越紧密，差值以 $\left(\frac{p}{q_\theta} - 1\right)^2$ 的二阶速度趋近于零——这正是 $k_3$ 在策略接近时方差较小的根本原因。

三者的设计逻辑对比如下：

| 估计器 |                        定义                         |              设计原理              |
| :----: | :-------------------------------------------------: | :--------------------------------: |
| $k_1$  |             $-\log \frac{p}{q_\theta}$              |             最朴素定义             |
| $k_2$  | $\frac{1}{2}\left(\log \frac{p}{q_\theta}\right)^2$ | 局部二阶行为与 KL 一致的 surrogate |
| $k_3$  | $\frac{p}{q_\theta} - 1 - \log \frac{p}{q_\theta}$  |      控制变量 + Bregman 散度       |

## 4. 数值估计：偏差与方差

假设从 $q_\theta$ 采样来估计反向 KL $D_{\mathrm{KL}}(q_\theta \| p)$：

### 4.1 无偏性分析

$$
\begin{aligned}
\mathbb{E}_{q_\theta}[k_1] &= \mathbb{E}_{q_\theta}\left[\log \frac{q_\theta}{p}\right] = D_{\mathrm{KL}}(q_\theta \| p) && \textbf{（无偏）} \\[8pt]
\mathbb{E}_{q_\theta}[k_3] &= \mathbb{E}_{q_\theta}\left[\frac{p}{q_\theta} - 1 - \log \frac{p}{q_\theta}\right] && \\
&= 1 - 1 + D_{\mathrm{KL}}(q_\theta \| p) && \\
&= D_{\mathrm{KL}}(q_\theta \| p) && \textbf{（无偏）} \\[8pt]
\mathbb{E}_{q_\theta}[k_2] &= \frac{1}{2}\mathbb{E}_{q_\theta}\left[\left(\log \frac{p}{q_\theta}\right)^2\right] \neq D_{\mathrm{KL}}(q_\theta \| p) && \textbf{（有偏）}
\end{aligned}
$$

在估计反向 KL 的数值时，$k_1$ 和 $k_3$ 是无偏估计，而 $k_2$ 是有偏估计。

### 4.2 方差特性分析

John Schulman 的实验（$q = \mathcal{N}(0,1)$，$p = \mathcal{N}(0.1,1)$，真实 KL = 0.005）显示：

| 估计器 | 偏差/真值 | 标准差/真值 |
| :----: | :-------: | :---------: |
| $k_1$  |     0     |     20      |
| $k_2$  |   0.002   |    1.42     |
| $k_3$  |     0     |    1.42     |

当 KL 较大时（$p = \mathcal{N}(1,1)$，真实 KL = 0.5）：

| 估计器 | 偏差/真值 | 标准差/真值 |
| :----: | :-------: | :---------: |
| $k_1$  |     0     |      2      |
| $k_2$  |   0.25    |    1.73     |
| $k_3$  |     0     |     1.7     |

直觉上看：

- $k_1 = -\log \frac{p}{q_\theta}$ 以一阶项起步，当 $\frac{p}{q_\theta}$ 接近 1 时波动较大，且可能取负值
- $k_3 = \frac{p}{q_\theta} - 1 - \log \frac{p}{q_\theta}$ 在 $\frac{p}{q_\theta}=1$ 处是二阶小量，始终非负，因此在策略接近时方差较小
- 但当离开小-KL、覆盖良好的局部区域后（$\frac{p}{q_\theta}$ 可能极大），$k_3$ 的方差也会因权重爆炸而迅速上升；这时 $k_1$ 和 $k_3$ 的优劣不能再简单一刀切

从纯数值估计的角度看，可以先记下面这张表：

| 估计器 |        对数值的偏差        |    方差特性    |
| :----: | :------------------------: | :------------: |
| $k_1$  |            无偏            | 高（可正可负） |
| $k_2$  | 有偏（小 KL 邻域通常较小） |   低（恒正）   |
| $k_3$  |            无偏            |   低（恒正）   |

如果只看 KL 数值本身，那么在 **小 KL、覆盖良好** 的常见局部场景下，$k_3$ 往往是更稳妥的选择。

> **注**：若要估计**正向 KL 的数值** $D_{\mathrm{KL}}(p \| q_\theta) = \mathbb{E}_p\left[\log \frac{p}{q_\theta}\right]$，且只能从 $q_\theta$ 采样，则可以使用重要性采样 $\mathbb{E}_{q_\theta}\left[\frac{p}{q_\theta} \log \frac{p}{q_\theta}\right]$。

## 5. KL 惩罚的两种使用方式

接下来真正分出岔路的，是 **KL 惩罚在实现里到底怎么用**。这一步决定了我们是只关心估计器的数值性质，还是还得把梯度性质一并算进去。

回顾 KL 正则化强化学习的目标函数（下式中 $\tau\sim q_\theta$ 表示“由策略 $q_\theta$ 诱导的轨迹分布”）：

$$
J(\theta) = \mathbb{E}_{\tau \sim q_\theta} \left[ \sum_{t=0}^T \gamma^t r(s_t, a_t) \right] - \beta \cdot D_{\mathrm{KL}}(q_\theta \| p)
$$

这个数学形式看似统一，但在基于策略梯度（Policy Gradient）的算法（如 PPO）中实现时，却衍生出两种截然不同的实现范式——它们在代码层面可能只差几行，却对应着完全不同的优化语义。

> **符号说明**：本节用 $\text{KL}_t$ 或 $\text{KL}(s)$ 泛指某个 token/state 级的 KL 估计器（如 $k_1, k_2, k_3$），具体定义见前文「三种估计器的定义与设计原理」一节。

### 5.1 作为损失项：KL 直接反传

```python
actor_loss = -advantage * log_prob + beta * kl  # kl 参与梯度计算
```

Critic 仅学习环境价值，KL 作为 actor 的正则项直接参与 loss 的反向传播。

### 5.2 作为奖励塑形项：KL 只改 reward，不直接反传

```python
kl = compute_kl(log_prob_q, log_prob_p).detach()
shaped_reward = reward - beta * kl
```

KL 被视为环境奖励的一部分，使用形塑后的奖励进行标准的 actor-critic 更新。KL 项本身不参与 loss 的反向传播。

很多实现里，这两种写法在代码上只差一个 `.detach()`；但从优化语义看，它们并不是同一种算法。先把差别说清楚：

- **KL 作为损失项**：需要 KL 估计器的正确显式梯度，关心梯度对应哪个优化目标
- **KL 作为奖励塑形项**：需要 KL 的准确数值估计，同时还要关注它诱导的策略梯度是否正确

## 6. 作为损失项时的梯度分析

当 KL 散度作为损失函数参与反向传播时，不同估计器对应的优化目标并不相同。这也是实践里最容易踩坑的地方。

下面沿用前文的统一框架，把 on-policy 与 off-policy 放进同一套推导里。回顾比率定义：

$$
\rho(x) := \frac{q_\theta(x)}{\text{sg}(\mu(x))}
$$

其中 $\mu$ 为采样策略。在此框架下：

- **On-policy**（$\mu = q_\theta$）：$\rho \equiv 1$，但 $\nabla_\theta \rho = s_\theta$
- **Off-policy**（$\mu \neq q_\theta$）：$\rho = \frac{q_\theta}{\mu}$，且 $\nabla_\theta \rho = \rho \cdot s_\theta$

### 6.1 三种估计器的基本梯度

首先计算三种估计器本身的梯度（不含 $\rho$），这些结果将在后续分析中反复使用。

**推导 $\nabla_\theta k_1$**：

$$
k_1 = -\log \frac{p(x)}{q_\theta(x)} = \log q_\theta(x) - \log p(x)
$$

$$
\nabla_\theta k_1 = \nabla_\theta \log q_\theta(x) - \nabla_\theta \log p(x) = s_\theta - 0 = s_\theta
$$

**推导 $\nabla_\theta k_2$**：

$$
k_2 = \frac{1}{2}\left(\log \frac{p}{q_\theta}\right)^2
$$

由链式法则：

$$
\begin{aligned}
\nabla_\theta k_2
&= \left(\log \frac{p}{q_\theta}\right) \cdot \nabla_\theta\left(\log \frac{p}{q_\theta}\right) \\
&= \left(\log \frac{p}{q_\theta}\right) \cdot \nabla_\theta(\log p(x) - \log q_\theta(x)) \\
&= \left(\log \frac{p}{q_\theta}\right)(-s_\theta) \\
&= - \left(\log \frac{p}{q_\theta}\right) s_\theta.
\end{aligned}
$$

**推导 $\nabla_\theta k_3$**：

$$
k_3 = \frac{p}{q_\theta} - 1 - \log \frac{p}{q_\theta}
$$

首先计算 $\nabla_\theta \frac{p}{q_\theta}$。由于 $\frac{p}{q_\theta} = p(x) \cdot q_\theta(x)^{-1}$：

$$
\nabla_\theta \frac{p}{q_\theta} = p(x) \cdot (-1) \cdot q_\theta(x)^{-2} \cdot \nabla_\theta q_\theta(x) = -\frac{p(x)}{q_\theta(x)} \cdot \frac{\nabla_\theta q_\theta(x)}{q_\theta(x)} = -\frac{p}{q_\theta} \cdot s_\theta
$$

再计算 $\nabla_\theta \log \frac{p}{q_\theta}$：

$$
\nabla_\theta \log \frac{p}{q_\theta} = \frac{q_\theta}{p} \nabla_\theta \frac{p}{q_\theta} = \frac{q_\theta}{p} \cdot \left(-\frac{p}{q_\theta} \cdot s_\theta\right) = -s_\theta
$$

因此：

$$
\nabla_\theta k_3 = \nabla_\theta \frac{p}{q_\theta} - 0 - \nabla_\theta \log \frac{p}{q_\theta} = -\frac{p}{q_\theta} \cdot s_\theta - (-s_\theta) = \left(1 - \frac{p}{q_\theta}\right) \cdot s_\theta
$$

三种估计器的梯度分别为：

- $\nabla_\theta k_1 = s_\theta$
- $\nabla_\theta k_2 = -\left(\log \frac{p}{q_\theta}\right) s_\theta = k_1 \cdot s_\theta$
- $\nabla_\theta k_3 = \left(1 - \frac{p}{q_\theta}\right) s_\theta$

这些基本梯度将在后续的统一框架分析中反复使用。

#### 「先期望后梯度」vs「先梯度后期望」：一个重要警示

在分析 KL 估计器的梯度时，有一个容易混淆的陷阱：**「先期望后梯度」与「先梯度后期望」可能给出不同的结果**。

如果从解析角度将 $\mathbb{E}_{q_\theta}[k_i]$ 视为 $\theta$ 的函数再求梯度（即「先期望后梯度」），根据「数值估计」一节的结论 $\mathbb{E}_{q_\theta}[k_1] = \mathbb{E}_{q_\theta}[k_3] = D_{\mathrm{KL}}(q_\theta \| p)$，我们有：

$$
\nabla_\theta \mathbb{E}_{q_\theta}[k_1] = \nabla_\theta D_{\mathrm{KL}}(q_\theta \| p)
$$

$$
\nabla_\theta \mathbb{E}_{q_\theta}[k_3] = \nabla_\theta D_{\mathrm{KL}}(q_\theta \| p)
$$

两者都给出反向 KL 的梯度。然而，在代码中直接对 $k_i$ 的样本均值进行反向传播时，自动微分执行的是「先梯度后期望」，得到 $\mathbb{E}_{q_\theta}[\nabla_\theta k_i]$——这与「先期望后梯度」的结果**可能不同**。

这种差异的根源在于：当采样分布 $q_\theta$ 本身依赖于 $\theta$ 时，期望与梯度不能随意交换。这正是 on-policy 场景最麻烦的地方，也是为什么需要把 $\rho$ 这条路径显式写出来。

### 6.2 统一框架下的梯度分析

现在，我们使用 $\rho$ 框架统一处理 on-policy 和 off-policy 场景。考虑损失函数形式 $L = \rho \cdot k$，其中 $\rho = \frac{q_\theta}{\text{sg}(\mu)}$。

下面的期望一律指对**固定采样分布** $\mu$ 的 $\mathbb{E}_\mu[\cdot]$。在这个前提下，由于 $\text{sg}(\mu)$ 不依赖于 $\theta$，对于任何关于 $\theta$ 可微的函数 $f_\theta(x)$，有

$$
\nabla_\theta \mathbb{E}_{\mu}[f_\theta(x)] = \mathbb{E}_{\mu}[\nabla_\theta f_\theta(x)]
$$

也就是说，在 $\rho$ 框架下，“先期望后梯度”和“先梯度后期望”对 $\mathbb{E}_\mu[\cdot]$ 是等价的；这并不意味着对 $\mathbb{E}_{q_\theta}[\cdot]$ 也能无条件交换微分与期望。

#### 统一框架下三种估计器的梯度推导

利用 $\nabla_\theta \rho = \rho \cdot s_\theta$（因为 $\rho = q_\theta / \text{sg}(\mu)$），结合前文推导的 $\nabla_\theta k_i$，应用乘积法则：

**$\nabla_\theta(\rho k_1)$**：

$$
\nabla_\theta(\rho k_1) = (\nabla_\theta \rho) k_1 + \rho (\nabla_\theta k_1) = \rho s_\theta k_1 + \rho s_\theta = \rho s_\theta (k_1 + 1)
$$

**$\nabla_\theta(\rho k_2)$**：

$$
\nabla_\theta(\rho k_2) = \rho s_\theta k_2 + \rho \left(-\log \frac{p}{q_\theta}\right) s_\theta = \rho s_\theta \left(k_2 - \log \frac{p}{q_\theta}\right) = \rho s_\theta (k_2 + k_1)
$$

**$\nabla_\theta(\text{sg}(\rho) k_2)$**（对 $\rho$ 施加 stop-gradient）：

$$
\nabla_\theta(\text{sg}(\rho) k_2) = \text{sg}(\rho) \cdot \nabla_\theta k_2 = \rho \cdot \left(-\log \frac{p}{q_\theta}\right) s_\theta = \rho s_\theta k_1
$$

**$\nabla_\theta(\rho k_3)$**：

$$
\nabla_\theta(\rho k_3) = \rho s_\theta k_3 + \rho \left(1-\frac{p}{q_\theta}\right) s_\theta = \rho s_\theta \left(k_3 + 1 - \frac{p}{q_\theta}\right)
$$

代入 $k_3 = \frac{p}{q_\theta} - 1 - \log \frac{p}{q_\theta}$：

$$
k_3 + 1 - \frac{p}{q_\theta} = \left(\frac{p}{q_\theta} - 1 - \log \frac{p}{q_\theta}\right) + 1 - \frac{p}{q_\theta} = -\log \frac{p}{q_\theta} = k_1
$$

这里有个很关键的消去：

$$
\boxed{\nabla_\theta(\rho k_3) = \rho s_\theta k_1}
$$

#### 梯度期望与优化目标

利用 $\mathbb{E}_\mu[\rho \cdot f] = \mathbb{E}_{q_\theta}[f]$ 和 $\mathbb{E}_{q_\theta}[s_\theta] = 0$：

**$\mathbb{E}_\mu[\nabla_\theta(\rho k_1)]$**：

$$
\mathbb{E}_\mu[\rho s_\theta (k_1 + 1)] = \mathbb{E}_{q_\theta}[s_\theta k_1] + \underbrace{\mathbb{E}_{q_\theta}[s_\theta]}_{=0} = \nabla_\theta D_{\mathrm{KL}}(q_\theta \| p) \quad \checkmark
$$

**$\mathbb{E}_\mu[\nabla_\theta(\rho k_2)]$**：

$$
\begin{aligned}
\mathbb{E}_\mu[\rho s_\theta (k_2 + k_1)]
&= \mathbb{E}_{q_\theta}[s_\theta k_2] + \mathbb{E}_{q_\theta}[s_\theta k_1] \\
&= \mathbb{E}_{q_\theta}[s_\theta k_2] + \mathbb{E}_{q_\theta}[\nabla_\theta k_2] && \text{（因为 } \nabla_\theta k_2 = k_1 s_\theta \text{）} \\
&= \nabla_\theta \mathbb{E}_{q_\theta}[k_2] && \text{（把 score-function 项与显式梯度项重新合并）}
\end{aligned}
$$

也就是说，$\rho k_2$ 的梯度期望对应的是“最小化 $\mathbb{E}_{q_\theta}[k_2]$”（一个与 KL 具有相同局部二阶行为的 surrogate），而**不是**反向 KL $D_{\mathrm{KL}}(q_\theta\|p)$ 的解析梯度；因此当目标是**精确优化反向 KL** 时，不宜直接使用 $\rho k_2$。

**$\mathbb{E}_\mu[\nabla_\theta(\text{sg}(\rho) k_2)]$**：

$$
\mathbb{E}_\mu[\rho s_\theta k_1] = \mathbb{E}_{q_\theta}[s_\theta k_1] = \nabla_\theta D_{\mathrm{KL}}(q_\theta \| p) \quad \checkmark
$$

**$\mathbb{E}_\mu[\nabla_\theta(\rho k_3)]$**：

$$
\mathbb{E}_\mu[\rho s_\theta k_1] = \mathbb{E}_{q_\theta}[s_\theta k_1] = \nabla_\theta D_{\mathrm{KL}}(q_\theta \| p) \quad \checkmark
$$

#### 梯度等价性：哪些方法产生相同的梯度随机变量

这一步也解释了为什么 $\text{sg}(\rho) k_2$ 与 $\rho k_3$ 经常一起出现：对同一个样本 $x$，它们反传得到的梯度向量其实完全一样，都是 $\rho s_\theta k_1$。不是只在期望上相同，而是样本级一致；因此均值、方差和更高阶统计量也都一致。

|      损失项写法       |        梯度随机变量         |                  梯度期望                  |          对应的优化目标          |
| :-------------------: | :-------------------------: | :----------------------------------------: | :------------------------------: |
|      $\rho k_1$       |   $\rho s_\theta (k_1+1)$   |  $\nabla D_{\mathrm{KL}}(q_\theta \| p)$   |            反向 KL ✓             |
|      $\rho k_2$       | $\rho s_\theta (k_2 + k_1)$ | $\nabla_\theta \mathbb{E}_{q_\theta}[k_2]$ | 局部二阶 surrogate（非反向 KL）✗ |
| $\text{sg}(\rho) k_2$ |     $\rho s_\theta k_1$     |  $\nabla D_{\mathrm{KL}}(q_\theta \| p)$   |            反向 KL ✓             |
|      $\rho k_3$       |     $\rho s_\theta k_1$     |  $\nabla D_{\mathrm{KL}}(q_\theta \| p)$   |            反向 KL ✓             |

### 6.3 On-policy 与 Off-policy 的统一视角

有了这一步，再回头看 on-policy 和 off-policy 的关系就会清楚很多。

**On-policy**（$\mu = q_\theta$）：

- $\rho = \frac{q_\theta}{\text{sg}(q_\theta)} \equiv 1$（数值恒为 1）
- $\rho k_1 = k_1$，$\rho k_2 = k_2$，$\rho k_3 = k_3$
- 但梯度不同！因为 $\nabla_\theta \rho = s_\theta \neq 0$

这解释了为什么 on-policy 时**朴素直接反向传播**（不显式构造 $\rho$）使用 $k_1$ 或 $k_3$ 作为损失函数会出问题：

- 直接使用 $k_1$：相当于没有 $\rho$ 的版本，$\mathbb{E}_{q_\theta}[\nabla k_1] = \mathbb{E}_{q_\theta}[s_\theta] = 0$，**完全无效**
- 直接使用 $k_3$：相当于没有 $\rho$ 的版本，$\mathbb{E}_{q_\theta}[\nabla k_3] = \nabla D_{\mathrm{KL}}(p \| q_\theta)$（正向 KL），**方向错误**
- 直接使用 $k_2$：$\mathbb{E}_{q_\theta}[\nabla k_2] = \nabla D_{\mathrm{KL}}(q_\theta \| p)$（反向 KL）✓ **在这种朴素实现下是与目标一致的选择**

但如果**显式构造** $\rho = \frac{q_\theta}{\text{sg}(q_\theta)}$，则：

- **可用**：$\rho k_1$（方差高）、$\text{sg}(\rho) k_2$（推荐）、$\rho k_3$（推荐）——三者均给出反向 KL 梯度
- **不宜直接用于精确优化反向 KL**：$\rho k_2$（$\rho$ 参与梯度）——优化的是与 KL 局部二阶行为一致的 surrogate，而非反向 KL

**Off-policy**（$\mu \neq q_\theta$）：

- $\rho = \frac{q_\theta}{\mu}$（标准重要性权重）
- **可用**：$\rho k_1$（方差高）、$\text{sg}(\rho) k_2$（推荐）、$\rho k_3$（推荐）——三者均给出反向 KL 梯度
- **不宜直接用于精确优化反向 KL**：$\rho k_2$（$\rho$ 参与梯度）——优化的是与 KL 局部二阶行为一致的 surrogate，而非反向 KL

值得注意的是，on-policy 时 $k_2$ 能直接工作，并不是一个可外推的普遍现象，而是 $\rho \equiv 1$ 时的一个特殊退化：因为这时 $\nabla_\theta k_2 = k_1 s_\theta$，恰好落在正确的反向 KL 梯度上；这个结论不能直接外推到一般 off-policy 情况。

关于大模型 off-policy 场景的深入分析，可以参考我之前的博客：[从两策略到三策略：LLM RL 中行为策略–参考策略不一致下的 TRPO 扩展](/reinforcement-learning/2025/11/15/three-policy-zh.html)。

### 6.4 方差分析

前面我们看到，给出反向 KL 无偏梯度的有三个选择：$\rho k_1$、$\text{sg}(\rho) k_2$、$\rho k_3$。它们的梯度随机变量分别为（注意 $s_\theta$ 是向量，因此梯度也是向量）：

$$
g_1(x) = \rho(x) s_\theta(x) (k_1(x) + 1), \quad g_\star(x) = \rho(x) s_\theta(x) k_1(x)
$$

其中 $g_\star$ 对应 $\text{sg}(\rho) k_2$ 和 $\rho k_3$（两者完全相同）。

为了避免“向量梯度的方差”这一表述的歧义，我们比较任意方向上的投影方差：取任意单位向量 $u$，定义标量随机变量

$$
g_1^{(u)} := u^\top g_1, \quad g_\star^{(u)} := u^\top g_\star.
$$

令 $A_u(x) := \rho(x)\, u^\top s_\theta(x)$，$B(x) := k_1(x)$，则

$$
g_1^{(u)} = A_u(B+1), \quad g_\star^{(u)} = A_u B.
$$

两者期望相同，且任意方向上的方差之差为

$$
\boxed{
\mathrm{Var}_\mu\big(g_1^{(u)}\big) - \mathrm{Var}_\mu\big(g_\star^{(u)}\big)
= \mathbb{E}_\mu\big[A_u(x)^2 \big(2B(x)+1\big)\big]
= \mathbb{E}_\mu\Big[\rho(x)^2\,\big(u^\top s_\theta(x)\big)^2\,\big(2k_1(x)+1\big)\Big].
}
$$

**在典型的 KL 惩罚场景下**（$q_\theta \approx p \approx \mu$），取 $\frac{p(x)}{q_\theta(x)} = 1 + \varepsilon(x)$，$|\varepsilon| \ll 1$：

- $k_1 = -\log \frac{p}{q_\theta} \approx -\varepsilon$
- $2k_1 + 1 \approx 1 - 2\varepsilon$，主导项为正的 $O(1)$ 常数

因此在足够小的邻域里，$\mathrm{Var}_\mu(g_1) > \mathrm{Var}_\mu(g_\star)$。

一旦离开这个小-KL 邻域，$2k_1+1$ 的符号就不再固定；这时整体比较还要结合 $\rho^2$ 与 score 项的加权来判断，不能再简单依赖这一局部展开。

直觉上看：

- $g_1 = \rho s_\theta (k_1 + 1)$ 包含一个量级为 $O(1)$ 的零均值噪声项 $\rho s_\theta$
- $g_\star = \rho s_\theta k_1$ 已将该常数噪声项消去，剩下与 $\varepsilon$ 成正比的一阶小量

方差对比如下：

|        估计器         |      梯度随机变量       | 系数量级（$\frac{p}{q_\theta}\approx1$） | 方差 |
| :-------------------: | :---------------------: | :--------------------------------------: | :--: |
|      $\rho k_1$       | $\rho s_\theta (k_1+1)$ |                  $O(1)$                  |  高  |
| $\text{sg}(\rho) k_2$ |   $\rho s_\theta k_1$   |             $O(\varepsilon)$             |  低  |
|      $\rho k_3$       |   $\rho s_\theta k_1$   |             $O(\varepsilon)$             |  低  |

$\text{sg}(\rho) k_2$ 与 $\rho k_3$ 给出的是同一个梯度随机变量；相比之下，$\rho k_1$ 多了一个零均值的常数噪声项，所以在典型的小-KL 场景里通常更吵。

> **实践建议**：若优化反向 KL，首选 $\rho k_3$ 或 $\text{sg}(\rho) k_2$（两者梯度等价且方差低）；$\rho k_1$ 虽无偏但方差高，可作为备选并需配合 clipping/正则化。

**极度 off-policy 时的警示**：

当 $\mu$ 与 $q_\theta$ 差异很大时——例如 $\mu$ 在 $q_\theta$ 的高密度区域几乎没有采样，或 $\rho = q_\theta / \mu$ 在尾部爆炸——任何基于 $\rho$ 的方法都会遭遇严重的方差问题。此时，$\rho k_3$（或 $\text{sg}(\rho) k_2$）相对于 $\rho k_1$ 的优势不再有理论保证，需要结合 clipping、正则化等策略综合处理。

不过，在 RL 实践中，我们通常会控制 KL 约束、限制 off-policy 程度（例如使用近邻策略 $\mu = q_{\theta_\text{old}}$）。在这个常见的场景中，可以相当有信心地说：

> **如果已经决定用重要性采样来优化反向 KL，推荐使用 $\rho k_3$ 或 $\text{sg}(\rho) k_2$（两者梯度等价且方差低）；相较之下，$\rho k_1$ 方差更高。**

与本文的分析一致，DeepSeek-V3.2 技术报告中使用的是 $\frac{q_\theta}{\mu} k_3$ 作为 off-policy KL 惩罚的估计器。在本文的记号下，这正是 $\rho k_3$：本质上是用当前策略与行为策略的比率修正 $k_3$，从而同时恢复无偏 KL 数值估计和无偏梯度。

<figure style="text-align:center;" markdown="0">
<img src="/assets/img/kl-estimators/dpsk-3d2-k3.png" style="width:95%;max-width:100%;">
<figcaption style="font-size:0.9em;color:gray;">图片来源：<a href="https://arxiv.org/pdf/2512.02556v1">DeepSeek-V3.2 技术报告 3.1 章节</a></figcaption>
</figure>

#### 梯度分析总览表

统一框架下各估计器对应的梯度目标如下：

|   采样类型    |         Loss          |          $\nabla_\theta$ Loss 的期望           |         对应的优化目标          | 能否用于优化反向 KL？ |
| :-----------: | :-------------------: | :--------------------------------------------: | :-----------------------------: | :-------------------: |
| on/off-policy |      $\rho k_1$       | $\nabla_\theta D_{\mathrm{KL}}(q_\theta \| p)$ |             反向 KL             |    ✓（但方差较高）    |
| on/off-policy |      $\rho k_2$       |   $\nabla_\theta \mathbb{E}_{q_\theta}[k_2]$   | 局部二阶 surrogate（非反向 KL） |           ✗           |
| on/off-policy | $\text{sg}(\rho) k_2$ | $\nabla_\theta D_{\mathrm{KL}}(q_\theta \| p)$ |             反向 KL             |   ✓（推荐，低方差）   |
| on/off-policy |      $\rho k_3$       | $\nabla_\theta D_{\mathrm{KL}}(q_\theta \| p)$ |             反向 KL             |   ✓（推荐，低方差）   |

其中 $\rho = \frac{q_\theta}{\text{sg}(\mu)}$。当 on-policy（$\mu = q_\theta$）时，$\rho \equiv 1$。

需要特别强调：**上表的结论针对的是 “loss 写成 $L=\rho\,k$ 且 $\rho$ 在计算图中保留梯度路径” 的统一框架**。在 on-policy 时，虽然数值上 $\rho\equiv 1$，但由于 $\rho=\frac{q_\theta}{\text{sg}(q_\theta)}$，仍有 $\nabla_\theta\rho=s_\theta\neq 0$，因此 $\rho k$ 与“直接对 $k$ 的样本均值反向传播”在梯度上并不等价。

如果你采用的是**朴素 on-policy 写法**（即从 $q_\theta$ 采样后，将 $\{k_i(x)\}$ 视为普通标量，对其样本均值直接反向传播；不显式构造 $\rho=\frac{q_\theta}{\text{sg}(q_\theta)}$ 来补上 score-function 路径），则会退化为：

- 直接使用 $k_1$：$\mathbb{E}_{q_\theta}[\nabla k_1]=0$（无效）
- 直接使用 $k_2$：$\mathbb{E}_{q_\theta}[\nabla k_2]=\nabla D_{\mathrm{KL}}(q_\theta\|p)$（反向 KL）✓
- 直接使用 $k_3$：$\mathbb{E}_{q_\theta}[\nabla k_3]=\nabla D_{\mathrm{KL}}(p\|q_\theta)$（正向 KL）✗

把上面的结果压缩成一句话：

1. **On-policy 优化反向 KL（朴素直接反向传播的实现）**：在本文限定的实现方式下，$k_2$ 是与目标最一致的选择
2. **Off-policy 优化反向 KL**：有三个正确选项：
   - $\rho k_1$：无偏但方差较高
   - $\text{sg}(\rho) k_2$：无偏，与 $\rho k_3$ **梯度完全等价**
   - $\rho k_3$：无偏且方差更低（与上一项等价，均为推荐选择）
3. **$\rho k_2$（权重参与梯度）不对应本文目标**：它优化的是一个与 KL 仅在局部二阶行为上一致的 surrogate，而不是反向 KL；这是一个容易被忽视的陷阱

## 7. 作为奖励塑形项时的梯度分析

最容易掉进去的坑在这里：既然 $k_1$ 和 $k_3$ 对反向 KL 的数值估计都是无偏的，那把它们加上 stop-gradient 之后放进奖励塑形里，是不是也应该没问题？

答案是否定的。数值上无偏，不推出放进 reward 后梯度仍然正确。因为一旦 KL 变成 shaped reward 的一部分，真正进入优化的是 $\mathbb{E}[s_\theta \hat{k}]$，而不是 $\mathbb{E}[\hat{k}]$ 本身。

### 7.1 真正的 KL 正则化策略梯度

考虑 KL 正则化的强化学习目标：

$$
J(\theta) = \mathbb{E}_{q_\theta}[R] - \beta \cdot D_{\mathrm{KL}}(q_\theta \| p)
$$

其解析梯度为：

$$
\nabla_\theta J = \mathbb{E}_{q_\theta}[s_\theta \cdot R] - \beta \cdot \nabla_\theta D_{\mathrm{KL}}(q_\theta \| p)
$$

利用前文「准备工作」章节的结论，反向 KL 的梯度为：

$$
\nabla_\theta D_{\mathrm{KL}}(q_\theta \| p) = \mathbb{E}_{q_\theta}\left[s_\theta \cdot \left(-\log \frac{p}{q_\theta}\right)\right] = \mathbb{E}_{q_\theta}[s_\theta \cdot k_1]
$$

因此，真正的 KL 正则化策略梯度是：

$$
\nabla_\theta J = \mathbb{E}_{q_\theta}\left[s_\theta \cdot \left(R - \beta \cdot k_1\right)\right]
$$

#### 使用估计器 $\hat{k}$ 时的梯度形式

下文只讨论**策略梯度主项本身**，暂不把 baseline、critic 拟合误差、GAE 与额外归一化一并纳入。此时，如果把某个估计器 $\hat{k}$（加 stop-gradient）放进奖励塑形，形塑后的奖励为 $\tilde{R} = R - \beta \cdot \text{sg}(\hat{k})$，策略梯度变为：

$$
\nabla_\theta \tilde{J} = \mathbb{E}_{q_\theta}\left[s_\theta \cdot (R - \beta \cdot \hat{k})\right]
$$

**在这个主项分析下，无偏条件是**：$\nabla_\theta \tilde{J} = \nabla_\theta J$ 当且仅当

$$
\mathbb{E}_{q_\theta}[s_\theta \cdot \hat{k}] = \mathbb{E}_{q_\theta}[s_\theta \cdot k_1]
$$

#### 使用 $k_1$ 作为惩罚：梯度无偏

当 $\hat{k} = k_1$ 时，条件自动满足：

$$
\mathbb{E}_{q_\theta}[s_\theta \cdot k_1] = \mathbb{E}_{q_\theta}[s_\theta \cdot k_1] \quad \checkmark
$$

因此，**就该策略梯度主项而言，把 $k_1$ 放进奖励塑形时诱导的是无偏梯度**。

#### 使用 $k_3$ 作为惩罚：梯度有偏

当 $\hat{k} = k_3 = \frac{p}{q_\theta} - 1 - \log \frac{p}{q_\theta}$ 时：

$$
\mathbb{E}_{q_\theta}[s_\theta \cdot k_3] = \mathbb{E}_{q_\theta}\left[s_\theta \cdot \left(\frac{p}{q_\theta} - 1\right)\right] + \mathbb{E}_{q_\theta}\left[s_\theta \cdot \left(-\log \frac{p}{q_\theta}\right)\right]
$$

第二项正是 $\mathbb{E}_{q_\theta}[s_\theta \cdot k_1]$。问题出在第一项：

$$
\mathbb{E}_{q_\theta}\left[s_\theta \cdot \left(\frac{p}{q_\theta} - 1\right)\right] = \mathbb{E}_{q_\theta}\left[s_\theta \cdot \frac{p}{q_\theta}\right] - \underbrace{\mathbb{E}_{q_\theta}[s_\theta]}_{=0} = \mathbb{E}_{q_\theta}\left[s_\theta \cdot \frac{p}{q_\theta}\right]
$$

而这个量可以改写为：

$$
\mathbb{E}_{q_\theta}\left[s_\theta \cdot \frac{p}{q_\theta}\right] = \int q_\theta(x) \cdot \nabla_\theta \log q_\theta(x) \cdot \frac{p(x)}{q_\theta(x)} dx = \int p(x) \cdot \nabla_\theta \log q_\theta(x) dx = \mathbb{E}_p[s_\theta]
$$

利用正向 KL 的梯度公式 $\nabla_\theta D_{\mathrm{KL}}(p \| q_\theta) = -\mathbb{E}_p[s_\theta]$，有：

$$
\mathbb{E}_{q_\theta}\left[s_\theta \cdot \frac{p}{q_\theta}\right] = -\nabla_\theta D_{\mathrm{KL}}(p \| q_\theta)
$$

因此：

$$
\mathbb{E}_{q_\theta}[s_\theta \cdot k_3] = \underbrace{-\nabla_\theta D_{\mathrm{KL}}(p \| q_\theta)}_{\text{偏差项}} + \nabla_\theta D_{\mathrm{KL}}(q_\theta \| p)
$$

**把 $k_3$ 放进奖励塑形时，梯度是有偏的**，偏差项等于正向 KL 梯度的负值。

更严格地说，把 $k_3$ 放进奖励塑形时，实际更新会在反向 KL 梯度之外额外混入一个与正向 KL 梯度相关的偏差项，因此不再对应纯粹的反向 KL 正则目标。这也是它在实践中容易变得不稳定的原因。

**实验现象**：Shah et al. (2025) 的实验表明，在其 on-policy LLM-RL 设定下：

- **$k_1$ in reward**：训练稳定
- **$k_3$ in reward**：出现显著不稳定，甚至训练崩溃

这与本文给出的偏差分析方向一致。

#### 使用 $k_2$ 作为惩罚：同样有偏

当 $\hat{k} = k_2 = \frac{1}{2}k_1^2$ 时，奖励塑形项对应的梯度项为

$$
\mathbb{E}_{q_\theta}[s_\theta \cdot k_2]
= \frac{1}{2}\mathbb{E}_{q_\theta}[s_\theta \cdot k_1^2],
$$

它一般**不等于** $\mathbb{E}_{q_\theta}[s_\theta \cdot k_1]$。因此，把 $k_2$ 放进奖励塑形时同样会诱导有偏的策略梯度。

#### Off-policy 场景下的结论

上述分析假设 on-policy 采样。在 off-policy 场景下，结论是否改变？

设样本来自行为策略 $\mu$，使用重要性加权的策略梯度：

$$
\nabla_\theta \tilde{J} = \mathbb{E}_\mu\left[\frac{q_\theta}{\mu} \cdot s_\theta \cdot (R - \beta \cdot k)\right]
$$

利用 $\mathbb{E}_\mu[\frac{q_\theta}{\mu} \cdot f] = \mathbb{E}_{q_\theta}[f]$，上式等于：

$$
= \mathbb{E}_{q_\theta}[s_\theta \cdot R] - \beta \cdot \mathbb{E}_{q_\theta}[s_\theta \cdot k]
$$

**无偏条件**仍然是 $\mathbb{E}_{q_\theta}[s_\theta \cdot k] = \mathbb{E}_{q_\theta}[s_\theta \cdot k_1]$，与 on-policy 完全相同。

这里需要单独强调一件事：在本文关注的当前样本 / 当前 token 级 off-policy 策略梯度主项里，重要性权重 $\frac{q_\theta}{\mu}$ 作用于整个策略梯度估计器，**不需要再对 shaped reward 中的 KL 估计器额外乘一层权重**。因此：

- Shaped reward 保持原形式：$\tilde{R} = R - \beta \cdot k_1$（不是 $R - \beta \cdot \frac{q_\theta}{\mu} k_1$）
- 在本文讨论的 **stop-grad reward shaping**（$\tilde{R}=R-\beta\,\text{sg}(k)$）且目标为 **反向 KL 正则** 的设定下：结论与 on-policy 的策略梯度主项相同，**只有 $k_1$ 能保持无偏的梯度主项**

> **注**：这里默认的是当前样本/当前 token 级的 reward shaping 写法；若回到一般多步 MDP 的严格 off-policy 推导，还需要配合逐步重要性采样或相应的值函数修正。

### 7.2 这一节的结论：只有 $k_1$ 保持无偏

| 估计器 | 数值无偏？ | 作为奖励塑形项时梯度主项无偏？ |  实际表现  |
| :----: | :--------: | :----------------------------: | :--------: |
| $k_1$  |     ✓      |               ✓                |    稳定    |
| $k_2$  |     ✗      |               ✗                |   不建议   |
| $k_3$  |     ✓      |               ✗                | 显著不稳定 |

回头看，评价 KL 估计器时，“数值无偏”和“梯度正确”其实是两个独立维度。对本文讨论的奖励塑形写法（stop-grad reward shaping，目标为反向 KL 正则；无论 on-policy 还是 off-policy）来说，就策略梯度主项而言，只有 $k_1$ 满足目标要求。$k_3$ 虽然数值无偏且方差更低，但放进奖励塑形后会导致梯度有偏，实践里也确实更容易不稳定。

> **补充说明**：一旦把 learned critic、GAE、baseline 归一化等实现细节一并纳入，实际更新中的偏差分析会更复杂。本节结论刻意只聚焦于 policy-gradient 主项，以避免把不同来源的偏差混在一起。

到这里容易产生一个“表面矛盾”：

- 在 **奖励塑形项**里我们强调“若目标是本文讨论的 reverse-KL stop-grad shaping，则只有 $k_1$ 在策略梯度主项上无偏”；
- 但在前文 **损失项反传**（尤其 off-policy）里，我们又推荐用 $\rho k_3$ 或 $\text{sg}(\rho)k_2$ 来获得更低方差的反向 KL 梯度。

下一节将解释：两者并不冲突——就“KL 正则项对应的 policy-gradient 随机变量”而言，它们甚至可以做到**样本级完全等价**；差异主要来自 KL 是否进入 advantage/baseline、以及信用分配（credit assignment）的路径。

## 8. $k_1$ 奖励塑形与低方差 KL 损失项的等价性与差异

说到这里，很容易追问一句：**KL 写进 loss，和 KL 写进 reward，到底在什么意义上等价，又在什么意义上不是一回事？**

### 8.1 KL 梯度项的样本级等价性

这里说的“等价”，只限于 KL 正则对应的那一项梯度随机变量；一旦把 learned critic、baseline、GAE、batch 中心化一起算进来，整体更新语义就会重新分叉。本节统一写成 **policy gradient 的上升方向** $\nabla_\theta J$（若你在代码里最小化 loss，则整体只差一个全局负号，不影响等价性结论），并继续使用前文的统一权重记号：样本来自 $x\sim\mu$，重要性权重 $\rho=\frac{q_\theta}{\text{sg}(\mu)}$ 作用在策略梯度估计器上。

**KL 作为损失项（低方差选择）**：前文已证明，采用 $\text{sg}(\rho) k_2$ 或 $\rho k_3$ 作为正则项时，梯度随机变量都化简为

$$
\nabla_\theta(\text{sg}(\rho) k_2) = \nabla_\theta(\rho k_3) = \rho s_\theta k_1
$$

**KL 作为奖励塑形项（$k_1$ in reward）**：shaped reward 为 $\tilde{R} = R - \beta \cdot k_1$（对 $k_1$ 做 stop-gradient 只是在实现上避免“KL 直接反传”，不改变它作为惩罚的数值）。在“策略梯度项”里，KL 惩罚贡献的是

$$
\mathbb{E}_\mu[\rho s_\theta \cdot (-\beta k_1)] = -\beta \cdot \mathbb{E}_\mu[\rho s_\theta k_1]
$$

这也是为什么前一节和第 6 节看起来像在推荐不同写法，实际上却并不冲突：两者的 KL 梯度项在这里**样本级完全相同**。

也就是说，在不考虑 baseline/advantage 的具体构造细节时：

- “把 KL 写进 loss 并用低方差实现（$\text{sg}(\rho)k_2$ 或 $\rho k_3$）”
- 与“把 KL 写进 reward 并选 $k_1$（stop-grad shaped reward）”

对策略更新施加的 KL 正则“力”可以是一模一样的。

具体来说，如果我们只看“最大化 $J$”时 KL 惩罚贡献的那一项梯度（惩罚项在 $J$ 里带负号，因此这项的上升方向自然带 $-\beta$）：

- **损失项写法（低方差实现）**：$-\beta \cdot \rho s_\theta k_1$
- **奖励塑形写法（$k_1$ in reward）**：$\rho s_\theta \cdot (-\beta k_1) = -\beta \cdot \rho s_\theta k_1$

它们是同一个随机变量，因此不仅期望相同，方差也完全相同。

#### 整体更新语义的差异

尽管 KL 梯度项在样本级等价，**两种方式的整体更新语义仍然不同**。差异主要体现在以下几个方面：

#### 1. KL 是否进入 Advantage/Baseline

**KL 作为损失项**（等价于最大化 $J(\theta)=\mathbb{E}[R]-\beta\,\mathrm{KL}$，但把 KL 项作为一个独立的、可控的“显式力”来实现）：

$$
\nabla_\theta J_{\text{loss-impl}} = \underbrace{\mathbb{E}_\mu[\rho s_\theta A_{\text{env}}]}_{\text{RL 上升方向}} + \underbrace{(-\beta) \cdot \mathbb{E}_\mu[\rho s_\theta k_1]}_{\text{独立的 KL 惩罚上升方向}}
$$

KL 是一个**独立的正则项**，与 advantage 完全解耦。KL 梯度的大小只取决于 $k_1$ 本身，不受 critic 质量或 baseline 选择的影响。

**KL 作为奖励塑形项**：

$$
\nabla_\theta J_{\text{reward-impl}} = \mathbb{E}_\mu[\rho s_\theta \tilde{A}], \quad \tilde{A} \text{ 基于 } (R - \beta \cdot k_1)
$$

KL 通过 shaped reward 进入 advantage 计算，会被 baseline 处理。这意味着：

- KL 的影响会被 advantage 的构造方式调制
- 如果使用 value function baseline，KL 的影响会被部分吸收

从实现角度看，这里的差别可以理解为：Loss 方案把“环境回报部分”和“KL 正则部分”分开估计；Reward 方案把 KL 视为回报的一部分，因此它会跟着你对回报做的所有处理（baseline、归一化、截断等）一起走。

#### 2. 信用分配：独立正则力 vs 混入 Shaped Reward

**KL 作为损失项**：每个 token/state 的 KL 梯度是「局部」的，只影响该位置的策略更新。

**KL 作为奖励塑形项**：KL 惩罚通过 return/advantage 的时间回传，可能影响到更早的决策。

#### 3. Reward 中心化 KL：对梯度无偏性的影响

在大模型 RL（如 GRPO、PPO for LLM）中，常见的 advantage 计算方式是 $A = r - \text{mean}(r)$。当 KL 作为奖励塑形项时，是否把 KL 也纳入 mean 会影响梯度的无偏性。

设采样 $x_1,\dots,x_n \overset{iid}{\sim} q_\theta$，记 $g_i = \nabla_\theta \log q_\theta(x_i)$，并用 $\mathrm{kl}_i$ 表示第 $i$ 个样本的 KL 惩罚标量，$\bar{\mathrm{kl}} = \frac{1}{n}\sum_j \mathrm{kl}_j$。

**不中心化（$-\beta\,\mathrm{kl}_i$）**：KL 梯度项的期望为

$$
-\beta \mathbb{E}[g_i\,\mathrm{kl}_i] = -\beta \nabla_\theta \mathbb{E}[\mathrm{KL}]
$$

这是对 $-\beta \mathbb{E}[\mathrm{KL}]$ 的**无偏梯度**。

**同 batch 均值中心化（$-\beta(\mathrm{kl}_i - \bar{\mathrm{kl}})$，含自身）**：由于 $\bar{\mathrm{kl}}$ 依赖所有样本（包括 $x_i$ 自身），期望梯度变为

$$
-\beta \left(1 - \frac{1}{n}\right) \nabla_\theta \mathbb{E}[\mathrm{KL}]
$$

即 KL 正则梯度被**缩小**了 $\frac{1}{n}$，等价于有效 $\beta$ 变小。这不是严格无偏的。

**Leave-one-out 中心化（$-\beta(\mathrm{kl}_i - \bar{\mathrm{kl}}_{-i})$）**：若改用 $\bar{\mathrm{kl}}_{-i} = \frac{1}{n-1}\sum_{j \neq i} \mathrm{kl}_j$，则 $\bar{\mathrm{kl}}_{-i}$ 与 $g_i$ 独立，有 $\mathbb{E}[g_i \bar{\mathrm{kl}}_{-i}] = 0$，因此

$$
-\beta \mathbb{E}[g_i (\mathrm{kl}_i - \bar{\mathrm{kl}}_{-i})] = -\beta \nabla_\theta \mathbb{E}[\mathrm{KL}]
$$

仍是**无偏梯度**，同时享受中心化带来的方差缩减。

同 batch 均值中心化引入的偏差为 $O(1/n)$，在 GRPO 等大 batch 场景下影响很小；若追求严格无偏，可改用 leave-one-out 均值，同时享受方差缩减。

### 8.2 何时选择哪种方式？

|     维度     |           KL 作为损失项           |                  KL 作为奖励塑形项                   |
| :----------: | :-------------------------------: | :--------------------------------------------------: |
| KL 梯度形态  | $\rho s_\theta k_1$（低方差选择） |                 $\rho s_\theta k_1$                  |
| 与 Advantage |             完全解耦              |               通过 shaped reward 耦合                |
|  KL 中心化   |          无（绝对惩罚）           |      有（$\text{KL} - \text{mean}(\text{KL})$）      |
|   信用分配   |          局部、per-token          |             可能有时间回传（取决于实现）             |
|   适用场景   |  希望 KL 作为显式正则项单独控制   | 希望 KL 随 shaped reward 一起进入 advantage / return |

**实践建议**：

1. **如果你希望 KL 作为显式正则项单独控制**——也就是尽量少受 advantage 构造、critic / baseline 质量影响，那么请选择 **KL 作为损失项**，使用 $\text{sg}(\rho) k_2$ 或 $\rho k_3$。注意，在 on-policy 场景下，如果不想显式构造 $\rho=\frac{q_\theta}{\text{sg}(q_\theta)}$，直接使用 $k_2$ 会更简单且不易出错。

2. **如果你希望 KL 随 shaped reward 一起进入 advantage / return**——也就是接受它与 baseline、信用分配路径耦合，那么请选择 **KL 作为奖励塑形项**，并使用 $k_1$。

基于上述“数值无偏 vs 梯度正确”以及“Loss 与 Reward 实现差异”的结论，下面进入可直接照抄到代码里的选型速查与常见踩坑点。

## 9. 实践指南与常见陷阱

### 9.1 三种估计器定义速查

$$
k_1 = \log \frac{q_\theta}{p}, \quad k_2 = \frac{1}{2}\left(\log \frac{p}{q_\theta}\right)^2, \quad k_3 = \frac{p}{q_\theta} - 1 - \log \frac{p}{q_\theta}
$$

### 9.2 数值估计性质

| 估计器 | 对反向 KL $D_{\mathrm{KL}}(q_\theta \| p)$ 数值无偏？ |       方差       |
| :----: | :---------------------------------------------------: | :--------------: |
| $k_1$  |                           ✓                           | 高（估计值可负） |
| $k_2$  |              ✗（小 KL 邻域通常偏差较小）              |        低        |
| $k_3$  |                           ✓                           |        低        |

### 9.3 选型速查表

#### On-policy 下把反向 KL 写成损失项

|                 损失项写法                 |                    优点                     |                     问题                      | 推荐 |
| :----------------------------------------: | :-----------------------------------------: | :-------------------------------------------: | :--: |
|                   $k_1$                    |                      —                      |   梯度期望为零，**完全无效**，不能用于优化    |  ✗✗  |
|                   $k_2$                    | 梯度正确（反向 KL），低方差，**实现最简单** |      数值有偏（小 KL 邻域通常偏差较小）       |  ✓✓  |
|                   $k_3$                    |                      —                      | 若目标是反向 KL，则它对应的是**正向 KL** 梯度 |  ✗✗  |
| $\frac{q_\theta}{\text{sg}(q_\theta)} k_3$ |    梯度正确（反向 KL），低方差，数值无偏    |         需显式构造 $\rho$，实现稍复杂         |  ✓   |

> **注**：$k_2$ 与 $\frac{q_\theta}{\text{sg}(q_\theta)} k_3$ 的梯度完全相同（样本级等价）。On-policy 时推荐直接用 $k_2$，实现最简单。

#### Off-policy 下把反向 KL 写成损失项

|                    损失项写法                    |                   优点                    |                           问题                            | 推荐 |
| :----------------------------------------------: | :---------------------------------------: | :-------------------------------------------------------: | :--: |
|            $\frac{q_\theta}{\mu} k_1$            |       梯度正确（反向 KL），数值无偏       |                       **方差较高**                        |  △   |
|            $\frac{q_\theta}{\mu} k_2$            |                     —                     | 梯度对应局部二阶 surrogate（非反向 KL），与本文目标不匹配 |  ✗✗  |
| $\text{sg}\left(\frac{q_\theta}{\mu}\right) k_2$ |      梯度正确（反向 KL），**低方差**      |            数值有偏（小 KL 邻域通常偏差较小）             |  ✓✓  |
|            $\frac{q_\theta}{\mu} k_3$            | 梯度正确（反向 KL），**低方差**，数值无偏 |                             —                             |  ✓✓  |

> **注**：$\text{sg}\left(\frac{q_\theta}{\mu}\right) k_2$ 与 $\frac{q_\theta}{\mu} k_3$ 的梯度完全相同（样本级等价）。两者均为推荐选择。

#### 把 KL 写成奖励塑形项（stop-grad shaped reward）

| 估计器 |               优点               |                                              问题                                              | 推荐 |
| :----: | :------------------------------: | :--------------------------------------------------------------------------------------------: | :--: |
| $k_1$  | 数值无偏，**诱导的梯度主项无偏** |                                            方差较高                                            |  ✓✓  |
| $k_2$  |             数值有偏             |                           诱导的策略梯度一般有偏（通常不等于目标项）                           |  ✗✗  |
| $k_3$  |         数值无偏，低方差         | **诱导的策略梯度有偏**，偏差项为 $-\nabla D_{\mathrm{KL}}(p\|q)$，实践中可能显著不稳定甚至崩溃 |  ✗✗  |

> **注**：在本文讨论的 stop-grad reward shaping 场景下，**只有 $k_1$ 能保持与反向 KL 正则一致的无偏策略梯度主项**。$k_2$ 和 $k_3$ 都会引入偏差；其中 $k_3$ 虽然数值无偏且方差低，但实践中曾观察到显著不稳定甚至崩溃。

#### 图例说明

- ✓✓：**强烈推荐**，理论正确且实践表现好
- ✓：推荐，理论正确但实现稍复杂或有小缺点
- △：可用但需谨慎，存在方差高等问题
- ✗✗：与本文当前优化目标不匹配，不建议这样用

### 9.4 常见陷阱

1. **On-policy 下使用 $k_1 = \log \frac{q_\theta}{p}$ 作为损失项**：梯度期望为零，完全无效。
2. **On-policy 下使用 $k_3 = \frac{p}{q_\theta} - 1 - \log \frac{p}{q_\theta}$ 作为损失项来优化反向 KL**：若目标是反向 KL，其梯度实际对应的是正向 KL $D_{\mathrm{KL}}(p \| q_\theta)$。
3. **Off-policy 下使用 $\frac{q_\theta}{\mu} k_2$（重要性权重不 detach）**：梯度对应一个与 KL 仅在局部二阶行为上一致的 surrogate，而非反向 KL。
4. **在奖励塑形项里使用 $k_3$**：虽然数值无偏，但诱导的策略梯度有偏，实践中可能显著不稳定甚至崩溃。
5. **On-policy 时将 $\rho$ 简单地设为常数 1**：必须显式构造 $\rho = \frac{q_\theta}{\text{sg}(q_\theta)}$（或等价地 $\exp(\log q_\theta - \text{sg}(\log q_\theta))$），否则会丢失 score-function 梯度路径，使 $\rho k_1$ 和 $\rho k_3$ 退化为朴素形式，从而不再对应前文的无偏结论。
6. **混淆「数值无偏」与「梯度正确」**：$k_3$ 对反向 KL 数值无偏，但作为奖励塑形项时诱导的策略梯度有偏；选择估计器时必须同时考虑这两个维度。

## 10. 总结

如果只记住四句话，可以记这四句：

1. **数值无偏不等于梯度正确。** 选 KL 估计器时，不仅要看它把 KL 数值估得准不准，还要看它在你的具体写法里到底在优化谁。
2. **若把 KL 写成可微损失项**：on-policy 的朴素实现直接用 $k_2$ 最省心；若显式构造 $\rho$，或者本来就是 off-policy，则推荐 $\rho k_3$ 或 $\mathrm{sg}(\rho)k_2$。
3. **若把 KL 写成 stop-grad reward shaping**：在本文讨论的策略梯度主项里，只有 $k_1$ 保持与反向 KL 正则一致的无偏梯度。
4. **低方差 KL loss 与 $k_1$ in reward 在 KL 那一项上可以样本级等价，但整体算法语义并不一样。** 前者把 KL 当成独立正则项；后者会把 KL 带进 advantage、baseline 和信用分配路径里。

## 11. 参考文献

1. Dibya Ghosh. "KL Divergence for Machine Learning". <https://dibyaghosh.com/blog/probability/kldivergence>

2. John Schulman. "Approximating KL Divergence". <https://joschu.net/blog/kl-approx.html>

3. Verl Documentation. "Proximal Policy Optimization (PPO)". <https://verl.readthedocs.io/en/latest/algo/ppo.html>

4. 初七123334. "RLHF/RLVR 训练中的 KL 近似方法浅析（k1 / k2 / k3）". <https://zhuanlan.zhihu.com/p/1966872846212010437>

5. Kezhao Liu, Jason Klein Liu, Mingtao Chen, Yiming Liu. "Rethinking KL Regularization in RLHF: From Value Estimation to Gradient Optimization". arXiv:2510.01555. <https://arxiv.org/abs/2510.01555>

6. Yifan Zhang, Yiping Ji, Gavin Brown, et al. "On the Design of KL-Regularized Policy Gradient Algorithms for LLM Reasoning". arXiv:2505.17508. <https://arxiv.org/abs/2505.17508>

7. Vedant Shah, Johan Obando-Ceron, Vineet Jain, Brian Bartoldson, Bhavya Kailkhura, Sarthak Mittal, Glen Berseth, Pablo Samuel Castro. "A Comedy of Estimators: On KL Regularization in RL Training of LLMs". arXiv:2512.21852. <https://arxiv.org/abs/2512.21852>

```bibtex
@misc{WangZhang2025KLEstimators,
  author       = {Wang, Xihuai and Zhang, Shao},
  title        = {Choosing KL Estimators in RL: From Value Unbiasedness to Gradient Correctness},
  year         = {2025},
  month        = dec,
  day          = {01},
  url          = {https://xihuai18.github.io/reinforcement-learning/2025/12/01/kl-estimators-zh.html}
}
```

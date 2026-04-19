---
layout: post
title: "驯服陈旧数据：LLM 强化学习的异策略训练与单调提升条件"
date: 2025-12-17
description: "本文从单策略采样的性能改进下界出发，推导 LLM 强化学习中的异策略训练问题，扩展到多策略静态/动态混合采样，并把单调提升条件拆成更新增量偏移、采样陈旧性、优势替换误差与支撑集条件。"
categories: reinforcement-learning
lang: zh
en_url: /reinforcement-learning/2025/12/17/offpolicy-en.html
---

> 本文研究一个在大规模 LLM 强化学习里反复出现的问题：当一个训练 batch 同时混入多个历史策略版本生成的数据时，能否仍为 PPO 式更新写出显式的单调改进下界？
>
> 先说结论：在动态混合采样下，基础下界可以概括为"代理目标 - 更新偏移惩罚 - 采样陈旧性惩罚"；若把实际优势估计也纳入分析，还需要再减去一个优势替换误差项。

## 1. 引言：为什么要关心"异策略"训练？

用强化学习训练大语言模型，最直接的做法是同策略（on-policy）训练：模型生成一批数据，立刻用这批数据更新，再用更新后的模型采样下一批。

但在大规模分布式训练里，数百个 GPU 并行采样，模型更新本身也有延迟。新版本发布时，旧版本生成的数据往往还留在队列里：直接丢掉太浪费，继续使用又担心数据已经过时。

这就是**异策略**（off-policy）训练面对的核心问题：**用旧策略采集的数据来更新新策略，什么条件仍足以保证一个可分析的单调提升下界？**

最终会看到，这个下界由三部分共同决定：一个可最大化的代理目标、一个由优化侧裁剪控制的更新偏移惩罚、以及一个由采样侧过滤控制的陈旧性惩罚。

在不少 RLHF / online alignment 设定里，若把 prompt 看作 context、response 看作 action，并忽略长程环境状态演化，问题常被近似为 contextual bandit。本文仍先在一般折扣 MDP 上推导，是为了把"多版本行为策略混合、采样陈旧性、裁剪机制"这些结构一次性写清楚；到第六部分再看 bandit 化之后哪些地方会明显简化、哪些结论会保留。

相关工作里，GePPO 讨论了样本复用下的 off-policy policy improvement guarantee，Decoupled PPO 则把 behavior policy 与 proximal policy 显式区分开。本文的着眼点不同：这里把行为侧进一步展开成多个历史策略版本的动态混合，并把风险拆成"更新增量偏移"与"采样陈旧性"两项。

也可以把这篇文章看成前一篇"三策略视角"文章的延伸：这里把行为侧从单一 $\mu$ 显式展开成历史策略混合 $\{\pi^{(i)}\}$，$\pi_k$ 与 $\pi_{k+1}$ 分别扮演当前参考策略与更新目标。即使没读过上一篇，也只需记住这里的核心思路：把当前更新可控的部分，和行为分布失配造成的部分分开分析。

### 1.1 本文讨论的异策略类型

为了避免把所有问题都称为"陈旧数据"，先把本文覆盖的理论错位拆开：

| 类型 | 数学形式 | 主要破坏的对象 |
| --- | --- | --- |
| 版本陈旧 | 样本来自 $\pi_{k-m}$，更新目标是 $\pi_{k+1}$ | 行为分布与当前近端分布错位 |
| 行为-近端不一致 | $\mu \neq \pi_k$ | PPO ratio 分母不再等于真实采样分布 |
| 多轮样本复用 | 同一批样本被多个更新反复使用 | 后续 epoch 相对原采样分布逐渐 off-policy |
| 混合行为策略 | $\mu=\sum_i w_i\pi^{(i)}$ 或扩展状态中的 $\beta$ | batch 不是来自单一旧策略 |
| 支撑集不一致 | 存在 $\mu(a\mid s)=0,\pi(a\mid s)>0$ | importance ratio 不可定义 |

后文的下界主要处理前四类；第五类是所有重要性修正都必须先满足的基础假设，因此会单独作为支撑覆盖条件讨论。

## 2. 理论基础

### 2.1 基本设定

考虑一个标准的马尔可夫决策过程（MDP），包括状态空间 $\mathcal{S}$、动作空间 $\mathcal{A}$、转移概率 $p(s'\mid s,a)$、奖励函数 $r(s,a)$、初始状态分布 $\rho_0$ 和折扣因子 $\gamma \in (0,1)$。

策略 $\pi$ 的期望累计折扣回报：

$$
J(\pi) := \mathbb{E}\left[\sum_{t=0}^{\infty} \gamma^t r(s_t, a_t) \mid \pi\right]
$$

#### 折扣状态访问分布

$$
d_\pi(s) := (1-\gamma) \sum_{t=0}^{\infty} \gamma^t \Pr(s_t = s \mid \pi)
$$

#### 优势函数

$$
A^\pi(s,a) := Q^\pi(s,a) - V^\pi(s)
$$

#### 全变差距离（TV 距离）

$$
D_{\mathrm{TV}}(\pi, \pi'; s) := \frac{1}{2} \sum_{a \in \mathcal{A}} |\pi(a \mid s) - \pi'(a \mid s)|
$$

本文统一用 $\mid$ 表示条件概率（例如 $\pi(a\mid s)$），$\|\cdot\|$ 保留给范数。

### 2.2 核心工具：性能差分引理

全文的起点是经典的 performance difference lemma：把 $J(\pi)-J(\pi_k)$ 精确写成新策略占据分布下对旧策略优势的期望。这一恒等式可追溯到 Kakade-Langford 的分析，也是 TRPO 推导的起点。

> **引理 2.1（性能差分引理）**
>
> 对任意旧策略 $\pi_k$ 和新策略 $\pi$，性能差异可写成：
>
> $$
> J(\pi) - J(\pi_k) = \frac{1}{1-\gamma} \mathbb{E}_{s \sim d_\pi}\left[ \mathbb{E}_{a \sim \pi(\cdot \mid s)}[A^{\pi_k}(s,a)] \right]
> $$

**直观理解**：新策略带来的改进，等于它自身访问到的状态分布下，按它选动作所得到的平均优势。

### 2.3 本文所有下界默认的理论条件

后文的每个下界都应理解为在若干标准条件下成立的结构性保证。为了不在每个定理里重复，先把核心假设列出来：

1. **支撑覆盖（common support）**：若目标策略可能选择某动作，则行为策略也必须给它正概率。典型写法是

   $$
   \pi(a\mid s)>0 \Rightarrow \mu(a\mid s)>0.
   $$

2. **优势有界**：存在常数 $A_{\max}$，使得 $|A^{\pi_k}(s,a)|\le A_{\max}$。这保证分布替换误差能被 TV / KL 类距离控制。
3. **比率有界或被约束**：重要性比率 $\pi(a\mid s)/\mu(a\mid s)$ 需要存在，并且在理论上不能任意大；若使用裁剪，则得到的是带偏的 surrogate，而不是原目标的无偏估计。
4. **混合索引可定义**：轨迹级混合中，每条轨迹的策略索引固定；若索引在 step/segment 级变化，需要额外建模 index transition。
5. **优势替换误差可控**：实际使用的 $\hat A$ 与理论中的 $A^{\pi_k}$ 之间需要有界偏差；第 5.4 节会单独讨论这一项。

这些假设不是技术细节，而是异策略理论能否成立的边界。特别是支撑覆盖一旦失败，importance ratio 本身就不再是合法对象。

## 3. 单策略采样的性能改进下界

### 3.1 分布不匹配与分布差异控制

性能差分引理有一个实际难题：右侧的期望是在新策略的状态分布 $d_\pi$ 下计算的，而我们只能从旧策略的分布 $d_{\pi_k}$ 中采样。

思路是：把期望分解成"旧分布下的期望"与"偏差项"两部分，再对偏差项加以控制。关键问题是：**状态分布的差异与策略的差异之间有怎样的定量关系？**

#### 状态分布差异的控制

> **引理 3.1（状态分布差异与策略 TV 距离的关系）**
>
> $$
> \|d_\pi - d_{\pi_k}\|_1 \leq \frac{2\gamma}{1-\gamma} \mathbb{E}_{s \sim d_{\pi_k}} \big[ D_{\mathrm{TV}}(\pi, \pi_k; s) \big]
> $$

这里采用的是 **average-divergence / average-TV** 风格的写法。它不是 TRPO 里更常见的 $\max_s D_{\mathrm{TV}}$ 版本，而更接近 CPO / Achiam et al. (2017) 那类"用平均散度刻画性能差距"的表述；这样写更容易落到样本平均，也更方便后文处理多源采样与陈旧性。关于从 $\|d_\pi-d_{\pi_k}\|_1$ 到 average TV 的 proof sketch 见附录 A。

#### 物理意义

策略在动作空间上的微小差异，会被环境动力学"放大"成状态访问分布的差异。上式中的系数 $\frac{2\gamma}{1-\gamma}$ 反映了**时间累积效应**——在长时域任务里（$\gamma$ 接近 1），放大效应更明显。

#### 证明思路

证明通常从折扣访问分布的不动点方程出发，再配合 average-divergence 风格的性能界完成。本文的重点不在复现这条证明链，而在于如何把它推广到多源采样与陈旧数据场景，所以正文只保留思路，附录 A 给出一个简短的 proof sketch。

### 3.2 策略性能改进下界

> **定理 3.2（策略性能改进下界）**
>
> 定义期望优势上界系数 $C_{\pi,\pi_k} := \max_{s} \lvert \mathbb{E}_{a \sim \pi}[A^{\pi_k}(s,a)] \rvert$，则：
>
> $$
> J(\pi) - J(\pi_k) \geq L_{\pi_k}(\pi) - \frac{2\gamma C_{\pi,\pi_k}}{(1-\gamma)^2} \mathbb{E}_{s \sim d_{\pi_k}} \big[ D_{\mathrm{TV}}(\pi, \pi_k; s) \big]
> $$
>
> 其中**代理目标**为：
>
> $$
> L_{\pi_k}(\pi) := \frac{1}{1-\gamma} \mathbb{E}_{s \sim d_{\pi_k}, a \sim \pi_k} \left[ \frac{\pi(a \mid s)}{\pi_k(a \mid s)} A^{\pi_k}(s,a) \right]
> $$

这里的 $L_{\pi_k}(\pi)$ 省略了与 $\pi$ 无关的常数项 $J(\pi_k)$；若写成更接近 TRPO 教科书的形式，就是 $J(\pi_k)+L_{\pi_k}(\pi)$。另外，$C_{\pi,\pi_k}$ 本身依赖新策略 $\pi$，所以它更像一个出现在下界表达式中的结构性系数，不宜在实践里直接当作固定超参数。

这个下界由两部分组成：

1. **代理目标** $L_{\pi_k}(\pi)$：可通过旧策略数据用重要性采样直接估计，它是 TRPO 的经典 surrogate，也是 PPO clipped / penalized 目标的出发点。

2. **策略偏移惩罚**：随着新旧策略 TV 距离的增大而增加，这解释了为何 PPO 等算法需要限制更新幅度。

**核心结论**：这条定理给出了一个显式的改进下界；当右侧为正时，就能保证性能改进。

### 3.3 LLM 有限序列形式与支撑集条件

若把 LLM-RL 写成 prompt-response 形式，令 prompt 为 $x$、回复为

$$
y=(a_1,\ldots,a_T),
$$

行为策略与目标策略的序列概率为

$$
\mu(y\mid x)=\prod_{t=1}^T \mu(a_t\mid x,a_{<t}),
\qquad
\pi(y\mid x)=\prod_{t=1}^T \pi(a_t\mid x,a_{<t}).
$$

对应的序列级重要性比率是

$$
\rho(y\mid x)=\frac{\pi(y\mid x)}{\mu(y\mid x)}
=\prod_{t=1}^T \frac{\pi(a_t\mid x,a_{<t})}{\mu(a_t\mid x,a_{<t})}.
$$

因此

$$
\log \rho(y\mid x)
=
\sum_{t=1}^T
\left[\log \pi(a_t\mid x,a_{<t})-\log \mu(a_t\mid x,a_{<t})\right].
$$

这说明 token 级 log-ratio 的小偏差会沿序列线性累积，而 sequence ratio 本身会以乘积形式放大。长回复、低概率 token、截断采样和 MoE routing 都会让这个比率更容易重尾化。

这里还暴露出一个基础条件：如果某个 token 在行为策略下被 top-$k$ / top-$p$ / mask / EOS 规则置为零概率，而目标策略仍给它正概率，那么 $\rho$ 不可定义。此时不能简单说"加一个 importance ratio 即可修正"；理论上必须重新定义目标支持集，或引入能保证 common support 的平滑 / 混合分布。

## 4. 多策略静态混合采样

### 4.1 问题设定与统一建模（静态混合）

实际训练中，一个批次的数据可能来自多个策略版本 $\{\pi^{(1)}, \ldots, \pi^{(M)}\}$，各版本占比分别为 $\alpha_1, \ldots, \alpha_M$。如何把定理 3.2 推广到这种情形？

**核心思想：扩展状态空间**

做法很直接：**把策略版本索引并入状态空间**。

定义扩展状态空间 $\tilde{\mathcal{S}} := \mathcal{S} \times \mathcal{I}$，其中 $\mathcal{I} = \{1, \ldots, M\}$ 是策略索引集合。在扩展状态 $(s, i)$ 下，**混合行为策略**定义为 $\beta(a \mid s, i) := \pi^{(i)}(a \mid s)$。

索引的演化由**索引转移核** $q(i' \mid i)$ 刻画。扩展 MDP 继承原始 MDP 的奖励和环境转移，索引按 $q(i'\mid i)$ 独立演化。

这个技巧之所以有效，是因为新策略 $\pi$ 在扩展 MDP 上的回报与原始 MDP 中的回报相同，因此可直接套用定理 3.2。

### 4.2 轨迹级混合：结构简化与改进下界

最常见的情形是**每条轨迹只使用一个旧策略**：在轨迹开始时采样索引 $I_0 \sim \alpha$，整条轨迹都用策略 $\pi^{(I_0)}$。此时索引转移核为恒等转移：$q(i' \mid i) = \mathbf{1}_{i'=i}$。

工程上，许多异步 actor-learner 架构中，采样端若按"整条轨迹归属于某个策略快照"的方式组织数据，learner 再混合使用不同版本的整条轨迹更新，这就大致对应这里的**轨迹级混合**。之所以说"大致"，是因为不同系统对"轨迹 / 采样单元"的切分边界未必完全一致。

> **引理 4.1（轨迹级混合的结构简化）**
>
> (a) 扩展状态访问分布分解为：$d_{\beta}(s, i) = \alpha_i \cdot d_{\pi^{(i)}}(s)$
>
> (b) 优势函数还原为：$A^{\beta}((s, i), a) = A^{\pi^{(i)}}(s, a)$

**(b) 的直观理解**：由于索引永不变，从扩展状态 $(s,i)$ 出发的**所有未来轨迹**都由同一个策略 $\pi^{(i)}$ 生成。因此未来的累计回报完全由 $\pi^{(i)}$ 决定，价值函数和优势函数自然还原为 $\pi^{(i)}$ 的对应量。

混合策略的回报也就是各旧策略回报的加权平均：$J_{\mathrm{mix}} = \sum_{i=1}^{M} \alpha_i J(\pi^{(i)})$。

**改进下界**

> **推论 4.2（轨迹级混合的性能改进下界）**
>
> $$
> J(\pi) - \sum_{i=1}^{M} \alpha_i J(\pi^{(i)}) \geq \sum_{i=1}^{M} \alpha_i L_{\pi^{(i)}}(\pi) - \frac{2\gamma \max_i C_{\pi, \pi^{(i)}}}{(1-\gamma)^2} \sum_{i=1}^{M} \alpha_i \mathbb{E}_{s \sim d_{\pi^{(i)}}} \big[ D_{\mathrm{TV}}(\pi, \pi^{(i)}; s) \big]
> $$

这里取 $\max_i C_{\pi,\pi^{(i)}}$ 只是为了写得更紧凑；更细致的写法是让每个分量各自带自己的 $C_i$ 再加权求和。也就是说，这里牺牲了一点紧度，换来更统一的展示形式。

结论是：只要对每条轨迹用对应旧策略的重要性比率构造损失，并控制新策略与各旧策略的偏移，混合训练就仍有明确的改进下界。

## 5. 动态混合采样与单调提升条件

### 5.1 问题与统一建模（动态混合）

上一节讨论的是**静态混合**——混合权重 $\alpha_i$ 固定不变。本节考虑更一般的**动态混合**——新策略发布后，采样逐步由新策略接管的过程。

前面的结论刻画了"新策略相对于混合行为策略"的改进。但实际训练中真正关心的是：**每轮更新后的最新策略 $\pi_{k+1}$ 相对于上一轮的 $\pi_k$ 是否单调提升？**

$$
J(\pi_{k+1}) \geq J(\pi_k)
$$

#### 统一建模框架

动态混合采样的两种典型形式都可以用索引转移核 $q(i'\mid i)$ 统一刻画：

**轨迹级混合**（对应常规异步训练的一种抽象；索引恒等转移）：$q(i'\mid i) = \mathbf{1}\{i'=i\}$

**步/段级混合**（partial rollout，也可理解为段式采样的一种抽象；允许切换）：$q(i'\mid i) = (1-\sigma(i))\mathbf{1}\{i'=i\} + \sigma(i)\kappa(i'\mid i)$

其中 $\sigma(i)$ 是切换概率，$\kappa(\cdot\mid i)$ 是目标索引分布。

### 5.2 分解与单调提升下界

引入混合回报 $J_{\mathrm{mix}}^{(k)}$ 作为中间桥梁，性能差异可以分解为：

$$
J(\pi_{k+1}) - J(\pi_k) = \underbrace{[J(\pi_{k+1}) - J_{\mathrm{mix}}^{(k)}]}_{\text{相对混合策略的改进}} + \underbrace{[J_{\mathrm{mix}}^{(k)} - J(\pi_k)]}_{\text{混合偏差项}}
$$

第一项可用定理 3.2 处理。第二项是**混合偏差项**：先把 $J_{\mathrm{mix}}^{(k)} - J(\pi_k)$ 写成各个 $J(\pi^{(i)}) - J(\pi_k)$ 的加权和，再对每一项套用基于 TV 距离的两策略下界，最后用 $\|A^{\pi_k}\|_\infty$ 统一收口，得到：

$$
J_{\mathrm{mix}}^{(k)} - J(\pi_k) \geq -\frac{2\|A^{\pi_k}\|_\infty}{1-\gamma} \mathbb{E}_{(s,i)\sim d_{\beta^{(k)}}} \big[ D_{\mathrm{TV}}(\pi^{(i)}, \pi_k; s) \big]
$$

#### 单调提升下界

合并以上结果，得到核心定理：

> **定理 5.1（动态混合采样下的单调提升下界）**
>
> $$
> \begin{aligned}
> J(\pi_{k+1}) - J(\pi_k) \geq\;& L_{\beta^{(k)}}(\pi_{k+1}) \\
> &- \frac{2\gamma C_{\pi_{k+1},\beta^{(k)}}}{(1-\gamma)^2} \mathbb{E}_{(s,i)\sim d_{\beta^{(k)}}} \big[ D_{\mathrm{TV}}(\pi_{k+1}, \pi^{(i)}; s) \big] \\
> &- \frac{2\|A^{\pi_k}\|_\infty}{1-\gamma} \mathbb{E}_{(s,i)\sim d_{\beta^{(k)}}} \big[ D_{\mathrm{TV}}(\pi^{(i)}, \pi_k; s) \big]
> \end{aligned}
> $$

其中 $L_{\beta^{(k)}}(\pi_{k+1})$ 表示"相对行为策略 $\beta^{(k)}$ 的代理目标"（与第三部分的 $L_{\pi_k}(\pi)$ 同形，只是把行为策略从单一 $\pi_k$ 推广到混合 $\beta^{(k)}$）。

更具体地，可写为

$$
L_{\beta^{(k)}}(\pi_{k+1}) := \frac{1}{1-\gamma} \mathbb{E}_{(s,i)\sim d_{\beta^{(k)}},\, a\sim \pi^{(i)}(\cdot\mid s)}\left[\frac{\pi_{k+1}(a\mid s)}{\pi^{(i)}(a\mid s)}\,A^{\beta^{(k)}}((s,i),a)\right].
$$

类似地，记

$$
C_{\pi_{k+1},\beta^{(k)}} := \max_{(s,i)}\left|\mathbb{E}_{a\sim \pi_{k+1}(\cdot\mid s)}\big[A^{\beta^{(k)}}((s,i),a)\big]\right|.
$$

下界里有两个惩罚项，对应**双重控制**：

- **更新偏移惩罚**：新策略 $\pi_{k+1}$ 相对于采样来源策略 $\pi^{(i)}$ 的偏移；
- **采样陈旧性惩罚**：采样来源策略 $\pi^{(i)}$ 相对于当前策略 $\pi_k$ 的陈旧性。

### 5.3 直接约束为何不可行：三角不等式分解

先加一句限定：下面讨论的不可行性针对的是"试图对每个历史来源都施加统一硬 trust-region 约束"这种解释，并不等于说 PPO-Clip 本身显式实现了这样的约束。

定理 5.1 中的更新偏移惩罚项看起来可以通过约束 $D_{\mathrm{TV}}(\pi_{k+1}, \pi^{(i)}; s)$ 控制；但若把它理解成上述统一硬约束，就会撞上一个实际的不可行性问题：

> **观察 5.2（统一硬 trust-region 的不可行性）**
>
> 设混合采样包含两个旧策略 $\pi^{(1)}$ 和 $\pi^{(2)}$，若存在某个状态 $s$ 使得 $D_{\mathrm{TV}}(\pi^{(1)}, \pi^{(2)}; s) > 2\delta$，则不存在任何策略 $\pi_{k+1}$ 能同时满足 $D_{\mathrm{TV}}(\pi_{k+1}, \pi^{(1)}; s) \leq \delta$ 和 $D_{\mathrm{TV}}(\pi_{k+1}, \pi^{(2)}; s) \leq \delta$。

#### 证明

由三角不等式，若同时满足这两个约束，则 $D_{\mathrm{TV}}(\pi^{(1)}, \pi^{(2)}; s) \leq 2\delta$，矛盾。

#### 问题根源

更新偏移惩罚项把 $\pi_{k+1}$ 与历史策略族 $\{\pi^{(i)}\}$ 直接耦合，而后者的内部结构是历史训练的产物，不受当前更新控制。

#### 三角不等式分解

解法是利用 TV 距离的三角不等式：

$$
D_{\mathrm{TV}}(\pi_{k+1}, \pi^{(i)}; s) \leq D_{\mathrm{TV}}(\pi_{k+1}, \pi_k; s) + D_{\mathrm{TV}}(\pi_k, \pi^{(i)}; s)
$$

把耦合约束拆成两个独立部分：

- **更新增量偏移** $D_{\mathrm{TV}}(\pi_{k+1}, \pi_k; s)$：新策略相对于当前策略的偏离，**由优化侧控制**；
- **采样陈旧性** $D_{\mathrm{TV}}(\pi_k, \pi^{(i)}; s)$：当前策略相对于各旧策略的偏离，**由采样侧控制**。

定义：

$$
U_k := \mathbb{E}_{(s,i)\sim d_{\beta^{(k)}}} \big[D_{\mathrm{TV}}(\pi_{k+1}, \pi_k; s)\big], \quad S_k := \mathbb{E}_{(s,i)\sim d_{\beta^{(k)}}} \big[D_{\mathrm{TV}}(\pi_k, \pi^{(i)}; s)\big]
$$

> **推论 5.3（分解后的单调提升下界）**
>
> $$
> J(\pi_{k+1}) - J(\pi_k) \geq L_{\beta^{(k)}}(\pi_{k+1}) - \frac{2\gamma C_{\pi_{k+1},\beta^{(k)}}}{(1-\gamma)^2} U_k - \left( \frac{2\gamma C_{\pi_{k+1},\beta^{(k)}}}{(1-\gamma)^2} + \frac{2\|A^{\pi_k}\|_\infty}{1-\gamma} \right) S_k
> $$

#### 为什么分解能解决问题？

关键在于：分解后的 $U_k$ 只涉及新策略 $\pi_{k+1}$ 和当前策略 $\pi_k$，**与旧策略族 $\{\pi^{(i)}\}$ 的结构完全无关**。因此无论旧策略之间差异多大，约束 $U_k$ 都是可行的——这正是观察 5.2 不可行性问题的解法。

对应的理论原则就是**职责分离**：

| 控制项 | 理论含义 | 约束方式 |
| --- | --- | --- |
| $U_k$ | 当前更新漂移 | 约束 $\pi_{k+1}$ 相对 $\pi_k$ 的偏移 |
| $S_k$ | 采样陈旧漂移 | 约束行为策略族相对 $\pi_k$ 的偏移 |

### 5.4 优势替换误差：异策略不只是 ratio 问题

上面的下界默认 surrogate 使用的是理论优势 $A^{\pi_k}(s,a)$。但在 LLM-RL 中，实际进入 loss 的往往是批次估计量、critic / GAE、verifier reward 的归一化，或 group-relative advantage。记实际使用的优势为 $\hat A(s,a)$，则即使 importance ratio 写对了，也还会出现优势替换误差：

$$
\mathbb{E}_{\mu}\left[\rho(s,a)\hat A(s,a)\right]
-
\mathbb{E}_{\mu}\left[\rho(s,a)A^{\pi_k}(s,a)\right]
=
\mathbb{E}_{\mu}\left[\rho(s,a)(\hat A(s,a)-A^{\pi_k}(s,a))\right].
$$

若假设 $|\rho(s,a)|\le M$ 且 $|\hat A(s,a)-A^{\pi_k}(s,a)|\le \epsilon_A$，则有

$$
\left|
\mathbb{E}_{\mu}\left[\rho(s,a)(\hat A(s,a)-A^{\pi_k}(s,a))\right]
\right|
\le M\epsilon_A.
$$

因此，异策略误差至少包含三类：目标更新漂移 $U_k$、行为采样陈旧性 $S_k$、以及优势替换误差 $\epsilon_A$。只讨论 actor ratio 会漏掉第三项；在理论上，$\hat A$ 是否仍接近 $A^{\pi_k}$ 是单调提升条件的一部分。

## 6. 轨迹级与步/段级混合的比较

### 6.1 机制差异与估计影响

两类混合机制的本质区别在于索引转移核的结构：

- **轨迹级混合**：$q(i'\mid i) = \mathbf{1}\{i'=i\}$，索引永不改变；
- **步/段级混合**：$\sigma(i) > 0$，允许轨迹内切换。

与常见工程术语的对应关系：

- **轨迹级混合**可大致理解为**常规异步训练**的理想化抽象：数据按整条轨迹 / episode 归属于某个策略版本；
- **步/段级混合**可大致理解为 **partial rollout（段式采样）** 的抽象：由于 actor 与 learner 异步，段边界处可能刷新到新策略版本，用索引转移核允许"轨迹内部版本切换"，可以更贴切地刻画这种现象。APRIL 提供了这类系统设计的一个代表性例子，但它的主要贡献是缓解长尾 rollout 的系统瓶颈，而不是给出本文所需的单调改进理论。

关键分水岭在于**引理 4.1 的结构简化是否成立**：轨迹级混合满足优势函数还原；步/段级混合一般不满足，因为未来回报会受索引转移核影响。

#### 采样陈旧性 $S_k$ 的差异

**轨迹级混合**的陈旧性来源于：混合权重 $\alpha_i^{(k)}$ 在新策略发布后仍对旧策略保留一定比例。

**步/段级混合**在一个简化模型下具有**指数压缩效应**：设索引一旦从旧版本切到新版本就不再切回，每一步以概率 $\sigma$ 完成切换，则折扣访问分布下旧索引的边缘质量为

$$
(1-\gamma)\sum_{t=0}^{\infty}[\gamma(1-\sigma)]^t = \frac{1-\gamma}{1-\gamma(1-\sigma)}.
$$

只要 $\sigma \gg 1-\gamma$，旧策略的权重就会被明显压缩。

#### 代理目标估计的差异

**轨迹级混合**：优势函数还原为 $A^{\pi^{(i)}}(s,a)$，估计路径清晰。

**步/段级混合的优势替代偏差**：若沿用单策略优势估计，会产生系统性偏差。原因是 $A^{\beta^{(k)}}((s,i),a)$ 需要对未来索引切换取期望，而 $A^{\pi^{(i)}}(s,a)$ 隐含了"未来始终沿用 $\pi^{(i)}$"的假设。

#### Bandit 设定下的统一

在单步 episode 的 LLM 训练中没有后续状态转移，两类机制的估计问题就此统一，不存在上述偏差。

### 6.2 风险与适用场景

步/段级混合还有一个隐患：即使单步重要性比值被裁剪，长轨迹下多步噪声叠加仍会放大梯度估计方差。每次更新的策略变化幅度较大时，轨迹内部的"行为突变"可能引出更重尾的比值分布。这也是表 6.1 中"每次更新策略变化幅度大"场景推荐轨迹级混合的原因。

#### 适用场景

#### 表 6.1　两类混合机制的适用场景

| 场景特征                 | 推荐机制 | 理由             |
| ------------------------ | -------- | ---------------- |
| 长轨迹、高频更新、强异步 | 步/段级  | 可显著压缩 $S_k$ |
| 短轨迹（非Bandit）       | 轨迹级   | $S_k$ 自然较低   |
| 每次更新策略变化幅度大   | 轨迹级   | 避免方差放大     |
| 单步episode（Bandit）    | 均可     | 按实现便利选择   |
| 需要折中方案             | 段级     | 在自然边界切换   |

**核心权衡**：步/段级混合在采样侧更强（快速去陈旧），轨迹级混合在估计侧更稳（代理目标更容易估计）。

## 7. 裁剪机制的理论基础

### 7.1 从 TV 距离到样本可控量

推论 5.3 告诉我们，要保证单调提升，需要控制更新增量偏移 $U_k = \mathbb{E}[D_{\mathrm{TV}}(\pi_{k+1}, \pi_k; s)]$。但 TV 距离是分布层面的量，如何用样本来控制？

连接理论和样本的是下面这个恒等式：

> **引理 7.1（TV 距离的比值差表示）**
>
> 设策略 $\pi_1$ 的支撑覆盖 $\pi$ 和 $\pi_2$ 的支撑，则对任意状态分布 $\mu$：
>
> $$
> \mathbb{E}_{s\sim \mu} \big[D_{\mathrm{TV}}(\pi, \pi_2; s)\big] = \frac{1}{2} \mathbb{E}_{s\sim \mu, a\sim\pi_1(\cdot\mid s)} \left| \frac{\pi(a\mid s)}{\pi_1(a\mid s)} - \frac{\pi_2(a\mid s)}{\pi_1(a\mid s)} \right|
> $$

注意：这里默认作为分母的行为策略在参与训练的动作上具有支撑覆盖。对 LLM 来说，这意味着若推理端使用带硬截断的 top-k / top-p 采样而不做平滑，一些比值可能根本无定义；第 8 节会回到这个问题。

#### 直观理解

左侧是两个分布之间的 TV 距离（要遍历所有动作），右侧是在 $\pi_1$ 下采样时两个重要性比值之差的绝对值。这样就能用样本估计和控制 TV 距离。

#### $U_k$ 的样本表示

利用引理 7.1，取 $\pi = \pi_{k+1}$，$\pi_2 = \pi_k$，$\pi_1 = \pi^{(i)}$（采样来源策略），可得：

$$
U_k = \frac{1}{2} \mathbb{E}_{(s,i) \sim d_{\beta^{(k)}}, a \sim \pi^{(i)}(\cdot\mid s)} \left| \frac{\pi_{k+1}(a\mid s)}{\pi^{(i)}(a\mid s)} - \frac{\pi_k(a\mid s)}{\pi^{(i)}(a\mid s)} \right|
$$

记 $\rho_{k+1} := \frac{\pi_{k+1}(a\mid s)}{\pi^{(i)}(a\mid s)}$ 和 $\rho_k := \frac{\pi_k(a\mid s)}{\pi^{(i)}(a\mid s)}$，则：

$$
U_k = \frac{1}{2} \mathbb{E}_{(s,i,a) \sim \text{训练数据}} \big| \rho_{k+1} - \rho_k \big|
$$

这意味着：**若在理论上硬性要求每个样本都满足 $\lvert\rho_{k+1} - \rho_k\rvert \leq \epsilon$，就能保证 $U_k \leq \epsilon/2$**。

### 7.2 约束 $U_k$：两种裁剪方式

#### 方法一：直接约束比值差

对每个样本 $(s, i, a)$ 要求：

$$
\left| \frac{\pi_{k+1}(a\mid s)}{\pi^{(i)}(a\mid s)} - \frac{\pi_k(a\mid s)}{\pi^{(i)}(a\mid s)} \right| \leq \epsilon
$$

即裁剪区间为 $\left[\frac{\pi_k(a\mid s)}{\pi^{(i)}(a\mid s)} - \epsilon, \frac{\pi_k(a\mid s)}{\pi^{(i)}(a\mid s)} + \epsilon\right]$，**裁剪中心是 $\rho_k$ 而不是 1**。

#### 方法二：约束增量比值

注意到 $\rho_{k+1} - \rho_k = \rho_k \cdot \left(\frac{\pi_{k+1}}{\pi_k} - 1\right)$，所以

$$
|\rho_{k+1} - \rho_k| = \rho_k \cdot \left|\frac{\pi_{k+1}(a\mid s)}{\pi_k(a\mid s)} - 1\right|
$$

如果进一步在理论上硬性约束 $\left\lvert\frac{\pi_{k+1}(a\mid s)}{\pi_k(a\mid s)} - 1\right\rvert \leq \epsilon$，由于 $\mathbb{E}_{a\sim\pi^{(i)}}[\rho_k] = 1$，可以证明 $U_k \leq \epsilon/2$。

这种方法直接对 $\pi_{k+1}/\pi_k$ 以 1 为中心裁剪，**裁剪约束本身不依赖旧策略 $\pi^{(i)}$**。但若采用后文的 $\hat{A}=\rho_k\cdot A^{\beta^{(k)}}$，仍需要每条样本的行为概率 $\pi^{(i)}(a\mid s)$（或记录的 logprob）来计算 $\rho_k$。

先强调一点：上面两条都是**理论上的逐样本硬约束**。下面写出的 clipped surrogate 只是实践中的近似实现，目的是把 $U_k$ 控制在可接受的范围内，而不是让每一步优化都自动满足严格保证。设当前样本来自旧策略 $\pi^{(i)}$，记：

- $\rho_{k+1} = \frac{\pi_{k+1}(a\mid s)}{\pi^{(i)}(a\mid s)}$（新策略相对采样策略的比值）；
- $\rho_k = \frac{\pi_k(a\mid s)}{\pi^{(i)}(a\mid s)}$（当前策略相对采样策略的比值）；
- $r = \frac{\pi_{k+1}(a\mid s)}{\pi_k(a\mid s)}$（新策略相对当前策略的增量比值）。

说明：若是**轨迹级混合**（索引不变），则 $A^{\beta^{(k)}}((s,i),a)=A^{\pi^{(i)}}(s,a)$，可直接用每条轨迹对应旧策略的优势估计；若是**步/段级混合**，直接用 $A^{\pi^{(i)}}$ 代替 $A^{\beta^{(k)}}$ 会引入优势替代偏差（第六部分详述），需要用能反映未来索引切换的优势 / 价值估计。

#### 标准 PPO（轨迹级混合时）

以 1 为中心裁剪 $\rho_{k+1}$

$$
L^{\mathrm{PPO}} = \mathbb{E} \left[ \min\left( \rho_{k+1} \cdot A^{\pi^{(i)}}, \; \mathrm{clip}(\rho_{k+1}, 1-\epsilon, 1+\epsilon) \cdot A^{\pi^{(i)}} \right) \right]
$$

#### 方法一

以 $\rho_k$ 为中心裁剪 $\rho_{k+1}$

$$
L^{\mathrm{M1}} = \mathbb{E} \left[ \min\left( \rho_{k+1} \cdot A^{\beta^{(k)}}, \; \mathrm{clip}(\rho_{k+1}, \rho_k-\epsilon, \rho_k+\epsilon) \cdot A^{\beta^{(k)}} \right) \right]
$$

#### 方法二

以 1 为中心裁剪增量比值 $r$

$$
L^{\mathrm{M2}} = \mathbb{E} \left[ \min\left( r \cdot \hat{A}, \; \mathrm{clip}(r, 1-\epsilon, 1+\epsilon) \cdot \hat{A} \right) \right]
$$

其中 $\hat{A} = \rho_k \cdot A^{\beta^{(k)}}$ 是经过重要性加权的优势估计。

### 7.3 方法对比与选型

#### 表 7.1　三种裁剪机制的对比

| 方法    | 裁剪变量                           | 裁剪中心                   | 裁剪区间                             | 更自然对应的偏移对象                |
| ------- | ---------------------------------- | -------------------------- | ------------------------------------ | ----------------------------------- |
| 标准PPO | $\rho_{k+1} = \pi_{k+1}/\pi^{(i)}$ | $1$                        | $[1-\epsilon, 1+\epsilon]$           | $\pi_{k+1}$ 相对 $\pi^{(i)}$ 的偏移 |
| 方法一  | $\rho_{k+1} = \pi_{k+1}/\pi^{(i)}$ | $\rho_k = \pi_k/\pi^{(i)}$ | $[\rho_k-\epsilon, \rho_k+\epsilon]$ | $\pi_{k+1}$ 相对 $\pi_k$ 的偏移     |
| 方法二  | $r = \pi_{k+1}/\pi_k$              | $1$                        | $[1-\epsilon, 1+\epsilon]$           | $\pi_{k+1}$ 相对 $\pi_k$ 的偏移     |

对前文的逐样本硬约束版本而言，方法一 / 二确实直接控制 $D_{\mathrm{TV}}(\pi_{k+1}, \pi_k)$；而这里的 clipped surrogate 对应的是更温和的说法：它们分别对不同的偏移对象施加主要的优化压力。

#### 标准 PPO 的根本问题（多策略混合）

沿用单源 PPO 的 trust-region 直觉，标准 PPO 的 clip 目标最自然的解读是：它试图抑制新策略相对每个采样来源策略 $\pi^{(i)}$ 的继续偏离。但 PPO-Clip 本身并不显式施加 TV / KL 约束，而是通过裁剪移除"继续远离行为策略"的收益；当各旧策略 $\pi^{(1)}, \pi^{(2)}, \ldots$ 之间差异显著时，这种优化压力容易被最陈旧的策略所牵制。

#### 方法一与方法二的共同优势

对前文的逐样本硬约束版本而言，方法一和方法二直接控制的是 $D_{\mathrm{TV}}(\pi_{k+1}, \pi_k)$。落到 clipped surrogate 的实践版本时，更准确的说法是：它们把主要优化压力从"同时贴近每个采样来源策略"改成"围绕当前策略 $\pi_k$ 控制更新增量"。由于 $\pi_k$ 是唯一确定的，这个目标对所有来源样本都一致，就绕开了统一硬 trust-region 约束的结构性困难。

#### 方法一 vs 方法二

| 比较维度                   | 方法一（自适应裁剪）                            | 方法二（增量裁剪）                                                                                             |
| -------------------------- | ----------------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| 陈旧样本（$\rho_k \gg 1$） | 自动收紧约束，更保守                            | 可能产生大梯度方差                                                                                             |
| LLM大词表低概率token       | 允许较大绝对变化（加法型）                      | 绝对变化受限（乘法型）                                                                                         |
| 实现复杂度                 | 需存储 $\pi^{(i)}(a\mid s)$ 和 $\pi_k(a\mid s)$ | 需 $\pi_k(a\mid s)$ 与 $\pi^{(i)}(a\mid s)$（或存储的 logprob）以计算 $\rho_k$；裁剪本身仅用 $\pi_{k+1}/\pi_k$ |
| 优势函数                   | 使用 $A^{\beta^{(k)}}$                          | 使用加权优势 $\rho_k \cdot A^{\beta^{(k)}}$                                                                    |

##### 维度一：陈旧样本处理

样本来自很旧的策略时，$\rho_k = \pi_k/\pi^{(i)}$ 可能很大。

- 方法二的被积函数为 $\rho_k \cdot \lvert r - 1\rvert$，即便 $\lvert r-1\rvert \leq \epsilon$，被积函数仍可能达到 $\epsilon \cdot \rho_k$，产生尖峰。
- 方法一直接约束 $\lvert\rho_{k+1} - \rho_k\rvert \leq \epsilon$，被积函数上界恒为 $\epsilon$，不会被 $\rho_k$ 放大。

##### 维度二：LLM 大词表问题

大语言模型词表规模巨大，大量 token 的概率极小。

- 方法二约束 $\pi_{k+1} \in [(1-\epsilon)\pi_k, (1+\epsilon)\pi_k]$，这是**乘法型约束**：若 $\pi_k(a\mid s) = 10^{-6}$，允许的绝对变化仅 $\epsilon \times 10^{-6}$。
- 方法一约束 $\lvert\pi_{k+1} - \pi_k\rvert \leq \epsilon \cdot \pi^{(i)}$，**尺度由采样策略概率 $\pi^{(i)}$ 决定**：若该 token 在旧策略下概率较高（例如 $\pi^{(i)}(a\mid s) = 0.1$），即便当前概率很低，也允许较快提升；当然，前提是该 token 在采样分布中有足够可观测的质量。

### 7.4 采样陈旧性控制与操作含义

前面讨论的都是优化侧的裁剪，但单调提升下界还含有采样陈旧性 $S_k$ 一项。本节处理采样侧的职责，并回到裁剪作为整体操作的准确含义。

#### 采样陈旧性的控制

推论 5.3 表明 $S_k$ 同样影响单调提升下界，但它**无法通过优化侧的裁剪来控制**，需要由采样系统实现：

- **丢弃陈旧数据**：设阈值 $\epsilon_{\mathrm{stale}}$，对每个样本计算 $\lvert\rho_k - 1\rvert = \lvert\pi_k(a\mid s)/\pi^{(i)}(a\mid s) - 1\rvert$，丢弃超过阈值的样本。
- **控制策略版本窗口**：限制混合采样的旧策略版本数，例如只使用最近 $W$ 个版本的数据。

#### 裁剪的操作含义

最后要澄清裁剪与理论下界的关系。

推论 5.3 里，$U_k$ 的系数 $C_{\pi_{k+1},\beta^{(k)}}$ 依赖于新策略 $\pi_{k+1}$，所以惩罚项**不能简单地换成常数**。正确的操作含义是：

> **在 $U_k \leq \epsilon/2$ 的约束下，最大化代理目标 $L_{\beta^{(k)}}(\pi_{k+1})$**

裁剪目标函数可以看作这一约束优化的**近似实现**——通过裁剪近似限制更新幅度，让 $U_k$ 保持可控；在此基础上再通过梯度上升提升代理目标，尽量贴近前述单调改进分析。

#### 本节小结

本节建立了裁剪机制的理论基础：

1. **引理 7.1** 把 TV 距离转化为样本层面的比值差，是连接理论与实现的桥梁；
2. **两种约束方法**：方法一（自适应裁剪中心）和方法二（固定增量裁剪）的**硬约束版本**都能保证 $U_k \leq \epsilon/2$；实践中的 clipped surrogate 是对这一思路的近似实现；
3. **与标准 PPO 对比**：沿用单源 trust-region 直觉，标准 PPO 的裁剪最自然地围绕"新策略相对行为策略的偏移"施加优化压力；方法一 / 二则把压力改为围绕当前策略 $\pi_k$ 控制更新增量，由此绕开多来源样本带来的结构性困难；
4. **方法选择**：陈旧性高或 LLM 大词表场景推荐方法一；若更希望"裁剪中心不再依赖旧策略族"，可选方法二（但仍需数据侧提供行为 logprob 来计算 $\rho_k$）；
5. **$S_k$ 的控制**由采样侧负责，通过数据过滤和版本窗口实现；
6. **裁剪是约束优化**：在 $U_k$ 约束下最大化代理目标。

## 8. 训推不一致的处理

### 8.1 背景与有效陈旧性

在大规模分布式训练中，推理端和训练端的策略可能不一致：

- **数值实现差异**：softmax 归一化、量化、核融合等；
- **解码规则差异**：温度缩放、top-p / top-k 采样等。

设训练侧建模的行为策略为 $\pi^{(i)}$，推理端实际采样的策略为 $\hat{\pi}^{(i)}$。

这里讨论的是"行为策略 vs. 当前训练策略"的失配，而不是 RLHF 里常见的"当前策略 vs. reference model"的 KL 正则——后者是另一条正则化轴线。

#### 有效陈旧性

定义**有效陈旧性**：

$$
\hat{S}_k := \mathbb{E}_{(s,i) \sim d_{\hat{\beta}^{(k)}}} \big[ D_{\mathrm{TV}}(\pi_k, \hat{\pi}^{(i)}; s) \big]
$$

这个定义同时涵盖了版本陈旧性与训推实现差异。

### 8.2 理论上的有效陈旧性控制

由引理 7.1，$\hat{S}_k$ 可写成样本级可计算的形式。给定阈值 $\epsilon_{\mathrm{stale}}$，若训练只使用满足 $\lvert\pi_k(a\mid s)/\hat{\pi}^{(i)}(a\mid s) - 1\rvert \leq \epsilon_{\mathrm{stale}}$ 的样本，则**被保留样本的条件分布**上的有效陈旧性（记为 $\hat{S}_k^{\mathrm{eff}}$）可被控制在 $\epsilon_{\mathrm{stale}}/2$ 以内。换言之，这里控制的是过滤后的训练分布，而不是原始采样分布上的 $\hat{S}_k$。

#### 关键理论条件

1. **行为分母对齐**：损失中的行为概率要使用推理端记录的 $\hat{\pi}^{(i)}(a\mid s)$；
2. **概率平滑**：若推理端有截断（如 top-k），需通过平滑等方式保证比值合法，并满足引理 7.1 所需的支撑覆盖条件。

## 9. 总结：理论判据

#### 核心理论框架

单调提升下界的结构为：

$$
J(\pi_{k+1}) - J(\pi_k) \geq \underbrace{L_{\beta^{(k)}}(\pi_{k+1})}_{\text{代理目标}} - \underbrace{C_1 \cdot U_k}_{\text{更新偏移惩罚}} - \underbrace{C_2 \cdot S_k}_{\text{采样陈旧性惩罚}}
$$

这里的 $C_1, C_2$ 只是对前文理论常数的压缩记号，用来概括下界结构。具体来说，

$$
C_1 = \frac{2\gamma C_{\pi_{k+1},\beta^{(k)}}}{(1-\gamma)^2},
\qquad
C_2 = C_1 + \frac{2\|A^{\pi_k}\|_\infty}{1-\gamma},
$$

正文为了突出结构，把记号压缩成了 $C_1, C_2$。它们不是实践里可以自由调节的超参数。若把第 5.4 节的优势替换误差显式纳入，则可把结构进一步写成

$$
J(\pi_{k+1}) - J(\pi_k)
\gtrsim
L_{\beta^{(k)}}(\pi_{k+1})
- C_1 U_k
- C_2 S_k
- C_3 \epsilon_A,
$$

其中 $C_3$ 由重要性比率上界等条件决定。这不是新的算法项，而是提醒我们：单调提升条件还依赖实际优势估计是否足够接近理论优势。

#### 理论职责分离

| 控制项 | 理论含义 | 约束方式 | 对应的分布对象 |
| ------ | -------- | -------- | -------------- |
| $U_k$  | 当前更新漂移 | 对 $\pi_{k+1}$ 相对 $\pi_k$ 的偏移加约束 | 目标策略与近端策略 |
| $S_k$  | 采样陈旧漂移 | 限制行为策略与近端策略的距离 | 行为分布与近端分布 |
| $\epsilon_A$ | 优势替换误差 | 控制 $\hat A$ 与 $A^{\pi_k}$ 的偏差 | 优势估计与理论优势 |

#### 裁剪项的理论角色

| 裁剪对象 | 理论作用 | 代价 |
| --- | --- | --- |
| 直接约束 $\pi_{k+1}/\pi^{(i)}$ | 同时限制更新漂移与部分陈旧性 | 对每个行为版本依赖更强 |
| 约束增量 $\pi_{k+1}/\pi_k$ | 只控制当前更新漂移 $U_k$ | 仍需额外控制 $S_k$ |
| 对比率做硬裁剪或过滤 | 得到更保守的 surrogate 上界 | 引入有偏目标，需重新解释单调条件 |

#### 训推不一致的理论处理

- 若训练侧建模分布是 $\pi^{(i)}$，但真实采样分布是 $\hat{\pi}^{(i)}$，则下界中的行为分布应使用 $\hat{\pi}^{(i)}$。
- 过滤或裁剪只能控制条件分布上的 $\hat S_k^{\mathrm{eff}}$，不等于原始采样分布上的 $\hat S_k$ 自动变小。
- 若解码截断导致 common support 失败，则需要先修正支持集假设，再讨论 importance correction。

## 附录

### A. 从状态分布差异到 average TV 的 proof sketch

证明引理 3.1 的常见起点是折扣状态访问分布的不动点方程：

$$
d_\pi = (1-\gamma)\rho_0 + \gamma P_\pi^\top d_\pi,
\qquad
d_{\pi_k} = (1-\gamma)\rho_0 + \gamma P_{\pi_k}^\top d_{\pi_k}.
$$

两式相减并整理，就可以把 $d_\pi-d_{\pi_k}$ 写成"策略诱导转移核差异"作用在旧分布上的结果。随后对 $\ell_1$ 范数取上界，利用马尔可夫核在 $\ell_1$ 下的不扩张性，以及

$$
\|(P_\pi-P_{\pi_k})(\cdot\mid s)\|_1 \le 2D_{\mathrm{TV}}(\pi,\pi_k;s),
$$

就能把状态分布差异控制到旧分布下的 average TV 上：

$$
\|d_\pi-d_{\pi_k}\|_1 \le \frac{2\gamma}{1-\gamma}\,\mathbb{E}_{s\sim d_{\pi_k}}\big[D_{\mathrm{TV}}(\pi,\pi_k;s)\big].
$$

这里省略了线性算子展开和常数整理；本文后续只用到最终的 average-TV 形式。

### B. 关键符号速查表

| 符号                                              | 含义                               |
| ------------------------------------------------- | ---------------------------------- |
| $\pi_k$, $\pi^{(i)}$                              | 第 $k$ 轮最新策略，第 $i$ 个旧策略 |
| $d_\pi(s)$, $A^\pi(s,a)$                          | 折扣状态访问分布，优势函数         |
| $D_{\mathrm{TV}}(\pi, \pi'; s)$                   | 两策略在状态 $s$ 上的 TV 距离      |
| $\beta^{(k)}(a \mid s, i) := \pi^{(i)}(a \mid s)$ | 第 $k$ 轮混合行为策略              |
| $q(i' \mid i)$, $\alpha_i^{(k)}$                  | 索引转移核，索引初始分布           |
| $U_k$, $S_k$                                      | 更新增量偏移，采样陈旧性           |
| $\epsilon$, $\epsilon_{\mathrm{stale}}$, $W$      | 裁剪半径，陈旧性阈值，版本窗口     |
| $C_{\pi,\pi_k}$                                   | 期望优势上界系数                   |

## 参考文献

1. John Schulman, Sergey Levine, Philipp Moritz, Michael I. Jordan, Pieter Abbeel. "Trust Region Policy Optimization" (TRPO). arXiv:1502.05477. <https://arxiv.org/abs/1502.05477>

2. Joshua Achiam, David Held, Aviv Tamar, Pieter Abbeel. "Constrained Policy Optimization" (CPO). arXiv:1705.10528. <https://arxiv.org/abs/1705.10528>

3. John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, Oleg Klimov. "Proximal Policy Optimization Algorithms" (PPO). arXiv:1707.06347. <https://arxiv.org/abs/1707.06347>

4. James Queeney, Ioannis Ch. Paschalidis, Christos G. Cassandras. "Generalized Proximal Policy Optimization with Sample Reuse" (GePPO). arXiv:2111.00072. <https://arxiv.org/abs/2111.00072>

5. Yuzhen Zhou, Jiajun Li, Yusheng Su, et al. "APRIL: Active Partial Rollouts in Reinforcement Learning to Tame Long-tail Generation" (APRIL; partial rollout). arXiv:2509.18521. <https://arxiv.org/abs/2509.18521>

6. Jacob Hilton, Karl Cobbe, John Schulman. "Batch size-invariance for policy optimization" (Decoupled PPO). arXiv:2110.00641. <https://arxiv.org/abs/2110.00641>

7. Sham Kakade, John Langford. "Approximately Optimal Approximate Reinforcement Learning". ICML 2002. <https://dl.acm.org/doi/10.5555/645531.657706>

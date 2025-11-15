---
layout: post
title: 从两策略到三策略：行为策略和参考策略不一致下的 TRPO 扩展
date: 2025-11-15
description: 在大模型强化学习中，因为推理框架和训练框架的不一致，以及异步训练框架下行为策略分布的多样性，行为策略与参考策略不一致的问题变得尤为突出。本文分析了行为策略与参考策略不一致问题在 TRPO 框架下的影响，并在这一分析基础上梳理了当前对这一问题的不同解决方法。
categories: reinforcement-learning
---

## 训推不一致和异步框架

最近看到不少关于大模型强化学习中“训推不一致”和“异步训推框架”的讨论，我自己的直觉是：这些看上去复杂多样的问题，很大一部分其实都在围绕一个更基础的矛盾打转——**行为策略（behavior policy）和参考策略（reference policy）不一致。**

本文先简单梳理一下我目前看到的相关工作，然后再尝试从“行为策略 vs 参考策略”的角度，把它们串到同一条线上，给各位读者提供一个补充视角。

在本文中我会用：

- **行为策略** $\mu$：实际负责生成 rollout 的策略，也就是“你在什么分布下采样到了这些数据”。在现代 LLM RL 系统里，它对应的是推理引擎里的那套实现（vLLM / SGLang 等），在异步框架下往往还是**多个 worker 策略的混合分布**。
- **参考策略** $\pi_{\theta_{\text{old}}}$：训练目标里拿来做重要性采样、Clipping 或 KL 约束的策略，典型地就是 PPO / GRPO 里的 “旧策略”（old policy）。
- **目标策略** $\pi_{\theta}$：训练目标里要优化的策略，也就是“你想让模型变成什么样”。典型地就是 PPO / GRPO 里的 “新策略”（new policy）。

在最经典、理想化的设定里，我们通常**默认** $\mu = \pi_{\theta_{\text{old}}}$。但在现实系统中，受异步更新、不同推理/训练后端、MoE 路由波动甚至硬件数值差异等因素影响，这二者往往会出现不同程度的偏离。

## 相关工作

下面按时间线简单列一下我印象比较深的一些工作（只代表我个人看到的片面子集）：

- [Decoupled PPO](https://arxiv.org/pdf/2110.00641) 率先指出在信赖域策略优化（TRPO 和 PPO）方法中，“旧策略”（old policy）实际承担了两个不同的角色：一是用于重要性采样进行异策略修正，在这个目的下，“旧策略”用于代表训练数据集所服从的行为策略（behavior policy）；二是用于限制新策略的更新幅度，在这个目的下，“旧策略”被用于衡量新旧策略的变化程度，称作近端策略（proximal policy，对应本文中的“参考策略”）。文章指出这两个目的下的“旧策略”可以是不同的策略，从而提出了 Decoupled PPO 更新目标，把“采样用谁”和“对谁做 trust region”在形式上解耦开来。
- [AReaL](https://arxiv.org/abs/2505.24298) 关注到了异步训练框架下行为策略与参考策略不一致的问题：rollout 往往由滞后的参数版本或不同 worker 产生。文章在异步框架下采用了 Decoupled PPO 风格的目标，将“行为策略分布”与“参考策略”显式区分开来，从而在异步场景下仍然维持类似 PPO 的优化性质。
- [GSPO](https://arxiv.org/abs/2507.18071) 从 GRPO 在长序列和 MoE 模型上的稳定性问题出发，指出 token-level 的 PPO / GRPO 在专家路由高度波动（尤其是新旧策略之间的路由差异）时会引入巨大的方差与不稳定。GSPO 提出在 **sequence-level** 定义 PPO-style 目标与比率约束，用整条序列的比率来约束更新，从而在 MoE 场景下显著缓解由路由不一致带来的训练崩溃问题。
- [Your Efficient RL Framework Secretly Brings You Off-Policy RL Training](https://fengyao.notion.site/off-policy-rl#28b721e3f6c480c3a756f8fb319e860d) 关注到了现有的一些大模型强化学习训练框架（如 VeRL）中，推理框架和训练框架在不少相同的功能模块上有不同的实现（例如 vLLM 和 FSDP / Megatron 等算子上的差异），导致行为策略 $\mu$ 与参考策略 $\pi_{\theta_{\text{old}}}$ 不一致。这种不一致使得原本假定为同策略（on-policy）的训练，实际上变成了带有明显偏差的异策略（off-policy）训练。文章总结了两种处理这一问题的现有方法：PPO-IS 与 vanilla-IS，并提出在 **token-level** 做截断重要性采样（truncated IS, TIS），以减少训推不一致程度较重的样本在训练中的影响。作者还写了两篇更为基础的分析文章，从原理上分析训推不一致问题：[Part I](https://fengyao.notion.site/pg-seq-token-part1-basics) 和 [Part II](https://fengyao.notion.site/pg-seq-token-part2-mismatch)。
- [Defeating Nondeterminism in LLM Inference](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference) 指出批处理大小不变性（batch-size invariance）的缺失是大模型推理框架随机性的核心来源之一：同一个输入在不同的 batch 组合和 kernel 路径下，得到的概率分布会发生可观差异。这意味着即便“名义上”是同一套参数，真实运行时的行为策略 $\mu$ 也会因为系统负载和调度差异而波动，从而进一步加剧训推不一致。
- [Small Leak Can Sink a Great Ship—Boost RL Training on MoE with 𝑰𝒄𝒆𝑷𝒐𝒑!](https://ringtech.notion.site/icepop) 观察到上述训推不一致问题在 MoE 模型上会进一步加剧：路由本身就对微小扰动高度敏感，再叠加推理/训练实现差异和异步采样，很容易放大偏差。文章提出 IcePop 方法：在 **token-level** 通过计算重要性采样比率，对过于大或者过于小的比率进行双侧掩码（masking），将这些“噪声较大”的数据从梯度中丢弃，从而稳定 MoE 上的 RL 训练。
- [When Speed Kills Stability: Demystifying RL Collapse from the Training-Inference Mismatch](https://yingru.notion.site/When-Speed-Kills-Stability-Demystifying-RL-Collapse-from-the-Training-Inference-Mismatch-271211a558b7808d8b12d403fd15edda) 系统性分析了训推不一致的各种成因，包括智能体工作流中引入的大量分布外和低概率信息、硬件和内核实现带来的计算不确定性，并分析了在 **token-level** 进行重要性采样如何在长序列上引入严重的偏差。文章进一步提出在 **sequence-level** 计算重要性采样掩码（sequence-level masked IS, sequence-level MIS）：只丢弃那些整条序列的重要性采样比率过大的数据，从而在控制偏差的同时，显著抑制由极端样本导致的训练崩溃。文中给出了较为完整的理论推导和丰富的实验支撑。
- [RL老训崩？训推差异是基石](https://zhuanlan.zhihu.com/p/1959976628290590602) 则更多从实践角度出发，分享了如何在实现上尽可能靠近“训推一致”的经验，包括如何选用一致的算子和精度配置、如何监控与约束训练端和推理端 log-prob 的偏差等，更着力于从训推框架层面入手，在工程上尽量从根本缓解训推差异问题。


## 三策略 TRPO 视角下的最小统一理解

上面列的这些工作，看上去各自解决的是：

- 算法层：PPO/GRPO 的目标怎么写，token-level 还是 sequence-level，用 clip 还是 mask；
- 系统层：推理框架和训练框架怎样对齐；
- 模型层：MoE 模型路由问题放大训练不稳定，等等。

但如果我们把“行为策略 vs 参考策略”这条线拉直，会发现绝大部分问题其实都可以塞进一个很简单的理论框架里：**三策略 TRPO**。

下面这节我会用尽量简单的数学，把这个三策略版 TRPO 摊开——它其实就是“TRPO + 三角不等式”，但非常好用：

- 一方面让我们重新理解“训推不一致”和“异步训练框架”到底在破坏什么；
- 另一方面，也帮我们统一理解 TIS、IcePop、sequence-level MIS 等，其实都是在实施下文的“**约束 2**”。

### 三个策略

沿用前文的记号，我们在一个折扣 MDP 上工作，折扣因子为 $\gamma\in(0,1)$：

- 状态 $s\in\mathcal{S}$，动作 $a\in\mathcal{A}$；
- 策略 $\pi(a\mid s)$；
- 折扣状态分布：
  $$
  d_\pi(s) := (1-\gamma)\sum_{t=0}^\infty \gamma^t \Pr_\pi(s_t = s).
  $$
- 回报（episode 视角）：
  $$
  \mathcal{J}(\pi) := \mathbb{E}_\pi\Big[\sum_{t=0}^\infty \gamma^t r_t\Big].
  $$
- 值函数 / 优势函数：
  $$
  V_\pi(s),\quad Q_\pi(s,a),\quad A_\pi(s,a) := Q_\pi(s,a) - V_\pi(s).
  $$

稍微赘述一下，在“三策略”设定里，我们有：

- **行为策略**（behavior policy）：$\mu$，真正用来 rollout 的策略；数据 $(s,a,r,\dots)$ 都是从它来的。  
- **参考策略**（reference policy）：$\pi_{\theta_{\text{old}}}$，优化目标里拿来做 ratio、clip 或 KL 约束的那一份“旧策略”。  
- **目标策略**（target policy）：$\pi_\theta$，我们这一步想要优化的策略。

在理想设定里我们默认 $\mu = \pi_{\theta_{\text{old}}}$；现实系统里这俩往往不等，这就是“训推不一致”的数学影子。

### 两策略 TRPO

> 有基础的读者可以直接跳到[三策略 TRPO](#三策略-trpo把参考策略--拉进来)。

TRPO 的所有理论保证，都是建立在**某个“基准策略”的优势函数**之上的。  既然实际能算清楚的**只有** $A_\mu$（数据是按 $\mu$ 采的），那我们就直接把 $\mu$ 当成基准。

一个经典的结论是 **性能差分引理（Performance Difference Lemma）**：

> 对任意两策略 $\mu$ 和 $\pi_\theta$，有  
> $$
> \mathcal{J}(\pi_\theta) - \mathcal{J}(\mu)
> = \frac{1}{1-\gamma}\;
> \mathbb{E}_{s\sim d_{\pi_\theta},\, a\sim\pi_\theta}[A_\mu(s,a)].
> $$

直觉非常简单：

- $A_\mu(s,a)$ 就是在说“如果在 $s$ 里本来按 $\mu$ 行动，现在换成动作 $a$，长期回报多/少多少”；  
- 把所有时刻、所有状态、所有动作的“增益”累积起来，就得到新策略比老策略总共赚了多少。

TRPO 的核心是：我们没法准确算
$$
\mathbb{E}_{s\sim d_{\pi_\theta}, a\sim\pi_\theta}[A_\mu(s,a)],
$$
因为 $d_{\pi_\theta}$ 是“新策略”的状态分布，我们没有在它下面采样过。

于是 TRPO 引入了一个替代目标：把状态分布换成旧策略的：

$$
L_\mu(\pi_\theta)
:= \mathcal{J}(\mu) + \frac{1}{1-\gamma}\mathbb{E}_{s\sim d_\mu,\,a\sim \pi_\theta}[A_\mu(s,a)].
$$

这份 $L_\mu$ 才是我们真正用数据能估出来的东西：“在旧策略的状态分布下，让新策略试着去选动作，看看优势有多大”。

从性能差分引理出发，两者之差是：

$$
\mathcal{J}(\pi_\theta) - L_\mu(\pi_\theta)
= \frac{1}{1-\gamma}\;
  \sum_s \big(d_{\pi_\theta}(s) - d_\mu(s)\big)
  \,\mathbb{E}_{a\sim\pi_\theta(\cdot\mid s)}[A_\mu(s,a)].
$$

如果我们定义

$$
\epsilon_\mu := \max_{s,a} |A_\mu(s,a)|,
$$

那么有一个非常直接的上界：

> **Lemma 1**  
> $$
> |\mathcal{J}(\pi_\theta) - L_\mu(\pi_\theta)|
> \le \frac{\epsilon_\mu}{1-\gamma}\;
>     \|d_{\pi_\theta} - d_\mu\|_1.
> $$

这里出现了第一个关键量：  
> **状态分布偏移** $\|d_{\pi_\theta} - d_\mu\|_1$，也就是“新策略和行为策略看到的世界，到底差了多少”。

我们通常不会直接对 $\|d_{\pi_\theta} - d_\mu\|_1$ 施加约束，反而是对“每一步 action 分布”的差异施加约束，比如信赖域、KL、clip 全是干这个的。

记总变差距离（total variation）：

$$
D_{\mathrm{TV}}(p,q) := \frac{1}{2}\|p-q\|_1.
$$

我们先假设一个条件：

> 对所有 $s$，行为策略和目标策略之间的 TV 被一个常数 $\beta$ 上界：  
> $$
> D_{\mathrm{TV}}\big(\mu(\cdot\mid s), \pi_\theta(\cdot\mid s)\big) \le \beta.
> $$

直观含义：在任意状态里，“新策略”和“生成数据的策略”选动作的分布都不会离太远。

一个经典结果（可以用 coupling 证明）是：

> **Lemma 2**  
> 在上述条件下有  
> $$
> \|d_{\pi_\theta} - d_\mu\|_1
> \le \frac{2\gamma}{1-\gamma}\,\beta.
> $$

把它和 Lemma 1 结合：

$$
|\mathcal{J}(\pi_\theta) - L_\mu(\pi_\theta)| \le \frac{\epsilon_\mu}{1-\gamma}\; \frac{2\gamma}{1-\gamma}\,\beta = \frac{2\epsilon_\mu\gamma}{(1-\gamma)^2}\,\beta.
$$

于是我们得到一个非常干净的**两策略 TRPO 下界（基准为行为策略）**：

> **Theorem 1（两策略版 TRPO 下界）**  
> $$
> \mathcal{J}(\pi_\theta)
> \;\ge\;
> L_\mu(\pi_\theta)
> \;-\;
> \frac{2\epsilon_\mu\gamma}{(1-\gamma)^2}\,\beta.
> $$

这就是最经典的“TRPO 单调性保证”的一种形式，只是我们把“旧策略”显式叫成了**行为策略** $\mu$。

### 三策略 TRPO：把参考策略 $\pi_{\theta_{\text{old}}}$ 拉进来

到这里，两策略 TRPO 已经给了我们一个非常干净的结论：

> **两策略 TRPO（基准为行为策略 $\mu$）：**  
> 记  
> $$\epsilon_\mu := \max_{s,a} |A_\mu(s,a)|,\quad \beta := \max_s D_{\mathrm{TV}}\big(\mu(\cdot\mid s), \pi_\theta(\cdot\mid s)\big),$$
> 则  
> $$\mathcal{J}(\pi_\theta) \;\ge\; L_\mu(\pi_\theta) \;-\; \underbrace{\frac{2\epsilon_\mu\gamma}{(1-\gamma)^2}}_{=:C} \,\beta,$$
> 其中  
> $$L_\mu(\pi_\theta) := \mathcal{J}(\mu) + \frac{1}{1-\gamma} \mathbb{E}_{s\sim d_\mu,a\sim\pi_\theta}[A_\mu(s,a)].$$

也就是说：

- **真正决定“替代目标 $L_\mu$ 靠不靠谱”的，是行为策略 $\mu$ 和目标策略 $\pi_\theta$ 的差异：**  
  $$
  \beta = \max_s D_{\mathrm{TV}}\big(\mu(\cdot\mid s), \pi_\theta(\cdot\mid s)\big).
  $$
- 如果你能直接约束住这个 $\beta$，就能直接把 TRPO 的单调性保证搬到行为策略视角下。

现实问题在于：**大模型强化学习训练里我们可能无法直接控制 $\beta$ 本身。**

在大部分 PPO / GRPO / GSPO / 现有 RLHF 框架里，实际发生的是：

- rollout 数据是由某个**行为策略** $\mu$ 产生的（推理引擎里的“那一版参数”+ 若干系统细节）；
- 更新时，我们希望利用**参考策略 $\pi_{\theta_{\text{old}}}$** 来限制**目标策略** $\pi_\theta$的更新幅度。

也就是说，实际可以“动手”的是两个量：

1. **参考 vs 目标**：我们可以通过 KL / clip 等手段控制
   $$\
   D_{\mathrm{TV}}\big(\pi_{\theta_{\text{old}}}(\cdot\mid s),\pi_\theta(\cdot\mid s)\big)
   $$
2. **行为 vs 参考**：我们希望**间接**控制
   $$
   D_{\mathrm{TV}}\big(\mu(\cdot\mid s),\pi_{\theta_{\text{old}}}(\cdot\mid s)\big)
   $$

于是自然就定义两个“proxy 差异”：

- **约束 1：参考 vs 目标**
  $$
  \alpha_0
  := \max_s D_{\mathrm{TV}}\big(\pi_{\theta_{\text{old}}}(\cdot\mid s),
                                \pi_\theta(\cdot\mid s)\big)
  $$
- **约束 2：行为 vs 参考**
  $$
  \alpha_1
  := \max_s D_{\mathrm{TV}}\big(\mu(\cdot\mid s),
                                \pi_{\theta_{\text{old}}}(\cdot\mid s)\big)
  $$

直觉上：

- $\alpha_0$：新策略到底离“你宣称的那份旧策略”有多远——这就是信赖域的那部分；
- $\alpha_1$：你用来训练的参考策略，到底跟真实采样时的行为策略差了多少——这就是训推不一致/异步的影子。

接下来只需要一行三角不等式。

### 一行三角不等式：从两策略到三策略 TRPO

对任意状态 $s$，有

$$
\begin{aligned}
D_{\mathrm{TV}}\big(\mu(\cdot\mid s),\pi_\theta(\cdot\mid s)\big)
&\le
D_{\mathrm{TV}}\big(\mu(\cdot\mid s),\pi_{\theta_{\text{old}}}(\cdot\mid s)\big)
\\
&\quad +
D_{\mathrm{TV}}\big(\pi_{\theta_{\text{old}}}(\cdot\mid s),\pi_\theta(\cdot\mid s)\big).
\end{aligned}
$$

对 $s$ 取上确界：

$$
\beta
:= \max_s D_{\mathrm{TV}}\big(\mu(\cdot\mid s),\pi_\theta(\cdot\mid s)\big)
\;\le\;
\alpha_1 + \alpha_0.
$$

把这个不等式塞回两策略 TRPO 的结论（Theorem 1）里：

$$
\mathcal{J}(\pi_\theta)
\;\ge\;
L_\mu(\pi_\theta)
\;-\;
C\,\beta
\;\ge\;
L_\mu(\pi_\theta)
\;-\;
C\,(\alpha_0 + \alpha_1),
$$

其中 $C = \frac{2\epsilon_\mu\gamma}{(1-\gamma)^2}$。

于是，我们得到一个非常干净的**三策略 TRPO 下界**：

> **Theorem 2（三策略 TRPO，下界形式）**  
> 记
> $$
> \epsilon_\mu := \max_{s,a} |A_\mu(s,a)|,\quad
> C := \frac{2\epsilon_\mu\gamma}{(1-\gamma)^2},
> $$
> 以及
> $$
> \alpha_0
> := \max_s D_{\mathrm{TV}}\big(\pi_{\theta_{\text{old}}}(\cdot\mid s),
>                               \pi_\theta(\cdot\mid s)\big),
> \quad
> \alpha_1
> := \max_s D_{\mathrm{TV}}\big(\mu(\cdot\mid s),
>                               \pi_{\theta_{\text{old}}}(\cdot\mid s)\big).
> $$
> 则对任意目标策略 $\pi_\theta$ 有
> $$
> \boxed{
> \mathcal{J}(\pi_\theta)
> \;\ge\;
> L_\mu(\pi_\theta)
> \;-\; C\,(\alpha_0 + \alpha_1),
> }
> $$
> 其中
> $$
> L_\mu(\pi_\theta)
> :=
> \mathcal{J}(\mu)
> + \frac{1}{1-\gamma}
>   \mathbb{E}_{s\sim d_\mu,a\sim\pi_\theta}[A_\mu(s,a)].
> $$

这句话的含义非常简单：

- **替代目标 $L_\mu(\pi_\theta)$ 与真实性能 $\mathcal{J}(\pi_\theta)$ 之间的 gap，被拆成了两部分：**
  - 参考 vs 目标 的偏移 $\alpha_0$；
  - 行为 vs 参考 的偏移 $\alpha_1$。

只要这两个量都小，**优化 $L_\mu$ 就能有效提升 $\mathcal{J}$**。

### 这两个差异各自怎么约束？

现在，我们可以从 Theorem 2 回头看各种实际方法：

- 绝大多数 “PPO/GRPO/GSPO” 类工作，其实是在控制 **约束 1：$\alpha_0$**；
- 绝大多数 “TIS / IcePop / MIS” 类工作，其实是在控制 **约束 2：$\alpha_1$**。

本文只讨论 **约束 2**。

约束 2的目标：**保证用来训练的数据，尽可能来自“接近参考策略”的行为策略。**

这里通常既有**系统层**的机制，也有**算法层（importance sampling）**的机制。

1. **系统层：让行为策略别飘太远**

   - 异步框架：
     - 给每个样本打上策略版本号，只能用 $\pi_{\theta_{\text{old}}}$ 不太远的版本采样的数据；
   - 训推对齐：
     - 强调训练框架和推理框架用相同精度、相同算子；

   这些机制的目标是：从“算法外部”让 $\mu$ 和 $\pi_{\theta_{\text{old}}}$ 靠近，从而压缩 $\alpha_1$。

2. **算法层：样本修正**

   在算法层，我们不再试图“纠正整个行为策略”，而是用重要性采样比率在**样本层面**做筛选和重加权，让 **“真正参与训练的样本子集”** 上的行为策略尽量接近参考策略，或者减小差异较大的样本在训练上的权重。

   具体来说就是下面这些方法，它们本质上全是“实现约束 2 的不同方式”。


### TIS、IcePop、sequence-level MIS：都是“约束 2”的不同实现
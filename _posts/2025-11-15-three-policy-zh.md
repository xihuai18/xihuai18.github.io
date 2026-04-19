---
layout: post
title: "从两策略到三策略：LLM RL 中行为策略–参考策略不一致下的 TRPO 扩展"
date: 2025-11-15
description: 在现代 LLM RL 流程中，训练里的"旧策略"可能已经不等于真正生成 rollout 的行为策略，破坏常见的同策略假设。本文把经典 TRPO 下界改写成行为策略、参考策略、目标策略的三策略形式，并说明 surrogate gap 同时被两个偏差来源控制。
og_image: /assets/img/three-policy/three-policy-mini-class-zh.jpg
categories: reinforcement-learning
lang: zh
en_url: /reinforcement-learning/2025/11/15/three-policy-en.html
zhihu_url: https://zhuanlan.zhihu.com/p/1973206684907365344
wechat_url: https://mp.weixin.qq.com/s/Gkjk_Fy8qWLkkdWAIuy9og
---

![Mini-class](/assets/img/three-policy/three-policy-mini-class-zh.jpg){: style="display:block;margin:0 auto;width:95%;max-width:100%;" }

> 本文的核心结论是：当行为策略 $\mu$、参考策略 $\pi_{\theta_{\text{old}}}$ 与目标策略 $\pi_\theta$ 不再重合时，TRPO / PPO 类 surrogate 的可靠性会同时受到"参考 vs 目标"和"行为 vs 参考"两个偏差来源的影响。

## 1. 引言：训推不一致与异步框架

最近不少 LLM 强化学习工作反复撞上同一个问题：**真正生成数据的行为策略（behavior policy）和训练里使用的参考策略（reference policy）并不一致。**

本文先梳理与后文最相关的一组工作，再把这一不一致放进同一个分析框架。

这里不打算证明一个更强的 TRPO 定理；我更关心的是把 LLM-RL 里经常被混在一起的三种策略拆开来写，用它解释训推不一致到底在破坏哪一部分分析。

具体来说，本文只做三件事：

- 先把经典 TRPO 的 surrogate gap 写成以行为策略为基准的形式；
- 再把行为策略、参考策略和目标策略用一个保守上界串起来；
- 最后用这个框架重读 LLM-RL 中的训推不一致、样本修正和 routing replay。

本文使用以下记号：

- **行为策略** $\mu$：实际负责生成 rollout 的策略，即"数据是在什么分布下采样的"。在现代 LLM-RL 系统中，它对应推理引擎里的实现（vLLM / SGLang 等）；在异步框架下，它往往还可以**近似看作多个 worker 策略诱导分布的混合**。
- **近端/参考策略** $\pi_{\theta_{\text{old}}}$：训练目标中用于重要性采样、clipping 或 trust-region 约束的策略，典型的就是 PPO / GRPO 里的"旧策略"（old policy）。为了避免和固定 KL 参考模型混淆，本文把它称为近端策略或参考策略；若后文提到固定 SFT 参考模型，会单独写成 $\pi_{\mathrm{ref}}$。
- **目标策略** $\pi_\theta$：训练目标中要优化的策略，即"希望模型变成什么样"，典型的就是 PPO / GRPO 里的"新策略"（new policy）。

为了把理论对象和工程变量对应起来，可以先看下面这张文字版坐标图：

| 理论对象 | 在分析中的角色 | 常见工程量 |
| --- | --- | --- |
| $\mu$ | 真实采样分布，决定数据来自哪里 | behavior log-prob、policy version、sampling config、routing trace |
| $\pi_{\theta_{\text{old}}}$ | 近端锚点，决定 ratio 分母与 trust region 参照 | old log-prob、clip anchor、proximal checkpoint |
| $\pi_\theta$ | 被优化的目标策略，决定更新方向 | new log-prob、当前 actor、当前 router |
| $\pi_{\mathrm{ref}}$ | 若存在固定 KL 参考模型，只负责 KL 正则 | ref log-prob、SFT/reference checkpoint |

在最理想化的设定里，我们通常**默认** $\mu = \pi_{\theta_{\text{old}}}$。但在真实系统里，由于异步更新、推理和训练后端不同、MoE 路由波动甚至硬件数值差异，二者往往会偏离，程度视系统而定。本文的理论坐标系就是要把"谁生成数据"、"谁作为近端锚点"、"谁正在被优化"这三件事分开。

## 2. 相关工作

下面按三条主线整理与后文三策略视角最相关的工作：

- **算法层**：如何写 trust region 目标、如何在样本层面处理"行为 vs 参考"偏移；
- **系统层**：如何在算法外部让行为策略与参考策略保持接近；
- **模型层**：MoE 路由带来的特殊问题。

同一篇工作可能兼具多个层面，这里按其主要贡献归位。

### 2.1 算法层：目标函数与样本级机制

- [Decoupled PPO](https://arxiv.org/abs/2110.00641) 较早指出，在信赖域策略优化（TRPO 和 PPO）方法中，"旧策略"（old policy）其实同时扮演两个角色：一是用于重要性采样以做异策略修正，此时"旧策略"代表数据所服从的行为策略（behavior policy）；二是用于限制新策略的更新幅度，此时"旧策略"用于衡量新旧策略的变化幅度，称为近端策略（proximal policy，对应本文中的"参考策略"）。文章指出这两个目的下的"旧策略"未必要是同一个，从而提出 Decoupled PPO 更新目标，把"用哪个策略采样"和"对哪个策略做 trust region"在形式上解耦开来。

- [GSPO](https://arxiv.org/abs/2507.18071) 从 GRPO 在长序列和 MoE 模型上的稳定性问题出发，指出 token-level 的 PPO / GRPO 在专家路由高度波动（尤其是新旧策略之间的路由差异）时，会引入巨大的方差与不稳定。GSPO 提出在 **sequence-level** 定义 PPO-style 目标与比率约束，用整条序列的比率来约束更新，在 MoE 场景下显著缓解由路由不一致带来的训练崩溃。

- [Your Efficient RL Framework Secretly Brings You Off-Policy RL Training](https://fengyao.notion.site/off-policy-rl#28b721e3f6c480c3a756f8fb319e860d) 关注现有的一些大模型 RL 训练框架（如 VeRL）中，推理框架和训练框架在同一功能模块上常常有不同实现（例如 vLLM 和 FSDP / Megatron 的算子差异），导致行为策略 $\mu$ 与参考策略 $\pi_{\theta_{\text{old}}}$ 不一致。这种不一致使得原本假定为同策略（on-policy）的训练，实际变成了带有明显偏差的异策略（off-policy）训练。文章总结了两种处理这一问题的现有方法：PPO-IS 与 vanilla-IS，并提出在 **token-level** 做截断重要性采样（truncated IS, TIS），减少训推不一致程度较重的样本对训练的影响。作者还写了两篇更基础的分析文章：[Part I](https://fengyao.notion.site/pg-seq-token-part1-basics) 和 [Part II](https://fengyao.notion.site/pg-seq-token-part2-mismatch)。

- [Small Leak Can Sink a Great Ship—Boost RL Training on MoE with 𝑰𝒄𝒆𝑷𝒐𝒑!](https://ringtech.notion.site/icepop) 观察到，上述训推不一致问题在 MoE 模型上会被进一步放大：路由本身对微小扰动就极为敏感，叠加推理/训练实现差异与异步采样后，偏差很容易失控。文章提出 IcePop 方法：在 **token-level** 计算重要性采样比率，对过大或过小的比率做双侧掩码（masking），把这些"噪声较大"的数据从梯度中剔除，稳定 MoE 上的 RL 训练。

- [When Speed Kills Stability: Demystifying RL Collapse from the Training-Inference Mismatch](https://yingru.notion.site/When-Speed-Kills-Stability-Demystifying-RL-Collapse-from-the-Training-Inference-Mismatch-271211a558b7808d8b12d403fd15edda) 系统性分析了训推不一致的各种成因，包括智能体工作流中引入的大量分布外和低概率 token、硬件和内核/kernel 实现带来的计算不确定性；并分析了 **token-level** 重要性采样在长序列上为何会引入严重偏差。文章进一步提出 **sequence-level** 重要性采样掩码（sequence-level masked IS, sequence-level MIS）：只丢弃整条序列重要性比率过大的数据，在控制偏差的同时明显抑制由极端样本引发的训练崩溃。文中给出了较完整的理论推导和较充分的实验支撑。

- [verl Rollout Importance Sampling](https://verl.readthedocs.io/en/latest/algo/rollout_corr.html) 在其 rollout correction 模块中引入 Token Veto（一票否决）机制：在 **token-level** 计算重要性比率 $\rho_t^{(\text{ref}\leftarrow\text{beh})}$，若轨迹中存在任意 token 使得 $\min_t \rho_t < \tau_{\text{veto}}$，则把整条序列从训练中剔除。"token 粒度检测、sequence 粒度否决"体现的是一种"一票否决"的保守策略。

- [INTELLECT-3 Technical Report](https://storage.googleapis.com/intellect-3-paper/INTELLECT_3_Technical_Report.pdf) 在其异步分布式 RL 训练框架中采用了类似的拒绝采样策略。INTELLECT-3 对每条 rollout 计算 **token-level** 重要性比率，若任意 token 的比率低于阈值（文中取 $10^{-5}$），就对整条轨迹做 masking。

### 2.2 系统层：异步与训推对齐

- [AReaL](https://arxiv.org/abs/2505.24298) 关注异步训练框架下行为策略与参考策略不一致的问题：rollout 往往由滞后的参数版本或不同 worker 产生。文章在异步框架下采用 Decoupled PPO 风格的目标，显式区分"行为策略分布"和"参考策略"，从而在异步场景下仍能维持类似 PPO 的优化性质。

- [Defeating Nondeterminism in LLM Inference](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference) 指出，批处理大小不变性（batch-size invariance）的缺失是大模型推理框架随机性的核心来源之一：同一个输入在不同的 batch 组合和 kernel 路径下，得到的概率分布会有明显差异。这意味着，即便"名义上"是同一套参数，真实运行时的行为策略 $\mu$ 也会因系统负载和调度不同而波动，进一步放大训推不一致。

- [RL 老训崩？训推差异是基石](https://zhuanlan.zhihu.com/p/1959976628290590602) 更多从实践角度出发，分享如何在实现上尽量靠近"训推一致"，包括选用一致的算子和精度配置、监控与约束训练端和推理端的 log-prob 偏差等，着重从训推框架层面入手，在工程上从源头缓解训推差异。

### 2.3 模型层：MoE 路由

- [Stabilizing MoE Reinforcement Learning by Aligning Training and Inference Routers](https://arxiv.org/abs/2510.11370) 聚焦 MoE 架构下特有的 **路由不一致（Routing Inconsistency）** 问题。文章发现，即使输入完全相同，推理端和训练端也可能因为算子实现或并行的微小差异，让 Router 选中不同的专家。这种"物理路径"上的不一致，使行为策略 $\mu$ 和参考策略 $\pi_{\theta_{\text{old}}}$ 的差异远超预期，很容易触发训练崩溃。文章提出 **Rollout Routing Replay (R3)**：在推理阶段记录每个 token 实际命中的专家索引，在训练阶段**强制回放**这些路由决策，不再重新计算。R3 通过这种方式在 MoE 拓扑上强制对齐训推两端的计算路径。

## 3. 三策略 TRPO：最简统一框架

上述工作分属不同层面，但沿着"行为策略 vs 参考策略"这条主线整理，大部分都可以放进同一个最简框架：**三策略 TRPO**。

下面把推导压到只保留后文需要的结构。核心步骤只有两步：先写出以行为策略为基准的 surrogate gap，再用一次三角不等式把它改写成两个偏差来源。这个分解对分析训推不一致尤其有用：

- 一方面帮助我们重新理解"训推不一致"和"异步训练框架"到底在影响什么；
- 另一方面，也帮助我们统一理解 TIS、IcePop、sequence-level MIS 等方法；在本文的视角下，它们更多是在**样本层面**缓解"行为 vs 参考"偏移带来的估计偏差与方差问题，而不是直接改变 worst-case 意义下的 $\alpha_1$。

### 3.1 三个策略

沿用前文记号，在一个折扣 MDP 上工作，折扣因子为 $\gamma\in(0,1)$：

> 对 LLM-RL 来说，更常见的往往是有限时域的序列决策。这里先沿用折扣 MDP 记号，只是为了直接承接经典 TRPO 的写法；后文的核心分解结构同样可以平移到有限时域版本。

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

沿用前文记号：行为策略为 $\mu$，参考策略为 $\pi_{\theta_{\text{old}}}$，目标策略为 $\pi_\theta$。

理想设定下通常默认 $\mu = \pi_{\theta_{\text{old}}}$；但在现实系统里二者往往不等，这就是"训推不一致"的数学体现。

### 3.2 两策略 TRPO：以行为策略为基准的 surrogate-gap 下界

> 熟悉 TRPO 的读者可以直接跳到后面的"三策略 TRPO"小节。

这里保留了 TRPO 的核心逻辑，但为了服务后文的三策略分解，我会用一条以行为策略 $\mu$ 为基准、常数较松的 surrogate-gap 下界，而不是逐字复现 Schulman et al. (2015) 的原始定理形式。

TRPO 的理论保证都建立在**某个"基准策略"的优势函数**之上。本文直接把 $\mu$ 当作基准：一是数据本就按 $\mu$ 采样；二是在真实的 LLM-RL 里，我们通常只能用基于 $\mu$ 数据估得的 critic / GAE / group-normalized reward 等 proxy 来近似 $A_\mu$，这部分估计误差本文暂不纳入下界。

一个经典的结论是 **性能差分引理（Performance Difference Lemma）**：

> 对任意两策略 $\mu$ 和 $\pi_\theta$，有
>
> $$
> \mathcal{J}(\pi_\theta) - \mathcal{J}(\mu)
> = \frac{1}{1-\gamma}\;
> \mathbb{E}_{s\sim d_{\pi_\theta},\, a\sim\pi_\theta}[A_\mu(s,a)].
> $$

TRPO 的难点在于，无法准确计算

$$
\mathbb{E}_{s\sim d_{\pi_\theta}, a\sim\pi_\theta}[A_\mu(s,a)],
$$

因为 $d_{\pi_\theta}$ 是"新策略"的状态分布，而我们并没有在这个分布下采样。

于是 TRPO 引入一个代理目标：把状态分布换成行为策略的：

$$
L_\mu(\pi_\theta)
:= \mathcal{J}(\mu) + \frac{1}{1-\gamma}\mathbb{E}_{s\sim d_\mu,\,a\sim \pi_\theta}[A_\mu(s,a)].
$$

实际计算时，$\mathbb{E}_{a\sim\pi_\theta}[A_\mu(s,a)]$ 往往通过重要性采样改写成基于行为策略样本的形式；这里先保留更直接的定义，后文再回到 PPO 风格的实现写法。

注意：这里的理论 surrogate $L_\mu$ 是以**行为策略** $\mu$ 为基准定义的；而实际 PPO / GRPO 风格的 loss 往往把 ratio 的分母写成**参考策略** $\pi_{\theta_{\text{old}}}$。这两者之间正隔着一层"行为 vs 参考"的不一致，也正是下文引入 $\alpha_1$ 的原因。

从性能差分引理出发，两者之差是：

$$
\mathcal{J}(\pi_\theta) - L_\mu(\pi_\theta)
= \frac{1}{1-\gamma}\;
  \sum_s \big(d_{\pi_\theta}(s) - d_\mu(s)\big)
  \,\mathbb{E}_{a\sim\pi_\theta(\cdot\mid s)}[A_\mu(s,a)].
$$

定义

$$
\epsilon_\mu := \max_{s,a} |A_\mu(s,a)|,
$$

便有一个直接的上界：

> **引理 1（状态分布偏移项的直接上界）**
>
> $$
> |\mathcal{J}(\pi_\theta) - L_\mu(\pi_\theta)|
> \le \frac{\epsilon_\mu}{1-\gamma}\;
>     \|d_{\pi_\theta} - d_\mu\|_1.
> $$

这里出现了第一个关键量：

> **状态分布偏移** $\|d_{\pi_\theta} - d_\mu\|_1$，也就是"新策略和行为策略看到的世界到底差了多少"。

通常不会直接约束 $\|d_{\pi_\theta} - d_\mu\|_1$，而是约束"每一步 action 分布"上的差异，比如 trust region、KL、clip 等。

记总变差距离（total variation）：

$$
D_{\mathrm{TV}}(p,q) := \frac{1}{2}\|p-q\|_1.
$$

假设存在常数 $\beta$，使得

> 对所有 $s$，行为策略与目标策略之间的 TV 被 $\beta$ 上界：
>
> $$
> D_{\mathrm{TV}}\big(\mu(\cdot\mid s), \pi_\theta(\cdot\mid s)\big) \le \beta.
> $$

直观含义：在任意状态下，"新策略"和"生成数据的策略"选动作的分布都不会离太远。

一个经典结果（用 coupling 可证）是：

> **引理 2（策略 TV 到状态分布偏移的传播界）**
> 在上述条件下有
>
> $$
> \|d_{\pi_\theta} - d_\mu\|_1
> \le \frac{2\gamma}{1-\gamma}\,\beta.
> $$

把它和引理 1 结合：

$$
|\mathcal{J}(\pi_\theta) - L_\mu(\pi_\theta)|
\le \frac{\epsilon_\mu}{1-\gamma}\; \frac{2\gamma}{1-\gamma}\,\beta
= \frac{2\epsilon_\mu\gamma}{(1-\gamma)^2}\,\beta.
$$

于是得到一条形式上很简洁的**以行为策略为基准的两策略 surrogate-gap 下界**：

> **定理 1（以行为策略为基准的两策略下界）**
>
> $$
> \mathcal{J}(\pi_\theta)
> \;\ge\;
> L_\mu(\pi_\theta)
> \;-\;
> \frac{2\epsilon_\mu\gamma}{(1-\gamma)^2}\,\beta.
> $$

这说明：

- **真正决定"代理目标 $L_\mu$ 靠不靠谱"的，是行为策略 $\mu$ 与目标策略 $\pi_\theta$ 的差异：**
  $$
  \beta = \max_s D_{\mathrm{TV}}\big(\mu(\cdot\mid s), \pi_\theta(\cdot\mid s)\big).
  $$

只要能直接约束住这个 $\beta$，就能把 TRPO 的单调性保证搬到行为策略视角下。这里采用便于展示的 worst-case TV 写法，常数不求最紧，只求把结构写清楚；若改用 average-TV 版本，也能得到更贴近样本平均的类似结论，只是常数和期望形式会相应变化。

### 3.3 三策略 TRPO

现实问题在于：**在大模型 RL 训练里，我们通常无法直接控制 $\beta$ 本身。**

在大部分 PPO / GRPO / GSPO 及现有 RLHF 框架中，实际发生的是：

- rollout 数据由某个**行为策略** $\mu$ 产生（推理引擎中的"那一版参数"加上若干系统细节）；
- 更新时希望用**参考策略** $\pi_{\theta_{\text{old}}}$ 来限制**目标策略** $\pi_\theta$ 的更新幅度。

也就是说，实际能直接控制或间接影响的是两个量：

1. **参考 vs 目标**：可以通过 KL / clip 等手段控制
   $$
   D_{\mathrm{TV}}\big(\pi_{\theta_{\text{old}}}(\cdot\mid s),\pi_\theta(\cdot\mid s)\big).
   $$
2. **行为 vs 参考**：希望**间接**控制
   $$
   D_{\mathrm{TV}}\big(\mu(\cdot\mid s),\pi_{\theta_{\text{old}}}(\cdot\mid s)\big).
   $$

于是自然就得到两个偏差来源，对应两个 TV 距离量：

- **偏差源 A：参考 vs 目标**
  $$
  \alpha_0
  := \max_s D_{\mathrm{TV}}\big(\pi_{\theta_{\text{old}}}(\cdot\mid s),
                                \pi_\theta(\cdot\mid s)\big);
  $$
- **偏差源 B：行为 vs 参考**
  $$
  \alpha_1
  := \max_s D_{\mathrm{TV}}\big(\mu(\cdot\mid s),
                                \pi_{\theta_{\text{old}}}(\cdot\mid s)\big).
  $$

直觉上：

- $\alpha_0$：新策略离训练里选定的参考策略有多远——这就是 trust region 控制的部分；
- $\alpha_1$：训练用的参考策略与真实采样时的行为策略差多少——这就是训推不一致或异步的影子。

现在把这两个量代回 TRPO 的下界。

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

把这个不等式代回定理 1，记

$$
C := \frac{2\epsilon_\mu\gamma}{(1-\gamma)^2},
$$

就有

$$
\mathcal{J}(\pi_\theta)
\;\ge\;
L_\mu(\pi_\theta)
\;-\;
C\,\beta
\;\ge\;
L_\mu(\pi_\theta)
\;-\;
C\,(\alpha_0 + \alpha_1).
$$

于是得到一条直接的**保守三策略 TRPO 下界**（也是定理 1 的直接推论）：

> **定理 2（三策略 TRPO）**
> 记
>
> $$
> \epsilon_\mu := \max_{s,a} |A_\mu(s,a)|,\quad
> C := \frac{2\epsilon_\mu\gamma}{(1-\gamma)^2},
> $$
>
> 以及
>
> $$
> \alpha_0
> := \max_s D_{\mathrm{TV}}\big(\pi_{\theta_{\text{old}}}(\cdot\mid s),
>                               \pi_\theta(\cdot\mid s)\big),
> \quad
> \alpha_1
> := \max_s D_{\mathrm{TV}}\big(\mu(\cdot\mid s),
>                               \pi_{\theta_{\text{old}}}(\cdot\mid s)\big).
> $$
>
> 则对任意目标策略 $\pi_\theta$ 有
>
> $$
> \boxed{
> \mathcal{J}(\pi_\theta)
> \;\ge\;
> L_\mu(\pi_\theta)
> \;-\; C\,(\alpha_0 + \alpha_1)
> }
> $$
>
> 其中
>
> $$
> L_\mu(\pi_\theta)
> :=
> \mathcal{J}(\mu) + \frac{1}{1-\gamma}
>   \mathbb{E}_{s\sim d_\mu,a\sim\pi_\theta}[A_\mu(s,a)].
> $$

定理 2 并不复杂：$L_\mu(\pi_\theta)$ 与 $\mathcal{J}(\pi_\theta)$ 之间的 gap 不是被精确拆成两项，而是被一条保守上界控制；这条上界恰好由"参考 vs 目标"的 $\alpha_0$ 和"行为 vs 参考"的 $\alpha_1$ 共同决定。若要真正推出性能提升，还需要 $L_\mu(\pi_\theta)$ 本身足够高，例如至少要超过 $\mathcal{J}(\mu) + C(\alpha_0 + \alpha_1)$。在 LLM 场景里，这个界的数值通常很松，所以我更把它当作一个结构工具，而不是性能证书。

### 3.4 LLM 有限序列形式：为什么长回复会放大三策略偏移？

上面的推导沿用折扣 MDP 记号，是为了直接承接 TRPO。若转写成 LLM-RL 中更常见的 prompt-response 形式，设 prompt 为 $x$，回复为

$$
y=(a_1,\ldots,a_T),
$$

则三类策略的序列概率分别为

$$
\mu(y\mid x)=\prod_{t=1}^T \mu(a_t\mid x,a_{<t}),
$$

$$
\pi_{\theta_{\text{old}}}(y\mid x)=\prod_{t=1}^T \pi_{\theta_{\text{old}}}(a_t\mid x,a_{<t}),
$$

$$
\pi_\theta(y\mid x)=\prod_{t=1}^T \pi_\theta(a_t\mid x,a_{<t}).
$$

于是行为到目标的序列级比率为

$$
\frac{\pi_\theta(y\mid x)}{\mu(y\mid x)}
=
\prod_{t=1}^T
\frac{\pi_\theta(a_t\mid x,a_{<t})}{\mu(a_t\mid x,a_{<t})},
$$

对应的 log-ratio 是 token 级 log-ratio 的求和。也就是说，token 级很小的三策略偏移会在长回复中累积；如果直接看 sequence ratio，则这种累积会以乘积形式表现出来。这也是为什么在 LLM-RL 里，$\alpha_0$ 和 $\alpha_1$ 不只是普通 MDP 中的抽象距离项，还会和回复长度、截断采样、路由决策等序列结构耦合。

这个小节的作用只是把理论对象翻译到有限序列设定：后文讨论 token-level TIS、sequence-level MIS、WTRS 和 routing replay 时，核心都是在不同粒度上处理同一个行为-近端-目标三角关系。

### 3.5 这两个偏差来源各自怎么控制？

回头看各种实际方法：

- 大多数 PPO / GRPO 类工作，主要是在控制"**参考 vs 目标**"这一侧的偏移，也就是 $\alpha_0$；
- GSPO 的方向也在这一侧，但它不只是把 $\alpha_0$ 控制得更稳，而是进一步把 ratio、clipping 和优化的基本单位从 token-level 改到 sequence-level，从而改变了这类偏移被度量和约束的粒度；
- TIS / IcePop / MIS / WTRS 则更多是在样本权重、样本子集或拒绝规则层面，缓解"**行为 vs 参考**"偏移带来的估计偏差与方差问题，而不是直接改变 $\alpha_1$ 的 worst-case 定义值。

至于我更关心哪一侧，基本总是后者：在真实系统里，trust region 通常还没先坏掉，行为策略和参考策略的失配就已经足以把训练拖离"近似 on-policy"的区域。本文下面只讨论"**行为 vs 参考**"这一侧。

这一侧的目标是：**让真正参与训练的数据尽量来自"接近参考策略"的行为分布，或至少不要让严重不一致的样本主导梯度。**

这里通常既有**系统层**的机制，也有**算法层（importance sampling）**的机制。

1. **系统层：让行为策略与参考策略保持接近**
   - 异步框架：给每个样本打上策略版本号，只使用与 $\pi_{\theta_{\text{old}}}$ 相差不大的参数版本采样的数据；
   - 训推对齐：让训练框架和推理框架用相同的精度、算子和相近的内核/kernel 行为。

   这些机制的目标是：从"算法外部"让 $\mu$ 和 $\pi_{\theta_{\text{old}}}$ 靠近，从而直接压低 $\alpha_1$。

2. **算法层：样本修正**

   在算法层，我们不再试图"纠正整个行为策略"，而是用重要性采样比率在**样本层面**做筛选和重加权，让真正参与训练的样本子集更接近参考策略，或减小差异较大的样本在训练中的权重。

   为了避免说法过强，后文把两个分布区分开来：

   $$
   \mu_{\mathrm{raw}} := \text{真实 rollout 行为分布},
   \qquad
   \mu_{\mathrm{eff}} := \text{经过重加权、掩码或拒绝后进入 surrogate 的有效训练分布}.
   $$

   TIS / IcePop / MIS / WTRS 通常不直接缩小 $D(\mu_{\mathrm{raw}},\pi_{\theta_{\text{old}}})$；它们改变的是 $\mu_{\mathrm{eff}}$，或改变不同样本在 surrogate 中的权重。因此，若仍用 $\alpha_1$ 表示原始行为策略距离，就不应把这些样本级机制简单说成"压低 $\alpha_1$"。更严谨的说法是：它们让实际被优化的有效目标更少受行为-参考偏移的极端样本主导。

   具体就是下面这些方法。它们的共同点是：都在样本层面缓解"行为 vs 参考"偏移带来的不良后果。

## 4. 重要性采样与掩码：四种围绕"行为 vs 参考"偏移的样本级机制

下面沿用前文的记号给出这四种方法的目标函数，只聚焦"行为策略 vs 参考策略"这一维。这里的统一写法主要突出"行为 vs 参考"偏移在训练 loss 中是如何被处理的，并不逐字复现各原论文中的所有实现细节（如优势估计、baseline 和额外正则项）。下面各个 $L_{\text{TIS}}, L_{\text{IcePop}}, L_{\text{MIS}}, L_{\text{WTRS}}$ 也只是为了方便比较而写出的训练 loss 抽象，不应理解为理论 surrogate $L_\mu$ 的逐字无偏估计。记 token 级的 PPO / GRPO 风格更新项为

$$
g_\theta(t)
= \min\big(r_t(\theta) A_t,\ \text{clip}(r_t(\theta),1-\epsilon,1+\epsilon) A_t\big),
$$

其中

$$
r_t(\theta) = \frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{\text{old}}}(a_t\mid s_t)},
\quad (s_t,a_t)\sim\mu.
$$

- $r_t(\theta)$ 是 **目标 vs 参考** 的比率；
- $A_t$ 表示与理论上的 $A_\mu(s_t,a_t)$ 对应的优势项；实践里它通常来自基于 $\mu$ 数据估得的 critic / GAE / group-normalized reward 等 proxy，因此不应理解为对 $A_\mu$ 的逐字无偏替代。

为了把 token 级的 $(s_t,a_t)$ 与序列级的 $(x,y)$ 记号打通，在以 RLHF（reinforcement learning from human feedback，人类反馈强化学习）为代表的 LLM-RL 设定下，我们约定：

- prompt 记为 $x$；回复记为 $y = (y_1,\dots,y_{|y|})$；
- token 级状态 $s_t := (x, y_{\lt t})$，动作 $a_t := y_t$；
- 因此行为策略和参考策略在序列上的分布可写成
  $$
  \mu(y\mid x) = \prod_{t=1}^{|y|}\mu(a_t=y_t\mid s_t),\quad
  \pi_{\theta_{\text{old}}}(y\mid x) = \prod_{t=1}^{|y|}\pi_{\theta_{\text{old}}}(a_t=y_t\mid s_t).
  $$

此外，为描述"参考 vs 行为"的偏移，统一定义 token 级重要性比率

$$
\rho_t^{(\text{ref}\leftarrow\text{beh})} :=
\frac{\pi_{\theta_{\text{old}}}(a_t\mid s_t)}{\mu(a_t\mid s_t)},
$$

以及其对应的序列级版本

$$
\rho(y\mid x) := \frac{\pi_{\theta_{\text{old}}}(y\mid x)}{\mu(y\mid x)}
= \prod_{t=1}^{|y|} \rho_t^{(\text{ref}\leftarrow\text{beh})}.
$$

接下来，TIS / IcePop / MIS 的区别就在于"如何用这些 $\rho$ 缓解行为 vs 参考偏移带来的训练问题"。

### 4.1 TIS：token-level 截断 IS

TIS 直接对上述 $\rho_t^{(\text{ref}\leftarrow\text{beh})}$ 做截断，记

$$
\color{blue}{w_t = \min\big(\rho_t^{(\text{ref}\leftarrow\text{beh})},\ C_{\text{IS}}\big)}
$$

更新目标写成

$$
L_{\text{TIS}}(\theta)
= - \mathbb{E}_{(s_t,a_t)\sim\mu}\big[\,\color{blue}{w_t}\; g_\theta(t)\big]
$$

- 蓝色的 $\color{blue}{w_t}$ 是被截断的 IS 权重：极端大的比率被压到常数 $C_{\text{IS}}$。
- 从三策略 TRPO 的角度看，这相当于在 **token 样本层面**"软削弱"行为策略与参考策略严重不一致的样本，减小它们对有效训练分布的影响。
- 留意：这里的 $w_t$ 截断的是"行为 $\to$ 参考"的比率，而 $g_\theta(t)$ 里的 clipping 控制的是"参考 $\to$ 目标"的比率。两者分别作用在两类不同的偏移上，是独立的两步。

### 4.2 IcePop：MoE 场景下的 token-level 双侧 Mask

IcePop 同样以 $\rho_t^{(\text{ref}\leftarrow\text{beh})}$ 为度量，但采用 **双侧掩码**：

$$
\color{blue}{m_t = \mathbf{1}\big[C_{\text{low}} \le \rho_t^{(\text{ref}\leftarrow\text{beh})} \le C_{\text{high}}\big]}
$$

更新目标写成

$$
L_{\text{IcePop}}(\theta)
= - \mathbb{E}_{(s_t,a_t)\sim\mu}\big[\,\color{blue}{m_t}\; g_\theta(t)\big]
$$

- 蓝色的 $\color{blue}{m_t}$ 决定某个 token 是否参与更新：比率太大或太小的 token 被直接丢弃。
- 这相当于硬性裁掉"行为策略与参考策略极度不一致"的 token，只在 $\rho_t$ 适中的区域上优化，从样本集合层面更强地压缩有效不一致。

### 4.3 sequence-level MIS：按整条序列 Mask 的重要性采样

MIS 的核心操作是：**只保留 IS 比率不超过阈值 $C$ 的序列，其余序列的损失直接置零**。写成

$$
\color{blue}{
\rho(y\mid x)
\leftarrow
\rho(y\mid x)\,\mathbf{1}\{\rho(y\mid x)\le C\}
}
$$

在统一的损失形式下，可以写成

$$
L_{\text{MIS}}(\theta)
=-\,\mathbb{E}_{(x,y)\sim\mu}
\Big[
\color{blue}{\rho(y\mid x)\,\mathbf{1}\{\rho(y\mid x)\le C\}}
\;\cdot\; \sum_{t=1}^{|y|}g_\theta(t)
\Big],
$$

和前两种方法不同，这里在 mask 之外仍显式保留了序列级的 $\rho(y\mid x)$，也就是同时做了 off-policy 修正与序列级筛选。这里采用的是突出核心结构的统一写法，并不逐字对应每篇原文的全部实现细节。

> **注**：在这个统一写法里，$\rho(y\mid x)$ 负责"行为 $\to$ 参考"的序列级修正，而 $g_\theta(t)$ 中的 $r_t(\theta)$ 负责"参考 $\to$ 目标"的 token 级更新控制。两者叠加后，数值上相当于把行为到目标的比率拆成两段处理；实际实现里往往还会配合额外截断来控制方差。

从三策略 TRPO 的角度看，MIS 不在 token 上做截断，而是直接在**序列级**筛掉"行为策略与参考策略严重不一致"的轨迹，只在 $\rho(y\mid x)\le C$ 的子分布上优化，从而在轨迹级的有效训练分布上更保守地应对"行为 vs 参考"的偏移。就我个人偏好而言，sequence-level 的处理通常比 token-level 的修补更可信，因为长序列里的 token 级权重太容易被极端值支配。

### 4.4 一类 veto-style 的最保守拒绝机制（本文称为 WTRS）

verl 的 Token Veto 与 INTELLECT-3 的 token masking 都属于某种 veto-style 的拒绝采样。为统一讨论，本文把这类"token 级检测、sequence 级否决"的机制统称为 **Worst Token Reject Sampling（WTRS）**。这个名字只是本文的分析记号，不是文献里的标准术语；两者的具体实现也不完全相同。

在这个统一抽象下：

- **verl Token Veto**：在其 rollout correction 模块中，若轨迹中存在任意 token 使得 $\min_t \rho_t < \tau_{\text{veto}}$，就通过 `response_mask` 把整条序列剔除。阈值 $\tau_{\text{veto}}$ 可由用户配置。

- **INTELLECT-3 Token Masking**：在其异步分布式 RL 框架中，若任意 token 的比率低于 $10^{-5}$，就对整条轨迹做 masking。

两者的核心操作一致：**若轨迹中存在任意 token 的 IS 比率低于阈值 $\tau$，就把整条序列从训练中剔除**。写成

$$
\color{blue}{
m(y\mid x) = \mathbf{1}\Big\{\min_{t=1}^{|y|} \rho_t^{(\text{ref}\leftarrow\text{beh})} \ge \tau\Big\}
}
$$

在统一的损失形式下，可以写成

$$
L_{\text{WTRS}}(\theta)
=-\,\mathbb{E}_{(x,y)\sim\mu}
\Big[
\color{blue}{m(y\mid x)}
\;\cdot\; \sum_{t=1}^{|y|}g_\theta(t)
\Big],
$$

从三策略 TRPO 的角度看，WTRS 采用"token 级检测、序列级否决"的混合策略：在 **token-level** 检测极端不一致的信号，一旦发现就在 **sequence-level** 执行拒绝。这种"一票否决"的设计相当保守，代价是样本利用率可能很差；但当系统噪声很重时，它往往比 token-level 的局部修补更稳。

## 5. MoE 路由回放：它在三策略 TRPO 中到底做了什么？

在 MoE（Mixture-of-Experts）模型上，训推不一致往往首先表现为**路由不一致（routing inconsistency）**：即便参数相同，推理端和训练端也可能因为算子、并行或数值细节的微小差异而路由到不同的专家。一个很自然的工程应对是**路由回放（routing replay）**：在 rollout（推理）时记录实际命中的专家路径，训练时强制复用这些路由决策。

**建模前提：** 本节为了便于分析，把路由选择 $z$ 视为扩展动作空间的一部分，也就是把"选专家"和"生成 token"一起看作策略的联合决策。下面关于 routing replay"改写 surrogate objective"的结论，都建立在这一建模选择之上；若把路由视为纯粹的内部实现细节而非显式决策，分析框架也需要相应调整。

这类方法经常被直觉性地理解为"在样本层面修正行为 vs 参考的偏移，甚至直接压小 $\alpha_1$"。但从三策略 TRPO 的视角看，更准确的说法是：

> **路由回放并不是在原 surrogate objective 上直接缩小原定义下的策略距离项，而是把 surrogate objective 改写成另一个"带路由条件/替换"的目标。**
> 它让路由不一致在 loss 里"不可见"，但通常并不会直接让原定义下的 $\alpha_0$ 或 $\alpha_1$ 变小。

下面用一个**尽量简单**、把路由视为显式中间决策的建模来把这件事写清楚。这个建模主要用来说明 replay 如何改写 surrogate objective，并不是原 MDP 上单调改进界的直接推论。

### 5.1 MoE 下的 surrogate objective：把"路由"和"token 生成"拆开

把 MoE 抽象成两阶段随机决策："先选专家 $z$，再在该专家条件下生成 token $a$"。
目标策略可分解为

$$
\pi_\theta(a,z\mid s)=\omega_\theta(z\mid s)\,\pi_\theta(a\mid s,z),
$$

其中：

- $\omega_\theta(z\mid s)$ 是路由器（router）的分布；
- $\pi_\theta(a\mid s,z)$ 是在专家 $z$ 条件下的 token 分布。

在三策略 TRPO 中，我们真正想优化的 surrogate objective 为

$$
L_\mu(\pi_\theta) = \mathcal{J}(\mu) + \frac{1}{1-\gamma}
\mathbb{E}_{s\sim d_\mu}
\bigg[
\sum_z \omega_\theta(z\mid s)\,F_\theta(s,z)
\bigg],
$$

其中我把专家层的优势聚合写成

$$
F_\theta(s,z)
:=
\sum_a \pi_\theta(a\mid s,z)\,A_\mu(s,a,z).
$$

这里的 $A_\mu(s,a,z)$ 把"选择专家 $z$ 并生成 token $a$"视为联合决策的优势函数；也就是说，这一节是在一个把路由显式纳入决策过程的扩展建模下讨论 replay 的作用。

关键点：**在原始的 $L_\mu(\pi_\theta)$ 里，路由分布就是当前要更新的 $\omega_\theta$**。换言之，MoE 的 RL 训练不仅在更新 token 生成分布，也在更新路由器本身。

### 5.2 回放行为策略的路由（behavior-router replay / R3 类）

R3 的做法是：rollout 时记录推理端实际命中的专家集合 $M_\mu(s)$，训练时强制当前策略**只在该集合内路由**。可以把它写成对路由分布的"条件化投影"：

$$
\omega_\theta^{\text{R3}}(z\mid s)
:=
\frac{\omega_\theta(z\mid s)\,\mathbf{1}\{z\in M_\mu(s)\}}
     {\sum_{z'\in M_\mu(s)}\omega_\theta(z'\mid s)} .
$$

从而训练时实际优化的 surrogate objective 变为

$$
L_\mu^{\text{R3}}(\pi_\theta) =
\mathcal{J}(\mu) +
\frac{1}{1-\gamma}
\mathbb{E}_{s\sim d_\mu}
\bigg[
\sum_{z\in M_\mu(s)} \omega_\theta^{\text{R3}}(z\mid s)\,F_\theta(s,z)
\bigg].
$$

和原始 $L_\mu(\pi_\theta)$ 对比可以看到，R3 并没有让 $\omega_\theta$ 逼近 $\omega_{\text{old}}$ 或 $\omega_\mu$；它做的是：

- **把对 $z\sim\omega_\theta$ 的期望改成了对 $z\sim\omega_\theta(\cdot\mid z\in M_\mu(s))$ 的条件期望**；
- 等价地说，把路由的可行 support 缩到了 $M_\mu(s)$。

因此 R3 训练的是一个"被行为路由集合条件化后的 surrogate objective"，而不是原来的 $L_\mu(\pi_\theta)$。
好处是显著降方差、提升稳定性；代价是**在每个状态上都压缩了路由器探索 / 更新的自由度**。

### 5.3 回放参考策略的路由（reference-router replay）

另一类 routing replay 复用参考策略（old policy）的路由器 $\omega_{\text{old}}$。这等价于训练一个混合策略

$$
\hat\pi_\theta(a,z\mid s)
:=
\omega_{\text{old}}(z\mid s)\,\pi_\theta(a\mid s,z),
$$

对应 surrogate objective 为

$$
L_\mu^{\text{ref-replay}}(\pi_\theta) =
\mathcal{J}(\mu) +
\frac{1}{1-\gamma}
\mathbb{E}_{s\sim d_\mu}
\bigg[
\sum_z \omega_{\text{old}}(z\mid s)\,F_\theta(s,z)
\bigg].
$$

这意味着：

- 在 surrogate objective 中，路由器被**固定为旧路由器**，路由相关的"参考 vs 目标"差异在 loss 里被直接抹掉；
- 训练对"新路由器 $\omega_\theta$ 是否偏离 $\omega_{\text{old}}$"不再敏感，由此绕开了路由不一致带来的不稳定。

但注意这同样是**换目标**：

- 真实策略空间里的 $\alpha_0$ 并没有因此变小，只是被"用旧路由器重定义目标"而在 loss 中不可见；
- 路由器的学习被强行冻结或大幅削弱。

### 5.4 路由回放：条件化 surrogate，而不是直接缩小 $\alpha_0/\alpha_1$

把两类 replay 放在一起看，它们的共同点是：

1. **优化的都不是原始的 $L_\mu(\pi_\theta)$**，而是某个"路由被条件化 / 替换后的 surrogate objective"。
2. **它们通常不会直接收缩三策略 TRPO 下界里的 $\alpha_0,\alpha_1$**。replay 让路由不匹配不再显式出现在 loss 中，但不匹配在真实策略距离里仍然存在。
3. **实践上是在"用偏差换方差"**：回放往往显著降低方差、提升稳定性，代价是可能限制 MoE 在 RL 目标下学到更优的路由模式。

所以，从三策略 TRPO 的视角，更准确的理解是：

> **在本文这套显式路由建模下，routing replay 更适合被理解为一种被行为路由条件化的 surrogate objective，而不宜直接理解为对 $\alpha_0$ 或 $\alpha_1$ 的直接约束。**

这和 R3 类方法报告的 training-inference mismatch 或 measured policy KL 下降并不矛盾：那些指标衡量的是引入路由回放后的条件化训练路径与推理路径是否更一致；而本文这里讨论的是原始联合策略空间中的 token-routing 分布距离。两个说法的度量对象不同。

## 6. 讨论

本文的核心判断可以概括为：

> **许多"大模型 RL 训推不一致"和"异步训练"问题，在本文的视角下，其实都可以这样理解：在 TRPO 框架下，当行为策略 $\mu$ 和参考策略 $\pi_{\theta_{\text{old}}}$ 不一致时，二者之间的偏移（$\alpha_1$）往往没有被显式建模，因此常被低估乃至默认忽略。**

从两策略到三策略，本文实际做的是：

- 把 TRPO 的下界从"旧策略 vs 新策略"的叙述改写成"**行为策略 – 参考策略 – 目标策略**"三者的关系；
- 显式地写出两类 TV 距离：
  - **参考 vs 目标** 的偏移 $\alpha_0$，对应 PPO / GRPO 类工作中最常见的 KL / clip / trust region；
  - **行为 vs 参考** 的偏移 $\alpha_1$，对应异步框架、训推差异、MoE 路由、kernel 非确定性等现实因素；
- 得到一个很直接的结论：
  在 $\epsilon_\mu$ 有界等前提下，代理目标 $L_\mu(\pi_\theta)$ 与真实性能 $\mathcal{J}(\pi_\theta)$ 的差距至多被 $C(\alpha_0 + \alpha_1)$ 所上界。

在这一视角下：

- Decoupled PPO / AReaL 可以看作在**形式上承认"三策略存在"**，并尝试在目标函数上将"行为分布"和"参考策略"解耦；
- PPO / GRPO 主要在"参考 vs 目标"这一侧控制更新幅度；GSPO 也在这一侧，但它进一步把 ratio 和优化的基本单位从 token 改到 sequence，从而改变了偏移被度量和约束的粒度；
- TIS、IcePop、MIS、WTRS 则是通过 IS 或掩码机制在样本层面缓解"行为 vs 参考"偏移带来的估计偏差与方差问题：
  - TIS：用 token-level 截断权重削弱比率过大的样本；
  - IcePop：在 MoE 场景下用 token-level 双侧掩码硬性丢弃"极端不一致"的 token；
  - MIS：在 sequence-level 直接屏蔽整条"比率过大"的轨迹；
  - WTRS：在 token-level 检测比率过小的信号，一旦发现就在 sequence-level 拒绝整条轨迹；
- **routing replay（路由回放）在三策略 TRPO 的视角下更适合看作"改写 surrogate objective"，而不宜简单理解为"直接缩小某个距离项"**：无论回放行为路由（R3 类）还是回放参考路由，它们都把原本的 $L_{\mu}(\pi_{\theta})$ 改成了一个路由被条件化/替换后的 surrogate objective，用**一定的目标偏差与路由学习自由度的收缩**换取**更低的方差与更高的稳定性**。因此它通常不会直接收缩 $\alpha_0$ 或 $\alpha_1$，而是让路由不一致在 loss 中"不可见"；
- 《RL 老训崩？训推差异是基石》以及前文提到的 _Defeating Nondeterminism in LLM Inference_ 等工程经验，则可以理解为在**系统侧和数值实现侧**尽量把 $\alpha_1$ 压低，让算法层的假设不至于完全失效。

还有一点值得单独强调：定理 2 里的 $\alpha_0$ 和 $\alpha_1$ 都是 worst-case TV，在 LLM 巨大的状态/动作空间里几乎无法直接观测。实践里更可操作的往往不是它们本身，而是一些工程 proxy，例如：logged states 上的平均 KL、token 或 sequence importance weights 的分位数、effective sample size（ESS）、rejection / masking rate，以及异步系统里的 staleness（版本差）。理论量告诉我们该关心哪两类偏差，工程指标则帮助我们判断这两类偏差是否已经开始失控。

如果只让我在 $\alpha_0$ 和 $\alpha_1$ 之间先盯一个，我大概率会先盯 $\alpha_1$。在今天的大模型 RL 系统里，后者往往更隐蔽，也更容易先把训练拖出"近似 PPO / TRPO"该有的工作区间。

从这个统一视角出发，我觉得至少有两个问题值得继续追：

- 在什么条件下，我们还能把"大模型 RL 训练"理解成某种意义上的"近似 TRPO / PPO"？
- 对一个具体的 RL 系统，应该先把精力投在更稳的 $\alpha_0$ 控制上，还是先把 $\alpha_1$ 压回可接受区间？我的偏好是先处理后者——如果行为分布已经飘掉，再漂亮的 trust region 也只是对错对象做约束。

如果这篇文章有什么实际作用，我希望至少是把一个常被忽略的问题说得更具体：很多看起来像"PPO 不稳"的现象，先坏掉的往往不是 clip 或 KL，而是 $\mu \neq \pi_{\theta_{\text{old}}}$。把这三种策略分开写，通常能更快看到真正的瓶颈。

## 参考文献与延伸阅读

1. John Schulman, Sergey Levine, Philipp Moritz, Michael I. Jordan, Pieter Abbeel. "Trust Region Policy Optimization" (TRPO). arXiv:1502.05477. <https://arxiv.org/abs/1502.05477>
2. Jacob Hilton, Karl Cobbe, John Schulman. "Batch size-invariance for policy optimization" (Decoupled PPO). arXiv:2110.00641. <https://arxiv.org/abs/2110.00641>
3. Joshua Achiam, David Held, Aviv Tamar, Pieter Abbeel. "Constrained Policy Optimization" (CPO). arXiv:1705.10528. <https://arxiv.org/abs/1705.10528>
4. James Queeney, Ioannis Ch. Paschalidis, Christos G. Cassandras. "Generalized Proximal Policy Optimization with Sample Reuse" (GePPO). arXiv:2111.00072. <https://arxiv.org/abs/2111.00072>
5. Wei Fu, Jiaxuan Gao, Xujie Shen, et al. "AReaL: A Large-Scale Asynchronous Reinforcement Learning System for Language Reasoning". arXiv:2505.24298. <https://arxiv.org/abs/2505.24298>
6. Chujie Zheng, Shixuan Liu, Mingze Li, et al. "Group Sequence Policy Optimization" (GSPO). arXiv:2507.18071. <https://arxiv.org/abs/2507.18071>
7. Wenhan Ma, Hailin Zhang, Liang Zhao, et al. "Stabilizing MoE Reinforcement Learning by Aligning Training and Inference Routers". arXiv:2510.11370. <https://arxiv.org/abs/2510.11370>

```bibtex
@misc{WangZhang2025ThreePolicyTRPO,
  author       = {Wang, Xihuai and Zhang, Shao},
  title        = {From Two Policies to Three: Extending TRPO under Behavior-Reference Policy Mismatch in LLM RL},
  year         = {2025},
  month        = nov,
  day          = {15},
  url          = {https://xihuai18.github.io/reinforcement-learning/2025/11/15/three-policy-zh.html},
  urldate      = {2025-11-23}
}
```

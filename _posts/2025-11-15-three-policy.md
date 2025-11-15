---
layout: post
title: 从两策略到三策略：行为策略和参考策略不一致下的 PPO 扩展
date: 2025-11-15
description: 在大模型强化学习中，因为推理框架和训练框架的不一致，以及异步训练框架下行为策略分布的多样性，行为策略与参考策略不一致的问题变得尤为突出。本文分析了行为策略与参考策略不一致问题在 TRPO 框架下的影响，并在这一分析基础上梳理了当前对这一问题的不同解决方法。
categories: reinforcement-learning
---

最近看到不少关于大模型强化学习中“训推不一致”和“异步训推框架”的讨论，我自己的直觉是：这些看上去复杂多样的问题，很大一部分其实都在围绕一个更基础的矛盾打转——**行为策略（behavior policy）和参考策略（reference policy）不一致。**

本文先简单梳理一下我*目前看到的*相关工作，然后再尝试从“行为策略 vs 参考策略”的角度，把它们串到同一条线上。

在本文中我会用：

- **行为策略** $\mu$：实际负责生成 rollout 的策略，也就是“你在什么分布下采样到了这些数据”。  
  在现代 LLM RL 系统里，它对应的是推理引擎里的那套实现（vLLM / SGLang 等），在异步框架下往往还是**多个 worker 策略的混合分布**。
- **参考策略** $\pi_{\theta_{\text{old}}}$：训练目标里拿来做重要性采样、Clipping 或 KL 约束的策略，  
  典型地就是 PPO / GRPO 里的 “旧策略”（old policy）。
- **目标策略** $\pi_{\theta}$：训练目标里要优化的策略，也就是“你想让模型变成什么样”。  
  典型地就是 PPO / GRPO 里的 “新策略”（new policy）。

在最经典、理想化的设定里，我们通常**默认** $\mu = \pi_{\theta_{\text{old}}}$。但在现实系统中，受异步更新、不同推理/训练后端、MoE 路由波动甚至硬件数值差异等因素影响，这二者往往会出现不同程度的偏离。

下面按时间线简单列一下我印象比较深的一些工作（只代表我个人看到的片面子集），也顺便标注它们主要是在解决哪一层面的不一致：

- （算法层）[Decoupled PPO](https://arxiv.org/pdf/2110.00641) 率先指出在信赖域策略优化（TRPO 和 PPO）方法中，“旧策略”（old policy）实际承担了两个不同的角色：一是用于重要性采样进行异策略修正，在这个目的下，“旧策略”用于代表训练数据集所服从的行为策略（behavior policy）；二是用于限制新策略的更新幅度，在这个目的下，“旧策略”被用于衡量新旧策略的变化程度，称作近端策略（proximal policy，对应本文中的“参考策略”）。文章指出这两个目的下的“旧策略”可以是不同的策略，从而提出了 Decoupled PPO 更新目标，把“采样用谁”和“对谁做 trust region”在形式上解耦开来。
- （异步框架）[AReaL](https://arxiv.org/abs/2505.24298) 关注到了异步训练框架下行为策略与参考策略不一致的问题：rollout 往往由滞后的参数版本或不同 worker 产生。文章在异步框架下采用了 Decoupled PPO 风格的目标，将“行为策略分布”与“参考策略”显式区分开来，从而在异步 setting 下仍然维持类似 PPO 的优化性质。
- （MoE & 序列级目标）[GSPO](https://arxiv.org/abs/2507.18071) 从 GRPO 在长序列和 MoE 模型上的稳定性问题出发，指出 token-level 的 PPO / GRPO 在专家路由高度波动（尤其是新旧策略之间的路由差异）时会引入巨大的方差与不稳定。GSPO 提出在 **sequence-level** 定义 PPO-style 目标与比率约束，用整条序列的比率来约束更新，从而在 MoE 场景下显著缓解由路由不一致带来的训练崩溃问题。
- （系统层训推不一致）[Your Efficient RL Framework Secretly Brings You Off-Policy RL Training](https://fengyao.notion.site/off-policy-rl#28b721e3f6c480c3a756f8fb319e860d) 关注到了现有的一些大模型强化学习训练框架（如 VeRL）中，推理框架和训练框架在不少相同的功能模块上有不同的实现（例如 vLLM 和 FSDP / Megatron 等算子上的差异），导致行为策略 $\mu$ 与参考策略 $\pi_{\theta_{\text{old}}}$ 不一致。这种不一致使得原本假定为同策略（on-policy）的训练，实际上变成了带有明显偏差的异策略（off-policy）训练。文章总结了两种处理这一问题的现有方法：PPO-IS 与 vanilla-IS，并提出在 **token-level** 做截断重要性采样（truncated IS, TIS），以减少训推不一致程度较重的样本在训练中的影响。作者还写了两篇更为基础的分析文章，从原理上分析训推不一致问题：[Part I](https://fengyao.notion.site/pg-seq-token-part1-basics) 和 [Part II](https://fengyao.notion.site/pg-seq-token-part2-mismatch)。
- （推理数值层）[Defeating Nondeterminism in LLM Inference](https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference) 指出批处理大小不变性（batch-size invariance）的缺失是大模型推理框架随机性的核心来源之一：同一个输入在不同的 batch 组合和 kernel 路径下，得到的概率分布会发生可观差异。这意味着即便“名义上”是同一套参数，真实运行时的行为策略 $\mu$ 也会因为系统负载和调度差异而波动，从而进一步加剧训推不一致。
- （MoE 纠偏）[Small Leak Can Sink a Great Ship—Boost RL Training on MoE with 𝑰𝒄𝒆𝑷𝒐𝒑!](https://ringtech.notion.site/icepop) 观察到上述训推不一致问题在 MoE 模型上会进一步加剧：路由本身就对微小扰动高度敏感，再叠加推理/训练实现差异和异步采样，很容易放大偏差。文章提出 IcePop 方法：在 **token-level** 通过计算重要性采样比率，对过于大或者过于小的比率进行双侧掩码（masking），将这些“噪声较大”的数据从梯度中丢弃，从而稳定 MoE 上的 RL 训练。
- （系统性视角）[When Speed Kills Stability: Demystifying RL Collapse from the Training-Inference Mismatch](https://yingru.notion.site/When-Speed-Kills-Stability-Demystifying-RL-Collapse-from-the-Training-Inference-Mismatch-271211a558b7808d8b12d403fd15edda) 系统性分析了训推不一致的各种成因，包括智能体工作流中引入的大量分布外和低概率信息、硬件和内核实现带来的计算不确定性，并分析了在 **token-level** 进行重要性采样如何在长序列上引入严重的偏差。文章进一步提出在 **sequence-level** 计算重要性采样掩码（sequence-level masked IS, sequence-level MIS）：只丢弃那些整条序列的重要性采样比率过大的数据，从而在控制偏差的同时，显著抑制由极端样本导致的训练崩溃。文中给出了较为完整的理论推导和丰富的实验支撑。
- （工程经验）[RL老训崩？训推差异是基石](https://zhuanlan.zhihu.com/p/1959976628290590602) 则更多从实践角度出发，分享了如何在实现上尽可能靠近“训推一致”的经验，包括如何选用一致的算子和精度配置、如何监控与约束训练端和推理端 log-prob 的偏差等，更着力于从训推框架层面入手，在工程上尽量从根本缓解训推差异问题。

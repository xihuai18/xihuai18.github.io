---
layout: post
title: "Understanding KL Divergence Estimators in RL: From Value Approximation to Gradient Estimation"
date: 2025-12-01
description: "How we approximate KL directly affects stability. This post dissects three classic estimators k1, k2, k3, covering on-policy and off-policy, and gives practical rules for using them for reward penalties vs. losses that backpropagate."
categories: reinforcement-learning
lang: en
zh_url: /reinforcement-learning/2025/12/01/kl-estimators-zh.html
zhihu_url: https://zhuanlan.zhihu.com/p/1978993413425763764
---



![Mini-class](/assets/img/kl-estimators/kl-estimator.png){: style="display:block;margin:0 auto;width:95%;max-width:100%;" }

> How we approximate KL divergence directly affects training stability. This post systematically analyzes three estimators $k_1, k_2, k_3$ in both on-policy and off-policy scenarios, and gives practical guidelines for choosing them when KL is used as a reward penalty versus when it is used as a loss for backpropagation.

## Introduction: The Role of KL Divergence in Reinforcement Learning

In policy optimization (PPO, GRPO, etc.) and alignment training (RLHF/RLAIF), **KL penalty** is the core mechanism to constrain the new policy from deviating too far from the reference policy, preventing training instability or policy collapse. However, implementing KL penalty involves multiple layers of choices: **which estimator** ($k_1$, $k_2$, $k_3$), **who to sample from** (on-policy vs off-policy), and **how to use it** (as a loss for gradient backpropagation or as a reward penalty). This post systematically dissects these choices and their interrelationships, helping readers clarify the relevant concepts.

### The Distinction Between Forward KL and Reverse KL

Let $q_\theta$ be the current actor, $p$ the reference policy. The two directions are:

**Reverse KL:**
$$
D_{\mathrm{KL}}(q_\theta \| p) = \mathbb{E}_{x \sim q_\theta}\left[\log \frac{q_\theta(x)}{p(x)}\right]
$$

<figure style="text-align:center;" markdown="0">
	<img src="/assets/img/kl-estimators/kl-estimator-reverse.png" style="width:80%;max-width:100%;">
	<figcaption style="font-size:0.9em;color:gray;">Image source: <a href="https://dibyaghosh.com/blog/probability/kldivergence/">Dibya Ghosh's Blog</a></figcaption>
</figure>

**Forward KL:**
$$
D_{\mathrm{KL}}(p \| q_\theta) = \mathbb{E}_{x \sim p}\left[\log \frac{p(x)}{q_\theta(x)}\right]
$$

<figure style="text-align:center;" markdown="0">
	<img src="/assets/img/kl-estimators/kl-estimator-forward.png" style="width:80%;max-width:100%;">
	<figcaption style="font-size:0.9em;color:gray;">Image source: <a href="https://dibyaghosh.com/blog/probability/kldivergence/">Dibya Ghosh's Blog</a></figcaption>
</figure>

**Intuition:**
- **Reverse KL** is mode-seeking: policy concentrates on high-probability regions of $p$, possibly sacrificing diversity.
- **Forward KL** is mass-covering: policy tries to cover the support of $p$.

RLHF typically uses **reverse KL** because we want the actor not to move too far from the reference, not necessarily to cover every mode.

### The Three Core Questions: Who to Sample From, What to Estimate, How to Use

When implementing KL penalty in practice, we need to answer three interrelated questions:

1. **Who to sample from?** Do samples come from the current policy $q_\theta$ (on-policy), or from a behavior policy $\mu$ (off-policy)?
2. **What to estimate?** Are we trying to estimate reverse KL $D_{\mathrm{KL}}(q_\theta \| p)$ or forward KL $D_{\mathrm{KL}}(p \| q_\theta)$?
3. **How to use it?** Is the KL term used as a loss for gradient backpropagation, or as a reward penalty (stop-gradient)?

These three questions' different combinations determine which estimator should be used. The goal of this post is to systematically clarify these choices and their interrelationships.

## Preliminaries: Notation and Basic Concepts

Before diving into the analysis, let's unify our notation and derive two fundamental results that will be used repeatedly.

### Notation

- $q_\theta$: Current actor policy (parameterized by $\theta$)
- $p$: Reference policy (independent of $\theta$)
- $\mu$: Behavior policy for off-policy sampling (independent of $\theta$)
- $s_\theta(x) = \nabla_\theta \log q_\theta(x)$: Score function
- $w(x) = \frac{q_\theta(x)}{\mu(x)}$: Importance weight
- $\text{sg}(\cdot)$: Stop-gradient operation (`.detach()` in code)

### Score Function and True KL Gradients

The score function has an important property: $\mathbb{E}_{q_\theta}[s_\theta] = 0$ (since $\int \nabla_\theta q_\theta dx = \nabla_\theta \int q_\theta dx = \nabla_\theta 1 = 0$).

Using this property, we can derive the **true gradients** of forward and reverse KL divergences.

**Reverse KL Gradient:**

$$
D_{\mathrm{KL}}(q_\theta \| p) = \int q_\theta(x) \log \frac{q_\theta(x)}{p(x)} dx
$$

Differentiating with respect to $\theta$ (using product rule):

$$
\nabla_\theta D_{\mathrm{KL}}(q_\theta \| p) = \int \nabla_\theta q_\theta \cdot \log \frac{q_\theta}{p} dx + \int q_\theta \cdot \nabla_\theta \log \frac{q_\theta}{p} dx
$$

Using $\nabla_\theta q_\theta = q_\theta \cdot s_\theta$, $\nabla_\theta \log q_\theta = s_\theta$, and $\nabla_\theta \log p = 0$:

$$
= \mathbb{E}_q\left[s_\theta \cdot \log \frac{q_\theta}{p}\right] + \mathbb{E}_q[s_\theta] = \mathbb{E}_q\left[s_\theta \cdot \log \frac{q_\theta}{p}\right]
$$

Thus:

$$
\boxed{\nabla_\theta D_{\mathrm{KL}}(q_\theta \| p) = \mathbb{E}_q\left[s_\theta \cdot \log \frac{q_\theta}{p}\right] = -\mathbb{E}_q\left[s_\theta \cdot \log \frac{p}{q}\right]}
$$

**Forward KL Gradient:**

$$
D_{\mathrm{KL}}(p \| q_\theta) = \int p(x) \log \frac{p(x)}{q_\theta(x)} dx
$$

Since $p(x)$ is independent of $\theta$:

$$
\nabla_\theta D_{\mathrm{KL}}(p \| q_\theta) = \int p(x) \cdot \nabla_\theta \left(-\log q_\theta(x)\right) dx = -\mathbb{E}_p[s_\theta]
$$

To estimate this using samples from $q$, apply importance sampling:

$$
-\mathbb{E}_p[s_\theta] = -\mathbb{E}_q\left[\frac{p}{q_\theta} \cdot s_\theta\right] = -\mathbb{E}_q\left[\frac{p}{q} \cdot s_\theta\right]
$$

Using $\mathbb{E}_q[s_\theta] = 0$, this can be rewritten as:

$$
\boxed{\nabla_\theta D_{\mathrm{KL}}(p \| q_\theta) = \mathbb{E}_q\left[\left(1-\frac{p}{q}\right) \cdot s_\theta\right]}
$$

With these two results, we can later determine which KL's true gradient each estimator's gradient expectation corresponds to.

## Three Estimators: Definitions and Design Principles

Let $r(x) = \dfrac{p(x)}{q_\theta(x)}$. John Schulman defined three single-sample estimators:

### $k_1$: the naive estimator

$$
k_1(x) = -\log r = \log q_\theta(x) - \log p(x)
$$

Direct log-ratio. It is unbiased for reverse KL, but **can be negative** while KL is always nonnegative, giving huge variance because positive and negative samples cancel.

### $k_2$: an f-divergence, lower variance

$$
k_2(x) = \frac{1}{2}(\log r)^2
$$

**Motivation:** $k_1$ can be positive or negative; $k_2$ squares it so **every sample is positive**, each telling you how far $p$ and $q$ differ.

**Why tiny bias?** $k_2$ is an **f-divergence** with $f(x) = \tfrac{1}{2}(\log x)^2$. All smooth f-divergences have the same second-order expansion near $q \approx p$:

$$
D_f(p, q_\theta) = \frac{f^{\prime\prime}(1)}{2} \theta^T F \theta + O(\theta^3)
$$

KL corresponds to $f(x) = -\log x$, so $f^{\prime\prime}(1) = 1$. For $k_2$, $f^{\prime\prime}(1) = 1$ as well. **When policies are close, $k_2$ tracks true KL almost identically**, bias only appears in higher-order terms.

### $k_3$: control variate, "optimal" shape

$$
k_3(x) = r - 1 - \log r
$$

**Motivation:** we want **unbiased and low variance**. Add a **control variate** to $k_1$: something zero-mean and negatively correlated.

Because $\mathbb{E}_q[r - 1] = 1 - 1 = 0$, for any $\lambda$:

$$
k_1 + \lambda(r - 1) = -\log r + \lambda(r - 1)
$$

is still unbiased.

**Why $\lambda = 1$?** By concavity of $\log$, $\log x \le x - 1$, so

$$
k_3 = (r - 1) - \log r \ge 0
$$

It is **always nonnegative**, avoiding the cancelation problem.

**Geometric view:** $k_3$ is a **Bregman divergence** for $\phi(x) = -\log x$. Its tangent at $x=1$ is $y = 1 - x$, so

$$
\begin{aligned}
D_\phi(r, 1) &= \phi(r) - \phi(1) - \phi'(1)(r - 1) \\
&= -\log r - 0 - (-1)(r - 1) \\
&= r - 1 - \log r = k_3.
\end{aligned}
$$

Convexity keeps $\phi$ above its tangent, so this gap is **nonnegative**. As $r \to 1$, the gap shrinks quadratically $(r-1)^2$, explaining the low variance when policies are close.

### Quick Comparison

| Estimator |        Definition        |             Design idea             |
| :-------: | :----------------------: | :---------------------------------: |
|   $k_1$   |        $-\log r$         |           Naive log-ratio           |
|   $k_2$   | $\tfrac{1}{2}(\log r)^2$ | f-divergence, KL-matching 2nd order |
|   $k_3$   |     $r - 1 - \log r$     |      Control variate + Bregman      |

These three estimators arise from different design philosophies. Next, we analyze their properties when **estimating KL values** — specifically, their bias and variance characteristics.

After understanding the definitions and design principles of the three estimators, we first analyze their properties in **estimating KL values** — that is, bias and variance.

## Value Estimation: Bias and Variance

This section analyzes the properties of the three estimators when **estimating KL values**. These properties are fundamental in any usage scenario.

Assume samples from $q_\theta$ to estimate reverse KL $D_{\mathrm{KL}}(q_\theta \| p)$.

### Unbiasedness Analysis

$$
\begin{aligned}
\mathbb{E}_{q}[k_1] &= \mathbb{E}_{q}\left[\log \tfrac{q}{p}\right] = D_{\mathrm{KL}}(q \| p) \quad \textbf{(unbiased)}\\
\mathbb{E}_{q}[k_3] &= \mathbb{E}_{q}\left[\frac{p}{q} - 1 - \log \frac{p}{q}\right] = 1 - 1 + D_{\mathrm{KL}}(q \| p) = D_{\mathrm{KL}}(q \| p) \quad \textbf{(unbiased)}\\
\mathbb{E}_{q}[k_2] &= \tfrac{1}{2}\mathbb{E}_{q}\left[\left(\log \frac{p}{q}\right)^2\right] \neq D_{\mathrm{KL}}(q \| p) \quad \textbf{(biased)}
\end{aligned}
$$

**Conclusion**: For estimating reverse KL **values**, $k_1$ and $k_3$ are unbiased estimators, while $k_2$ is biased.

### Variance Characteristics

John Schulman's toy experiments ($q = \mathcal{N}(0,1)$, $p = \mathcal{N}(0.1,1)$, true KL = 0.005):

| Estimator | bias/true | stdev/true |
| :-------: | :-------: | :--------: |
|   $k_1$   |     0     |     20     |
|   $k_2$   |   0.002   |    1.42    |
|   $k_3$   |     0     |    1.42    |

When KL is large ($p = \mathcal{N}(1,1)$, true KL = 0.5):

| Estimator | bias/true | stdev/true |
| :-------: | :-------: | :--------: |
|   $k_1$   |     0     |     2      |
|   $k_2$   |   0.25    |    1.73    |
|   $k_3$   |     0     |    1.7     |

**Core intuitive understanding**:
- $k_1 = -\log \frac{p}{q}$ starts with a first-order term and when $\frac{p}{q}$ is close to 1, it fluctuates greatly and can be negative
- $k_3 = \frac{p}{q} - 1 - \log \frac{p}{q}$ is a second-order quantity near $\frac{p}{q}=1$ and always non-negative, thus having lower variance when policies are close
- But when coverage is severely insufficient ($\frac{p}{q}$ can explode), $k_3$'s variance can increase due to weight explosion; in this case, $k_1$ is actually more stable

### Summary of Value Estimation

| Estimator |  Bias for value  | Variance characteristics |
| :-------: | :--------------: | :----------------------: |
|   $k_1$   |     Unbiased     | High (can be +/-)        |
|   $k_2$   | Biased (minimal) | Low (always positive)    |
|   $k_3$   |     Unbiased     | Low (always positive)    |

From the perspective of value estimation, $k_3$ is the optimal choice as "unbiased + low variance".

> **Note**: To estimate the **forward KL value** $D_{\mathrm{KL}}(p \| q) = \mathbb{E}_p\left[\log \frac{p}{q}\right]$, but only sample from $q$, use importance sampling $\mathbb{E}_q\left[\frac{p}{q} \log \frac{p}{q}\right]$.

However, before choosing an estimator, there's a more fundamental question to answer: **should KL be added to rewards, or be part of the loss?** This choice fundamentally affects optimization behavior and credit assignment.

## Two Ways to Use KL Penalty

Having understood the value properties of these estimators, we must address a more fundamental question: **How should the KL penalty be integrated into the RL algorithm?** This implementation choice determines whether we need only consider the estimator's value properties, or must also account for its gradient behavior.

Recall the objective function for KL-regularized reinforcement learning:

$$
J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta} \left[ \sum_{t=0}^T \gamma^t r(s_t, a_t) \right] - \beta \cdot D_{\mathrm{KL}}(\pi_\theta \| \pi_{\text{ref}})
$$

This mathematical form looks unified, but when implementing it in Actor-Critic based algorithms (like PPO), it gives rise to two fundamentally different implementation paradigms — they may differ by only a few lines of code, but correspond to completely different optimization semantics.

> **Notation**: In this section, we use $\text{KL}_t$ or $\text{KL}(s)$ to generically refer to a token/state-level KL estimator (such as $k_1, k_2, k_3$), with specific definitions from the earlier section "Three Estimators: Definitions and Design Principles".

### As Loss: KL Participates in Gradient Backpropagation

```python
actor_loss = -advantage * log_prob + beta * kl  # kl participates in gradient
```

The critic only learns environment value; KL as a regularization term for the actor directly participates in loss gradient backpropagation.

### As Reward: KL Added to Reward Shaping

```python
kl = compute_kl(log_prob_q, log_prob_p).detach()
shaped_reward = reward - beta * kl
```

KL is treated as part of the environment reward, using shaped reward for standard actor-critic updates. The KL term does not participate in loss gradient backpropagation.

These two approaches may seem like just a `.detach()` difference in code, but in reality they correspond to fundamentally different optimization semantics.

### Core Differences Between the Two Approaches

#### Different Optimization Targets

**KL as Loss**: Optimizes **original task + supervised regularization**. KL doesn't change the MDP definition, it's just an external constraint term.

**KL as Reward**: Optimizes a **regularized new MDP** where the reward function becomes $\tilde{r}(s,a) = r(s,a) - \beta \cdot \text{KL}(s)$.

**Intuition**: The former is "adding constraints under the original rules"; the latter is "changing the game rules".

#### Different Gradient Paths

**KL as Loss**: The gradient splits into two independent paths:

$$
g_{\text{loss}} = \underbrace{\mathbb{E}\left[\nabla_\theta \log \pi_\theta \cdot A_t^{\text{env}}\right]}_{\text{RL gradient}} + \underbrace{\beta \cdot \nabla_\theta \text{KL}}_{\text{KL explicit gradient}}
$$

**KL as Reward**: Single policy gradient, KL influence is **reflected indirectly through advantage**:

$$
g_{\text{reward}} = \mathbb{E}\left[\nabla_\theta \log \pi_\theta \cdot \tilde{A}_t\right], \quad \tilde{A}_t \text{ based on } (r_t - \beta \cdot \text{KL}_t)
$$

**Key distinction**: Is KL's force "a separate force" or "multiplied on advantage"? The former's KL gradient is deterministic, unaffected by critic quality.

#### Different Value Functions and Credit Assignment

**Value Function**:

**KL as Loss**: Critic only learns environment value

$$
V^{\text{env}}(s) = \mathbb{E}\left[\sum_t \gamma^t r_t\right]
$$

Cleaner separation of concerns, making it easier to monitor task return and KL divergence separately.

**KL as Reward**: Critic learns mixed value

$$
V^{\text{reg}}(s) = \mathbb{E}\left[\sum_t \gamma^t (r_t - \beta \cdot \text{KL}_t)\right]
$$

**Credit Assignment**:

Consider a scenario: first few steps are routing behavior, final step has high reward but also high KL.

**KL as Loss**: The terminal state's KL only appears in that state's gradient term; the policy is still willing to **visit high-reward regions but locally correct** behavior.

**KL as Reward**: The terminal state's large KL is **propagated back to all previous steps** through TD, so the policy tends to **fundamentally avoid** high-KL regions — this is "planning-based KL budget allocation".

### Why This Distinction Matters

|      Dimension      |        KL as Loss (gradient backprop)        |        KL as Reward (stop-grad)         |
| :-----------------: | :------------------------------------------: | :-------------------------------------: |
| Optimization target |   Original task + supervised regularization  |          Regularized new MDP            |
|   Actor gradient    |       RL gradient + explicit KL gradient     |  Single PG, based on shaped advantage   |
|       Critic        | Learns $V^{\text{env}}$: only environment reward | Learns $V^{\text{reg}}$: reward + KL mixed |
|  Credit Assignment  |          Local per-state, no planning        |      Multi-step backprop, planning-capable |
|     Focus on        | Estimator's **explicit gradient** (corresponds to which optimization objective) | KL **value** + whether induced **policy gradient** is correct |

**One-liner**: KL as loss makes the agent "visit but locally correct" — constraints are more local and flexible; KL as reward makes the agent "plan to avoid high-KL paths" — constraints are more global and thorough.

**Selection Guidelines**:
- If you want constraints to be "**corrective**", allowing the agent to explore but locally correct behavior, choose **KL as Loss**
- If you want constraints to be "**preventive**", making the agent fundamentally avoid high-KL regions, choose **KL as Reward**

After understanding the difference between these two paradigms, we can clarify:
- **KL as Loss**: Needs correct explicit gradients of the KL estimator, caring about which optimization objective the gradient corresponds to
- **KL as Reward**: Needs accurate value estimation of KL, while also caring about whether the induced policy gradient is correct

Below we deeply analyze the gradient properties of estimators according to the two usage modes of "as Loss" and "as Reward".

## Gradient Analysis When Used as Loss

When KL serves as a loss term participating in gradient backpropagation, we must understand the optimization objective each estimator corresponds to. This is the most subtle yet critical aspect in practical implementations.

### On-policy Scenario

We start the analysis from the on-policy scenario, i.e., samples come from the current policy $q_\theta$.

#### Two Differentiation Orders: Grad-then-Expectation vs. Expectation-then-Grad

In code implementation, there are two paths:

1. **Grad-then-expectation**: Compute gradient for each sample's $k_i(x)$, then take expectation of gradients (Monte Carlo estimation)
2. **Expectation-then-grad**: Treat $\mathbb{E}_q[k_i]$ as a loss function, then differentiate the analytical expression

**In typical deep learning code, we actually execute "grad-then-expectation"** — autograd computes gradients for each sample, then averages over the batch.

#### Gradient Derivations for the Three Estimators

Now we compute the gradients of the three estimators to see which KL's true gradient their expectations correspond to (refer to the "Preliminaries" section).

**Deriving $\nabla_\theta k_1$**:

$$
k_1 = -\log \frac{p(x)}{q_\theta(x)} = \log q_\theta(x) - \log p(x)
$$

$$
\nabla_\theta k_1 = \nabla_\theta \log q_\theta(x) - \nabla_\theta \log p(x) = s_\theta - 0 = s_\theta
$$

**Deriving $\nabla_\theta k_2$**:

$$
k_2 = \frac{1}{2}\left(\log \frac{p}{q}\right)^2
$$

By the chain rule:

$$
\begin{aligned}
\nabla_\theta k_2 
&= \left(\log \frac{p}{q}\right) \cdot \nabla_\theta\left(\log \frac{p}{q}\right) \\
&= \left(\log \frac{p}{q}\right) \cdot \nabla_\theta(\log p(x) - \log q_\theta(x)) \\
&= \left(\log \frac{p}{q}\right)(-s_\theta) \\
&= - \left(\log \frac{p}{q}\right) s_\theta.
\end{aligned}
$$

**Deriving $\nabla_\theta k_3$**:

$$
k_3 = \frac{p}{q} - 1 - \log \frac{p}{q}
$$

First, compute $\nabla_\theta \frac{p}{q}$. Since $\frac{p}{q} = p(x) \cdot q_\theta(x)^{-1}$:

$$
\nabla_\theta \frac{p}{q} = p(x) \cdot (-1) \cdot q_\theta(x)^{-2} \cdot \nabla_\theta q_\theta(x) = -\frac{p(x)}{q_\theta(x)} \cdot \frac{\nabla_\theta q_\theta(x)}{q_\theta(x)} = -\frac{p}{q} \cdot s_\theta
$$

Then compute $\nabla_\theta \log \frac{p}{q}$:

$$
\nabla_\theta \log \frac{p}{q} = \frac{q}{p} \nabla_\theta \frac{p}{q} = \frac{q}{p} \cdot \left(-\frac{p}{q} \cdot s_\theta\right) = -s_\theta
$$

Therefore:

$$
\nabla_\theta k_3 = \nabla_\theta \frac{p}{q} - 0 - \nabla_\theta \log \frac{p}{q} = -\frac{p}{q} \cdot s_\theta - (-s_\theta) = \left(1 - \frac{p}{q}\right) \cdot s_\theta
$$

Taking expectations under $q_\theta$:

| Estimator |                                        $\mathbb{E}_{q}[\nabla_\theta k_i]$                                         |       Equivalent to       |
| :----: | :----------------------------------------------------------------------------------------------------------------: | :----------------: |
| $k_1$  |                                           $\mathbb{E}_{q}[s_\theta] = 0$                                           | Zero (useless as loss) |
| $k_2$  | $-\mathbb{E}_{q}\left[\left(\log \frac{p}{q}\right) \cdot s_\theta\right] = \nabla_\theta D_{\mathrm{KL}}(q \| p)$ |   Reverse KL gradient   |
| $k_3$  |   $\mathbb{E}_{q}\left[\left(1-\frac{p}{q}\right) \cdot s_\theta\right] = \nabla_\theta D_{\mathrm{KL}}(p \| q)$   |   Forward KL gradient   |

**Key insights**:
- **$k_2$'s gradient** equals the true reverse KL gradient — the correct choice for optimizing "constraining policy not to deviate from reference"
- **$k_3$'s gradient** equals the true forward KL gradient — corresponding to a "coverage" objective
- **$k_1$'s gradient expectation is always zero** — backpropagating as loss is meaningless!

#### Key Finding: $k_1$ Ineffective, $k_2$ for Reverse KL, $k_3$ for Forward KL

**"Expectation-then-grad" vs. "Grad-then-expectation"**:

If we analytically treat $\mathbb{E}_q[k_i]$ as a function of $\theta$ and then differentiate (i.e., "expectation-then-grad"), then:

$$
\nabla_\theta \mathbb{E}_q[k_1] = \nabla_\theta D_{\mathrm{KL}}(q \| p)
$$

$$
\nabla_\theta \mathbb{E}_q[k_3] = \nabla_\theta D_{\mathrm{KL}}(q \| p)
$$

Both give reverse KL's gradient. However, when directly calling backpropagation on sample means of $k_3$ in code, autograd executes "grad-then-expectation", yielding $\mathbb{E}_q[\nabla_\theta k_3]$, i.e., **the forward KL gradient**.

**Key conclusion for on-policy scenario**: For the same estimator, the two differentiation orders can give completely different results. Specifically:
- To optimize **reverse KL**: must use $k_2$
- To optimize **forward KL**: use $k_3$
- $k_1$'s gradient expectation is always zero, useless as loss

### Off-policy Scenario

Now consider the off-policy scenario where samples come from a behavior policy $\mu \neq q_\theta$. This situation is very common in practical RL training when using old/mixed policies to generate data or in offline RL with fixed sample distributions.

For an in-depth analysis of off-policy scenarios in large language models, refer to: [From Two-Policy to Three-Policy: TRPO Extension Under Behavior-Reference Mismatch in LLM RL](/reinforcement-learning/2025/11/15/three-policy-en.html).

#### Importance Weighting and Gradient Equivalence

When sampling from $\mu$ independent of $\theta$, we use importance weight $w(x) = \frac{q_\theta(x)}{\mu(x)}$ in the loss. A key difference emerges:

> **Before** (on-policy): expectation $\mathbb{E}_{q_{\theta}}[\cdot]$ depends on $\theta$  
> **Now** (off-policy): expectation $\mathbb{E}_{\mu}[\cdot]$ is independent of $\theta$

This makes "expectation-then-grad" and "grad-then-expectation" **equivalent**:

$$
\nabla_\theta \mathbb{E}_{\mu}[f_\theta(x)] = \mathbb{E}_{\mu}[\nabla_\theta f_\theta(x)]
$$

#### Gradient Derivations with Importance Weights

Since $\nabla_\theta w = w s_\theta$ and using product rule with previously derived $\nabla_\theta k_i$:

$$
\nabla_\theta(w k_1) = w s_\theta (k_1 + 1), \quad
\nabla_\theta(w k_2) = w s_\theta (k_2 - k_1), \quad
\nabla_\theta(w k_3) = w s_\theta k_1
$$

Note that $\nabla_\theta(w k_3) = w s_\theta k_1 = \nabla_\theta(\text{sg}(w) k_2)$  — these two are **gradient-identical**.

Taking expectations and using $\mathbb{E}_\mu[w \cdot f] = \mathbb{E}_{q}[f]$ and $\mathbb{E}_{q}[s_\theta]=0$:

|                    Weighted estimator                    |          Value target          |                    Gradient target                     |
| :------------------------------------------------------: | :----------------------------: | :---------------------------------------------------: |
|            $\frac{q_\theta}{\mu} k_1$                    | $D_{\mathrm{KL}}(q \| p)$     | $\nabla D_{\mathrm{KL}}(q \| p)$ ✓ (higher var)     |
|            $\frac{q_\theta}{\mu} k_2$                    |  $\mathbb{E}_q[k_2]$ (f-div)   |      $\nabla \mathbb{E}_q[k_2]$ ✗ (not reverse KL)       |
| $\text{sg}\left(\frac{q_\theta}{\mu}\right) k_2$         |  $\mathbb{E}_q[k_2]$ (f-div)   | $\nabla D_{\mathrm{KL}}(q \| p)$ ✓ (low var) |
|            $\frac{q_\theta}{\mu} k_3$                    | $D_{\mathrm{KL}}(q \| p)$     | $\nabla D_{\mathrm{KL}}(q \| p)$ ✓ (low var) |

**Key finding**: $\frac{q}{\mu} k_3$ and $\text{sg}\left(\frac{q}{\mu}\right) k_2$ have **identical gradients** $w s_\theta k_1$ and are statistically equivalent.

#### Variance Analysis

In the typical regime where $q_\theta \approx p \approx \mu$, setting $\frac{p}{q}=1+\varepsilon$ with $|\varepsilon|\ll1$:

- $\nabla_\theta(w k_1) \approx w s_\theta(1-\varepsilon)$ — contains $O(1)$ noise term $w s_\theta$
- $\nabla_\theta(w k_3) = \nabla_\theta(\text{sg}(w) k_2) \approx w s_\theta(-\varepsilon)$ — only $O(\varepsilon)$ terms

The variance difference is:

$$
\mathrm{Var}_\mu(g_1) - \mathrm{Var}_\mu(g_\star) = \mathbb{E}_\mu\big[w^2 s_\theta^2 (2k_1+1)\big] \approx \mathbb{E}_\mu[w^2 s_\theta^2] > 0
$$

Therefore, $w k_1$ has roughly one order of magnitude higher variance than $w k_3$ or $\text{sg}(w) k_2$.

This is why DeepSeek v3.2 uses $\frac{q_\theta}{\mu} k_3$ for off-policy KL penalty:

<figure style="text-align:center;" markdown="0">
<img src="/assets/img/kl-estimators/dpsk-3d2-k3.png" style="width:95%;max-width:100%;">
<figcaption style="font-size:0.9em;color:gray;">Source: <a href="https://arxiv.org/pdf/2512.02556v1">DeepSeek v3.2 Technical Report Section 3.1</a></figcaption>
</figure>

#### Practical Recommendations

**For on-policy reverse KL optimization:**
- **Use $k_2$ as loss**: $\mathcal{L} = \tfrac{1}{2}(\log r)^2$
- Gradient: $\mathbb{E}_q[\nabla k_2] = \nabla D_{\mathrm{KL}}(q \| p)$

**For on-policy forward KL optimization** (coverage objectives):
- **Use $k_3$ as loss**: Autograd gives $\mathbb{E}_q[(1-r) s_\theta] = \nabla D_{\mathrm{KL}}(p \| q)$

**For off-policy reverse KL optimization** (samples from $\mu$):
- **Recommended**: $\dfrac{q_\theta}{\mu} k_3$ or $\text{sg}\left(\dfrac{q_\theta}{\mu}\right) k_2$ (identical, low variance)
- **Fallback**: $\dfrac{q_\theta}{\mu} k_1$ (unbiased but ~10× higher variance)
- **Avoid**: $\dfrac{q_\theta}{\mu} k_2$ with weight in gradient (biased for reverse KL)

### Comprehensive Gradient Analysis Summary

|  Sampling   |         Loss          |       Gradient expectation       |  Optimizes  | Usable for reverse KL? |
| :---------: | :-------------------: | :------------------------------: | :---------: | :--------------------: |
|  $q$ (on)   |         $k_1$         |      $\mathbb{E}_q[s_\theta]=0$       | None (zero) |           ✗            |
|  $q$ (on)   |         $k_2$         | $\nabla D_{\mathrm{KL}}(q \| p)$ | Reverse KL  |           ✓            |
|  $q$ (on)   |         $k_3$         | $\nabla D_{\mathrm{KL}}(p \| q)$ | Forward KL  |           ✗            |
| $\mu$ (off) |   $\frac{q}{\mu} k_1$   | $\nabla D_{\mathrm{KL}}(q \| p)$ | Reverse KL  |    ✓ (high variance)    |
| $\mu$ (off) |   $\frac{q}{\mu} k_2$   |    $\nabla \mathbb{E}_q[k_2]$    | f-div (wrong) |           ✗            |
| $\mu$ (off) | $\text{sg}(\frac{q}{\mu}) k_2$ | $\nabla D_{\mathrm{KL}}(q \| p)$ | Reverse KL  |    ✓ (low variance)     |
| $\mu$ (off) |   $\frac{q}{\mu} k_3$   | $\nabla D_{\mathrm{KL}}(q \| p)$ | Reverse KL  |    ✓ (low variance, recommended)     |

## Gradient Analysis When Used as Reward

Having analyzed the value estimation properties of the three estimators in the previous section, one might naturally assume: since $k_1$ and $k_3$ are both unbiased for reverse KL value, incorporating them (with stop-gradient) as reward penalties should be equally valid.

**This intuition, however, is incorrect.**

The issue is: when KL is used as a reward penalty, although the KL term itself doesn't backpropagate gradients, it will indirectly affect the policy gradient through advantage. Therefore, to evaluate whether an estimator "can be used for reward penalty", we shouldn't just look at value bias, but whether **the policy gradient it induces is correct**.

### The True KL-Regularized Policy Gradient

Consider the KL-regularized reinforcement learning objective:

$$
J(\theta) = \mathbb{E}_{q_\theta}[R] - \beta \cdot D_{\mathrm{KL}}(q_\theta \| p)
$$

Its true gradient is:

$$
\nabla_\theta J = \mathbb{E}_{q_\theta}[s_\theta \cdot R] - \beta \cdot \nabla_\theta D_{\mathrm{KL}}(q_\theta \| p)
$$

Using the result from the "Preliminaries" section, the reverse KL gradient is:

$$
\nabla_\theta D_{\mathrm{KL}}(q_\theta \| p) = \mathbb{E}_{q_\theta}\left[s_\theta \cdot \left(-\log \frac{p}{q}\right)\right] = \mathbb{E}_{q_\theta}[s_\theta \cdot k_1]
$$

Therefore, the true KL-regularized policy gradient is:

$$
\nabla_\theta J = \mathbb{E}_{q_\theta}\left[s_\theta \cdot \left(R - \beta \cdot k_1\right)\right]
$$

### Gradient Form When Using Estimator $\hat{k}$

When we use some estimator $\hat{k}$ (with stop-gradient) as a reward penalty, the shaped reward is $\tilde{R} = R - \beta \cdot \text{sg}(\hat{k})$, and the policy gradient becomes:

$$
\nabla_\theta \tilde{J} = \mathbb{E}_{q_\theta}\left[s_\theta \cdot (R - \beta \cdot \hat{k})\right]
$$

**Unbiasedness condition**: $\nabla_\theta \tilde{J} = \nabla_\theta J$ if and only if

$$
\mathbb{E}_{q_\theta}[s_\theta \cdot \hat{k}] = \mathbb{E}_{q_\theta}[s_\theta \cdot k_1]
$$

### Using $k_1$ as Penalty: Gradient Unbiased

When $\hat{k} = k_1$, the condition is automatically satisfied:

$$
\mathbb{E}_{q_\theta}[s_\theta \cdot k_1] = \mathbb{E}_{q_\theta}[s_\theta \cdot k_1] \quad \checkmark
$$

Therefore, **when $k_1$ is used as a reward penalty, the induced policy gradient is unbiased**.

### Using $k_3$ as Penalty: Gradient Biased

When $\hat{k} = k_3 = \frac{p}{q} - 1 - \log \frac{p}{q}$:

$$
\mathbb{E}_{q_\theta}[s_\theta \cdot k_3] = \mathbb{E}_{q_\theta}\left[s_\theta \cdot \left(\frac{p}{q} - 1\right)\right] + \mathbb{E}_{q_\theta}\left[s_\theta \cdot \left(-\log \frac{p}{q}\right)\right]
$$

The second term is exactly $\mathbb{E}_{q_\theta}[s_\theta \cdot k_1]$. The problem lies in the first term:

$$
\mathbb{E}_{q_\theta}\left[s_\theta \cdot \left(\frac{p}{q} - 1\right)\right] = \mathbb{E}_{q_\theta}\left[s_\theta \cdot \frac{p}{q}\right] - \underbrace{\mathbb{E}_{q_\theta}[s_\theta]}_{=0} = \mathbb{E}_{q_\theta}\left[s_\theta \cdot \frac{p}{q}\right]
$$

This can be rewritten as:

$$
\mathbb{E}_{q_\theta}\left[s_\theta \cdot \frac{p}{q}\right] = \int q_\theta(x) \cdot \nabla_\theta \log q_\theta(x) \cdot \frac{p(x)}{q_\theta(x)} dx = \int p(x) \cdot \nabla_\theta \log q_\theta(x) dx = \mathbb{E}_p[s_\theta]
$$

Using the forward KL gradient formula $\nabla_\theta D_{\mathrm{KL}}(p \| q_\theta) = -\mathbb{E}_p[s_\theta]$, we have:

$$
\mathbb{E}_{q_\theta}\left[s_\theta \cdot \frac{p}{q}\right] = -\nabla_\theta D_{\mathrm{KL}}(p \| q_\theta)
$$

Therefore:

$$
\mathbb{E}_{q_\theta}[s_\theta \cdot k_3] = \underbrace{-\nabla_\theta D_{\mathrm{KL}}(p \| q_\theta)}_{\text{bias term}} + \nabla_\theta D_{\mathrm{KL}}(q_\theta \| p)
$$

**When $k_3$ is used as a reward penalty, the gradient is biased**, with the bias term equal to the negative of the forward KL gradient.

**Geometric meaning of the bias**: Using $k_3$ as a reward penalty is equivalent to optimizing a "wrong mixed objective":
- Penalizing reverse KL (hoping policy doesn't deviate from reference)
- But also **wrongly encouraging forward KL to increase** (hoping reference doesn't cover policy)

These two directions conflict, potentially causing optimization instability.

**Experimental verification**: Shah et al. (2025)'s experiments show that in on-policy RL fine-tuning of LLMs:
- **$k_1$ in reward**: Training is stable
- **$k_3$ in reward**: **Training collapses**

This is completely consistent with our theoretical analysis.

### Off-policy Scenario

The above analysis assumes on-policy sampling. Does the conclusion change in off-policy scenarios?

Let samples come from behavior policy $\mu$, using importance-weighted policy gradient:

$$
\nabla_\theta \tilde{J} = \mathbb{E}_\mu\left[\frac{q_\theta}{\mu} \cdot s_\theta \cdot (R - \beta \cdot k)\right]
$$

Using $\mathbb{E}_\mu[\frac{q_\theta}{\mu} \cdot f] = \mathbb{E}_{q_\theta}[f]$, this equals:

$$
= \mathbb{E}_{q_\theta}[s_\theta \cdot R] - \beta \cdot \mathbb{E}_{q_\theta}[s_\theta \cdot k]
$$

**The unbiasedness condition** remains $\mathbb{E}_{q_\theta}[s_\theta \cdot k] = \mathbb{E}_{q_\theta}[s_\theta \cdot k_1]$, exactly the same as on-policy.

**Key insight**: In the off-policy policy gradient framework, the importance weight $\frac{q_\theta}{\mu}$ acts on the entire policy gradient estimator, **no need to separately weight the KL estimator in shaped reward**. Therefore:

- Shaped reward keeps its original form: $\tilde{R} = R - \beta \cdot k_1$ (not $R - \beta \cdot \frac{q_\theta}{\mu} k_1$)
- Conclusion same as on-policy: **only use $k_1$, not $k_3$**

### Key Finding: Only $k_1$ Can Be Used for Reward Penalty

| Estimator | Value unbiased? | Gradient unbiased when used as reward penalty? | Actual performance |
| :-------: | :-------------: | :--------------------------------------------: | :----------------: |
|   $k_1$   |        ✓        |                       ✓                        |       Stable       |
|   $k_3$   |        ✓        |                       ✗                        |     Collapses      |

**Core lesson**: When evaluating KL estimators, "value unbiasedness" and "gradient correctness" are two independent dimensions. For reward penalty scenarios (whether on-policy or off-policy), **only $k_1$ is the correct choice**. Although $k_3$ is value-unbiased and has lower variance, using it as a reward penalty causes biased gradients and may lead to training collapse.

## Practical Guide and Common Pitfalls

With the preceding theoretical analysis, this section provides selection recommendations for specific scenarios, convenient for direct reference.

### Quick Reference Table

The table below provides recommended estimator choices along three dimensions: "target KL direction" × "sampling source" × "usage mode". "For **Loss**" corresponds to KL as loss (gradient backpropagation needed); "For **Reward**" corresponds to KL as reward penalty (stop-gradient).

|                Target                |      Sampling      |                    For Loss (gradient backprop)                     |                For Reward (stop-grad)                 |
| :----------------------------------: | :----------------: | :-----------------------------------------------------------------: | :---------------------------------------------------: |
| Reverse KL $D_{\mathrm{KL}}(q \| p)$ |  $q$ (on-policy)   |                                $k_2$                                |                         $k_1$                         |
| Reverse KL $D_{\mathrm{KL}}(q \| p)$ | $\mu$ (off-policy) | $\tfrac{q}{\mu} k_3$ or $\text{sg}\left(\tfrac{q}{\mu}\right) k_2$ |                         $k_1$                         |
| Forward KL $D_{\mathrm{KL}}(p \| q)$ |        $q$         |                                $k_3$                                | $\mathbb{E}_q\left[\frac{p}{q} \log \frac{p}{q}\right]$ |

### KL as Loss (Needs Gradient Backpropagation)

When KL is used as part of a loss for backpropagation, we must consider gradient correctness.

#### On-policy: Optimize Reverse KL (Most Common)

Goal: Control actor to not deviate from reference policy.

**Correct approach**: Use **$k_2$** as loss.

$$
\mathcal{L}_{k_2} = \frac{1}{2}\left(\log \frac{p}{q}\right)^2
$$

Its gradient expectation $\mathbb{E}_q[\nabla k_2] = \nabla_\theta D_{\mathrm{KL}}(q \| p)$ is exactly the true gradient of reverse KL.

#### On-policy: Optimize Forward KL (Coverage Scenario)

Goal: Make policy cover the support of the reference distribution (offline RL, imitation learning, etc.).

**Correct approach**: Use **$k_3$** as loss.

$$
\mathbb{E}_q[\nabla k_3] = \mathbb{E}_q\left[\left(1-\frac{p}{q}\right) \cdot s_\theta\right] = \nabla_\theta D_{\mathrm{KL}}(p \| q)
$$

Directly calling loss gradient backprop on the sample mean of $k_3$, autograd computes $\mathbb{E}_q[\nabla_\theta k_3]$, which is the forward KL gradient, no additional processing needed.

#### Off-policy: Optimize Reverse KL

Goal: Data comes from behavior policy $\mu$, still want to optimize reverse KL.

**Recommended approach**: Use **$\dfrac{q_\theta}{\mu} k_3$** or **$\text{sg}\left(\dfrac{q_\theta}{\mu}\right) k_2$** as loss (both have identical gradients).

$$
\mathcal{L} = \dfrac{q_\theta(x)}{\mu(x)} \cdot \left(\dfrac{p(x)}{q_\theta(x)} - 1 - \log \dfrac{p(x)}{q_\theta(x)}\right)
$$

or

$$
\mathcal{L} = \text{sg}\left(\dfrac{q_\theta(x)}{\mu(x)}\right) \cdot \dfrac{1}{2}\left(\log \dfrac{p(x)}{q_\theta(x)}\right)^2
$$

- Gradients are unbiased
- When $q_\theta \approx p$, both have much lower variance

**Fallback**: Use $\dfrac{q_\theta}{\mu} k_1$ (gradient also unbiased but higher variance)

**Avoid**: Using $\dfrac{q_\theta}{\mu} k_2$ (with weight in gradient) — gradient is biased, not the correct direction for reverse KL

### KL as Reward Penalty (Stop-gradient)

When KL is used as a scalar penalty added to reward, although the KL term itself doesn't backpropagate gradients, it will indirectly affect the policy gradient through advantage. Based on the earlier section "Gradient Analysis When Used as Reward":

**Recommend**:
- Use **$k_1$** (value-unbiased and induced policy gradient is also unbiased)
- Conclusion is the same whether on-policy or off-policy

**Avoid**:
- Using $k_3$ (although value-unbiased and lower variance, the induced policy gradient is biased and may cause training collapse)

> **Note**: In off-policy policy gradient, the importance weight $\frac{q_\theta}{\mu}$ acts on the entire $s_\theta \cdot \tilde{R}$; the shaped reward itself can keep the form $\tilde{R} = R - \beta \cdot k_1$.

### Common Pitfalls

#### Pitfall 1: Using $k_1$ Directly as Loss (On-policy)

The gradient expectation of $k_1$ is always zero ($\mathbb{E}_q[\nabla k_1] = \mathbb{E}_q[s_\theta] = 0$); using it as a loss is completely ineffective.

> **Solution**: Use $k_2$ for on-policy reverse KL optimization; use $k_3$ for forward KL optimization.

#### Pitfall 2: Confusing $k_3$'s Value Unbiasedness with Gradient Behavior

$k_3$ is value-unbiased for **reverse KL value**, but its **gradient** corresponds to **forward KL** — these are completely different.

|                    Scenario                    |                                     Problem                                      |
| :--------------------------------------------: | :------------------------------------------------------------------------------: |
|  Using $k_3$ as Loss (targeting reverse KL)   |            $\nabla k_3$ corresponds to forward KL, optimizing wrong direction            |
| Using $k_3$ as Reward penalty (targeting reverse KL) | Induces biased policy gradient (bias term $-\nabla D_{\mathrm{KL}}(p\|q)$), may cause training collapse |

> **Solution**:
> - Use as **Loss** to optimize reverse KL → use $k_2$; use $k_3$ only for forward KL
> - Use as **Reward** penalty → only use $k_1$ (whether on-policy or off-policy)

#### Pitfall 3: Off-policy Detach Handling of Importance Weights

In off-policy scenarios, whether to detach the importance weight $w = q_\theta / \mu$ leads to completely different results. The following table summarizes the correct detach strategies:

|      Estimator       | Detach $w$? | Gradient corresponds to |
| :------------------: | :---------: | :---------------------: |
|       $w k_1$        |  No detach  |     Reverse KL ✓     |
|       $w k_3$        |  No detach  |     Reverse KL ✓     |
|       $w k_2$        |  No detach  |   f-divergence ✗    |
| $\text{sg}(w) k_2$ |   Detach    |     Reverse KL ✓     |

> **Solution**: For off-policy reverse KL optimization, recommend using $w k_3$ or $\text{sg}(w) k_2$ (both have identical gradients). If using $w k_1$, gradient is unbiased but variance is higher.

#### Pitfall 4: Using $k_3$ in Reward Penalty

Although $k_3$ is value-unbiased for reverse KL and has lower variance, using it as a reward penalty causes biased policy gradient (bias term $-\nabla D_{\mathrm{KL}}(p\|q)$), potentially leading to training collapse.

> **Solution**: Whether on-policy or off-policy, reward penalty should only use $k_1$.

## Summary

This post systematically analyzes the three KL estimators $k_1, k_2, k_3$ around three core questions: **who to sample from**, **whose value to estimate**, **whose gradient is needed**.

**Core takeaways**:

1. **First clarify usage mode**: KL as Loss (gradient backprop) or as Reward (stop-grad)?
2. **KL as Loss (on-policy)**: Use $k_2$ for reverse KL; use $k_3$ for forward KL
3. **KL as Loss (off-policy)**: Use $\frac{q}{\mu} k_3$ or $\text{sg}\left(\frac{q}{\mu}\right) k_2$ (note detach strategy!)
4. **KL as Reward**: Only use $k_1$ ($k_3$ although value-unbiased causes biased policy gradient)

Clarify these points, and the three estimators will no longer be confusing.

**One-liners:**

- **Only value (reward penalty):** use $k_1$ or $k_3$ (both unbiased for reverse KL value); off-policy multiply by $\tfrac{q_\theta}{\mu}$.
- **Need gradients (loss):**
	- **On-policy:** reverse KL -> $k_2$; forward KL -> $k_3$.
	- **Off-policy:** reverse KL -> $\tfrac{q_\theta}{\mu} k_3$ or $\text{sg}(\tfrac{q_\theta}{\mu}) k_2$ (same gradient, low variance); fallback $\tfrac{q_\theta}{\mu} k_1$ (unbiased but noisier).

Keep three questions clear: **who do we sample from, whose value do we estimate, whose gradient do we need?** Especially note: **on-policy vs. off-policy choose different estimators for reverse KL** — on-policy use $k_2$, off-policy use $\tfrac{q_\theta}{\mu} k_3$ or $\text{sg}(\tfrac{q_\theta}{\mu}) k_2$.

Additionally, don't forget to determine **the KL usage mode** before choosing an estimator:
- **KL as reward:** Constraints act on the policy indirectly through shaped advantage, with cross-timestep credit assignment capability; agent will "plan to avoid high-KL paths"
- **KL as loss:** Constraints act on the policy directly as an independent gradient term; agent will "visit but locally correct"

This choice is more fundamental than the estimator itself, depending on whether you want constraints to be "preventive" or "corrective".

## References

1. Dibya Ghosh. "KL Divergence for Machine Learning". <https://dibyaghosh.com/blog/probability/kldivergence>

2. John Schulman. "Approximating KL Divergence". <https://joschu.net/blog/kl-approx.html>

3. Verl Documentation. "Proximal Policy Optimization (PPO)". <https://verl.readthedocs.io/en/latest/algo/ppo.html>

4. 初七123334. RLHF/RLVR 训练中的 KL 近似方法浅析（k1 / k2 / k3）. <https://zhuanlan.zhihu.com/p/1966872846212010437>

5. Kezhao Liu, Jason Klein Liu, Mingtao Chen, Yiming Liu. "Rethinking KL Regularization in RLHF: From Value Estimation to Gradient Optimization". <https://arxiv.org/abs/2510.01555>

6. Yifan Zhang, Yiping Ji, Gavin Brown, et al. "On the Design of KL-Regularized Policy Gradient Algorithms for LLM Reasoning". <https://arxiv.org/abs/2505.17508>

7. Vedant Shah, Johan Obando-Ceron, Vineet Jain, Brian Bartoldson, Bhavya Kailkhura, Sarthak Mittal, Glen Berseth, Pablo Samuel Castro. "A Comedy of Estimators: On KL Regularization in RL Training of LLMs". <https://arxiv.org/abs/2512.21852>

```bibtex
@misc{WangZhang2025KLEstimators,
  author       = {Wang, Xihuai and Zhang, Shao},
  title        = {Understanding KL Divergence Estimators in RL: From Value Approximation to Gradient Estimation},
  year         = {2025},
  month        = dec,
  day          = {01},
  url          = {https://xihuai18.github.io/reinforcement-learning/2025/12/01/kl-estimators-en.html}
}
```
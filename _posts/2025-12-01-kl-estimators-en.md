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



![Mini-class](/assets/img/kl-estimators/kl-estimator-en.png){: style="display:block;margin:0 auto;width:95%;max-width:100%;" }

> How we approximate KL divergence directly affects training stability. This post systematically analyzes three estimators $k_1, k_2, k_3$ in both on-policy and off-policy scenarios, and gives practical guidelines for choosing them when KL is used as a reward penalty versus when it is used as a loss for backpropagation.

## Introduction: What KL Does in RL

In policy optimization (PPO, GRPO, etc.) and alignment training (RLHF/RLAIF), **KL penalty** keeps the new policy from drifting too far from a reference policy, preventing instability or collapse. However, implementing KL penalty involves multiple layers of choices: **which estimator** ($k_1$, $k_2$, $k_3$), **who to sample from** (on-policy vs off-policy), and **how to use it** (as reward shaping or as a loss for backpropagation). This post systematically dissects these choices and their interrelationships.

### Forward vs. reverse KL

Let $q_\theta$ be the current actor, $p$ the reference policy. The two directions are:

**Reverse KL:**
$$
D_{\mathrm{KL}}(q_\theta \| p) = \mathbb{E}_{x \sim q_\theta}\left[\log \frac{q_\theta(x)}{p(x)}\right]
$$

<figure style="text-align:center;">
	<img src="/assets/img/kl-estimators/kl-estimator-reverse.png" style="width:95%;max-width:100%;">
	<figcaption style="font-size:0.9em;color:gray;">Image source: <a href="https://dibyaghosh.com/blog/probability/kldivergence/">Dibya Ghosh's Blog</a></figcaption>
</figure>

**Forward KL:**
$$
D_{\mathrm{KL}}(p \| q_\theta) = \mathbb{E}_{x \sim p}\left[\log \frac{p(x)}{q_\theta(x)}\right]
$$

<figure style="text-align:center;">
	<img src="/assets/img/kl-estimators/kl-estimator-forward.png" style="width:95%;max-width:100%;">
	<figcaption style="font-size:0.9em;color:gray;">Image source: <a href="https://dibyaghosh.com/blog/probability/kldivergence/">Dibya Ghosh's Blog</a></figcaption>
</figure>

**Intuition:**
- **Reverse KL** is mode-seeking: policy concentrates on high-probability regions of $p$, possibly sacrificing diversity.
- **Forward KL** is mass-covering: policy tries to cover the support of $p$.

RLHF typically uses **reverse KL** because we want the actor not to move too far from the reference, not necessarily to cover every mode.

## Three estimators: definitions and design

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

### Quick comparison

| Estimator | Definition | Design idea | Bias (value) | Variance |
| :---: | :---: | :---: | :---: | :---: |
| $k_1$ | $\log r$ | Naive log-ratio | Unbiased | High (can be negative) |
| $k_2$ | $\tfrac{1}{2}(\log r)^2$ | f-divergence, KL-matching 2nd order | Biased (very small) | Low (always positive) |
| $k_3$ | $r - 1 - \log r$ | Control variate + Bregman | Unbiased | Low (always positive) |

For estimating the KL **value**, $k_3$ is "unbiased + low variance"; but as we'll analyze, **the gradient story is completely different** — different estimators' gradients may correspond to different optimization objectives. Moreover, whether KL is added to the reward for shaping or used as a loss for direct gradient backpropagation will fundamentally affect training behavior.

## Core analysis

### Bias and variance for KL values

Assume samples from $q_\theta$ to estimate reverse KL $D_{\mathrm{KL}}(q_\theta \| p)$.

**Unbiasedness:**

$$
\begin{aligned}
\mathbb{E}_{q}[k_1] &= \mathbb{E}_{q}\left[\log \tfrac{q}{p}\right] = D_{\mathrm{KL}}(q \| p) \quad \textbf{(unbiased)}\\
\mathbb{E}_{q}[k_3] &= \mathbb{E}_{q}[r - 1 - \log r] = 1 - 1 + D_{\mathrm{KL}}(q \| p) = D_{\mathrm{KL}}(q \| p) \quad \textbf{(unbiased)}\\
\mathbb{E}_{q}[k_2] &= \tfrac{1}{2}\mathbb{E}_{q}[(\log r)^2] \neq D_{\mathrm{KL}}(q \| p) \quad \textbf{(biased)}
\end{aligned}
$$

**Variance trade-off:**

John Schulman's toy experiments ($q = \mathcal{N}(0,1)$, $p = \mathcal{N}(0.1,1)$, true KL = 0.005):

| Estimator | bias/true | stdev/true |
| :---: | :---: | :---: |
| $k_1$ | 0 | 20 |
| $k_2$ | 0.002 | 1.42 |
| $k_3$ | 0 | 1.42 |

When KL is large ($p = \mathcal{N}(1,1)$, true KL = 0.5):

| Estimator | bias/true | stdev/true |
| :---: | :---: | :---: |
| $k_1$ | 0 | 2 |
| $k_2$ | 0.25 | 1.73 |
| $k_3$ | 0 | 1.7 |

**Intuition:**
- $k_1 = -\log r$ is first-order around $r=1$, can be negative, so variance explodes when close.
- $k_3 = r - 1 - \log r$ is second-order near $r=1$ and always positive, so lower variance when close.
- When coverage is poor (heavy tails in $r$), $k_3$ can explode; then $k_1$ can be more stable.

> **Note:** To estimate **forward KL value** $D_{\mathrm{KL}}(p \| q) = \mathbb{E}_p[\log r]$ but only sample from $q$, use importance sampling $\mathbb{E}_q[r \log r]$.

### Gradient estimation: the crucial distinction

This is the easiest part to get wrong. First analyze **on-policy** (samples from $q_\theta$), then extend to **off-policy** (samples from behavior $\mu$).

#### True gradients for reference

Let score function $s_\theta(x) = \nabla_\theta \log q_\theta(x)$, with key property $\mathbb{E}_{q_\theta}[s_\theta] = 0$.

**Reverse KL gradient:**

$$
D_{\mathrm{KL}}(q_\theta \| p) = \int q_\theta(x) \log \frac{q_\theta(x)}{p(x)} dx
$$

Product rule and $\nabla_\theta q_\theta = q_\theta s_\theta$, $\nabla_\theta \log p = 0$ give

$$
\nabla_\theta D_{\mathrm{KL}}(q_\theta \| p) = \mathbb{E}_q\left[s_\theta \log \tfrac{q_\theta}{p}\right] = -\mathbb{E}_q[s_\theta \log r].
$$

**Forward KL gradient:**

$$
D_{\mathrm{KL}}(p \| q_\theta) = \int p(x) \log \frac{p(x)}{q_\theta(x)} dx
$$

Since $p$ is $\theta$-independent,

$$
\nabla_\theta D_{\mathrm{KL}}(p \| q_\theta) = -\mathbb{E}_p[s_\theta] = -\mathbb{E}_q[r s_\theta] = \mathbb{E}_q[(1-r) s_\theta].
$$

These baselines tell us what each estimator's expected gradient really targets.

#### Two differentiation orders

1) **Grad then expectation:** autograd on each sample, then batch average (what DL code actually does).
2) **Expectation then grad:** treat $\mathbb{E}_q[k_i]$ as a function of $\theta$ and differentiate analytically.

Typical code does (1).

#### Gradients of the three estimators (on-policy)

$$
\nabla_\theta k_1 = s_\theta
$$

$$
\nabla_\theta k_2 = (\log r) \nabla_\theta(\log r) = (\log r)(-s_\theta) = - (\log r) s_\theta
$$

$$
\nabla_\theta k_3 = (1 - r) s_\theta
$$

Taking expectation under $q_\theta$:

| Estimator | $\mathbb{E}_{q}[\nabla_\theta k_i]$ | Equals |
| :---: | :---: | :---: |
| $k_1$ | $\mathbb{E}_{q}[s_\theta] = 0$ | Zero (useless as loss) |
| $k_2$ | $-\mathbb{E}_{q}[(\log r) s_\theta] = \nabla_\theta D_{\mathrm{KL}}(q \| p)$ | Gradient of reverse KL |
| $k_3$ | $\mathbb{E}_{q}[(1-r) s_\theta] = \nabla_\theta D_{\mathrm{KL}}(p \| q)$ | Gradient of forward KL |

**Key takeaways:**
- **$k_2$ gradient** matches reverse KL gradient (the usual "stay near ref" objective).
- **$k_3$ gradient** matches forward KL gradient (coverage objective).
- **$k_1$ gradient expectation is zero** — useless as a loss.

#### Expectation-then-grad vs. grad-then-expectation

If you first form $\mathbb{E}_q[k_i]$ and then differentiate (expectation-then-grad):

$$
\nabla_\theta \mathbb{E}_q[k_1] = \nabla_\theta D_{\mathrm{KL}}(q \| p), \quad \nabla_\theta \mathbb{E}_q[k_3] = \nabla_\theta D_{\mathrm{KL}}(q \| p).
$$

Both give reverse KL. But autograd on per-sample $k_3$ averages (grad-then-expectation) yields **forward KL gradient**. Same estimator, different order, different result.

### Off-policy gradients with importance sampling

Real RL often samples from a behavior policy $\mu$ (old or mixed policy, replay buffer). To optimize **reverse KL** you need **importance weights**.

See also my earlier post: [Three-policy TRPO extension for LLM RL](/reinforcement-learning/2025/11/15/three-policy-zh.html).

#### Setup

Define importance weight

$$
w(x) = \frac{q_\theta(x)}{\mu(x)}.
$$

Using batch loss $w(x) k_i(x)$ with autograd, what gradients do we get?

A key difference:
- Previously expectations were under $q_\theta$ (depends on $\theta$).
- Now expectations are under $\mu$ (independent of $\theta$).

#### Crucial observation: the two orders coincide

Because $\mu$ is $\theta$-independent,

$$
\nabla_\theta \mathbb{E}_{\mu}[f_\theta] = \mathbb{E}_{\mu}[\nabla_\theta f_\theta].
$$

So autograd on sample means (grad-then-expectation) equals expectation-then-grad. For $k_1$ and $k_3$, both value-unbiased for reverse KL, their gradient expectations also match reverse KL.

#### Value unbiasedness remains

By $\mathbb{E}_\mu[w f] = \mathbb{E}_q[f]$:

$$
\mathbb{E}_\mu[w k_1] = D_{\mathrm{KL}}(q_\theta \| p), \quad \mathbb{E}_\mu[w k_3] = D_{\mathrm{KL}}(q_\theta \| p) \quad \textbf{(unbiased)}
$$

$$
\mathbb{E}_\mu[w k_2] = \mathbb{E}_{q_\theta}[k_2] \neq D_{\mathrm{KL}}(q_\theta \| p) \quad \textbf{(biased)}
$$

#### Gradients with weights

Gradient of weight: $\nabla_\theta w = w s_\theta$. Using product rule:

$$
\nabla_\theta(w k_1) = w s_\theta (k_1 + 1)
$$
$$
\nabla_\theta(w k_2) = w s_\theta (k_2 - \log r)
$$
$$
\nabla_\theta(w k_3) = w s_\theta (k_3 + 1 - r) = w s_\theta k_1
$$

Which give expected gradients:

| Weighted estimator | Value target | Expected gradient |
| :---: | :---: | :---: |
| $\tfrac{q_\theta}{\mu} k_1$ | $D_{\mathrm{KL}}(q_\theta \| p)$ | $\nabla_\theta D_{\mathrm{KL}}(q_\theta \| p)$ (reverse KL) ✓ |
| $\tfrac{q_\theta}{\mu} k_2$ | $\mathbb{E}_q[k_2]$ (f-divergence) | $\nabla_\theta \mathbb{E}_q[k_2]$, not reverse KL ✗ |
| $\text{sg}\left(\tfrac{q_\theta}{\mu}\right) k_2$ | $\mathbb{E}_q[k_2]$ (f-divergence) | $\nabla_\theta D_{\mathrm{KL}}(q_\theta \| p)$ (reverse KL) ✓ |
| $\tfrac{q_\theta}{\mu} k_3$ | $D_{\mathrm{KL}}(q_\theta \| p)$ | $\nabla_\theta D_{\mathrm{KL}}(q_\theta \| p)$ (reverse KL) ✓ |

**Interesting reversal vs. on-policy:**
- On-policy: $k_2$ as loss gives reverse KL gradient; $k_1$ gradient is zero.
- Off-policy + weights: $\tfrac{q}{\mu}k_1$ and $\tfrac{q}{\mu}k_3$ give reverse KL gradients; $\tfrac{q}{\mu}k_2$ (with weight in grad) fails.
- Detaching the weight makes $\text{sg}(\tfrac{q}{\mu}) k_2$ also give reverse KL gradient.

#### Variance of the three unbiased off-policy gradient estimators

Unbiased reverse-KL gradient estimators (off-policy + IS):

$$
L_1 = w k_1, \quad L_2 = \bar w k_2, \quad L_3 = w k_3,
$$

With $w = \tfrac{q_\theta}{\mu}$, $\bar w = \mathrm{sg}(w)$. Using $\nabla_\theta w = w s_\theta$, $\nabla_\theta k_1 = s_\theta$, $\nabla_\theta k_2 = k_1 s_\theta$, $\nabla_\theta k_3 = (1-r) s_\theta$:

$$
\begin{aligned}
g_1 &= w s_\theta (k_1+1),\\
g_2 &= w s_\theta k_1,\\
g_3 &= w s_\theta k_1.
\end{aligned}
$$

So **$g_2 \equiv g_3$**. Only two distinct variance behaviors: $g_1$ vs. $g_\star := g_2 = g_3$.

Let $A = w s_\theta, B = k_1$. Then

$$
g_1 = A(B+1), \quad g_\star = A B.
$$

Variance difference:

$$
\boxed{\mathrm{Var}_\mu(g_1) - \mathrm{Var}_\mu(g_\star) = \mathbb{E}_\mu[A^2(2B+1)]} = \mathbb{E}_\mu\big[w^2 s_\theta^2 (2k_1+1)\big].
$$

In the typical KL-penalty regime $q_\theta \approx p \approx \mu$, write $r = 1 + \varepsilon$, $\lvert\varepsilon\rvert \ll 1$, so $k_1 \approx -\varepsilon$, $2k_1+1 \approx 1 - 2\varepsilon > 0$. Thus $\mathrm{Var}(g_1) > \mathrm{Var}(g_\star)$.

Intuition:
- $g_1$ includes an $O(1)$ zero-mean noise term $w s_\theta$.
- $g_\star$ cancels that term; remaining magnitude is $O(\varepsilon)$, giving much lower variance.

Table summary:

| Estimator | Gradient rv | Scale ($r\approx1$) | Variance |
| :---: | :---: | :---: | :---: |
| $w k_1$ | $w s_\theta (k_1+1)$ | $O(1)$ | High |
| $\mathrm{sg}(w) k_2$ | $w s_\theta k_1$ | $O(\varepsilon)$ | Low |
| $w k_3$ | $w s_\theta k_1$ | $O(\varepsilon)$ | Low |

Conclusion: off-policy IS with reverse-KL gradients has three unbiased options: $w k_1$, $\bar w k_2$, $w k_3$. The latter two are identical in gradient and variance and are preferred; $w k_1$ is unbiased but noisier.

**When far off-policy:** If $w$ explodes (little overlap), any $\tfrac{q}{\mu}$ method suffers. Then the variance advantage of $k_3$ over $k_1$ is not guaranteed; clipping/regularization becomes necessary.

### Gradient cheat sheet

| Sampling | Loss | $\mathbb{E}[\nabla_\theta \text{Loss}]$ | Optimizes | Right for reverse KL? |
| :---: | :---: | :---: | :---: | :---: |
| $q$ (on) | $k_1$ | $\mathbb{E}_q[s_\theta] = 0$ | None (zero grad) | ✗ |
| $q$ (on) | $k_2$ | $\nabla_\theta D_{\mathrm{KL}}(q \| p)$ | Reverse KL | ✓ |
| $q$ (on) | $k_3$ | $\nabla_\theta D_{\mathrm{KL}}(p \| q)$ | Forward KL | ✗ |
| $\mu$ (off) | $\tfrac{q}{\mu} k_1$ | $\nabla_\theta D_{\mathrm{KL}}(q \| p)$ | Reverse KL | ✓ (higher var) |
| $\mu$ (off) | $\tfrac{q}{\mu} k_2$ | $\nabla_\theta \mathbb{E}_q[k_2]$ | f-divergence (not KL) | ✗ |
| $\mu$ (off) | $\text{sg}\left(\tfrac{q}{\mu}\right) k_2$ | $\nabla_\theta D_{\mathrm{KL}}(q \| p)$ | Reverse KL | ✓ |
| $\mu$ (off) | $\tfrac{q}{\mu} k_3$ | $\nabla_\theta D_{\mathrm{KL}}(q \| p)$ | Reverse KL | ✓ (recommended, low var) |

**Key conclusions:**
1) **On-policy reverse KL:** use $k_2$ (only correct choice).
2) **Off-policy reverse KL:** three correct options: $\tfrac{q}{\mu} k_1$ (unbiased, higher var); $\text{sg}(\tfrac{q}{\mu}) k_2$ (unbiased, equals next); $\tfrac{q}{\mu} k_3$ (unbiased, lower var; equals previous).
3) **$\tfrac{q}{\mu} k_2$ with weight in grad is wrong** for reverse KL.

However, before choosing an estimator, there's a more fundamental question to answer: **should KL be added to rewards, or be part of the loss?** This choice fundamentally affects optimization behavior and credit assignment.

## Two Ways to Use KL: As Reward vs. As Loss

In practice, KL penalty can be used in two fundamentally different ways: added to rewards for shaping (no gradient backpropagation needed), or as part of the loss for backpropagation (gradient needed).

These two approaches may seem like just a `detach` difference in code, but they correspond to completely different optimization behaviors.

### Definitions

**KL as Reward (stop-gradient):**

```python
kl = compute_kl(log_prob_q, log_prob_p).detach()
shaped_reward = reward - beta * kl
```

Use shaped reward for standard actor-critic updates.

**KL as Loss (backprop):**

```python
actor_loss = -advantage * log_prob + beta * kl  # kl participates in gradient
```

Critic only learns environment value; KL is a regularization term for the actor that backpropagates gradients.

### Key Difference 1: Optimization Target

**KL as Reward:** Optimizes a **regularized new MDP** where the reward function becomes $\tilde{r}(s,a) = r(s,a) - \beta \cdot \text{KL}(s)$.

**KL as Loss:** Optimizes the **original task + supervised regularization**; KL doesn't change the MDP definition, it's just an external constraint term.

**Intuition:** The former "changes the game rules"; the latter "adds constraints under the original rules".

### Key Difference 2: Actor Gradient

**KL as Reward:** Single policy gradient, KL influence is **reflected indirectly through advantage**:

$$
g_{\text{reward}} = \mathbb{E}\left[s_\theta \cdot \tilde{A}_t\right], \quad \tilde{A}_t \text{ based on } (r_t - \beta \cdot \text{KL}_t)
$$

**KL as Loss:** Gradient splits into two independent paths:

$$
g_{\text{loss}} = \underbrace{\mathbb{E}\left[s_\theta \cdot A_t^{\text{env}}\right]}_{\text{RL gradient}} + \underbrace{\beta \cdot \mathbb{E}\left[\nabla_\theta \text{KL}_t\right]}_{\text{KL explicit gradient}}
$$

**Key distinction:** Is KL's force "multiplied on advantage" or "a separate force"? The latter's KL gradient is deterministic, unaffected by critic quality.

### Key Difference 3: Critic Learning Target

**KL as Reward:** Critic learns mixed value

$$
V^{\text{reg}}(s) = \mathbb{E}\left[\sum_t \gamma^t (r_t - \beta \cdot \text{KL}_t)\right]
$$

**KL as Loss:** Critic only learns environment value

$$
V^{\text{env}}(s) = \mathbb{E}\left[\sum_t \gamma^t r_t\right]
$$

The latter has cleaner separation, making it easier to monitor task return and KL divergence separately.

### Key Difference 4: Credit Assignment

Consider a scenario: first few steps are routing behavior, final step has high reward but also high KL.

**KL as Reward:** The large KL at the terminal state is **propagated back to all previous steps** through TD, so the policy tends to **fundamentally avoid** high-KL regions — this is "planning-based KL budget allocation".

**KL as Loss:** The terminal state's KL only appears in that state's gradient term; the policy is still willing to **visit high-reward regions but locally correct** behavior.

### Summary

| Dimension | KL as Reward (stop-grad) | KL as Loss (backprop) |
| :---: | :---: | :---: |
| Optimization target | Regularized new MDP | Original task + supervised regularization |
| Actor gradient | Single PG, based on shaped advantage | RL gradient + explicit KL gradient |
| Critic | Learns $V^{\text{reg}}$: reward + KL mixed | Learns $V^{\text{env}}$: only environment reward |
| Credit Assignment | Multi-step backprop, planning-capable | Local per-state, no planning |

**One-liner:** KL as reward makes the agent "plan to avoid high-KL paths" — constraints are more global and thorough; KL as loss makes the agent "visit but locally correct" — constraints are more local and flexible. The choice depends on whether you need cross-timestep KL budget allocation capability, and whether you want constraints to be "preventive" or "corrective".

## RL practice guide

Combining the preceding analysis of "estimator mathematical properties" and "usage modes", this section provides practical recommendations for specific scenarios.

### KL as reward penalty (no gradient needed)

When KL is a scalar penalty in rewards, we only need accurate **values**, no backprop. Refer to the earlier section on "Bias and variance for KL values".

**Recommend:**
- Use **$k_1$** or **$k_3$** (both unbiased for reverse KL value).
- When policies are close, $k_3$ is typically lower variance.
- With poor coverage or heavy tails, $k_1$ is more robust.
- Off-policy: multiply by $\tfrac{q_\theta}{\mu}$.

> For a **forward KL penalty**, use $\mathbb{E}_q[r \log r]$ or (if sampling from $p$) $\mathbb{E}_p[\log r]$.

### KL as loss (needs gradients)

#### On-policy: optimize reverse KL (most common)

Goal: keep actor near reference.

**Use $k_2$ as loss.**

$$
\mathcal{L}_{k_2} = \tfrac{1}{2}(\log r)^2
$$

Then $\mathbb{E}_q[\nabla k_2] = \nabla_\theta D_{\mathrm{KL}}(q \| p)$.

#### On-policy: optimize forward KL (coverage)

Goal: cover the reference distribution (offline RL, imitation, etc.).

**Use $k_3$ as loss.** Autograd on sample means gives $\mathbb{E}_q[(1-r) s_\theta] = \nabla_\theta D_{\mathrm{KL}}(p \| q)$.

#### Off-policy: optimize reverse KL

Goal: samples from behavior $\mu$, still optimize reverse KL.

**Recommended:** $\dfrac{q_\theta}{\mu} k_3$ or $\mathrm{sg}\left(\dfrac{q_\theta}{\mu}\right) k_2$ (identical gradients).

$$
\mathcal{L} = \dfrac{q_\theta(x)}{\mu(x)} \Big( \dfrac{p(x)}{q_\theta(x)} - 1 - \log \dfrac{p(x)}{q_\theta(x)} \Big)
$$

or

$$
\mathcal{L} = \mathrm{sg}\left(\dfrac{q_\theta(x)}{\mu(x)}\right) \cdot \tfrac{1}{2}\left(\log \dfrac{p(x)}{q_\theta(x)}\right)^2.
$$

- Gradients are unbiased.
- When $q_\theta \approx p$, both have much lower variance.

**Fallback:** $\dfrac{q_\theta}{\mu} k_1$ (unbiased but higher variance).

**Avoid:** $\dfrac{q_\theta}{\mu} k_2$ with weight in gradient — biased for reverse KL.

## "Grab-and-use" crib sheet

The table below provides recommended estimator choices along three dimensions: "target KL direction" × "sampling source" × "usage mode". "For **value**" corresponds to KL as reward penalty (no gradient needed); "For **gradient**" corresponds to KL as loss (gradient backpropagation needed).

| Target | Sampling | For value (KL as Reward) | For gradient (KL as Loss) |
| :---: | :---: | :---: | :---: |
| Reverse KL $D_{\mathrm{KL}}(q \| p)$ | $q$ (on-policy) | $k_1$ or $k_3$ (unbiased) | $k_2$ |
| Reverse KL $D_{\mathrm{KL}}(q \| p)$ | $\mu$ (off-policy) | $\tfrac{q}{\mu} k_1$ or $\tfrac{q}{\mu} k_3$ (unbiased) | $\tfrac{q}{\mu} k_3$ (recommended) or $\text{sg}(\tfrac{q}{\mu}) k_2$ |
| Forward KL $D_{\mathrm{KL}}(p \| q)$ | $q$ | $\mathbb{E}_q[r\log r]$ | $k_3$ |

## Common implementation traps

**Trap 1: Using $k_1$ directly as loss (on-policy)**

When KL is used as a loss, $k_1$ gradient expectation is zero ($\mathbb{E}_q[s_\theta]=0$); as a loss it does nothing.

> **Fix:** First clarify the KL usage mode. For reward shaping (no gradient needed), both $k_1$ and $k_3$ work; for losses (gradient needed), use $k_2$ (reverse KL) or $k_3$ (forward KL) on-policy.

**Trap 2: Mixing up $k_3$ value-unbiasedness vs. gradient target**

$k_3$ is value-unbiased for reverse KL, but its **gradient** is **forward KL**. If you want reverse KL and backprop $k_3$, you are actually optimizing forward KL.

> **Fix:** be explicit: reverse KL -> $k_2$; forward KL -> $k_3$.

**Trap 3: Heavy-tailed $r$ blows up variance**

If $r = p/q$ has extreme values, $k_3$ variance can explode.

> **Fix:** enforce KL constraint or clip $r$.

**Trap 4: Off-policy but still using $k_2$ or $\tfrac{q_\theta}{\mu} k_2$ (with grad on weight)**

If $\mu \neq q_\theta$:
- Plain $k_2$ (no weight): expectation is under $\mu$, estimator fails.
- $\tfrac{q_\theta}{\mu} k_2$ with weight in grad: gradient is biased (f-divergence), not reverse KL.

> **Fix:** off-policy reverse KL -> use $\tfrac{q_\theta}{\mu} k_3$ (recommended), $\text{sg}(\tfrac{q_\theta}{\mu}) k_2$, or $\tfrac{q_\theta}{\mu} k_1$.

**Trap 5: Wrong detach on importance weights**

$w = q_\theta / \mu$ often comes from `log_prob_q - log_prob_mu` then `exp`. Detaching $w$ matters:

- **Using $k_1$ or $k_3$:** $w$ **must participate in gradient** (do not detach), otherwise you drop $\nabla_\theta w = w s_\theta$ and get wrong gradients.
- **Using $k_2$:** **detach $w$** to get reverse KL gradient. If $w$ stays in the graph, you get f-divergence gradient instead.

> **Summary:** match estimator with the right detach strategy.

## Summary

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
4. 初七123334. "RLHF/RLVR 训练中的 KL 近似方法浅析（k1 / k2 / k3)". <https://zhuanlan.zhihu.com/p/1966872846212010437>
5. Kezhao Liu, Jason Klein Liu, Mingtao Chen, Yiming Liu. "Rethinking KL Regularization in RLHF: From Value Estimation to Gradient Optimization". <https://arxiv.org/abs/2510.01555>
6. Yifan Zhang, Yiping Ji, Gavin Brown, et al. "On the Design of KL-Regularized Policy Gradient Algorithms for LLM Reasoning". <https://arxiv.org/abs/2505.17508>

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
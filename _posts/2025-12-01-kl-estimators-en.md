---
layout: post
title: "Understanding KL Divergence Estimators in RL: From Value Approximation to Gradient Estimation"
date: 2025-12-01
description: "In reinforcement learning, how we approximate KL divergence directly affects training stability. This post systematically dissects three classic estimators k1, k2, and k3, and gives practical guidelines for choosing them for reward penalties vs. gradient-based losses."
categories: reinforcement-learning
lang: en
---

* TOC
{:toc}

> In reinforcement learning, how we approximate KL divergence directly affects training stability. This post systematically analyzes the differences between three classic estimators $k_1, k_2, k_3$ and provides practical guidelines for choosing them when KL is used as a reward penalty versus when it is used as a loss for backpropagation.

[中文版](https://xihuai18.github.io/reinforcement-learning/2025/12/01/kl-estimators-cn.html) \| [知乎版本 ![Zhihu](https://static.zhihu.com/heifetz/favicon.ico)](https://zhuanlan.zhihu.com/p/1978993413425763764)

![Mini-class](/assets/img/kl-estimator-en.png){: style="display:block;margin:0 auto;width:95%;max-width:100%;" }

## Introduction: The Role of KL Divergence in Reinforcement Learning

In policy optimization (PPO, GRPO, etc.) or alignment training (RLHF/RLAIF), **KL regularization** is the core mechanism that constrains the new policy from drifting too far away from a reference policy, in order to prevent unstable training or policy collapse.

### Forward vs. Reverse KL

Let $q_\theta$ be the current actor policy and $p$ be the reference policy. The two directions of KL divergence are

**Reverse KL**:
$$
D_{\mathrm{KL}}(q_\theta \| p) = \mathbb{E}_{x \sim q_\theta}\left[\log \frac{q_\theta(x)}{p(x)}\right]
$$

<figure style="text-align:center;">
	<img src="/assets/img/kl-estimator-reverse.png" style="width:95%;max-width:100%;">
	<figcaption style="font-size:0.9em;color:gray;">Image credit: <a href="https://dibyaghosh.com/blog/probability/kldivergence/">Dibya Ghosh's Blog</a></figcaption>
</figure>

**Forward KL**:
$$
D_{\mathrm{KL}}(p \| q_\theta) = \mathbb{E}_{x \sim p}\left[\log \frac{p(x)}{q_\theta(x)}\right]
$$

<figure style="text-align:center;">
	<img src="/assets/img/kl-estimator-forward.png" style="width:95%;max-width:100%;">
	<figcaption style="font-size:0.9em;color:gray;">Image credit: <a href="https://dibyaghosh.com/blog/probability/kldivergence/">Dibya Ghosh's Blog</a></figcaption>
</figure>

**Intuition**:
- **Reverse KL** is typically **mode-seeking** – the policy concentrates on high-density regions of the reference distribution and may sacrifice diversity.
- **Forward KL** is typically **mass-covering** – the policy tries to cover the full support of the reference distribution.

In mainstream RLHF implementations, **reverse KL** is more common, because we want the actor not to drift too far away from the reference policy, rather than forcing it to cover all modes.


## Three Estimators: Definitions and Design Principles

Let the ratio be $r(x) = \frac{p(x)}{q_\theta(x)}$. John Schulman introduced the following three single-sample estimators:

### $k_1$: The Most Naive Estimator

$$
k_1(x) = -\log r = \log q_\theta(x) - \log p(x)
$$

This is the most direct definition – simply taking the negative log-ratio. It is an **unbiased** estimator of reverse KL, but it has a fatal issue: it **can be negative**, whereas KL divergence is always non-negative. This leads to extremely high variance because positive and negative samples can cancel each other out.

### $k_2$: A Low-Variance Estimator from f-Divergences

$$
k_2(x) = \frac{1}{2}(\log r)^2
$$

**Design motivation**: The problem with $k_1$ is that it can be both positive and negative. $k_2$ squares the log-ratio, ensuring that **every sample is positive**. Intuitively, each sample tells you “how far apart” $p$ and $q$ are.

**Why is the bias small?** $k_2$ is essentially an **f-divergence** with $f(x) = \frac{1}{2}(\log x)^2$. f-divergences have a nice property: **for any differentiable $f$-divergence, when $q \approx p$, the second-order expansion has the form**

$$
D_f(p, q_\theta) = \frac{f''(1)}{2} \theta^T F \theta + O(\theta^3)
$$

where $F$ is the Fisher information matrix. KL divergence corresponds to $f(x) = -\log x$, with $f''(1) = 1$; $k_2$ corresponds to $f(x) = \frac{1}{2}(\log x)^2$, which also satisfies $f''(1) = 1$. This means that **when the policies are close, $k_2$ behaves almost identically to the true KL**, and the bias only appears in higher-order terms.

### $k_3$: A “Best of Both Worlds” Estimator via Control Variates

$$
k_3(x) = r - 1 - \log r
$$

**Design motivation**: We would like an estimator that is **both unbiased and low variance**. A standard trick is to add a **control variate** – a zero-mean term that is negatively correlated with the original estimator.

Note that $\mathbb{E}_q[r - 1] = \mathbb{E}_q\left[\frac{p}{q}\right] - 1 = 1 - 1 = 0$. Therefore, for any $\lambda$,

$$
k_1 + \lambda(r - 1) = -\log r + \lambda(r - 1)
$$

remains an unbiased estimator.

**Why choose $\lambda = 1$?** Since $\log$ is concave, we have $\log x \leq x - 1$, so

$$
k_3 = (r - 1) - \log r \geq 0
$$

which is **always non-negative**. This guarantees that each sample contributes information in the “same direction” and eliminates the cancellation problem of $k_1$.

**Geometric intuition**: $k_3$ is in fact a **Bregman divergence**. Consider the convex function $\phi(x) = -\log x$. The tangent at $x = 1$ is $y = 1 - x$. The Bregman divergence between $r$ and 1 is

$$
D_\phi(r, 1) = \phi(r) - \phi(1) - \phi'(1)(r - 1) = -\log r - 0 - (-1)(r - 1) = r - 1 - \log r = k_3.
$$

Since a convex function always lies above its tangents, this difference is **naturally non-negative**. More importantly, as $r \to 1$, the function and its tangent “stick together” more tightly, and the gap shrinks at the rate of $(r-1)^2$. This is exactly why $k_3$ has small variance when the policies are close.


### Summary: Comparing the Three Estimators

| Estimator | Definition              | Design principle                     |   Value bias   | Variance           |
| :-------: | :---------------------- | :----------------------------------- | :------------: | :----------------- |
|   $k_1$   | $-\log r$               | Naive definition                     |    Unbiased    | High (can be neg.) |
|   $k_2$   | $\frac{1}{2}(\log r)^2$ | f-divergence, 2nd-order matches KL   | Biased (small) | Low (always pos.)  |
|   $k_3$   | $r - 1 - \log r$        | Control variate + Bregman divergence |    Unbiased    | Low (always pos.)  |

From a pure **value estimation** perspective, $k_3$ looks like the “best of both worlds”: **unbiased + low variance**. However, as we will see, the **story is completely different at the gradient level**.


## Core Analysis

### Bias and Variance for Estimating the KL Value

Assume we sample from $q_\theta$ to estimate the reverse KL $D_{\mathrm{KL}}(q_\theta \| p)$.

**Unbiasedness**:

$$
\mathbb{E}_q[k_1] = \mathbb{E}_q\left[\log \frac{q}{p}\right] = D_{\mathrm{KL}}(q \| p) \quad \textbf{(unbiased)}
$$

$$
\mathbb{E}_q[k_3] = \mathbb{E}_q[r - 1 - \log r] = 1 - 1 + D_{\mathrm{KL}}(q \| p) = D_{\mathrm{KL}}(q \| p) \quad \textbf{(unbiased)}
$$

$$
\mathbb{E}_q[k_2] = \frac{1}{2}\mathbb{E}_q[(\log r)^2] \neq D_{\mathrm{KL}}(q \| p) \quad \textbf{(biased)}
$$

**Conclusion**: For estimating the **value** of the reverse KL, $k_1$ and $k_3$ are unbiased, whereas $k_2$ is biased.

**Bias–variance trade-off**:

In John Schulman’s experiment with $q = \mathcal{N}(0,1)$, $p = \mathcal{N}(0.1,1)$ and true KL = 0.005, the statistics are

| Estimator | bias/true | stdev/true |
| :-------: | :-------: | :--------: |
|   $k_1$   |     0     |     20     |
|   $k_2$   |   0.002   |    1.42    |
|   $k_3$   |     0     |    1.42    |

When KL is larger ($p = \mathcal{N}(1,1)$, true KL = 0.5):

| Estimator | bias/true | stdev/true |
| :-------: | :-------: | :--------: |
|   $k_1$   |     0     |     2      |
|   $k_2$   |   0.25    |    1.73    |
|   $k_3$   |     0     |    1.7     |

**Intuition**:
- $k_1 = -\log r$ starts with a first-order term. When $r$ is close to 1, its fluctuations are large and it can be negative.
- $k_3 = r - 1 - \log r$ is second-order around $r = 1$ and always non-negative, so it has smaller variance when policies are close.
- When coverage is very poor (i.e., $r$ can explode), the variance of $k_3$ can blow up due to the heavy tails of $r$; in that regime, $k_1$ can be more stable.

> **Note**: To estimate the **forward KL value** $D_{\mathrm{KL}}(p \| q) = \mathbb{E}_p[\log r]$ using samples from $q$, you can use importance sampling $\mathbb{E}_q[r \log r]$.


### The Crucial Distinction When Estimating KL Gradients

**This is the most confusing yet practically important part.**

#### True Gradients of Forward and Reverse KL

Before analyzing the estimators, let us derive the **true gradients** of forward and reverse KL with respect to $\theta$.

Denote the score function $s_\theta(x) = \nabla_\theta \log q_\theta(x)$. A key property is $\mathbb{E}_{q_\theta}[s_\theta] = 0$ (since $\int \nabla_\theta q_\theta dx = \nabla_\theta 1 = 0$).

**Gradient of reverse KL**:

$$
D_{\mathrm{KL}}(q_\theta \| p) = \int q_\theta(x) \log \frac{q_\theta(x)}{p(x)} dx.
$$

Taking the gradient with respect to $\theta$ (using the product rule):

$$
\nabla_\theta D_{\mathrm{KL}}(q_\theta \| p) = \int \nabla_\theta q_\theta \cdot \log \frac{q_\theta}{p} dx + \int q_\theta \cdot \nabla_\theta \log \frac{q_\theta}{p} dx.
$$

Using $\nabla_\theta q_\theta = q_\theta s_\theta$, $\nabla_\theta \log q_\theta = s_\theta$, and $\nabla_\theta \log p = 0$ gives

$$
= \mathbb{E}_q\left[s_\theta \cdot \log \frac{q_\theta}{p}\right] + \mathbb{E}_q[s_\theta] = \mathbb{E}_q\left[s_\theta \cdot \log \frac{q_\theta}{p}\right].
$$

Thus

$$
\boxed{\nabla_\theta D_{\mathrm{KL}}(q_\theta \| p) = \mathbb{E}_q\left[s_\theta \cdot \log \frac{q_\theta}{p}\right] = -\mathbb{E}_q[s_\theta \cdot \log r]}
$$

**Gradient of forward KL**:

$$
D_{\mathrm{KL}}(p \| q_\theta) = \int p(x) \log \frac{p(x)}{q_\theta(x)} dx.
$$

Since $p(x)$ is independent of $\theta$,

$$
\nabla_\theta D_{\mathrm{KL}}(p \| q_\theta) = \int p(x) \cdot \nabla_\theta(-\log q_\theta(x)) dx = -\mathbb{E}_p[s_\theta].
$$

To estimate this using samples from $q$, we use importance sampling:

$$
-\mathbb{E}_p[s_\theta] = -\mathbb{E}_q\left[\frac{p}{q_\theta} s_\theta\right] = -\mathbb{E}_q[r s_\theta].
$$

Using $\mathbb{E}_q[s_\theta] = 0$, we can rewrite this as

$$
\boxed{\nabla_\theta D_{\mathrm{KL}}(p \| q_\theta) = \mathbb{E}_q[(1 - r) s_\theta]}
$$

These two ground-truth gradients will be our reference when judging what each estimator’s gradient actually corresponds to.

#### Two Orders of Operations for Gradients

In implementation, there are two conceptual orders of operations:

1. **Gradient-then-expectation**: compute $\nabla_\theta k_i(x)$ for each sample and then average (Monte Carlo estimator).
2. **Expectation-then-gradient**: treat $\mathbb{E}_q[k_i]$ as a function of $\theta$ and differentiate analytically.

**In typical deep learning code, we do (1)**: autodiff computes the gradient per sample and then the batch average.

#### Gradients of the Three Estimators

Now we compute the gradients of the three estimators and see which KL gradient each one matches in expectation.

**Gradient of $k_1$**:

$$
k_1 = -\log r = -\log \frac{p(x)}{q_\theta(x)} = \log q_\theta(x) - \log p(x),
$$

so

$$
\nabla_\theta k_1 = \nabla_\theta \log q_\theta(x) - \nabla_\theta \log p(x) = s_\theta - 0 = s_\theta.
$$

**Gradient of $k_2$**:

$$
k_2 = \frac{1}{2}(\log r)^2.
$$

By the chain rule

$$
\nabla_\theta k_2 = (\log r) \cdot \nabla_\theta(\log r) = (\log r) \cdot \nabla_\theta(\log p(x) - \log q_\theta(x)) = (\log r)(-s_\theta) = - (\log r) s_\theta.
$$

**Gradient of $k_3$**:

$$
k_3 = r - 1 - \log r.
$$

First compute $\nabla_\theta r$. Since $r = p(x) q_\theta(x)^{-1}$,

$$
\nabla_\theta r = p(x)(-1) q_\theta(x)^{-2} \nabla_\theta q_\theta(x) = -\frac{p(x)}{q_\theta(x)} \cdot \frac{\nabla_\theta q_\theta(x)}{q_\theta(x)} = -r s_\theta.
$$

Then

$$
\nabla_\theta \log r = \frac{1}{r} \nabla_\theta r = \frac{1}{r}(-r s_\theta) = -s_\theta,
$$

so

$$
\nabla_\theta k_3 = \nabla_\theta r - 0 - \nabla_\theta \log r = -r s_\theta - (-s_\theta) = (1 - r) s_\theta.
$$

Taking expectations under $q_\theta$:

| Estimator | $\mathbb{E}_q[\nabla_\theta k_i]$                                          | Equals                     |
| :-------: | :------------------------------------------------------------------------- | :------------------------- |
|   $k_1$   | $\mathbb{E}_q[s_\theta] = 0$                                               | **Zero (useless as loss)** |
|   $k_2$   | $-\mathbb{E}_q[(\log r) s_\theta] = \nabla_\theta D_{\mathrm{KL}}(q \| p)$ | **Gradient of reverse KL** |
|   $k_3$   | $\mathbb{E}_q[(1 - r) s_\theta] = \nabla_\theta D_{\mathrm{KL}}(p \| q)$   | **Gradient of forward KL** |

**Key takeaways**:
- The **gradient of $k_2$** matches the true gradient of **reverse KL** – this is the correct choice if your goal is to constrain the policy not to drift from the reference.
- The **gradient of $k_3$** matches the true gradient of **forward KL** – this corresponds to a coverage-style objective.
- The **expected gradient of $k_1$ is always zero** – using $k_1$ directly as a loss is pointless.

#### “Expectation-then-gradient” vs. “Gradient-then-expectation”

If, analytically, you first compute $\mathbb{E}_q[k_i]$ and then differentiate (i.e., **expectation-then-gradient**), you obtain

$$
\nabla_\theta \mathbb{E}_q[k_1] = \nabla_\theta D_{\mathrm{KL}}(q \| p)
$$

and

$$
\nabla_\theta \mathbb{E}_q[k_3] = \nabla_\theta D_{\mathrm{KL}}(q \| p).
$$

Both give the gradient of reverse KL. But when you implement $k_3$ as a **sample-wise loss in code** and call `backward` on the batch mean, autodiff is effectively computing $\mathbb{E}_q[\nabla_\theta k_3]$, which, as shown above, is actually the gradient of **forward KL**.

This subtle difference is crucial: **for the same estimator, changing the order of expectation and gradient can lead to completely different optimization objectives**.


## Practical Guidelines for RL

### KL as a Reward Penalty (No Gradient Needed)

When KL is only used as a scalar penalty in reward shaping, we only care about an accurate **value estimate**, and we do not backpropagate through it.

**Recommendations**:
- Use **$k_1$** or **$k_3$** (both are unbiased for the reverse KL value).
- When the policy is already close to the reference, $k_3$ often has lower variance.
- When coverage is poor or there is severe tail mismatch, $k_1$ can be more robust.

> **Note**: If you want a **forward KL penalty** (to encourage coverage of the behavior distribution), you can use $\mathbb{E}_q[r \log r]$ or, if you can sample from $p$, directly use $\mathbb{E}_p[\log r]$.

### KL as a Loss (Gradient Required)

When KL is part of the loss that you differentiate, you must ensure that the gradient matches your intended objective.

#### Optimizing Reverse KL (Most Common Case)

Goal: constrain the actor not to drift far from the reference policy.

**Correct choice**: use **$k_2$** as the loss.

$$
\mathcal{L}_{k_2} = \frac{1}{2}(\log r)^2.
$$

Its gradient expectation $\mathbb{E}_q[\nabla k_2] = \nabla_\theta D_{\mathrm{KL}}(q \| p)$ is exactly the true gradient of reverse KL.

#### Optimizing Forward KL (Coverage-Oriented Settings)

Goal: make the policy cover the support of the reference distribution (e.g., in offline RL or imitation learning).

**Correct choice**: use **$k_3$** as the loss.

$$
\mathbb{E}_q[\nabla k_3] = \mathbb{E}_q[(1 - r) s_\theta] = \nabla_\theta D_{\mathrm{KL}}(p \| q).
$$

If you backpropagate through the batch mean of $k_3$, autodiff computes exactly this forward-KL gradient – no extra tricks needed.


## A Ready-to-Use Cheat Sheet

| Objective                          | Sampling dist. | For **value estimate**    | For **gradient (loss)** |
| :--------------------------------- | :------------: | :------------------------ | :---------------------- |
| Reverse KL $D_{\mathrm{KL}}(q\|p)$ |      $q$       | $k_1$ or $k_3$ (unbiased) | $k_2$                   |
| Forward KL $D_{\mathrm{KL}}(p\|q)$ |      $q$       | $\mathbb{E}_q[r\log r]$   | $k_3$                   |


## Common Implementation Pitfalls

**Pitfall 1: Using $k_1$ Directly as a Loss**

The expected gradient of $k_1$ is zero ($\mathbb{E}_q[\nabla k_1] = \mathbb{E}_q[s_\theta] = 0$), so as a loss it is ineffective.

> **Fix**: Use $k_1$ or $k_3$ only when you need a scalar KL penalty in rewards (no gradient), and use $k_2$ or $k_3$ when you actually want a loss with a meaningful gradient.

**Pitfall 2: Confusing $k_3$’s Unbiased Value with Its Gradient Objective**

$k_3$ is an **unbiased value estimator of the reverse KL**, but its **gradient** corresponds to the **forward KL**. If your goal is to optimize reverse KL but you use $k_3$ as a loss, you are in fact optimizing forward KL.

> **Fix**: Be explicit about your objective. Use $k_2$ when optimizing reverse KL; use $k_3$ only when you intentionally optimize forward KL.

**Pitfall 3: Heavy-Tailed $r$ Causing Variance Explosion**

When the policy and reference distribution are very different, $r = p/q$ can have extreme values, causing the variance of $k_3$ (and importance-sampling-based estimators) to blow up.


## Conclusion

**One-line summary**:

- **KL for value only (reward penalty)**: use $k_1$ or $k_3$ (both are unbiased for reverse KL).
- **KL as a differentiable loss (needs gradients)**:
	- To optimize **reverse KL**, use $k_2$.
	- To optimize **forward KL**, use $k_3$.

Once you keep clear **who you sample from**, **which KL you estimate**, and **with respect to which quantity you differentiate**, the three estimators become much less confusing.


## References

1. Dibya Ghosh. "KL Divergence for Machine Learning". https://dibyaghosh.com/blog/probability/kldivergence
2. John Schulman. "Approximating KL Divergence". https://joschu.net/blog/kl-approx.html
3. Verl Documentation. "Proximal Policy Optimization (PPO)". https://verl.readthedocs.io/en/latest/algo/ppo.html
4. 初七123334. "Approximate KL in RLHF/RLVR Training: A Brief Analysis of k1 / k2 / k3" (in Chinese). https://zhuanlan.zhihu.com/p/1966872846212010437
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


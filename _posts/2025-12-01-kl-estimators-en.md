---
layout: post
title: "Understanding KL Divergence Estimators in RL: From Value Approximation to Gradient Estimation"
date: 2025-12-01
description: "In reinforcement learning, how we approximate KL divergence directly affects training stability. This post systematically dissects three classic estimators k1, k2, and k3, covering both on-policy and off-policy scenarios, and gives practical guidelines for choosing them for reward penalties vs. gradient-based losses."
categories: reinforcement-learning
lang: en
---

* TOC
{:toc}

![Mini-class](/assets/img/kl-estimators/kl-estimator-en.png){: style="display:block;margin:0 auto;width:95%;max-width:100%;" }

> In reinforcement learning, how we approximate KL divergence directly affects training stability. This post systematically analyzes the differences between three classic estimators $k\_1, k\_2, k\_3$ in both on-policy and off-policy scenarios, and provides practical guidelines for choosing them when KL is used as a reward penalty versus when it is used as a loss for backpropagation.

[中文版](/reinforcement-learning/2025/12/01/kl-estimators-cn.html) \| [知乎版本 ![Zhihu](https://static.zhihu.com/heifetz/favicon.ico)](https://zhuanlan.zhihu.com/p/1978993413425763764)

## Introduction: The Role of KL Divergence in Reinforcement Learning

In policy optimization (PPO, GRPO, etc.) or alignment training (RLHF/RLAIF), **KL regularization** is the core mechanism that constrains the new policy from drifting too far away from a reference policy, in order to prevent unstable training or policy collapse.

### Forward vs. Reverse KL

Let $q\_\theta$ be the current actor policy and $p$ be the reference policy. The two directions of KL divergence are

**Reverse KL**:
$$
D_{\mathrm{KL}}(q_\theta \| p) = \mathbb{E}_{x \sim q_\theta}\left[\log \frac{q_\theta(x)}{p(x)}\right]
$$

<figure style="text-align:center;">
	<img src="/assets/img/kl-estimators/kl-estimator-reverse.png" style="width:95%;max-width:100%;">
	<figcaption style="font-size:0.9em;color:gray;">Image credit: <a href="https://dibyaghosh.com/blog/probability/kldivergence/">Dibya Ghosh's Blog</a></figcaption>
</figure>

**Forward KL**:
$$
D_{\mathrm{KL}}(p \| q_\theta) = \mathbb{E}_{x \sim p}\left[\log \frac{p(x)}{q_\theta(x)}\right]
$$

<figure style="text-align:center;">
	<img src="/assets/img/kl-estimators/kl-estimator-forward.png" style="width:95%;max-width:100%;">
	<figcaption style="font-size:0.9em;color:gray;">Image credit: <a href="https://dibyaghosh.com/blog/probability/kldivergence/">Dibya Ghosh's Blog</a></figcaption>
</figure>

**Intuition**:
- **Reverse KL** is typically **mode-seeking** – the policy concentrates on high-density regions of the reference distribution and may sacrifice diversity.
- **Forward KL** is typically **mass-covering** – the policy tries to cover the full support of the reference distribution.

In mainstream RLHF implementations, **reverse KL** is more common, because we want the actor not to drift too far away from the reference policy, rather than forcing it to cover all modes.


## Three Estimators: Definitions and Design Principles

Let the ratio be $r(x) = \frac{p(x)}{q\_\theta(x)}$. John Schulman introduced the following three single-sample estimators:

### $k\_1$: The Most Naive Estimator

$$
k_1(x) = -\log r = \log q_\theta(x) - \log p(x)
$$

This is the most direct definition – simply taking the negative log-ratio. It is an **unbiased** estimator of reverse KL, but it has a fatal issue: it **can be negative**, whereas KL divergence is always non-negative. This leads to extremely high variance because positive and negative samples can cancel each other out.

### $k\_2$: A Low-Variance Estimator from f-Divergences

$$
k_2(x) = \frac{1}{2}(\log r)^2
$$

**Design motivation**: The problem with $k\_1$ is that it can be both positive and negative. $k\_2$ squares the log-ratio, ensuring that **every sample is positive**. Intuitively, each sample tells you "how far apart" $p$ and $q$ are.

**Why is the bias small?** $k\_2$ is essentially an **f-divergence** with $f(x) = \frac{1}{2}(\log x)^2$. f-divergences have a nice property: **for any differentiable $f$-divergence, when $q \approx p$, the second-order expansion has the form**

$$
D_f(p, q_\theta) = \frac{f^{\prime\prime}(1)}{2} \theta^T F \theta + O(\theta^3)
$$

where $F$ is the Fisher information matrix. KL divergence corresponds to $f(x) = -\log x$, with $f^{\prime\prime}(1) = 1$; $k\_2$ corresponds to $f(x) = \frac{1}{2}(\log x)^2$, which also satisfies $f^{\prime\prime}(1) = 1$. This means that **when the policies are close, $k\_2$ behaves almost identically to the true KL**, and the bias only appears in higher-order terms.

### $k\_3$: A "Best of Both Worlds" Estimator via Control Variates

$$
k_3(x) = r - 1 - \log r
$$

**Design motivation**: We would like an estimator that is **both unbiased and low variance**. A standard trick is to add a **control variate** – a zero-mean term that is negatively correlated with the original estimator.

Note that $\mathbb{E}\_q[r - 1] = \mathbb{E}\_q\left[\frac{p}{q}\right] - 1 = 1 - 1 = 0$. Therefore, for any $\lambda$,

$$
k_1 + \lambda(r - 1) = -\log r + \lambda(r - 1)
$$

remains an unbiased estimator.

**Why choose $\lambda = 1$?** Since $\log$ is concave, we have $\log x \leq x - 1$, so

$$
k_3 = (r - 1) - \log r \geq 0
$$

which is **always non-negative**. This guarantees that each sample contributes information in the "same direction" and eliminates the cancellation problem of $k\_1$.

**Geometric intuition**: $k\_3$ is in fact a **Bregman divergence**. Consider the convex function $\phi(x) = -\log x$. The tangent at $x = 1$ is $y = 1 - x$. The Bregman divergence between $r$ and 1 is

$$
\begin{aligned}
D_\phi(r, 1) &= \phi(r) - \phi(1) - \phi'(1)(r - 1) \\
&= -\log r - 0 - (-1)(r - 1) \\
&= r - 1 - \log r \\
&= k_3.
\end{aligned}
$$

Since a convex function always lies above its tangents, this difference is **naturally non-negative**. More importantly, as $r \to 1$, the function and its tangent "stick together" more tightly, and the gap shrinks at the rate of $(r-1)^2$. This is exactly why $k\_3$ has small variance when the policies are close.


### Summary: Comparing the Three Estimators

<div class="table-responsive" markdown="0">
<table class="table table-bordered" style="font-size: 0.95em;">
  <thead>
    <tr style="background-color: var(--global-bg-color);">
      <th style="text-align: center;">Estimator</th>
      <th style="text-align: center;">Definition</th>
      <th style="text-align: center;">Design Principle</th>
      <th style="text-align: center;">Value Bias</th>
      <th style="text-align: center;">Variance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="text-align: center;">$k\_1$</td>
      <td style="text-align: center;">$-\log r$</td>
      <td style="text-align: center;">Naive definition</td>
      <td style="text-align: center;">Unbiased</td>
      <td style="text-align: center;">High (can be neg.)</td>
    </tr>
    <tr>
      <td style="text-align: center;">$k\_2$</td>
      <td style="text-align: center;">$\frac{1}{2}(\log r)^2$</td>
      <td style="text-align: center;">f-divergence, 2nd-order matches KL</td>
      <td style="text-align: center;">Biased (small)</td>
      <td style="text-align: center;">Low (always pos.)</td>
    </tr>
    <tr>
      <td style="text-align: center;">$k\_3$</td>
      <td style="text-align: center;">$r - 1 - \log r$</td>
      <td style="text-align: center;">Control variate + Bregman divergence</td>
      <td style="text-align: center;">Unbiased</td>
      <td style="text-align: center;">Low (always pos.)</td>
    </tr>
  </tbody>
</table>
</div>

From a pure **value estimation** perspective, $k\_3$ looks like the "best of both worlds": **unbiased + low variance**. However, as we will see, the **story is completely different at the gradient level**.


## Core Analysis

### Bias and Variance for Estimating the KL Value

Assume we sample from $q\_\theta$ to estimate the reverse KL $D\_{\mathrm{KL}}(q\_\theta \| p)$.

**Unbiasedness**:

$$
\begin{aligned}
\mathbb{E}_{q}[k_1] &= \mathbb{E}_{q}\left[\log \frac{q}{p}\right] = D_{\mathrm{KL}}(q \| p) \quad \textbf{(Unbiased)}\\
\mathbb{E}_{q}[k_3] &= \mathbb{E}_{q}[r - 1 - \log r] \\
&= 1 - 1 + D_{\mathrm{KL}}(q \| p) \\
&= D_{\mathrm{KL}}(q \| p) \quad \textbf{(Unbiased)}\\
\mathbb{E}_{q}[k_2] &= \frac{1}{2}\mathbb{E}_{q}[(\log r)^2] \neq D_{\mathrm{KL}}(q \| p) \quad \textbf{(Biased)}
\end{aligned}
$$

**Conclusion**: For estimating the **value** of the reverse KL, $k\_1$ and $k\_3$ are unbiased, whereas $k\_2$ is biased.

**Bias–variance trade-off**:

In John Schulman's experiment with $q = \mathcal{N}(0,1)$, $p = \mathcal{N}(0.1,1)$ and true KL = 0.005, the statistics are

<div class="table-responsive" markdown="0">
<table class="table table-bordered" style="font-size: 0.95em;">
  <thead>
    <tr style="background-color: var(--global-bg-color);">
      <th style="text-align: center;">Estimator</th>
      <th style="text-align: center;">bias/true</th>
      <th style="text-align: center;">stdev/true</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="text-align: center;">$k\_1$</td>
      <td style="text-align: center;">0</td>
      <td style="text-align: center;">20</td>
    </tr>
    <tr>
      <td style="text-align: center;">$k\_2$</td>
      <td style="text-align: center;">0.002</td>
      <td style="text-align: center;">1.42</td>
    </tr>
    <tr>
      <td style="text-align: center;">$k\_3$</td>
      <td style="text-align: center;">0</td>
      <td style="text-align: center;">1.42</td>
    </tr>
  </tbody>
</table>
</div>

When KL is larger ($p = \mathcal{N}(1,1)$, true KL = 0.5):

<div class="table-responsive" markdown="0">
<table class="table table-bordered" style="font-size: 0.95em;">
  <thead>
    <tr style="background-color: var(--global-bg-color);">
      <th style="text-align: center;">Estimator</th>
      <th style="text-align: center;">bias/true</th>
      <th style="text-align: center;">stdev/true</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="text-align: center;">$k\_1$</td>
      <td style="text-align: center;">0</td>
      <td style="text-align: center;">2</td>
    </tr>
    <tr>
      <td style="text-align: center;">$k\_2$</td>
      <td style="text-align: center;">0.25</td>
      <td style="text-align: center;">1.73</td>
    </tr>
    <tr>
      <td style="text-align: center;">$k\_3$</td>
      <td style="text-align: center;">0</td>
      <td style="text-align: center;">1.7</td>
    </tr>
  </tbody>
</table>
</div>

**Intuition**:
- $k\_1 = -\log r$ starts with a first-order term. When $r$ is close to 1, its fluctuations are large and it can be negative.
- $k\_3 = r - 1 - \log r$ is second-order around $r = 1$ and always non-negative, so it has smaller variance when policies are close.
- When coverage is very poor (i.e., $r$ can explode), the variance of $k\_3$ can blow up due to the heavy tails of $r$; in that regime, $k\_1$ can be more stable.

> **Note**: To estimate the **forward KL value** $D\_{\mathrm{KL}}(p \| q) = \mathbb{E}\_p[\log r]$ using samples from $q$, you can use importance sampling $\mathbb{E}\_q[r \log r]$.


### The Crucial Distinction When Estimating KL Gradients

**This is the most confusing yet practically important part.**

#### True Gradients of Forward and Reverse KL

Before analyzing the estimators, let us derive the **true gradients** of forward and reverse KL with respect to $\theta$.

Denote the score function $s\_\theta(x) = \nabla\_\theta \log q\_\theta(x)$. A key property is $\mathbb{E}\_{q\_\theta}[s\_\theta] = 0$ (since $\int \nabla\_\theta q\_\theta dx = \nabla\_\theta 1 = 0$).

**Gradient of reverse KL**:

$$
D_{\mathrm{KL}}(q_\theta \| p) = \int q_\theta(x) \log \frac{q_\theta(x)}{p(x)} dx.
$$

Taking the gradient with respect to $\theta$ (using the product rule):

$$
\nabla_\theta D_{\mathrm{KL}}(q_\theta \| p) = \int \nabla_\theta q_\theta \cdot \log \frac{q_\theta}{p} dx + \int q_\theta \cdot \nabla_\theta \log \frac{q_\theta}{p} dx.
$$

Using $\nabla\_\theta q\_\theta = q\_\theta s\_\theta$, $\nabla\_\theta \log q\_\theta = s\_\theta$, and $\nabla\_\theta \log p = 0$ gives

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

Using $\mathbb{E}\_q[s\_\theta] = 0$, we can rewrite this as

$$
\boxed{\nabla_\theta D_{\mathrm{KL}}(p \| q_\theta) = \mathbb{E}_q[(1 - r) s_\theta]}
$$

These two ground-truth gradients will be our reference when judging what each estimator’s gradient actually corresponds to.

#### Two Orders of Operations for Gradients

In implementation, there are two conceptual orders of operations:

1. **Gradient-then-expectation**: compute $\nabla\_\theta k\_i(x)$ for each sample and then average (Monte Carlo estimator).
2. **Expectation-then-gradient**: treat $\mathbb{E}\_q[k\_i]$ as a function of $\theta$ and differentiate analytically.

**In typical deep learning code, we do (1)**: autodiff computes the gradient per sample and then the batch average.

#### Gradients of the Three Estimators

Now we compute the gradients of the three estimators and see which KL gradient each one matches in expectation.

**Gradient of $k\_1$**:

$$
k_1 = -\log r = -\log \frac{p(x)}{q_\theta(x)} = \log q_\theta(x) - \log p(x),
$$

so

$$
\nabla_\theta k_1 = \nabla_\theta \log q_\theta(x) - \nabla_\theta \log p(x) = s_\theta - 0 = s_\theta.
$$

**Gradient of $k\_2$**:

$$
k_2 = \frac{1}{2}(\log r)^2.
$$

By the chain rule

$$
\begin{aligned}
\nabla_\theta k_2 
&= (\log r) \cdot \nabla_\theta(\log r) \\
&= (\log r) \cdot \nabla_\theta(\log p(x) - \log q_\theta(x)) \\
&= (\log r)(-s_\theta) \\
&= - (\log r) s_\theta.
\end{aligned}
$$

**Gradient of $k\_3$**:

$$
k_3 = r - 1 - \log r.
$$

First compute $\nabla\_\theta r$. Since $r = p(x) q\_\theta(x)^{-1}$,

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

Taking expectations under $q\_\theta$:

<div class="table-responsive" markdown="0">
<table class="table table-bordered" style="font-size: 0.95em;">
  <thead>
    <tr style="background-color: var(--global-bg-color);">
      <th style="text-align: center;">Estimator</th>
      <th style="text-align: center;">$\mathbb{E}\_{q}[\nabla\_\theta k\_i]$</th>
      <th style="text-align: center;">Equals</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="text-align: center;">$k\_1$</td>
      <td style="text-align: center;">$\mathbb{E}\_{q}[s\_\theta] = 0$</td>
      <td style="text-align: center;"><strong>Zero (useless as loss)</strong></td>
    </tr>
    <tr>
      <td style="text-align: center;">$k\_2$</td>
      <td style="text-align: center;">$-\mathbb{E}\_{q}[(\log r) \cdot s\_\theta] = \nabla\_\theta D\_{\mathrm{KL}}(q \| p)$</td>
      <td style="text-align: center;"><strong>Gradient of reverse KL</strong></td>
    </tr>
    <tr>
      <td style="text-align: center;">$k\_3$</td>
      <td style="text-align: center;">$\mathbb{E}\_{q}[(1-r) \cdot s\_\theta] = \nabla\_\theta D\_{\mathrm{KL}}(p \| q)$</td>
      <td style="text-align: center;"><strong>Gradient of forward KL</strong></td>
    </tr>
  </tbody>
</table>
</div>

**Key takeaways**:
- The **gradient of $k\_2$** matches the true gradient of **reverse KL** – this is the correct choice if your goal is to constrain the policy not to drift from the reference.
- The **gradient of $k\_3$** matches the true gradient of **forward KL** – this corresponds to a coverage-style objective.
- The **expected gradient of $k\_1$ is always zero** – using $k\_1$ directly as a loss is pointless.

#### “Expectation-then-gradient” vs. “Gradient-then-expectation”

If, analytically, you first compute $\mathbb{E}\_q[k\_i]$ and then differentiate (i.e., **expectation-then-gradient**), you obtain

$$
\nabla_\theta \mathbb{E}_q[k_1] = \nabla_\theta D_{\mathrm{KL}}(q \| p)
$$

and

$$
\nabla_\theta \mathbb{E}_q[k_3] = \nabla_\theta D_{\mathrm{KL}}(q \| p).
$$

Both give the gradient of reverse KL. But when you implement $k\_3$ as a **sample-wise loss in code** and call `backward` on the batch mean, autodiff is effectively computing $\mathbb{E}\_q[\nabla\_\theta k\_3]$, which, as shown above, is actually the gradient of **forward KL**.

This subtle difference is crucial: **for the same estimator, changing the order of expectation and gradient can lead to completely different optimization objectives**.


### Extension: KL Gradient Estimation with Off-Policy Sampling

The previous analysis assumed **samples come from the current policy $q\_\theta$** (on-policy). However, in practical RL training, we often encounter off-policy scenarios:

- Using old policies or mixed policies to generate data, then updating the current actor $q\_\theta$.
- In offline RL / experience replay, the sample distribution is fixed to $\mu$, not the current $q\_\theta$.

In this case, if we still want to optimize the **reverse KL** $D\_{\mathrm{KL}}(q\_\theta \| p)$, we must introduce **importance weights**.

For a deeper analysis of off-policy scenarios in large models, you can refer to my previous blog post: [From Two Policies to Three: Extending TRPO under Behavior–Reference Policy Mismatch in LLM RL](/reinforcement-learning/2025/11/15/three-policy-en.html).

#### Setup and Notation

Continuing with the previous notation, we now add the sampling distribution $\mu(x)$ and define the **importance weight**:

$$
w(x) = \frac{q_\theta(x)}{\mu(x)}
$$

When sampling from $x \sim \mu$, we use the batch mean of $w(x) k\_i(x)$ as the loss and then call autodiff. What gradients do the three estimators provide?

A key difference is:

> **Previously**, the expectation was $\mathbb{E}\_{q\_{\theta}}[\cdot]$, where the distribution itself depended on $\theta$.
> **Now**, the expectation is $\mathbb{E}\_{\mu}[\cdot]$, and $\mu$ is independent of $\theta$.

This fundamentally changes the relationship between "expectation-then-gradient" and "gradient-then-expectation".

#### Key Observation: Equivalence of the Two Orders

Since $\mu$ is independent of $\theta$, for any function $f\_\theta(x)$ differentiable with respect to $\theta$, we have

$$
\nabla_\theta \mathbb{E}_{\mu}[f_\theta(x)] = \mathbb{E}_{\mu}[\nabla_\theta f_\theta(x)]
$$

In other words, **backpropagating through the sample mean in code (gradient-then-expectation) is equivalent to differentiating the analytical form (expectation-then-gradient)**. It no longer splits into two different results as in the on-policy case.

**Therefore, in the off-policy + importance weighting case, for estimators $k\_1$ and $k\_3$ that are unbiased for the reverse KL value, their expected gradients will both correspond to the true gradient of the reverse KL.**

This is a fundamental difference from the on-policy case.

#### Numerical Level: Unbiasedness Holds

From the standard importance sampling relation $\mathbb{E}\_\mu[w \cdot f] = \mathbb{E}\_{q\_\theta}[f]$, we have

$$
\mathbb{E}_\mu[w k_1] = D_{\mathrm{KL}}(q_\theta \| p), \quad
\mathbb{E}_\mu[w k_3] = D_{\mathrm{KL}}(q_\theta \| p) \quad \textbf{(Unbiased)}
$$

$$
\mathbb{E}_\mu[w k_2] = \mathbb{E}_{q_\theta}[k_2] \neq D_{\mathrm{KL}}(q_\theta \| p) \quad \textbf{(Biased)}
$$

This is exactly consistent with the on-policy case.

#### Gradient Derivation

First, compute the gradient of the importance weight. Since $w = q\_\theta / \mu$ and $\mu$ does not depend on $\theta$:

$$
\nabla_\theta w(x) = w(x) s_\theta(x)
$$

Combining with the previously derived $\nabla\_\theta k\_i$ and using the product rule:

**$\nabla_\theta(w k_1)$**:

$$
\nabla_\theta(w k_1) = (\nabla_\theta w) k_1 + w (\nabla_\theta k_1) = w s_\theta k_1 + w s_\theta = w s_\theta (k_1 + 1)
$$

**$\nabla_\theta(w k_2)$**:

$$
\nabla_\theta(w k_2) = w s_\theta k_2 + w (-\log r) s_\theta = w s_\theta (k_2 - \log r)
$$

**$\nabla_\theta(w k_3)$**:

$$
\nabla_\theta(w k_3) = w s_\theta k_3 + w (1-r) s_\theta = w s_\theta (k_3 + 1 - r)
$$

Substituting $k\_3 = r - 1 - \log r$:

$$
k_3 + 1 - r = (r - 1 - \log r) + 1 - r = -\log r = k_1
$$

Thus, we have a beautiful simplification:

$$
\boxed{\nabla_\theta(w k_3) = w s_\theta k_1 = -w s_\theta \log r}
$$

#### Which Ones Give the Unbiased Reverse KL Gradient?

Using $\mathbb{E}\_\mu[w \cdot f] = \mathbb{E}\_{q\_\theta}[f]$ and $\mathbb{E}\_{q\_\theta}[s\_\theta] = 0$:

**$\mathbb{E}_\mu[\nabla_\theta(w k_1)]$**:

$$
\mathbb{E}_\mu[w s_\theta (k_1 + 1)] = \mathbb{E}_{q}[s_\theta k_1] + \underbrace{\mathbb{E}_{q}[s_\theta]}_{=0} = \nabla_\theta D_{\mathrm{KL}}(q_\theta \| p) \quad \checkmark
$$

**$\mathbb{E}_\mu[\nabla_\theta(w k_2)]$**:

$$
\mathbb{E}_\mu[w s_\theta (k_2 - \log r)] = \mathbb{E}_{q}[s_\theta (k_2 - \log r)] = \nabla_\theta \mathbb{E}_{q}[k_2]
$$

This is the true gradient of the f-divergence $\mathbb{E}\_q[k\_2]$, **not** the gradient of reverse KL.

**$\mathbb{E}\_\mu[\nabla\_\theta(\bar{w} k\_2)]$** (where $\bar{w} = \text{sg}(w)$ denotes detached weights):

If we treat the importance weight as a constant (detach it in code), then:

$$
\nabla_\theta(\bar{w} k_2) = \bar{w} \cdot \nabla_\theta k_2 = \bar{w} \cdot (-\log r) s_\theta
$$

Taking expectation:

$$
\mathbb{E}_\mu[\bar{w} \cdot (-\log r) s_\theta] = \mathbb{E}_{q}[(-\log r) s_\theta] = \nabla_\theta D_{\mathrm{KL}}(q_\theta \| p) \quad \checkmark
$$

This is exactly the true gradient of reverse KL!

**$\mathbb{E}_\mu[\nabla_\theta(w k_3)]$**:

$$
\mathbb{E}_\mu[w s_\theta k_1] = \mathbb{E}_{q}[s_\theta k_1] = \nabla_\theta D_{\mathrm{KL}}(q_\theta \| p) \quad \checkmark
$$

**Summary Table**:

<div class="table-responsive" markdown="0">
<table class="table table-bordered" style="font-size: 0.95em;">
  <thead>
    <tr style="background-color: var(--global-bg-color);">
      <th style="text-align: center;">Weighted Estimator</th>
      <th style="text-align: center;">Expectation Target</th>
      <th style="text-align: center;">Expected Gradient Target</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="text-align: center;">$\frac{q\_\theta}{\mu} k\_1$</td>
      <td style="text-align: center;">$D\_{\mathrm{KL}}(q\_\theta \| p)$</td>
      <td style="text-align: center;">$\nabla\_\theta D\_{\mathrm{KL}}(q\_\theta \| p)$ (Reverse KL) ✓</td>
    </tr>
    <tr>
      <td style="text-align: center;">$\frac{q\_\theta}{\mu} k\_2$</td>
      <td style="text-align: center;">$\mathbb{E}\_q[k\_2]$ (f-divergence)</td>
      <td style="text-align: center;">$\nabla\_\theta \mathbb{E}\_q[k\_2]$, <strong>NOT</strong> Reverse KL ✗</td>
    </tr>
    <tr>
      <td style="text-align: center;">$\text{sg}\left(\frac{q\_\theta}{\mu}\right) k\_2$</td>
      <td style="text-align: center;">$\mathbb{E}\_q[k\_2]$ (f-divergence)</td>
      <td style="text-align: center;">$\nabla\_\theta D\_{\mathrm{KL}}(q\_\theta \| p)$ (Reverse KL) ✓</td>
    </tr>
    <tr>
      <td style="text-align: center;">$\frac{q\_\theta}{\mu} k\_3$</td>
      <td style="text-align: center;">$D\_{\mathrm{KL}}(q\_\theta \| p)$</td>
      <td style="text-align: center;">$\nabla\_\theta D\_{\mathrm{KL}}(q\_\theta \| p)$ (Reverse KL) ✓</td>
    </tr>
  </tbody>
</table>
</div>

**Comparison with On-Policy Case — An Interesting Reversal**:

- In on-policy, the gradient of $k\_2$ as a loss is the reverse KL, while the expected gradient of $k\_1$ is zero.
- In off-policy + importance weighting, $\frac{q\_\theta}{\mu} k\_1$ and $\frac{q\_\theta}{\mu} k\_3$ give the true gradient of reverse KL, while $\frac{q\_\theta}{\mu} k\_2$ (with weights in the gradient) **is no longer applicable**.
- However, if we **detach** the importance weights, $\text{sg}\left(\frac{q\_\theta}{\mu}\right) k\_2$ also gives the true gradient of reverse KL.

#### Do the Three Unbiased Gradient Estimators Differ in Variance?

In the off-policy + importance sampling setting, **three losses give an unbiased gradient of reverse KL**:

$$
L_1(x) = w(x) k_1(x), \qquad
L_2(x) = \bar{w}(x) k_2(x), \qquad
L_3(x) = w(x) k_3(x),
$$

where $w = \dfrac{q\_\theta}{\mu}$ and $\bar{w} = \mathrm{sg}(w)$ denotes a detached weight. Their gradient random variables are

$$
g_1(x) := \nabla_\theta L_1(x), \qquad
g_2(x) := \nabla_\theta L_2(x), \qquad
g_3(x) := \nabla_\theta L_3(x).
$$

Using previously derived results ($\nabla\_\theta w = w s\_\theta$, $\nabla\_\theta k\_1 = s\_\theta$, $\nabla\_\theta k\_2 = - (\log r) s\_\theta = k\_1 s\_\theta$, $\nabla\_\theta k\_3 = (1 - r) s\_\theta$), we get

$$
g_1(x) = w(x) s_\theta(x)\big(k_1(x)+1\big),
$$

$$
g_2(x) = \bar{w}(x)\, k_1(x) s_\theta(x) = w(x) s_\theta(x) k_1(x),
$$

$$
g_3(x) = w(x) s_\theta(x)\big(k_3(x) + 1 - r(x)\big) = w(x) s_\theta(x) k_1(x).
$$

So in gradient space **$g\_2(x) \equiv g\_3(x)$ exactly** (same mean, same variance, same higher moments). Both share the correct expectation $\nabla\_\theta D\_{\mathrm{KL}}(q\_\theta \| p)$. Compared with $g\_1$, the only difference is a zero-mean extra term $w s\_\theta$:

$$
g\_1(x) - g\_3(x) = w(x) s\_\theta(x), \qquad \mathbb{E}\_\mu[w s\_\theta] = \mathbb{E}\_{q\_\theta}[s\_\theta] = 0.
$$

Define $A(x) := w(x) s\_\theta(x)$ and $B(x) := k\_1(x)$. Then $g\_1 = A(B+1)$ and $g\_\star := g\_2 = g\_3 = A B$. Their variance difference is

$$
\mathrm{Var}_\mu(g_1) - \mathrm{Var}_\mu(g_\star) = \mathbb{E}_\mu\big[A^2 \big((B+1)^2 - B^2\big)\big] = \mathbb{E}_\mu\big[A^2 (2B + 1)\big]
$$

or explicitly

$$
\mathrm{Var}\_\mu(g\_1) - \mathrm{Var}\_\mu(g\_\star) = \mathbb{E}\_\mu\Big[w(x)^2 s\_\theta(x)^2 \big(2 k\_1(x) + 1\big)\Big].
$$

In the typical KL-penalty regime $q\_\theta \approx p \approx \mu$, let $r(x) = 1 + \varepsilon(x)$ with $\lvert\varepsilon\rvert \ll 1$. Then $k\_1 = -\log r \approx -\varepsilon$, so $2k\_1 + 1 \approx 1 - 2\varepsilon$, with the leading term being a positive $O(1)$ constant. This means the right-hand side is approximately $\mathbb{E}\_\mu[w^2 s\_\theta^2] > 0$, and therefore $\mathrm{Var}\_\mu(g\_1) > \mathrm{Var}\_\mu(g\_\star)$.

More specifically, with first-order approximation $k\_1 \approx -\varepsilon$ and $k\_1 + 1 \approx 1 - \varepsilon$, we have

$$
g_1(x) \approx w(x) s_\theta(x)\big(1 - \varepsilon(x)\big), \qquad
g_\star(x) \approx w(x) s_\theta(x)\big(-\varepsilon(x)\big).
$$

Intuition: $g\_1$ keeps an $O(1)$ zero-mean noise term $w s\_\theta$, while $g\_\star$ is an $O(\varepsilon)$ term. When policies are close, $\lvert\varepsilon\rvert$ is small, so $g\_\star$ (i.e., $\bar{w} k\_2$ or $w k\_3$) has much lower variance than $g\_1$ ($w k\_1$).

Summary table:

<div class="table-responsive" markdown="0">
<table class="table table-bordered" style="font-size: 0.95em;">
  <thead>
    <tr style="background-color: var(--global-bg-color);">
      <th style="text-align: center; white-space: nowrap;">Estimator</th>
      <th style="text-align: center; white-space: nowrap;">Gradient RV</th>
      <th style="text-align: center; white-space: nowrap;">Scale ($r\approx1$)</th>
      <th style="text-align: center;">Variance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="text-align: center;">$w k\_1$</td>
      <td style="text-align: center;">$w s\_\theta (k\_1+1)$</td>
      <td style="text-align: center;">$O(1)$</td>
      <td style="text-align: center;">High</td>
    </tr>
    <tr>
      <td style="text-align: center;">$\mathrm{sg}(w) k\_2$</td>
      <td style="text-align: center;">$w s\_\theta k\_1$</td>
      <td style="text-align: center;">$O(\varepsilon)$</td>
      <td style="text-align: center;">Low</td>
    </tr>
    <tr>
      <td style="text-align: center;">$w k\_3$</td>
      <td style="text-align: center;">$w s\_\theta k\_1$</td>
      <td style="text-align: center;">$O(\varepsilon)$</td>
      <td style="text-align: center;">Low</td>
    </tr>
  </tbody>
</table>
</div>

**Warning for Extreme Off-Policy**:

When $\mu$ differs greatly from $q\_\theta$ — for example, $\mu$ has almost no samples in the high-density regions of $q\_\theta$, or $w = q\_\theta / \mu$ explodes in the tails — any method based on $\frac{q\_\theta}{\mu}$ will suffer from severe variance problems. In this case, the advantage of $w k\_3$ (or $\mathrm{sg}(w) k\_2$) over $w k\_1$ is no longer theoretically guaranteed, and strategies like clipping or regularization are needed.

However, in RL practice, we usually control the KL constraint and limit the degree of off-policy (e.g., using a proximal policy $\mu = q\_{\theta\_\text{old}}$). In this common regime, we can say with considerable confidence:

> **If you have decided to use off-policy + importance sampling to optimize reverse KL, prefer $w k\_3$ or $\mathrm{sg}(w) k\_2$; $w k\_1$ is unbiased but noisier.**

This is why the DeepSeek v3.2 technical report uses $\frac{q\_\theta}{\mu} k\_3$ as the estimator for off-policy KL penalty.

<figure style="text-align:center;">
  <img src="/assets/img/kl-estimators/dpsk-3d2-k3.png" style="width:95%;max-width:100%;">
  <figcaption style="font-size:0.9em;color:gray;">Image source: <a href="https://arxiv.org/pdf/2512.02556v1">DeepSeek v3.2 Technical Report Section 3.1</a></figcaption>
</figure>

#### Summary

- When sampling from a behavior policy $\mu$, the natural off-policy KL estimator is $\frac{q\_\theta}{\mu} k\_i$.
- **Numerically**, $\frac{q\_\theta}{\mu} k\_1$ and $\frac{q\_\theta}{\mu} k\_3$ remain unbiased for reverse KL; $\frac{q\_\theta}{\mu} k\_2$ is biased.
- **Gradient-wise**, because $\mu$ is independent of $\theta$:
  - $\mathbb{E}\_\mu[\nabla\_\theta(\frac{q\_\theta}{\mu} k\_1)] = \nabla\_\theta D\_{\mathrm{KL}}(q\_\theta \| p)$
  - $\mathbb{E}\_\mu[\nabla\_\theta(\mathrm{sg}(\frac{q\_\theta}{\mu}) k\_2)] = \nabla\_\theta D\_{\mathrm{KL}}(q\_\theta \| p)$
  - $\mathbb{E}\_\mu[\nabla\_\theta(\frac{q\_\theta}{\mu} k\_3)] = \nabla\_\theta D\_{\mathrm{KL}}(q\_\theta \| p)$
- **Variance-wise**, $\mathrm{sg}(w) k\_2$ and $w k\_3$ have identical gradients and lower variance; $w k\_1$ is unbiased but noisier unless clipped.

### Gradient Estimation Overview

The following table summarizes the expected gradients and corresponding optimization objectives for each estimator in both on-policy and off-policy scenarios:

<div class="table-responsive" markdown="0">
<table class="table table-bordered" style="font-size: 0.95em;">
  <thead>
    <tr style="background-color: var(--global-bg-color);">
      <th style="text-align: center; white-space: nowrap;">Sampling Source</th>
      <th style="text-align: center;">Loss</th>
      <th style="text-align: center;">Expected $\nabla\_\theta$ Loss</th>
      <th style="text-align: center;">Corresponding Objective</th>
      <th style="text-align: center; white-space: nowrap;">Can Optimize Reverse KL?</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="text-align: center;">$q$ (on)</td>
      <td style="text-align: center;">$k\_1$</td>
      <td style="text-align: center;">$\mathbb{E}\_q[s\_\theta] = 0$</td>
      <td style="text-align: center;">None (Gradient is zero)</td>
      <td style="text-align: center;">✗</td>
    </tr>
    <tr>
      <td style="text-align: center;">$q$ (on)</td>
      <td style="text-align: center;">$k\_2$</td>
      <td style="text-align: center;">$\nabla\_\theta D\_{\mathrm{KL}}(q \| p)$</td>
      <td style="text-align: center;"><strong>Reverse KL</strong></td>
      <td style="text-align: center;">✓</td>
    </tr>
    <tr>
      <td style="text-align: center;">$q$ (on)</td>
      <td style="text-align: center;">$k\_3$</td>
      <td style="text-align: center;">$\nabla\_\theta D\_{\mathrm{KL}}(p \| q)$</td>
      <td style="text-align: center;">Forward KL</td>
      <td style="text-align: center;">✗</td>
    </tr>
    <tr>
      <td style="text-align: center;">$\mu$ (off)</td>
      <td style="text-align: center;">$\frac{q}{\mu} k\_1$</td>
      <td style="text-align: center;">$\nabla\_\theta D\_{\mathrm{KL}}(q \| p)$</td>
      <td style="text-align: center;"><strong>Reverse KL</strong></td>
      <td style="text-align: center;">✓ (High Variance)</td>
    </tr>
    <tr>
      <td style="text-align: center;">$\mu$ (off)</td>
      <td style="text-align: center;">$\frac{q}{\mu} k\_2$</td>
      <td style="text-align: center;">$\nabla\_\theta \mathbb{E}\_q[k\_2]$</td>
      <td style="text-align: center;">f-divergence (Not KL)</td>
      <td style="text-align: center;">✗</td>
    </tr>
    <tr>
      <td style="text-align: center;">$\mu$ (off)</td>
      <td style="text-align: center;">$\text{sg}\left(\frac{q}{\mu}\right) k\_2$</td>
      <td style="text-align: center;">$\nabla\_\theta D\_{\mathrm{KL}}(q \| p)$</td>
      <td style="text-align: center;"><strong>Reverse KL</strong></td>
      <td style="text-align: center;">✓</td>
    </tr>
    <tr>
      <td style="text-align: center;">$\mu$ (off)</td>
      <td style="text-align: center;">$\frac{q}{\mu} k\_3$</td>
      <td style="text-align: center;">$\nabla\_\theta D\_{\mathrm{KL}}(q \| p)$</td>
      <td style="text-align: center;"><strong>Reverse KL</strong></td>
      <td style="text-align: center;">✓ (Recommended, Low Var)</td>
    </tr>
  </tbody>
</table>
</div>

**Key Conclusions**:

1. **On-policy optimizing Reverse KL**: The only correct choice is $k\_2$.
2. **Off-policy optimizing Reverse KL**: Three correct options:
   - $\frac{q}{\mu} k\_1$: Unbiased but high variance
   - $\text{sg}\left(\frac{q}{\mu}\right) k\_2$: Unbiased, similar behavior to on-policy $k\_2$
   - $\frac{q}{\mu} k\_3$: Unbiased and lower variance (recommended)
3. **$\frac{q}{\mu} k\_2$ (weights in gradient) fails in off-policy**: This is a trap that is easily overlooked.


## Practical Guidelines for RL

### KL as a Reward Penalty (No Gradient Needed)

When KL is only used as a scalar penalty in reward shaping, we only care about an accurate **value estimate**, and we do not backpropagate through it.

**Recommendations**:
- Use **$k\_1$** or **$k\_3$** (both are unbiased for the reverse KL value).
- When the policy is already close to the reference, $k\_3$ often has lower variance.
- When coverage is poor or there is severe tail mismatch, $k\_1$ can be more robust.
- In off-policy settings, simply add the importance weight $\frac{q\_\theta}{\mu}$.

> **Note**: If you want a **forward KL penalty** (to encourage coverage of the behavior distribution), you can use $\mathbb{E}\_q[r \log r]$ or, if you can sample from $p$, directly use $\mathbb{E}\_p[\log r]$.

### KL as a Loss (Gradient Required)

When KL is part of the loss that you differentiate, you must ensure that the gradient matches your intended objective.

#### On-policy: Optimizing Reverse KL (Most Common Case)

Goal: constrain the actor not to drift far from the reference policy.

**Correct choice**: use **$k\_2$** as the loss.

$$
\mathcal{L}\_{k\_2} = \frac{1}{2}(\log r)^2.
$$

Its gradient expectation $\mathbb{E}\_q[\nabla k\_2] = \nabla\_\theta D\_{\mathrm{KL}}(q \| p)$ is exactly the true gradient of reverse KL.

#### On-policy: Optimizing Forward KL (Coverage-Oriented Settings)

Goal: make the policy cover the support of the reference distribution (e.g., in offline RL or imitation learning).

**Correct choice**: use **$k\_3$** as the loss.

$$
\mathbb{E}\_q[\nabla k\_3] = \mathbb{E}\_q[(1 - r) s\_\theta] = \nabla\_\theta D\_{\mathrm{KL}}(p \| q).
$$

If you backpropagate through the batch mean of $k\_3$, autodiff computes exactly this forward-KL gradient – no extra tricks needed.

#### Off-policy: Optimizing Reverse KL

Goal: Data comes from behavior policy $\mu$, but we still want to optimize reverse KL.

**Recommended**: use **$\frac{q\_\theta}{\mu} k\_3$** as the loss.

$$
\mathcal{L} = \frac{q\_\theta(x)}{\mu(x)} \cdot \left(\frac{p(x)}{q\_\theta(x)} - 1 - \log \frac{p(x)}{q\_\theta(x)}\right)
$$

- Unbiased gradient.
- Significantly lower variance when $q\_\theta \approx p$.

**Alternative 1**: Use $\text{sg}\left(\frac{q\_\theta}{\mu}\right) k\_2$ (detach the importance weights).

$$
\mathcal{L} = \text{sg}\left(\frac{q\_\theta(x)}{\mu(x)}\right) \cdot \frac{1}{2}\left(\log \frac{p(x)}{q\_\theta(x)}\right)^2
$$

This way, the gradient becomes $\bar{w} \cdot (-\log r) s\_\theta$, whose expectation is still the true gradient of reverse KL. This approach is similar in form to on-policy $k\_2$, just with an additional importance weight that does not participate in the gradient.

**Alternative 2**: Use $\frac{q\_\theta}{\mu} k\_1$ (gradient is also unbiased, but variance is higher).

**Avoid**: Using $\frac{q\_\theta}{\mu} k\_2$ (with weights in gradient) — the gradient is biased, not the correct direction for reverse KL.


## A Ready-to-Use Cheat Sheet

<div class="table-responsive" markdown="0">
<table class="table table-bordered" style="font-size: 0.95em;">
  <thead>
    <tr style="background-color: var(--global-bg-color);">
      <th style="text-align: center;">Objective</th>
      <th style="text-align: center;">Sampling Source</th>
      <th style="text-align: center;">For <strong>Value Estimate</strong></th>
      <th style="text-align: center;">For <strong>Gradient (Loss)</strong></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="text-align: center;">Reverse KL $D_{\mathrm{KL}}(q \| p)$</td>
      <td style="text-align: center;">$q$ (on-policy)</td>
      <td style="text-align: center;">$k\_1$ or $k\_3$ (unbiased)</td>
      <td style="text-align: center;">$k\_2$</td>
    </tr>
    <tr>
      <td style="text-align: center;">Reverse KL $D_{\mathrm{KL}}(q \| p)$</td>
      <td style="text-align: center;">$\mu$ (off-policy)</td>
      <td style="text-align: center;">$\frac{q}{\mu} k\_1$ or $\frac{q}{\mu} k\_3$ (unbiased)</td>
      <td style="text-align: center;">$\frac{q}{\mu} k\_3$ (recommended) or $\text{sg}\left(\frac{q}{\mu}\right) k\_2$</td>
    </tr>
    <tr>
      <td style="text-align: center;">Forward KL $D_{\mathrm{KL}}(p \| q)$</td>
      <td style="text-align: center;">$q$</td>
      <td style="text-align: center;">$\mathbb{E}\_q[r\log r]$</td>
      <td style="text-align: center;">$k\_3$</td>
    </tr>
  </tbody>
</table>
</div>


## Common Implementation Pitfalls

**Pitfall 1: Using $k\_1$ Directly as a Loss (On-Policy)**

The expected gradient of $k\_1$ is zero ($\mathbb{E}\_q[\nabla k\_1] = \mathbb{E}\_q[s\_\theta] = 0$), so as a loss it is ineffective.

> **Fix**: Use $k\_1$ or $k\_3$ only when you need a scalar KL penalty in rewards (no gradient), and use $k\_2$ or $k\_3$ when you actually want a loss with a meaningful gradient.

**Pitfall 2: Confusing $k\_3$'s Unbiased Value with Its Gradient Objective**

$k\_3$ is an **unbiased value estimator of the reverse KL**, but its **gradient** corresponds to the **forward KL**. If your goal is to optimize reverse KL but you use $k\_3$ as a loss, you are in fact optimizing forward KL.

> **Fix**: Be explicit about your objective. Use $k\_2$ when optimizing reverse KL; use $k\_3$ only when you intentionally optimize forward KL.

**Pitfall 3: Heavy-Tailed $r$ Causing Variance Explosion**

When the policy and reference distribution are very different, $r = p/q$ can have extreme values, causing the variance of $k\_3$ (and importance-sampling-based estimators) to blow up.

> **Fix**: Control the KL constraint or clip $r$.

**Pitfall 4: Using $k\_2$ or $\frac{q\_\theta}{\mu} k\_2$ (with weights in gradient) in Off-Policy Settings**

In on-policy settings, $k\_2$ is the correct choice for optimizing reverse KL. However, if data comes from $\mu \neq q\_\theta$:
- Using $k\_2$ directly (unweighted): The expectation is not over $q\_\theta$, so the estimator fails.
- Using $\frac{q\_\theta}{\mu} k\_2$ (with weights in gradient): The gradient is biased and does not point to the reverse KL direction.

> **Fix**: In off-policy scenarios, switch to $\frac{q\_\theta}{\mu} k\_3$ (recommended), $\text{sg}\left(\frac{q\_\theta}{\mu}\right) k\_2$ (detach weights), or $\frac{q\_\theta}{\mu} k\_1$.

**Pitfall 5: Improper Handling of Importance Weight Detachment**

In implementation, $w = q\_\theta / \mu$ is usually computed via `exp(log_prob_q - log_prob_mu)`. Whether to detach $w$ leads to completely different results:

- **When using $k\_1$ or $k\_3$**: $w$ **should participate in gradient computation** (do not detach), otherwise you lose the $\nabla\_\theta w = w s\_\theta$ term, leading to incorrect gradients.
- **When using $k\_2$**: $w$ **should be detached**, so that you get the true gradient of reverse KL. If $w$ participates in gradient computation, you get the gradient of an f-divergence, not reverse KL.

> **Summary**: When choosing different estimators, make sure to match the correct detach strategy.


## Conclusion

**One-line summary**:

- **KL for value only (reward penalty)**: use $k\_1$ or $k\_3$ (both are unbiased for reverse KL); add importance weights if off-policy.
- **KL as a differentiable loss (needs gradients)**:
	- **On-policy**: To optimize **reverse KL**, use $k\_2$; to optimize **forward KL**, use $k\_3$.
	- **Off-policy**: To optimize **reverse KL**, use $\frac{q\_\theta}{\mu} k\_3$ (recommended, unbiased + low variance) or $\text{sg}\left(\frac{q\_\theta}{\mu}\right) k\_2$ (detach weights).

Once you keep clear **who you sample from**, **which KL you estimate**, and **with respect to which quantity you differentiate**, the three estimators become much less confusing. Especially note: **the correct choice for optimizing reverse KL differs between on-policy ($k\_2$) and off-policy ($\frac{q\_\theta}{\mu} k\_3$ or $\text{sg}\left(\frac{q\_\theta}{\mu}\right) k\_2$)**.


## References

1. Dibya Ghosh. "KL Divergence for Machine Learning". https://dibyaghosh.com/blog/probability/kldivergence
2. John Schulman. "Approximating KL Divergence". https://joschu.net/blog/kl-approx.html
3. Verl Documentation. "Proximal Policy Optimization (PPO)". https://verl.readthedocs.io/en/latest/algo/ppo.html
4. 初七123334. "Approximate KL in RLHF/RLVR Training: A Brief Analysis of k1 / k2 / k3" (in Chinese). https://zhuanlan.zhihu.com/p/1966872846212010437
5. Kezhao Liu, Jason Klein Liu, Mingtao Chen, Yiming Liu. "Rethinking KL Regularization in RLHF: From Value Estimation to Gradient Optimization". https://arxiv.org/abs/2510.01555
6. Yifan Zhang, Yiping Ji, Gavin Brown, et al. "On the Design of KL-Regularized Policy Gradient Algorithms for LLM Reasoning". https://arxiv.org/abs/2505.17508

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


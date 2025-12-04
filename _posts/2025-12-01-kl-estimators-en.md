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

> In reinforcement learning, how we approximate KL divergence directly affects training stability. This post systematically analyzes the differences between three classic estimators $k_1, k_2, k_3$ in both on-policy and off-policy scenarios, and provides practical guidelines for choosing them when KL is used as a reward penalty versus when it is used as a loss for backpropagation.

[中文版](/reinforcement-learning/2025/12/01/kl-estimators-cn.html) \| [知乎版本 ![Zhihu](https://static.zhihu.com/heifetz/favicon.ico)](https://zhuanlan.zhihu.com/p/1978993413425763764)

## Introduction: The Role of KL Divergence in Reinforcement Learning

In policy optimization (PPO, GRPO, etc.) or alignment training (RLHF/RLAIF), **KL regularization** is the core mechanism that constrains the new policy from drifting too far away from a reference policy, in order to prevent unstable training or policy collapse.

### Forward vs. Reverse KL

Let $q_\theta$ be the current actor policy and $p$ be the reference policy. The two directions of KL divergence are

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
D_f(p, q_\theta) = \frac{f^{\prime\prime}(1)}{2} \theta^T F \theta + O(\theta^3)
$$

where $F$ is the Fisher information matrix. KL divergence corresponds to $f(x) = -\log x$, with $f^{\prime\prime}(1) = 1$; $k_2$ corresponds to $f(x) = \frac{1}{2}(\log x)^2$, which also satisfies $f^{\prime\prime}(1) = 1$. This means that **when the policies are close, $k_2$ behaves almost identically to the true KL**, and the bias only appears in higher-order terms.

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
\begin{aligned}
D_\phi(r, 1) &= \phi(r) - \phi(1) - \phi'(1)(r - 1) \\
&= -\log r - 0 - (-1)(r - 1) \\
&= r - 1 - \log r \\
&= k_3.
\end{aligned}
$$

Since a convex function always lies above its tangents, this difference is **naturally non-negative**. More importantly, as $r \to 1$, the function and its tangent “stick together” more tightly, and the gap shrinks at the rate of $(r-1)^2$. This is exactly why $k_3$ has small variance when the policies are close.


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
      <td style="text-align: center;">$k_1$</td>
      <td style="text-align: center;">$-\log r$</td>
      <td style="text-align: center;">Naive definition</td>
      <td style="text-align: center;">Unbiased</td>
      <td style="text-align: center;">High (can be neg.)</td>
    </tr>
    <tr>
      <td style="text-align: center;">$k_2$</td>
      <td style="text-align: center;">$\frac{1}{2}(\log r)^2$</td>
      <td style="text-align: center;">f-divergence, 2nd-order matches KL</td>
      <td style="text-align: center;">Biased (small)</td>
      <td style="text-align: center;">Low (always pos.)</td>
    </tr>
    <tr>
      <td style="text-align: center;">$k_3$</td>
      <td style="text-align: center;">$r - 1 - \log r$</td>
      <td style="text-align: center;">Control variate + Bregman divergence</td>
      <td style="text-align: center;">Unbiased</td>
      <td style="text-align: center;">Low (always pos.)</td>
    </tr>
  </tbody>
</table>
</div>

From a pure **value estimation** perspective, $k_3$ looks like the "best of both worlds": **unbiased + low variance**. However, as we will see, the **story is completely different at the gradient level**.


## Core Analysis

### Bias and Variance for Estimating the KL Value

Assume we sample from $q_\theta$ to estimate the reverse KL $D_{\mathrm{KL}}(q_\theta \| p)$.

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

**Conclusion**: For estimating the **value** of the reverse KL, $k_1$ and $k_3$ are unbiased, whereas $k_2$ is biased.

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
      <td style="text-align: center;">$k_1$</td>
      <td style="text-align: center;">0</td>
      <td style="text-align: center;">20</td>
    </tr>
    <tr>
      <td style="text-align: center;">$k_2$</td>
      <td style="text-align: center;">0.002</td>
      <td style="text-align: center;">1.42</td>
    </tr>
    <tr>
      <td style="text-align: center;">$k_3$</td>
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
      <td style="text-align: center;">$k_1$</td>
      <td style="text-align: center;">0</td>
      <td style="text-align: center;">2</td>
    </tr>
    <tr>
      <td style="text-align: center;">$k_2$</td>
      <td style="text-align: center;">0.25</td>
      <td style="text-align: center;">1.73</td>
    </tr>
    <tr>
      <td style="text-align: center;">$k_3$</td>
      <td style="text-align: center;">0</td>
      <td style="text-align: center;">1.7</td>
    </tr>
  </tbody>
</table>
</div>

**Intuition**:
- $k_1 = -\log r$ starts with a first-order term. When $r$ is close to 1, its fluctuations are large and it can be negative.
- $k_3 = r - 1 - \log r$ is second-order around $r = 1$ and always non-negative, so it has smaller variance when policies are close.
- When coverage is very poor (i.e., $r$ can explode), the variance of $k_3$ can blow up due to the heavy tails of $r$; in that regime, $k_1$ can be more stable.

> **Note**: To estimate the **forward KL value** $D_{\mathrm{KL}}(p \| q) = \mathbb{E}_p[\log r]$ using samples from $q$, you can use importance sampling $\mathbb{E}_q[r \log r]$.


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

Using $\mathbb{E}\_q[s\_\theta] = 0$, we can rewrite this as

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
\begin{aligned}
\nabla_\theta k_2 
&= (\log r) \cdot \nabla_\theta(\log r) \\
&= (\log r) \cdot \nabla_\theta(\log p(x) - \log q_\theta(x)) \\
&= (\log r)(-s_\theta) \\
&= - (\log r) s_\theta.
\end{aligned}
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

<div class="table-responsive" markdown="0">
<table class="table table-bordered" style="font-size: 0.95em;">
  <thead>
    <tr style="background-color: var(--global-bg-color);">
      <th style="text-align: center;">Estimator</th>
      <th style="text-align: center;">$\mathbb{E}_{q}[\nabla_\theta k_i]$</th>
      <th style="text-align: center;">Equals</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="text-align: center;">$k_1$</td>
      <td style="text-align: center;">$\mathbb{E}_{q}[s_\theta] = 0$</td>
      <td style="text-align: center;"><strong>Zero (useless as loss)</strong></td>
    </tr>
    <tr>
      <td style="text-align: center;">$k_2$</td>
      <td style="text-align: center;">$-\mathbb{E}_{q}[(\log r) \cdot s_\theta] = \nabla_\theta D_{\mathrm{KL}}(q \| p)$</td>
      <td style="text-align: center;"><strong>Gradient of reverse KL</strong></td>
    </tr>
    <tr>
      <td style="text-align: center;">$k_3$</td>
      <td style="text-align: center;">$\mathbb{E}_{q}[(1-r) \cdot s_\theta] = \nabla_\theta D_{\mathrm{KL}}(p \| q)$</td>
      <td style="text-align: center;"><strong>Gradient of forward KL</strong></td>
    </tr>
  </tbody>
</table>
</div>

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

Both give the gradient of reverse KL. But when you implement $k_3$ as a **sample-wise loss in code** and call `backward` on the batch mean, autodiff is effectively computing $\mathbb{E}\_q[\nabla\_\theta k\_3]$, which, as shown above, is actually the gradient of **forward KL**.

This subtle difference is crucial: **for the same estimator, changing the order of expectation and gradient can lead to completely different optimization objectives**.


### Extension: KL Gradient Estimation with Off-Policy Sampling

The previous analysis assumed **samples come from the current policy $q_\theta$** (on-policy). However, in practical RL training, we often encounter off-policy scenarios:

- Using old policies or mixed policies to generate data, then updating the current actor $q_\theta$.
- In offline RL / experience replay, the sample distribution is fixed to $\mu$, not the current $q_\theta$.

In this case, if we still want to optimize the **reverse KL** $D_{\mathrm{KL}}(q_\theta \| p)$, we must introduce **importance weights**.

For a deeper analysis of off-policy scenarios in large models, you can refer to my previous blog post: [From Two Policies to Three: Extending TRPO under Behavior–Reference Policy Mismatch in LLM RL](/reinforcement-learning/2025/11/15/three-policy-en.html).

#### Setup and Notation

Continuing with the previous notation, we now add the sampling distribution $\mu(x)$ and define the **importance weight**:

$$
w(x) = \frac{q_\theta(x)}{\mu(x)}
$$

When sampling from $x \sim \mu$, we use the batch mean of $w(x) k_i(x)$ as the loss and then call autodiff. What gradients do the three estimators provide?

A key difference is:

> **Previously**, the expectation was $\mathbb{E}_{q_{\theta}}[\cdot]$, where the distribution itself depended on $\theta$.
> **Now**, the expectation is $\mathbb{E}_{\mu}[\cdot]$, and $\mu$ is independent of $\theta$.

This fundamentally changes the relationship between "expectation-then-gradient" and "gradient-then-expectation".

#### Key Observation: Equivalence of the Two Orders

Since $\mu$ is independent of $\theta$, for any function $f_\theta(x)$ differentiable with respect to $\theta$, we have

$$
\nabla_\theta \mathbb{E}_{\mu}[f_\theta(x)] = \mathbb{E}_{\mu}[\nabla_\theta f_\theta(x)]
$$

In other words, **backpropagating through the sample mean in code (gradient-then-expectation) is equivalent to differentiating the analytical form (expectation-then-gradient)**. It no longer splits into two different results as in the on-policy case.

**Therefore, in the off-policy + importance weighting case, for estimators $k_1$ and $k_3$ that are unbiased for the reverse KL value, their expected gradients will both correspond to the true gradient of the reverse KL.**

This is a fundamental difference from the on-policy case.

#### Numerical Level: Unbiasedness Holds

From the standard importance sampling relation $\mathbb{E}_\mu[w \cdot f] = \mathbb{E}_{q_\theta}[f]$, we have

$$
\mathbb{E}_\mu[w k_1] = D_{\mathrm{KL}}(q_\theta \| p), \quad
\mathbb{E}_\mu[w k_3] = D_{\mathrm{KL}}(q_\theta \| p) \quad \textbf{(Unbiased)}
$$

$$
\mathbb{E}_\mu[w k_2] = \mathbb{E}_{q_\theta}[k_2] \neq D_{\mathrm{KL}}(q_\theta \| p) \quad \textbf{(Biased)}
$$

This is exactly consistent with the on-policy case.

#### Gradient Derivation

First, compute the gradient of the importance weight. Since $w = q_\theta / \mu$ and $\mu$ does not depend on $\theta$:

$$
\nabla_\theta w(x) = w(x) s_\theta(x)
$$

Combining with the previously derived $\nabla_\theta k_i$ and using the product rule:

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

Substituting $k_3 = r - 1 - \log r$:

$$
k_3 + 1 - r = (r - 1 - \log r) + 1 - r = -\log r = k_1
$$

Thus, we have a beautiful simplification:

$$
\boxed{\nabla_\theta(w k_3) = w s_\theta k_1 = -w s_\theta \log r}
$$

#### Which Ones Give the Unbiased Reverse KL Gradient?

Using $\mathbb{E}_\mu[w \cdot f] = \mathbb{E}_{q_\theta}[f]$ and $\mathbb{E}_{q_\theta}[s_\theta] = 0$:

**$\mathbb{E}_\mu[\nabla_\theta(w k_1)]$**:

$$
\mathbb{E}_\mu[w s_\theta (k_1 + 1)] = \mathbb{E}_{q}[s_\theta k_1] + \underbrace{\mathbb{E}_{q}[s_\theta]}_{=0} = \nabla_\theta D_{\mathrm{KL}}(q_\theta \| p) \quad \checkmark
$$

**$\mathbb{E}_\mu[\nabla_\theta(w k_2)]$**:

$$
\mathbb{E}_\mu[w s_\theta (k_2 - \log r)] = \mathbb{E}_{q}[s_\theta (k_2 - \log r)] = \nabla_\theta \mathbb{E}_{q}[k_2]
$$

This is the true gradient of the f-divergence $\mathbb{E}_q[k_2]$, **not** the gradient of reverse KL.

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
      <td style="text-align: center;">$\frac{q_\theta}{\mu} k_1$</td>
      <td style="text-align: center;">$D_{\mathrm{KL}}(q_\theta \| p)$</td>
      <td style="text-align: center;">$\nabla_\theta D_{\mathrm{KL}}(q_\theta \| p)$ (Reverse KL) ✓</td>
    </tr>
    <tr>
      <td style="text-align: center;">$\frac{q_\theta}{\mu} k_2$</td>
      <td style="text-align: center;">$\mathbb{E}_q[k_2]$ (f-divergence)</td>
      <td style="text-align: center;">$\nabla_\theta \mathbb{E}_q[k_2]$, <strong>NOT</strong> Reverse KL ✗</td>
    </tr>
    <tr>
      <td style="text-align: center;">$\frac{q_\theta}{\mu} k_3$</td>
      <td style="text-align: center;">$D_{\mathrm{KL}}(q_\theta \| p)$</td>
      <td style="text-align: center;">$\nabla_\theta D_{\mathrm{KL}}(q_\theta \| p)$ (Reverse KL) ✓</td>
    </tr>
  </tbody>
</table>
</div>

**Comparison with On-Policy Case — An Interesting Reversal**:

- In on-policy, the gradient of $k_2$ as a loss is the reverse KL, while the expected gradient of $k_1$ is zero.
- In off-policy + importance weighting, $\frac{q_\theta}{\mu} k_1$ and $\frac{q_\theta}{\mu} k_3$ give the true gradient of reverse KL, while $\frac{q_\theta}{\mu} k_2$ **is no longer applicable**.

#### Does $\frac{q_\theta}{\mu} k_3$ Have Lower Gradient Variance?

Now we care about the gradient random variables:

$$
g_1(x) := \nabla_\theta(w k_1) = w(x) s_\theta(x) (k_1(x) + 1)
$$

$$
g_3(x) := \nabla_\theta(w k_3) = w(x) s_\theta(x) k_1(x)
$$

Both have the same expectation $\nabla_\theta D_{\mathrm{KL}}(q_\theta \| p)$, but there is a simple relationship:

$$
g_1(x) - g_3(x) = w(x) s_\theta(x)
$$

And since $\mathbb{E}_\mu[w s_\theta] = \mathbb{E}_{q_\theta}[s_\theta] = 0$, we know that $w s_\theta$ is a **zero-mean "pure noise term"**.

**First-Order Expansion Analysis**:

In the common regime of KL penalties, the three distributions are close: $q_\theta \approx p \approx \mu$. Let $r(x) = 1 + \varepsilon(x)$ with $\vert \varepsilon \vert \ll 1$, and perform a first-order expansion:

$$
k_1 = -\log r \approx -\varepsilon + O(\varepsilon^2)
$$

Substituting into the coefficients of $g_1, g_3$:

$$
k_1 + 1 \approx 1 - \varepsilon + O(\varepsilon^2), \quad k_1 \approx -\varepsilon + O(\varepsilon^2)
$$

Thus:

$$
g_1(x) \approx w(x) s_\theta(x) \cdot \big(1 - \varepsilon(x) + O(\varepsilon^2)\big)
$$

$$
g_3(x) \approx w(x) s_\theta(x) \cdot \big(-\varepsilon(x) + O(\varepsilon^2)\big)
$$

**Core Intuition**:

- $g_1$ contains a constant term $w s_\theta$ with "magnitude 1 but expectation 0". The true gradient is the **tiny difference** after this part cancels out with other terms, so the single-sample variance is large.
- $g_3$ analytically eliminates this constant term, leaving a **first-order small quantity** proportional to the bias $\varepsilon(x) = r(x) - 1$. When policies are close, $\vert \varepsilon \vert$ is small, so the fluctuation of $g_3$ is naturally significantly smaller.

This is completely consistent with the intuition in the "Value Estimation" section: $k_3$ is a second-order small quantity at $r=1$, while $k_1$ is a first-order quantity. With importance weights, this property is preserved in the gradient estimation.

> **Conclusion**: In the typical KL penalty scenario where $q_\theta \approx p \approx \mu$, the gradient variance of $\frac{q_\theta}{\mu} k_3$ is **strictly lower order** than that of $\frac{q_\theta}{\mu} k_1$, making it the "unbiased + low variance" choice.

**Warning for Extreme Off-Policy**:

When $\mu$ differs greatly from $q_\theta$ — for example, $\mu$ has almost no samples in the high-density regions of $q_\theta$, or $w = q_\theta / \mu$ explodes in the tails — any method based on $\frac{q_\theta}{\mu}$ will suffer from severe variance problems. In this case, the advantage of $\frac{q_\theta}{\mu} k_3$ over $\frac{q_\theta}{\mu} k_1$ is no longer theoretically guaranteed, and strategies like clipping or regularization are needed.

However, in RL practice, we usually control the KL constraint and limit the degree of off-policy (e.g., using a proximal policy $\mu = q_{\theta_\text{old}}$). In this common regime, we can say with considerable confidence:

> **If you have decided to use off-policy + importance sampling to optimize reverse KL, using $\frac{q_\theta}{\mu} k_3$ as the loss usually yields lower gradient variance than $\frac{q_\theta}{\mu} k_1$.**

This is why the DeepSeek v3.2 technical report uses $\frac{q_\theta}{\mu} k_3$ as the estimator for off-policy KL penalty.

<figure style="text-align:center;">
  <img src="/assets/img/kl-estimators/dpsk-3d2-k3.png" style="width:95%;max-width:100%;">
  <figcaption style="font-size:0.9em;color:gray;">Image source: <a href="https://arxiv.org/pdf/2512.02556v1">DeepSeek v3.2 Technical Report Section 3.1</a></figcaption>
</figure>

#### Summary

- When sampling from a behavior policy $\mu$, the natural off-policy KL estimator is $\frac{q_\theta}{\mu} k_i$.
- **Numerically**, $\frac{q_\theta}{\mu} k_1$ and $\frac{q_\theta}{\mu} k_3$ remain unbiased estimates of reverse KL.
- **Gradient-wise**, since $\mu$ is independent of $\theta$, "expectation-then-gradient" and "gradient-then-expectation" are equivalent:
  - $\mathbb{E}_\mu[\nabla_\theta(\frac{q_\theta}{\mu} k_1)] = \nabla_\theta D_{\mathrm{KL}}(q_\theta \| p)$
  - $\mathbb{E}_\mu[\nabla_\theta(\frac{q_\theta}{\mu} k_3)] = \nabla_\theta D_{\mathrm{KL}}(q_\theta \| p)$
  - $\mathbb{E}_\mu[\nabla_\theta(\frac{q_\theta}{\mu} k_2)] \neq \nabla_\theta D_{\mathrm{KL}}(q_\theta \| p)$
- **Variance-wise**, the gradient of $\frac{q_\theta}{\mu} k_3$ can be seen as the gradient of $\frac{q_\theta}{\mu} k_1$ minus a zero-mean noise term $w s_\theta$. When $q_\theta \approx p \approx \mu$ and importance weights are not too extreme, **the gradient of $\frac{q_\theta}{\mu} k_3$ is more stable and has lower variance**.

### Gradient Estimation Overview

The following table summarizes the expected gradients and corresponding optimization objectives for each estimator in both on-policy and off-policy scenarios:

<div class="table-responsive" markdown="0">
<table class="table table-bordered" style="font-size: 0.95em;">
  <thead>
    <tr style="background-color: var(--global-bg-color);">
      <th style="text-align: center; white-space: nowrap;">Sampling Source</th>
      <th style="text-align: center;">Loss</th>
      <th style="text-align: center;">Expected $\nabla_\theta$ Loss</th>
      <th style="text-align: center;">Corresponding Objective</th>
      <th style="text-align: center; white-space: nowrap;">Can Optimize Reverse KL?</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="text-align: center;">$q$ (on)</td>
      <td style="text-align: center;">$k_1$</td>
      <td style="text-align: center;">$\mathbb{E}_q[s_\theta] = 0$</td>
      <td style="text-align: center;">None (Gradient is zero)</td>
      <td style="text-align: center;">✗</td>
    </tr>
    <tr>
      <td style="text-align: center;">$q$ (on)</td>
      <td style="text-align: center;">$k_2$</td>
      <td style="text-align: center;">$\nabla_\theta D_{\mathrm{KL}}(q \| p)$</td>
      <td style="text-align: center;"><strong>Reverse KL</strong></td>
      <td style="text-align: center;">✓</td>
    </tr>
    <tr>
      <td style="text-align: center;">$q$ (on)</td>
      <td style="text-align: center;">$k_3$</td>
      <td style="text-align: center;">$\nabla_\theta D_{\mathrm{KL}}(p \| q)$</td>
      <td style="text-align: center;">Forward KL</td>
      <td style="text-align: center;">✗</td>
    </tr>
    <tr>
      <td style="text-align: center;">$\mu$ (off)</td>
      <td style="text-align: center;">$\frac{q}{\mu} k_1$</td>
      <td style="text-align: center;">$\nabla_\theta D_{\mathrm{KL}}(q \| p)$</td>
      <td style="text-align: center;"><strong>Reverse KL</strong></td>
      <td style="text-align: center;">✓ (High Variance)</td>
    </tr>
    <tr>
      <td style="text-align: center;">$\mu$ (off)</td>
      <td style="text-align: center;">$\frac{q}{\mu} k_2$</td>
      <td style="text-align: center;">$\nabla_\theta \mathbb{E}_q[k_2]$</td>
      <td style="text-align: center;">f-divergence (Not KL)</td>
      <td style="text-align: center;">✗</td>
    </tr>
    <tr>
      <td style="text-align: center;">$\mu$ (off)</td>
      <td style="text-align: center;">$\frac{q}{\mu} k_3$</td>
      <td style="text-align: center;">$\nabla_\theta D_{\mathrm{KL}}(q \| p)$</td>
      <td style="text-align: center;"><strong>Reverse KL</strong></td>
      <td style="text-align: center;">✓ (Recommended, Low Var)</td>
    </tr>
  </tbody>
</table>
</div>

**Key Conclusions**:

1. **On-policy optimizing Reverse KL**: The only correct choice is $k_2$.
2. **Off-policy optimizing Reverse KL**: Both $\frac{q}{\mu} k_1$ and $\frac{q}{\mu} k_3$ are correct, but $\frac{q}{\mu} k_3$ has lower variance.
3. **$k_2$ fails in off-policy**: This is a trap that is easily overlooked.


## Practical Guidelines for RL

### KL as a Reward Penalty (No Gradient Needed)

When KL is only used as a scalar penalty in reward shaping, we only care about an accurate **value estimate**, and we do not backpropagate through it.

**Recommendations**:
- Use **$k_1$** or **$k_3$** (both are unbiased for the reverse KL value).
- When the policy is already close to the reference, $k_3$ often has lower variance.
- When coverage is poor or there is severe tail mismatch, $k_1$ can be more robust.
- In off-policy settings, simply add the importance weight $\frac{q_\theta}{\mu}$.

> **Note**: If you want a **forward KL penalty** (to encourage coverage of the behavior distribution), you can use $\mathbb{E}_q[r \log r]$ or, if you can sample from $p$, directly use $\mathbb{E}_p[\log r]$.

### KL as a Loss (Gradient Required)

When KL is part of the loss that you differentiate, you must ensure that the gradient matches your intended objective.

#### On-policy: Optimizing Reverse KL (Most Common Case)

Goal: constrain the actor not to drift far from the reference policy.

**Correct choice**: use **$k_2$** as the loss.

$$
\mathcal{L}_{k_2} = \frac{1}{2}(\log r)^2.
$$

Its gradient expectation $\mathbb{E}\_q[\nabla k\_2] = \nabla\_\theta D\_{\mathrm{KL}}(q \| p)$ is exactly the true gradient of reverse KL.

#### On-policy: Optimizing Forward KL (Coverage-Oriented Settings)

Goal: make the policy cover the support of the reference distribution (e.g., in offline RL or imitation learning).

**Correct choice**: use **$k_3$** as the loss.

$$
\mathbb{E}_q[\nabla k_3] = \mathbb{E}_q[(1 - r) s_\theta] = \nabla_\theta D_{\mathrm{KL}}(p \| q).
$$

If you backpropagate through the batch mean of $k_3$, autodiff computes exactly this forward-KL gradient – no extra tricks needed.

#### Off-policy: Optimizing Reverse KL

Goal: Data comes from behavior policy $\mu$, but we still want to optimize reverse KL.

**Correct choice**: use **$\frac{q_\theta}{\mu} k_3$** as the loss.

$$
\mathcal{L} = \frac{q_\theta(x)}{\mu(x)} \cdot \left(\frac{p(x)}{q_\theta(x)} - 1 - \log \frac{p(x)}{q_\theta(x)}\right)
$$

- Unbiased gradient.
- Significantly lower variance when $q_\theta \approx p$.

**Alternative**: Use $\frac{q_\theta}{\mu} k_1$ (gradient is also unbiased, but variance is higher).

**Avoid**: Using $\frac{q_\theta}{\mu} k_2$ (gradient is biased, not the correct direction for reverse KL).


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
      <td style="text-align: center;">$k_1$ or $k_3$ (unbiased)</td>
      <td style="text-align: center;">$k_2$</td>
    </tr>
    <tr>
      <td style="text-align: center;">Reverse KL $D_{\mathrm{KL}}(q \| p)$</td>
      <td style="text-align: center;">$\mu$ (off-policy)</td>
      <td style="text-align: center;">$\frac{q}{\mu} k_1$ or $\frac{q}{\mu} k_3$ (unbiased)</td>
      <td style="text-align: center;">$\frac{q}{\mu} k_3$</td>
    </tr>
    <tr>
      <td style="text-align: center;">Forward KL $D_{\mathrm{KL}}(p \| q)$</td>
      <td style="text-align: center;">$q$</td>
      <td style="text-align: center;">$\mathbb{E}_q[r\log r]$</td>
      <td style="text-align: center;">$k_3$</td>
    </tr>
  </tbody>
</table>
</div>


## Common Implementation Pitfalls

**Pitfall 1: Using $k_1$ Directly as a Loss (On-Policy)**

The expected gradient of $k_1$ is zero ($\mathbb{E}_q[\nabla k_1] = \mathbb{E}_q[s_\theta] = 0$), so as a loss it is ineffective.

> **Fix**: Use $k_1$ or $k_3$ only when you need a scalar KL penalty in rewards (no gradient), and use $k_2$ or $k_3$ when you actually want a loss with a meaningful gradient.

**Pitfall 2: Confusing $k_3$’s Unbiased Value with Its Gradient Objective**

$k_3$ is an **unbiased value estimator of the reverse KL**, but its **gradient** corresponds to the **forward KL**. If your goal is to optimize reverse KL but you use $k_3$ as a loss, you are in fact optimizing forward KL.

> **Fix**: Be explicit about your objective. Use $k_2$ when optimizing reverse KL; use $k_3$ only when you intentionally optimize forward KL.

**Pitfall 3: Heavy-Tailed $r$ Causing Variance Explosion**

When the policy and reference distribution are very different, $r = p/q$ can have extreme values, causing the variance of $k_3$ (and importance-sampling-based estimators) to blow up.

> **Fix**: Control the KL constraint or clip $r$.

**Pitfall 4: Using $k_2$ or $\frac{q_\theta}{\mu} k_2$ in Off-Policy Settings**

In on-policy settings, $k_2$ is the correct choice for optimizing reverse KL. However, if data comes from $\mu \neq q_\theta$:
- Using $k_2$ directly (unweighted): The expectation is not over $q_\theta$, so the estimator fails.
- Using $\frac{q_\theta}{\mu} k_2$: The gradient is biased and does not point to the reverse KL direction.

> **Fix**: In off-policy scenarios, switch to $\frac{q_\theta}{\mu} k_3$ (recommended) or $\frac{q_\theta}{\mu} k_1$.

**Pitfall 5: Detaching Importance Weights**

In implementation, $w = q_\theta / \mu$ is usually computed via `exp(log_prob_q - log_prob_mu)`. If you treat $w$ as a constant (detach it), you lose the $\nabla_\theta w = w s_\theta$ term, leading to incorrect gradients.

> **Fix**: Ensure $w$ is part of the computation graph so that autodiff correctly computes the full $\nabla_\theta(w k_i)$.


## Conclusion

**One-line summary**:

- **KL for value only (reward penalty)**: use $k_1$ or $k_3$ (both are unbiased for reverse KL); add importance weights if off-policy.
- **KL as a differentiable loss (needs gradients)**:
	- **On-policy**: To optimize **reverse KL**, use $k_2$; to optimize **forward KL**, use $k_3$.
	- **Off-policy**: To optimize **reverse KL**, use $\frac{q_\theta}{\mu} k_3$ (unbiased + low variance).

Once you keep clear **who you sample from**, **which KL you estimate**, and **with respect to which quantity you differentiate**, the three estimators become much less confusing. Especially note: **the correct choice for optimizing reverse KL differs between on-policy ($k_2$) and off-policy ($\frac{q_\theta}{\mu} k_3$)**.


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


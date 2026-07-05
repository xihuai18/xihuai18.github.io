---
layout: post
title: 渲染测试页（kitchen sink）
description: Markdown + LaTeX + 图片渲染管线的回归测试页。覆盖标题、列表、引用、代码、公式、表格、图片。不进导航、不进 sitemap。
date: 2026-07-05
lang: zh
permalink: /render-test/
sitemap: false
noindex: true
nav: false
cite_this_post: false
---

本页是渲染管线的 kitchen-sink 回归测试。改动 `_plugins/`、`_sass/` 或 MathJax 配置后,用本页做视觉验证:浅色/深色 × 桌面/移动四种组合。

## 1. 标题层级

下面从 h1 到 h6 各出现一次(h1 仅测样式,正文写作请从 h2 开始)。

# 一级标题 H1

## 二级标题 H2

### 三级标题 H3

#### 四级标题 H4

##### 五级标题 H5

###### 六级标题 H6

## 2. 带内联公式的标题:直接缩小 $\alpha_0/\alpha_1$

这个 h2 标题含内联公式,用于回归 heading id 占位符泄漏问题:生成的 `id` 不应含 `inlmath`/`dispmath` 片段,且右侧目录(TOC)点击本条目应能跳转到这里。

### 含角括号下标的标题:条件前缀 $a_{<t}$ 的记号

h3 同样带公式且含 `<`,历史上角括号会吞掉后续 HTML,现在应实体化为 `&lt;`。

## 3. 列表

无序列表:

- 第一项:普通文本
- 第二项:带行内公式 $\pi_\theta(a\mid s)$ 与行内代码 `token_ids`
- 第三项:嵌套
  - 嵌套项 A,含长英文词 hyperparameter-regularization
  - 嵌套项 B
    - 第三层嵌套

有序列表:

1. 收集 rollout 数据
2. 计算重要性比率 $\rho_t = \pi_\theta / \mu$
3. 更新策略
   1. 嵌套有序 1
   2. 嵌套有序 2

## 4. 引用块

普通 blockquote:

> 这是一段普通引用。行内公式 $D_{\mathrm{KL}}(\pi \| \mu)$ 和 `inline code` 都应正常渲染。

定理式 blockquote(含块级公式):

> **定理 1(示例)**
> 设 $\epsilon_\mu = \max_{s,a}\lvert A_\mu(s,a)\rvert < \infty$,则对任意策略 $\pi$,
>
> $$
> J(\pi) \ge J(\mu) + \frac{1}{1-\gamma}\,\mathbb{E}_{s\sim d_\mu,\,a\sim \mu}\!\left[\rho(s,a)\,A_\mu(s,a)\right] - \frac{2\gamma\,\epsilon_\mu}{(1-\gamma)^2}\,\beta
> $$
>
> 其中 $\beta$ 为策略间的最大总变差距离。

技术注记 blockquote:

> **技术注记(占位符还原)**:本页所有公式先被 `math_protection.rb` 换成占位符再交给 kramdown,post_render 阶段还原;若页面上出现形如"inl-math + 数字 + math-end"(去掉连字符)的占位符残留,说明管线坏了。

## 5. 代码

行内代码:`bundle exec jekyll serve --port 4000`,以及一个很长的行内 token:`very_long_identifier_that_should_not_break_layout_2026`。

代码块(带语言):

```python
def importance_ratio(pi_theta, mu, tokens):
    """Token-level importance sampling ratio."""
    log_rho = pi_theta.log_prob(tokens) - mu.log_prob(tokens)
    return log_rho.exp().clamp(max=5.0)
```

## 6. 公式

行内公式:$\rho_t = \dfrac{\pi_\theta(a_t \mid s_t)}{\mu(a_t \mid s_t)}$,以及含角括号下标的 $a_{<t}$、$s_{>k}$(历史 bug 回归项)。

独立公式块:

$$
J(\theta) = \mathbb{E}_{\tau \sim \mu}\left[ \sum_{t=0}^{T} \gamma^t \, \rho_t \, A_\mu(s_t, a_t) \right]
$$

`aligned` 环境:

$$
\begin{aligned}
\mathcal{L}(\theta) &= \mathbb{E}_t\left[ \min\!\left( \rho_t A_t,\ \mathrm{clip}(\rho_t, 1-\epsilon, 1+\epsilon) A_t \right) \right] \\
&\le \mathbb{E}_t\left[ \rho_t A_t \right] \\
&= \mathcal{L}_{\mathrm{IS}}(\theta)
\end{aligned}
$$

`cases` 环境:

$$
w_t =
\begin{cases}
\rho_t & \text{if } \rho_t \le c \\
c & \text{if } \rho_t > c \\
0 & \text{if } \rho_t < \epsilon_{\min}
\end{cases}
$$

`pmatrix` 环境:

$$
\Sigma =
\begin{pmatrix}
\sigma_1^2 & \rho\sigma_1\sigma_2 \\
\rho\sigma_1\sigma_2 & \sigma_2^2
\end{pmatrix}
$$

站点 accent 色高亮(`\color{#4F9143}`)与 `\underbrace`:

$$
J(\pi) \ge \underbrace{J(\mu) + \frac{1}{1-\gamma}\,\mathbb{E}\left[\rho A_\mu\right]}_{\text{surrogate}} - {\color{#4F9143} \frac{2\gamma\,\epsilon_\mu}{(1-\gamma)^2}\,\beta}
$$

## 7. 表格

单元格含公式的表格(对照 three-policy §4.5,注意 "sequence" 一词不应从中间折断):

| 机制 | 检测粒度 | 处置粒度 | 带内权重 | 防哪一侧 |
| --- | --- | --- | --- | --- |
| TIS | token | token | 截断后的 $\rho_t$ | 大比率(截断封顶) |
| IcePop | token | token | 保留 $\rho_t$ | 两侧(直接丢弃) |
| Seq-MIS | sequence | sequence | 保留 $\rho(y\mid x)$ | 大比率(整条丢弃) |
| WTRS | token | sequence | 无(纯否决) | 小比率(一票否决) |

窄表(两列,应保持 100% 宽、无横向滚动):

| 符号 | 含义 |
| --- | --- |
| $\mu$ | 行为策略 |
| $\pi_{\theta_{\text{old}}}$ | 参考策略 |

超宽表(9 列,移动端 390px 应出现横向滚动而不是把单词折成竖条):

| 算法 | 采样策略 | 参考策略 | 目标策略 | 权重形式 | 裁剪范围 | 陈旧性容忍 | throughput 影响 | regularization 强度 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PPO | $\pi_{\text{old}}$ | $\pi_{\text{old}}$ | $\pi_\theta$ | $\rho_t$ | $[1-\epsilon, 1+\epsilon]$ | 低 | baseline throughput | 中等 |
| GePPO | 混合 $\mu$ | $\pi_{\text{old}}$ | $\pi_\theta$ | $\rho_t$ 分量加权 | 按分量 | 中 | 提升约 20% | 中等 |
| Three-policy TRPO | $\mu$ | $\pi_{\text{ref}}$ | $\pi_\theta$ | 条件化 surrogate | 双侧约束 | 高 | 依赖异步框架 | 较强 |

## 8. 图片

Markdown 属性列表写法:

![KL estimator 示意图](/assets/img/_archive/kl-estimator.png){: style="display:block;margin:0 auto;width:80%;max-width:100%;" }

HTML figure/figcaption 写法(自 `_plugins/kramdown_phrasing_fix.rb` 修复 figcaption 的 content model 后,`markdown="0"` 不再是硬性要求,但仍是最保险的写法):

<figure style="text-align:center;margin:1.5rem auto;" markdown="0">
  <img src="/assets/img/_archive/three-policy-mini-class-zh.jpg" alt="三策略小课堂" style="width:90%;max-width:100%;" />
  <figcaption style="font-size:0.875em;color:var(--global-text-color-light);margin-top:0.5rem;">HTML figure 写法的图注:三策略 mini-class 示意图。</figcaption>
</figure>

不带 `markdown="0"` 的 figure(回归 kramdown_phrasing_fix:此块若解析失败,会把后续内容全部吞进 figure、下一节标题 id 退化为 `section`):

<figure style="text-align:center;margin:1.5rem auto;">
  <img src="/assets/img/robot.png" alt="robot" style="width:120px;max-width:100%;" />
  <figcaption style="font-size:0.875em;color:var(--global-text-color-light);margin-top:0.5rem;">无 markdown="0" 的图注,含行内公式 $\rho_t$ 与 **粗体**。</figcaption>
</figure>

## 9. 折叠块(details)

原生 HTML 写法(`markdown="1"`,summary 与闭合标签同行;回归 `</summary>`/`</details>` 泄漏 bug):

<details markdown="1">
<summary>推导细节:$\alpha_0/\alpha_1$ 条件化的展开(点击展开)</summary>

正文可写完整 Markdown:**粗体**、`行内代码`、行内公式 $a_{<t}$,以及列表:

- surrogate 按路由条件化拆分
- 对每个分支分别应用下界

$$
\begin{aligned}
J(\theta) &= \mathbb{E}_{\tau\sim\mu}\left[\sum_{t} \gamma^t \rho_t A_t\right] \\
&\le \mathcal{L}_{\mathrm{cond}}(\theta)
\end{aligned}
$$

</details>

Liquid tag 写法(站点自有 details 块标签):

{% details "定理 A:$k_1$ 无偏性的证明(tag 写法)" %}
tag 正文同样支持 Markdown 与公式:$\hat{k}_1 = \rho - 1 - \log\rho$。

$$
\mathbb{E}_{x\sim\mu}\left[\hat{k}_1\right] = D_{\mathrm{KL}}(\mu \| \pi)
$$
{% enddetails %}

## 10. 相关文章式块

**相关文章**

- [从两策略到三策略：LLM RL 中行为策略–参考策略不一致下的 TRPO 扩展](/reinforcement-learning/2025/11/15/three-policy-zh.html)
- [RL 中的 KL 估计器选型：从数值无偏到梯度正确](/reinforcement-learning/2025/12/01/kl-estimators-zh.html)
- [驯服陈旧数据：LLM 异策略强化学习的单调提升条件](/reinforcement-learning/2025/12/17/offpolicy-zh.html)

## 11. 收尾

若以上各节在浅色/深色、桌面/移动四种组合下都渲染正常(公式无 `mjx-merror`、表格可读、图片居中、TOC 可跳转),渲染管线即视为健康。

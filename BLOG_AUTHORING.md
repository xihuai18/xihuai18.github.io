# 博客写作对照文档

新增或修改博客文章时对照本文档。这里记录的是三篇现有文章（three-policy / kl-estimators / offpolicy）已经统一的格式惯例，以及本站渲染管线的已知坑。本文件已在 `_config.yml` 中 exclude，不会被构建进站点。

## 1. 新增一篇文章的流程

1. 在 `_posts/` 下建一对文件：`YYYY-MM-DD-slug-zh.md` 和 `YYYY-MM-DD-slug-en.md`。
2. 按第 2 节模板写 front matter，`zh_url`/`en_url` 互相指向。
3. 按第 3–7 节的惯例写正文，zh 先行，en 从 zh 回译（口径以 zh 为准）。
4. 在**已有文章**的「相关文章 / Related posts」块里加上新文章的链接（zh 链 zh、en 链 en）。
5. 本地预览 + 按第 9 节检查清单过一遍。
6. 发布到知乎/公众号之后，把 `zhihu_url`/`wechat_url` 补进两个文件的 front matter。

## 2. Front matter 模板

```yaml
---
layout: post
title: "中文标题：副标题"
date: YYYY-MM-DD
description: "一两句话摘要，会显示在列表页和 og 标签里。"
og_image: /assets/img/<slug>/<cover>.png   # 可选；zh/en 可用不同语言版本的图
categories: reinforcement-learning
lang: zh                                    # en 版写 en
en_url: /reinforcement-learning/YYYY/MM/DD/<slug>-en.html   # en 版对应写 zh_url
zhihu_url: https://zhuanlan.zhihu.com/p/...  # 发布后补
wechat_url: https://mp.weixin.qq.com/s/...   # 发布后补
---
```

permalink 规则是 `/:categories/:year/:month/:day/:title.html`，`title` 取文件名里的 slug（含 `-zh`/`-en` 后缀）。

## 3. 开篇结构

按顺序：

1. 封面图（可选）：
   ```markdown
   ![Mini-class](/assets/img/<slug>/<cover>.png){: style="display:block;margin:0 auto;width:95%;max-width:100%;" }
   ```
2. 摘要 blockquote（必有）：一段 `>` 引用，写清核心结论。可以两段（问题 + 结论），offpolicy 篇是这种写法。
3. 直接进入 `## 1. 引言：...`。

## 4. 章节与标题

- 一级小节 `## N. 标题`，二级 `### N.M 标题`，更细的小标题用**无编号 h4**（`#### 标题`）。
- 表格标题用无编号 h4（如 `#### 两类混合机制的适用场景`），**不要用「表 X.Y」编号**；正文引用写「上表 / 下表」。
- 标题里尽量少放行内公式；确需使用时可以放（锚点里的公式占位符由 `_plugins/heading_id_cleanup.rb` 自动清理），发布前点一下 TOC 验证跳转即可。
- 结尾固定三块：总结/讨论一节 → 「相关文章」块 → `## 参考文献`（en 版 `## References`）。

## 5. 正文格式惯例

**定理 / 引理 / 推论 / 观察**：blockquote + 粗体标签 + 括号命名：

```markdown
> **定理 2（三策略 TRPO）**
> 内容……
```

编号在一篇文章内部保持一致即可（three-policy 用全局编号，offpolicy 用分节编号；两种都可，但一篇内不要混用）。zh/en 两版编号必须一一对应。

**技术注记**（可跳过的补充推导）：blockquote + `**技术注记：…**` 开头，en 版 `**Technical note: …**`。

**公式高亮**：统一用站点 accent 同族绿色 `\color{#4F9143}{...}`（浅色/深色主题都可读）。不要用 `\color{blue}` 等命名色（深色主题下对比度差）。正文提及时说「绿色的…」。

**表格**：
- 宽表会由 `_plugins/table_wrapper.rb` 自动包进横向滚动容器（移动端横滚），不需要手动加 wrapper。
- 窄列避免放长英文单词（不会再被从中折断，但会把列撑宽/触发横滚）；列多时考虑精简列或拆表。
- 单元格里可以放行内公式。
- 对照表建议带「出处」列（可放链接）。

**图片**：
- 常规配图用 Markdown + 属性列表（同封面图写法）。懒加载由插件自动注入，不用手写 `loading`。
- 引用外部来源的图可用 HTML figure。`_plugins/kramdown_phrasing_fix.rb` 已修复 `<figcaption>` 的解析问题；仍推荐保留 `markdown="0"`，因为这会让整块 HTML 明确按原样输出，最稳：
  ```html
  <figure style="text-align:center;" markdown="0">
  <img src="/assets/img/<slug>/<name>.png" style="width:95%;max-width:100%;">
  <figcaption style="font-size:0.9em;color:gray;">Source: <a href="...">...</a></figcaption>
  </figure>
  ```

## 6. 数学写法与已知坑

- 行内 `$...$`；独立公式块 `$$` 各占一行，前后留空行。
- 多行推导用一个 `aligned` 环境，不要连续多个 `$$` 块。
- math 内的 `<`/`>`（如 `$a_{<t}$`）由 `_plugins/math_protection.rb` 处理成实体，可以直接写；但**任何涉及角括号的公式改动都要在渲染后亲眼验证**（历史上出过公式被浏览器当 HTML 标签吞掉的 bug）。
- 改 `_plugins/` 下任何文件后必须重启 `jekyll serve`（watch 不重载插件）。
- 快速静态自查：
  ```bash
  # $$ 定界符必须成对（输出应为偶数）
  grep -c '^\$\$' _posts/<file>.md
  # 生成页面里不应有裸角括号残留
  grep -c '_{<[a-zA-Z]' _site/reinforcement-learning/**/*.html
  ```

## 7. 参考文献与相关文章

参考文献节之前放「相关文章」块（zh 链 zh、en 链 en，列出另外几篇，按时间序）：

```markdown
**相关文章**

- [文章标题一](/reinforcement-learning/YYYY/MM/DD/<slug>-zh.html)
- [文章标题二](/reinforcement-learning/YYYY/MM/DD/<slug>-zh.html)

## 参考文献
```

参考文献格式（编号列表、条目间**不空行**）：

```markdown
1. 作者1, 作者2, 作者3, et al. "标题" (简称/备注). arXiv:XXXX.XXXXX. <https://arxiv.org/abs/XXXX.XXXXX>
2. 作者. "标题". 博客. <https://...>
```

- 作者超过约 5 人可列前三位 + `et al.`，但**名字必须与原文一致**（历史上出过把不存在的作者写进引用的问题，务必对原文核对）。
- 转述他人方法/公式时逐字对照原文，简化处理要加「有意简化」说明。
- 正文里内联引用过的工作都要进编号参考文献。

## 8. 双语同步规则

- 论述口径以 zh 为基准；en 不得比 zh 更绝对（如 zh 写「通常」，en 不要写 "always/only"）。
- 公式、定理编号、表格、参考文献逐条对应；一版改了另一版当场同步，不要攒。
- 术语不强行翻译（surrogate、rollout、on-policy 等保留英文）；zh 行文避免翻译腔。

## 9. 发布前检查清单

- [ ] 本地预览（命令见第 10 节），zh/en 两页都打开。
- [ ] MathJax 渲染完成后逐屏看一遍新增/改动的公式（**浅色 + 深色两个主题**），重点：角括号下标、`\color` 高亮、underbrace、表格内公式。
- [ ] 移动端宽度（~390px）抽查：宽表、长公式不撑破版面。
- [ ] 第 6 节的两条 grep 自查通过。
- [ ] 「相关文章」块：新文章里有旧文章，旧文章里补了新文章。
- [ ] front matter 的 `zh_url`/`en_url` 互指正确（点一下语言切换链接）。
- [ ] zh/en 逐节对照过一遍（口径、编号、引用）。

## 10. 本地预览与验证命令

```bash
export PATH="/opt/homebrew/opt/ruby/bin:/opt/homebrew/lib/ruby/gems/4.0.0/bin:$PATH"
bundle exec jekyll serve --port 4000
```

- gems 在 `vendor/bundle`（`.bundle/config` 已配 ruby-china mirror）；缺 stdlib gem 时学 `ostruct` 的做法加进 Gemfile。
- 渲染回归基准页：`http://localhost:4000/render-test/`（`_pages/render-test.md`，覆盖全部格式要素；改渲染管线后先看这页）。

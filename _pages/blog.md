---
layout: default
permalink: /blog/
title: Blog
nav: false
nav_order: 1
disable_share_image: true
twitter_card: summary
---

<div class="post blog-index">

  <header class="post-header">
    <p class="eyebrow">The Blog</p>
    <h1 class="post-title has-dropcap">Xihuai&rsquo;s <span class="blog-title__dropcap">B</span>log</h1>
    <p class="lead">Notes on reinforcement learning, multi-agent systems, and LLM reasoning &mdash; written to clarify my own thinking, shared in case they help yours.</p>
  </header>

  {% assign uv_enabled = site.data.post_uv_meta.generated_at %}
  {% assign uv_data_all = site.data.post_uv.all %}
  {% assign uv_data_d30 = site.data.post_uv.d30 %}
  {% assign uv_has_modes = uv_data_all.size | plus: 0 %}

  {%- comment -%} Build dedup post list — when an EN post has a zh_url, skip the ZH version on the index.
    Pair zh back onto EN via candidate loop. {%- endcomment -%}
  {%- assign postlist = site.posts -%}

  {%- comment -%} Count posts + collect categories with counts {%- endcomment -%}
  {%- assign visible_count = 0 -%}
  {%- assign cat_slug_list = "" -%}
  {%- assign cat_name_list = "" -%}
  {%- assign cat_count_str = "" -%}
  {%- for post in postlist -%}
    {%- if post.lang == 'zh' and post.en_url -%}{%- continue -%}{%- endif -%}
    {%- assign visible_count = visible_count | plus: 1 -%}
    {%- for category in post.categories -%}
      {%- assign cat_slug = category | slugify -%}
      {%- unless cat_slug_list contains cat_slug -%}
        {%- if cat_slug_list == "" -%}
          {%- assign cat_slug_list = cat_slug -%}
          {%- assign cat_name_list = category -%}
        {%- else -%}
          {%- assign cat_slug_list = cat_slug_list | append: "|" | append: cat_slug -%}
          {%- assign cat_name_list = cat_name_list | append: "|" | append: category -%}
        {%- endif -%}
      {%- endunless -%}
    {%- endfor -%}
  {%- endfor -%}
  {%- assign cat_slugs = cat_slug_list | split: "|" -%}
  {%- assign cat_names = cat_name_list | split: "|" -%}

  <nav class="blog-controls" role="group" aria-label="Filter posts by category" markdown="0">
    <button type="button" class="blog-pill is-active" data-filter-link data-filter-type="category" data-filter-value="">
      All posts <span class="blog-pill__count">{{ visible_count }}</span>
    </button>
    {%- for i in (0..cat_slugs.size) -%}
      {%- if i >= cat_slugs.size -%}{%- break -%}{%- endif -%}
      {%- assign this_slug = cat_slugs[i] -%}
      {%- assign this_name = cat_names[i] -%}
      {%- assign this_count = 0 -%}
      {%- for post in postlist -%}
        {%- if post.lang == 'zh' and post.en_url -%}{%- continue -%}{%- endif -%}
        {%- for c in post.categories -%}
          {%- if c | slugify == this_slug -%}{%- assign this_count = this_count | plus: 1 -%}{%- break -%}{%- endif -%}
        {%- endfor -%}
      {%- endfor -%}
      {%- assign display_name = this_name | replace: "-", " " | capitalize -%}
      <button type="button" class="blog-pill" data-filter-link data-filter-type="category" data-filter-value="{{ this_slug }}">
        {{ display_name }} <span class="blog-pill__count">{{ this_count }}</span>
      </button>
    {%- endfor -%}

    {% if uv_enabled and uv_has_modes > 0 %}
    <span class="blog-controls__spacer"></span>
    <span class="blog-views-toggle mono-meta">
      Views:
      <a href="#" class="uv-toggle" data-uv-mode="all">All time</a>
      /
      <a href="#" class="uv-toggle" data-uv-mode="d30">30 days</a>
    </span>
    {% endif %}
  </nav>

  <ol class="post-rows">
    {%- for post in postlist -%}
      {%- if post.lang == 'zh' and post.en_url -%}{%- continue -%}{%- endif -%}

      {%- assign post_year = post.date | date: "%Y" -%}
      {%- assign post_categories_slug = "" -%}
      {%- for category in post.categories -%}
        {%- assign cs = category | slugify -%}
        {%- if post_categories_slug == "" -%}
          {%- assign post_categories_slug = cs -%}
        {%- else -%}
          {%- assign post_categories_slug = post_categories_slug | append: " " | append: cs -%}
        {%- endif -%}
      {%- endfor -%}

      {%- assign zh_post = nil -%}
      {%- if post.lang == 'en' and post.zh_url -%}
        {%- for candidate in postlist -%}
          {%- if candidate.lang == 'zh' and candidate.en_url == post.url -%}
            {%- assign zh_post = candidate -%}
            {%- break -%}
          {%- endif -%}
        {%- endfor -%}
      {%- endif -%}

      {%- comment -%} Compute read_time for each language — word count differs between
        English and the ZH translation, so reading-time estimates differ too. Views
        (post_uv) are also keyed per URL, so EN and ZH have distinct view counts. {%- endcomment -%}
      {%- if post.external_source == blank -%}
        {%- assign read_time_en = post.content | number_of_words | divided_by: 180 | plus: 1 -%}
      {%- else -%}
        {%- assign read_time_en = post.feed_content | strip_html | number_of_words | divided_by: 180 | plus: 1 -%}
      {%- endif -%}
      {%- if zh_post -%}
        {%- assign read_time_zh = zh_post.content | number_of_words | divided_by: 180 | plus: 1 -%}
      {%- endif -%}

    <li class="post-row blog-post-list-item" data-year="{{ post_year }}" data-categories="{{ post_categories_slug }}">
      <time class="post-row__date mono-meta" datetime="{{ post.date | date_to_xmlschema }}">
        {{ post.date | date: "%b %-d, %Y" }}
      </time>
      <div class="post-row__body">
        {%- if zh_post -%}
        {%- assign pair_id = post.id | slugify -%}
        <div class="post-row__langbar" role="tablist" aria-label="Language">
          <button type="button" class="post-row__langtab is-active" data-lang-pair="{{ pair_id }}" data-lang-target="en" role="tab" aria-selected="true">EN</button>
          <button type="button" class="post-row__langtab" data-lang-pair="{{ pair_id }}" data-lang-target="zh" role="tab" aria-selected="false">中</button>
        </div>

        <div class="post-row__lang-pane is-active" data-lang-pair="{{ pair_id }}" data-lang="en">
          <h3 class="post-row__title">
            {%- if post.redirect == blank -%}
              <a href="{{ post.url | relative_url }}">{{ post.title }}</a>
            {%- elsif post.redirect contains '://' -%}
              <a href="{{ post.redirect }}" target="_blank" rel="noopener">{{ post.title }} <span class="post-row__external" aria-hidden="true">↗</span></a>
            {%- else -%}
              <a href="{{ post.redirect | relative_url }}">{{ post.title }}</a>
            {%- endif -%}
          </h3>
          {%- if post.description -%}
          <p class="post-row__desc">{{ post.description }}</p>
          {%- endif -%}
          <p class="post-row__meta mono-meta">
            <span class="post-row__read">{{ read_time_en }} min read</span>
            {%- for category in post.categories -%}
            <span class="post-row__sep" aria-hidden="true">·</span>
            <a class="post-row__cat" href="{{ '/blog/' | relative_url }}?category={{ category | slugify }}" data-filter-link data-filter-type="category" data-filter-value="{{ category | slugify }}">{{ category | replace: "-", " " | capitalize }}</a>
            {%- endfor -%}
            {%- if uv_enabled and uv_has_modes > 0 -%}
            {%- include post_uv.html url=post.url -%}
            <span class="post-row__sep" aria-hidden="true">·</span>
            <span class="post-row__views post-uv" data-uv-all="{{ uv_all }}" data-uv-d30="{{ uv_d30 }}">{{ uv_all }} views</span>
            {%- endif -%}
          </p>
        </div>

        <div class="post-row__lang-pane" data-lang-pair="{{ pair_id }}" data-lang="zh" hidden>
          <h3 class="post-row__title">
            <a href="{{ zh_post.url | relative_url }}">{{ zh_post.title }}</a>
          </h3>
          {%- if zh_post.description -%}
          <p class="post-row__desc">{{ zh_post.description }}</p>
          {%- endif -%}
          <p class="post-row__meta mono-meta">
            <span class="post-row__read">{{ read_time_zh }} 分钟阅读</span>
            {%- for category in zh_post.categories -%}
            <span class="post-row__sep" aria-hidden="true">·</span>
            <a class="post-row__cat" href="{{ '/blog/' | relative_url }}?category={{ category | slugify }}" data-filter-link data-filter-type="category" data-filter-value="{{ category | slugify }}">{{ category | replace: "-", " " | capitalize }}</a>
            {%- endfor -%}
            {%- if uv_enabled and uv_has_modes > 0 -%}
            {%- include post_uv.html url=zh_post.url -%}
            <span class="post-row__sep" aria-hidden="true">·</span>
            <span class="post-row__views post-uv" data-uv-all="{{ uv_all }}" data-uv-d30="{{ uv_d30 }}" data-uv-label="阅读">{{ uv_all }} 阅读</span>
            {%- endif -%}
          </p>
        </div>

        {%- else -%}
        <h3 class="post-row__title">
          {%- if post.redirect == blank -%}
            <a href="{{ post.url | relative_url }}">{{ post.title }}</a>
          {%- elsif post.redirect contains '://' -%}
            <a href="{{ post.redirect }}" target="_blank" rel="noopener">{{ post.title }} <span class="post-row__external" aria-hidden="true">↗</span></a>
          {%- else -%}
            <a href="{{ post.redirect | relative_url }}">{{ post.title }}</a>
          {%- endif -%}
        </h3>
        {%- if post.description -%}
        <p class="post-row__desc">{{ post.description }}</p>
        {%- endif -%}
        {%- assign _lbl_read = "min read" -%}
        {%- assign _lbl_views = "views" -%}
        {%- if post.lang == 'zh' -%}
          {%- assign _lbl_read = "分钟阅读" -%}
          {%- assign _lbl_views = "阅读" -%}
        {%- endif -%}
        <p class="post-row__meta mono-meta">
          <span class="post-row__read">{{ read_time_en }} {{ _lbl_read }}</span>
          {%- for category in post.categories -%}
          <span class="post-row__sep" aria-hidden="true">·</span>
          <a class="post-row__cat" href="{{ '/blog/' | relative_url }}?category={{ category | slugify }}" data-filter-link data-filter-type="category" data-filter-value="{{ category | slugify }}">{{ category | replace: "-", " " | capitalize }}</a>
          {%- endfor -%}
          {%- if uv_enabled and uv_has_modes > 0 -%}
          {%- include post_uv.html url=post.url -%}
          <span class="post-row__sep" aria-hidden="true">·</span>
          <span class="post-row__views post-uv" data-uv-all="{{ uv_all }}" data-uv-d30="{{ uv_d30 }}">{{ uv_all }} {{ _lbl_views }}</span>
          {%- endif -%}
        </p>
        {%- endif -%}
      </div>
    </li>
    {%- endfor -%}
  </ol>

{% if page.pagination.enabled %}
{% include pagination.html %}
{% endif %}

</div>

<script defer src="{{ '/assets/js/blog_filter.js' | relative_url }}"></script>

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
{%- assign cat_slug_list = "|" -%}
{%- assign cat_name_list = "|" -%}
{%- assign tag_slug_list = "|" -%}
{%- assign tag_name_list = "|" -%}
{%- assign year_list_raw = "" -%}
{%- for post in postlist -%}
{%- assign post_lang = post.lang | default: site.lang | downcase | replace: '_', '-' -%}
{%- if post_lang == 'zh-cn' or post_lang == 'cn' -%}{%- assign post_lang = 'zh' -%}{%- endif -%}
{%- if post_lang == 'zh' and post.en_url -%}{%- continue -%}{%- endif -%}
{%- assign zh_post = nil -%}
{%- if post_lang == 'en' and post.zh_url -%}
{%- for candidate in postlist -%}
{%- assign candidate_lang = candidate.lang | default: site.lang | downcase | replace: '_', '-' -%}
{%- if candidate_lang == 'zh-cn' or candidate_lang == 'cn' -%}{%- assign candidate_lang = 'zh' -%}{%- endif -%}
{%- if candidate_lang == 'zh' and candidate.en_url == post.url -%}
{%- assign zh_post = candidate -%}
{%- break -%}
{%- endif -%}
{%- endfor -%}
{%- endif -%}
{%- assign visible_count = visible_count | plus: 1 -%}
{%- assign post_year = post.date | date: "%Y" -%}
{%- assign year_list_raw = year_list_raw | append: post_year | append: "|" -%}
{%- for category in post.categories -%}
{%- assign cat_slug = category | slugify -%}
{%- assign cat_slug_token = "|" | append: cat_slug | append: "|" -%}
{%- unless cat_slug_list contains cat_slug_token -%}
{%- if cat_name_list == "|" -%}
{%- assign cat_slug_list = cat_slug_list | append: cat_slug | append: "|" -%}
{%- assign cat_name_list = cat_name_list | append: category | append: "|" -%}
{%- else -%}
{%- assign cat_slug_list = cat_slug_list | append: cat_slug | append: "|" -%}
{%- assign cat_name_list = cat_name_list | append: category | append: "|" -%}
{%- endif -%}
{%- endunless -%}
{%- endfor -%}
{%- if zh_post -%}
{%- for category in zh_post.categories -%}
{%- assign cat_slug = category | slugify -%}
{%- assign cat_slug_token = "|" | append: cat_slug | append: "|" -%}
{%- unless cat_slug_list contains cat_slug_token -%}
{%- assign cat_slug_list = cat_slug_list | append: cat_slug | append: "|" -%}
{%- assign cat_name_list = cat_name_list | append: category | append: "|" -%}
{%- endunless -%}
{%- endfor -%}
{%- endif -%}
{%- for tag in post.tags -%}
{%- assign tag_slug = tag | slugify -%}
{%- assign tag_slug_token = "|" | append: tag_slug | append: "|" -%}
{%- unless tag_slug_list contains tag_slug_token -%}
{%- assign tag_slug_list = tag_slug_list | append: tag_slug | append: "|" -%}
{%- assign tag_name_list = tag_name_list | append: tag | append: "|" -%}
{%- endunless -%}
{%- endfor -%}
{%- if zh_post -%}
{%- for tag in zh_post.tags -%}
{%- assign tag_slug = tag | slugify -%}
{%- assign tag_slug_token = "|" | append: tag_slug | append: "|" -%}
{%- unless tag_slug_list contains tag_slug_token -%}
{%- assign tag_slug_list = tag_slug_list | append: tag_slug | append: "|" -%}
{%- assign tag_name_list = tag_name_list | append: tag | append: "|" -%}
{%- endunless -%}
{%- endfor -%}
{%- endif -%}
{%- endfor -%}
{%- assign cat_slugs = cat_slug_list | split: "|" -%}
{%- assign cat_names = cat_name_list | split: "|" -%}
{%- assign tag_slugs = tag_slug_list | split: "|" -%}
{%- assign tag_names = tag_name_list | split: "|" -%}
{%- assign years_list = year_list_raw | split: "|" | uniq -%}

  <nav class="blog-controls" role="group" aria-label="Filter posts by category" markdown="0">
    <button type="button" class="blog-pill is-active" data-filter-link data-filter-type="category" data-filter-value="">
      All posts <span class="blog-pill__count">{{ visible_count }}</span>
    </button>
    {%- for i in (0..cat_slugs.size) -%}
      {%- if i >= cat_slugs.size -%}{%- break -%}{%- endif -%}
      {%- assign this_slug = cat_slugs[i] -%}
      {%- assign this_name = cat_names[i] -%}
      {%- if this_slug == "" -%}{%- continue -%}{%- endif -%}
      {%- assign this_count = 0 -%}
      {%- for post in postlist -%}
        {%- assign post_lang = post.lang | default: site.lang | downcase | replace: '_', '-' -%}
        {%- if post_lang == 'zh-cn' or post_lang == 'cn' -%}{%- assign post_lang = 'zh' -%}{%- endif -%}
        {%- if post_lang == 'zh' and post.en_url -%}{%- continue -%}{%- endif -%}
        {%- assign zh_post = nil -%}
        {%- if post_lang == 'en' and post.zh_url -%}
          {%- for candidate in postlist -%}
            {%- assign candidate_lang = candidate.lang | default: site.lang | downcase | replace: '_', '-' -%}
            {%- if candidate_lang == 'zh-cn' or candidate_lang == 'cn' -%}{%- assign candidate_lang = 'zh' -%}{%- endif -%}
            {%- if candidate_lang == 'zh' and candidate.en_url == post.url -%}
              {%- assign zh_post = candidate -%}
              {%- break -%}
            {%- endif -%}
          {%- endfor -%}
        {%- endif -%}
        {%- assign matched_category = false -%}
        {%- for c in post.categories -%}
          {%- if c | slugify == this_slug -%}{%- assign matched_category = true -%}{%- break -%}{%- endif -%}
        {%- endfor -%}
        {%- unless matched_category -%}
          {%- for c in zh_post.categories -%}
            {%- if c | slugify == this_slug -%}{%- assign matched_category = true -%}{%- break -%}{%- endif -%}
          {%- endfor -%}
        {%- endunless -%}
        {%- if matched_category -%}{%- assign this_count = this_count | plus: 1 -%}{%- endif -%}
      {%- endfor -%}
      {%- assign display_name = this_name | replace: "-", " " | capitalize -%}
      <button type="button" class="blog-pill" data-filter-link data-filter-type="category" data-filter-value="{{ this_slug }}">
        {{ display_name }} <span class="blog-pill__count">{{ this_count }}</span>
      </button>
    {%- endfor -%}
    {%- for i in (0..tag_slugs.size) -%}
      {%- if i >= tag_slugs.size -%}{%- break -%}{%- endif -%}
      {%- assign this_slug = tag_slugs[i] -%}
      {%- assign this_name = tag_names[i] -%}
      {%- if this_slug == "" -%}{%- continue -%}{%- endif -%}
      {%- assign this_count = 0 -%}
      {%- for post in postlist -%}
        {%- assign post_lang = post.lang | default: site.lang | downcase | replace: '_', '-' -%}
        {%- if post_lang == 'zh-cn' or post_lang == 'cn' -%}{%- assign post_lang = 'zh' -%}{%- endif -%}
        {%- if post_lang == 'zh' and post.en_url -%}{%- continue -%}{%- endif -%}
        {%- assign zh_post = nil -%}
        {%- if post_lang == 'en' and post.zh_url -%}
          {%- for candidate in postlist -%}
            {%- assign candidate_lang = candidate.lang | default: site.lang | downcase | replace: '_', '-' -%}
            {%- if candidate_lang == 'zh-cn' or candidate_lang == 'cn' -%}{%- assign candidate_lang = 'zh' -%}{%- endif -%}
            {%- if candidate_lang == 'zh' and candidate.en_url == post.url -%}
              {%- assign zh_post = candidate -%}
              {%- break -%}
            {%- endif -%}
          {%- endfor -%}
        {%- endif -%}
        {%- assign matched_tag = false -%}
        {%- for t in post.tags -%}
          {%- if t | slugify == this_slug -%}{%- assign matched_tag = true -%}{%- break -%}{%- endif -%}
        {%- endfor -%}
        {%- unless matched_tag -%}
          {%- for t in zh_post.tags -%}
            {%- if t | slugify == this_slug -%}{%- assign matched_tag = true -%}{%- break -%}{%- endif -%}
          {%- endfor -%}
        {%- endunless -%}
        {%- if matched_tag -%}{%- assign this_count = this_count | plus: 1 -%}{%- endif -%}
      {%- endfor -%}
      {%- assign display_name = this_name | replace: "-", " " | capitalize -%}
      <button type="button" class="blog-pill" data-filter-link data-filter-type="tag" data-filter-value="{{ this_slug }}">
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

{%- for year in years_list -%}
{%- if year == "" -%}{%- continue -%}{%- endif -%}
{%- assign count_in_year = 0 -%}
{%- for post in postlist -%}
{%- assign post_lang = post.lang | default: site.lang | downcase | replace: '_', '-' -%}
{%- if post_lang == 'zh-cn' or post_lang == 'cn' -%}{%- assign post_lang = 'zh' -%}{%- endif -%}
{%- if post_lang == 'zh' and post.en_url -%}{%- continue -%}{%- endif -%}
{%- assign post_year = post.date | date: "%Y" -%}
{%- if post_year == year -%}{%- assign count_in_year = count_in_year | plus: 1 -%}{%- endif -%}
{%- endfor -%}

  <section class="blog-year-group" data-year-group="{{ year }}">
    <h2 class="blog-year-head">
      <span class="blog-year-num">{{ year }}</span>
      <span class="blog-year-rule" aria-hidden="true"></span>
      <span class="blog-year-count">{{ count_in_year }} {% if count_in_year == 1 %}post{% else %}posts{% endif %}</span>
    </h2>

    <ol class="post-rows">
      {%- for post in postlist -%}
        {%- assign post_lang = post.lang | default: site.lang | downcase | replace: '_', '-' -%}
        {%- if post_lang == 'zh-cn' or post_lang == 'cn' -%}{%- assign post_lang = 'zh' -%}{%- endif -%}
        {%- if post_lang == 'zh' and post.en_url -%}{%- continue -%}{%- endif -%}
        {%- assign post_year = post.date | date: "%Y" -%}
        {%- unless post_year == year -%}{%- continue -%}{%- endunless -%}

        {%- assign post_categories_slug = "" -%}
        {%- assign post_category_tokens = "|" -%}
        {%- for category in post.categories -%}
          {%- assign cs = category | slugify -%}
          {%- assign cs_token = "|" | append: cs | append: "|" -%}
          {%- unless post_category_tokens contains cs_token -%}
            {%- assign post_category_tokens = post_category_tokens | append: cs | append: "|" -%}
            {%- if post_categories_slug == "" -%}
              {%- assign post_categories_slug = cs -%}
            {%- else -%}
              {%- assign post_categories_slug = post_categories_slug | append: " " | append: cs -%}
            {%- endif -%}
          {%- endunless -%}
        {%- endfor -%}
        {%- assign post_tags_slug = "" -%}
        {%- assign post_tag_tokens = "|" -%}
        {%- for tag in post.tags -%}
          {%- assign ts = tag | slugify -%}
          {%- assign ts_token = "|" | append: ts | append: "|" -%}
          {%- unless post_tag_tokens contains ts_token -%}
            {%- assign post_tag_tokens = post_tag_tokens | append: ts | append: "|" -%}
            {%- if post_tags_slug == "" -%}
              {%- assign post_tags_slug = ts -%}
            {%- else -%}
              {%- assign post_tags_slug = post_tags_slug | append: " " | append: ts -%}
            {%- endif -%}
          {%- endunless -%}
        {%- endfor -%}

        {%- assign zh_post = nil -%}
        {%- if post_lang == 'en' and post.zh_url -%}
          {%- for candidate in postlist -%}
            {%- assign candidate_lang = candidate.lang | default: site.lang | downcase | replace: '_', '-' -%}
            {%- if candidate_lang == 'zh-cn' or candidate_lang == 'cn' -%}{%- assign candidate_lang = 'zh' -%}{%- endif -%}
            {%- if candidate_lang == 'zh' and candidate.en_url == post.url -%}
              {%- assign zh_post = candidate -%}
              {%- break -%}
            {%- endif -%}
          {%- endfor -%}
        {%- endif -%}

        {%- comment -%} Compute read_time for each language — word count differs between
          English and the ZH translation, so reading-time estimates differ too. Views
          (post_uv) are also keyed per URL, so EN and ZH have distinct view counts. {%- endcomment -%}
        {%- if post.external_source == blank -%}
          {%- if post_lang == 'zh' -%}
            {%- assign read_units_en = post.content | strip_html | number_of_words: 'cjk' -%}
          {%- else -%}
            {%- assign read_units_en = post.content | strip_html | number_of_words -%}
          {%- endif -%}
        {%- else -%}
          {%- if post_lang == 'zh' -%}
            {%- assign read_units_en = post.feed_content | strip_html | number_of_words: 'cjk' -%}
          {%- else -%}
            {%- assign read_units_en = post.feed_content | strip_html | number_of_words -%}
          {%- endif -%}
        {%- endif -%}
        {%- assign read_time_en = read_units_en | plus: 179 | divided_by: 180 -%}
        {%- if read_time_en < 1 -%}{%- assign read_time_en = 1 -%}{%- endif -%}
        {%- if zh_post -%}
          {%- for category in zh_post.categories -%}
            {%- assign cs = category | slugify -%}
            {%- assign cs_token = "|" | append: cs | append: "|" -%}
            {%- unless post_category_tokens contains cs_token -%}
              {%- assign post_category_tokens = post_category_tokens | append: cs | append: "|" -%}
              {%- if post_categories_slug == "" -%}
                {%- assign post_categories_slug = cs -%}
              {%- else -%}
                {%- assign post_categories_slug = post_categories_slug | append: " " | append: cs -%}
              {%- endif -%}
            {%- endunless -%}
          {%- endfor -%}
          {%- for tag in zh_post.tags -%}
            {%- assign ts = tag | slugify -%}
            {%- assign ts_token = "|" | append: ts | append: "|" -%}
            {%- unless post_tag_tokens contains ts_token -%}
              {%- assign post_tag_tokens = post_tag_tokens | append: ts | append: "|" -%}
              {%- if post_tags_slug == "" -%}
                {%- assign post_tags_slug = ts -%}
              {%- else -%}
                {%- assign post_tags_slug = post_tags_slug | append: " " | append: ts -%}
              {%- endif -%}
            {%- endunless -%}
          {%- endfor -%}
          {%- if zh_post.external_source == blank -%}
            {%- assign read_units_zh = zh_post.content | strip_html | number_of_words: 'cjk' -%}
          {%- else -%}
            {%- assign read_units_zh = zh_post.feed_content | strip_html | number_of_words: 'cjk' -%}
          {%- endif -%}
          {%- assign read_time_zh = read_units_zh | plus: 179 | divided_by: 180 -%}
          {%- if read_time_zh < 1 -%}{%- assign read_time_zh = 1 -%}{%- endif -%}
        {%- endif -%}
        {%- if post.redirect == blank -%}
          {%- assign post_href = post.url | relative_url -%}
          {%- assign post_target = '' -%}
          {%- assign post_is_external = false -%}
        {%- elsif post.redirect contains '://' -%}
          {%- assign post_href = post.redirect -%}
          {%- assign post_target = ' target="_blank" rel="noopener"' -%}
          {%- assign post_is_external = true -%}
        {%- else -%}
          {%- assign post_href = post.redirect | relative_url -%}
          {%- assign post_target = '' -%}
          {%- assign post_is_external = false -%}
        {%- endif -%}
        {%- if zh_post -%}
          {%- if zh_post.redirect == blank -%}
            {%- assign zh_href = zh_post.url | relative_url -%}
            {%- assign zh_target = '' -%}
            {%- assign zh_is_external = false -%}
          {%- elsif zh_post.redirect contains '://' -%}
            {%- assign zh_href = zh_post.redirect -%}
            {%- assign zh_target = ' target="_blank" rel="noopener"' -%}
            {%- assign zh_is_external = true -%}
          {%- else -%}
            {%- assign zh_href = zh_post.redirect | relative_url -%}
            {%- assign zh_target = '' -%}
            {%- assign zh_is_external = false -%}
          {%- endif -%}
        {%- endif -%}

      <li class="post-row blog-post-list-item" data-year="{{ post_year }}" data-categories="{{ post_categories_slug }}" data-tags="{{ post_tags_slug }}">
        <time
          class="post-row__date mono-meta"
          datetime="{{ post.date | date_to_xmlschema }}"
          data-date-en="{{ post.date | date: '%b %-d, %Y' }}"
          data-date-zh="{{ post.date | date: '%Y年%-m月%-d日' }}"
        >
          {%- if post_lang == 'zh' and zh_post == nil -%}
            {{ post.date | date: "%Y年%-m月%-d日" }}
          {%- else -%}
            {{ post.date | date: "%b %-d, %Y" }}
          {%- endif -%}
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
              <a href="{{ post_href }}"{{ post_target }}>{{ post.title }}{%- if post_is_external -%} <span class="post-row__external" aria-hidden="true">↗</span>{%- endif -%}</a>
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
              {%- for tag in post.tags -%}
              <span class="post-row__sep" aria-hidden="true">·</span>
              <a class="post-row__cat" href="{{ '/blog/' | relative_url }}?tag={{ tag | slugify }}" data-filter-link data-filter-type="tag" data-filter-value="{{ tag | slugify }}">{{ tag }}</a>
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
              <a href="{{ zh_href }}"{{ zh_target }}>{{ zh_post.title }}{%- if zh_is_external -%} <span class="post-row__external" aria-hidden="true">↗</span>{%- endif -%}</a>
            </h3>
            {%- if zh_post.description -%}
            <p class="post-row__desc">{{ zh_post.description }}</p>
            {%- endif -%}
            <p class="post-row__meta mono-meta">
              <span class="post-row__read">{{ read_time_zh }} 分钟阅读</span>
              {%- for category in zh_post.categories -%}
              <span class="post-row__sep" aria-hidden="true">·</span>
              <a class="post-row__cat" href="{{ '/blog/' | relative_url }}?category={{ category | slugify }}&lang=zh" data-filter-link data-filter-type="category" data-filter-value="{{ category | slugify }}">{{ category | replace: "-", " " | capitalize }}</a>
              {%- endfor -%}
              {%- for tag in zh_post.tags -%}
              <span class="post-row__sep" aria-hidden="true">·</span>
              <a class="post-row__cat" href="{{ '/blog/' | relative_url }}?tag={{ tag | slugify }}&lang=zh" data-filter-link data-filter-type="tag" data-filter-value="{{ tag | slugify }}">{{ tag }}</a>
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
            <a href="{{ post_href }}"{{ post_target }}>{{ post.title }}{%- if post_is_external -%} <span class="post-row__external" aria-hidden="true">↗</span>{%- endif -%}</a>
          </h3>
          {%- if post.description -%}
          <p class="post-row__desc">{{ post.description }}</p>
          {%- endif -%}
          {%- assign _lbl_read = "min read" -%}
          {%- assign _lbl_views = "views" -%}
          {%- if post_lang == 'zh' -%}
            {%- assign _lbl_read = "分钟阅读" -%}
            {%- assign _lbl_views = "阅读" -%}
          {%- endif -%}
          <p class="post-row__meta mono-meta">
            <span class="post-row__read">{{ read_time_en }} {{ _lbl_read }}</span>
            {%- for category in post.categories -%}
            <span class="post-row__sep" aria-hidden="true">·</span>
            <a class="post-row__cat" href="{{ '/blog/' | relative_url }}?category={{ category | slugify }}{% if post_lang == 'zh' %}&lang=zh{% endif %}" data-filter-link data-filter-type="category" data-filter-value="{{ category | slugify }}">{{ category | replace: "-", " " | capitalize }}</a>
            {%- endfor -%}
            {%- for tag in post.tags -%}
            <span class="post-row__sep" aria-hidden="true">·</span>
            <a class="post-row__cat" href="{{ '/blog/' | relative_url }}?tag={{ tag | slugify }}{% if post_lang == 'zh' %}&lang=zh{% endif %}" data-filter-link data-filter-type="tag" data-filter-value="{{ tag | slugify }}">{{ tag }}</a>
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

  </section>
  {%- endfor -%}

{% if page.pagination.enabled %}
{% include pagination.html %}
{% endif %}

</div>

<script defer src="{{ '/assets/js/blog_filter.js' | relative_url }}"></script>

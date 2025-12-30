---
layout: default
permalink: /blog/
title: Xihuai's Blog
nav: false
nav_order: 1
---

<div class="post">

{% assign uv_enabled = site.data.post_uv_meta.generated_at %}
{% assign uv_data_all = site.data.post_uv.all %}
{% assign uv_data_d30 = site.data.post_uv.d30 %}
{% assign uv_has_modes = uv_data_all.size | plus: 0 %}

{% assign blog_name_size = site.blog_name | size %}
{% assign blog_description_size = site.blog_description | size %}

{% if blog_name_size > 0 or blog_description_size > 0 %}

  <div class="header-bar">
    <h1>{{ site.blog_name }}</h1>
    <h2>{{ site.blog_description }}</h2>
  </div>
  {% endif %}

{% if site.display_tags and site.display_tags.size > 0 or site.display_categories and site.display_categories.size > 0 %}

  <div class="tag-category-list">
    <ul class="p-0 m-0">
      {% for tag in site.display_tags %}
        <li>
          #️⃣ <a href="{{ tag | slugify | prepend: '/blog/tag/' | relative_url }}">{{ tag }}</a>
        </li>
        {% unless forloop.last %}
          <p>&bull;</p>
        {% endunless %}
      {% endfor %}
      {% if site.display_categories.size > 0 and site.display_tags.size > 0 %}
        <p>&bull;</p>
      {% endif %}
      {% for category in site.display_categories %}
        <li>
          🏷️ <a href="{{ category | slugify | prepend: '/blog/category/' | relative_url }}">{{ category }}</a>
        </li>
        {% unless forloop.last %}
          <p>&bull;</p>
        {% endunless %}
      {% endfor %}
    </ul>
  </div>
  {% endif %}

{% if uv_enabled and uv_has_modes > 0 %}
  <p class="post-meta mb-2">
    Views:
    <a href="#" class="uv-toggle" data-uv-mode="all">All time</a>
    /
    <a href="#" class="uv-toggle" data-uv-mode="d30">Last 30 days</a>
  </p>
{% endif %}

{% assign featured_posts = site.posts | where: "featured", "true" %}
{% if featured_posts.size > 0 %}
<br>

<div class="container featured-posts">
{% assign is_even = featured_posts.size | modulo: 2 %}
<div class="row row-cols-{% if featured_posts.size <= 2 or is_even == 0 %}2{% else %}3{% endif %}">
{% for post in featured_posts %}
<div class="col mb-4">
<a href="{{ post.url | relative_url }}">
<div class="card hoverable">
<div class="row g-0">
<div class="col-md-12">
<div class="card-body">
<div class="float-right">
📌
</div>
<h3 class="card-title text-lowercase">{{ post.title }}</h3>
<p class="card-text">{{ post.description }}</p>

                    {% if post.external_source == blank %}
                      {% assign read_time = post.content | number_of_words | divided_by: 180 | plus: 1 %}
                    {% else %}
                      {% assign read_time = post.feed_content | strip_html | number_of_words | divided_by: 180 | plus: 1 %}
                    {% endif %}
                    {% assign year = post.date | date: "%Y" %}

                    <p class="post-meta">
                      {{ read_time }} min read &nbsp; &middot; &nbsp;
                      <a href="{{ year | prepend: '/blog/' | relative_url }}">
                        📅 {{ year }} </a>
                      {% if uv_enabled and uv_has_modes > 0 %}
                        {% include post_uv.html url=post.url %}
                        &nbsp; &middot; &nbsp;
                        <span class="post-uv" data-uv-all="{{ uv_all }}" data-uv-d30="{{ uv_d30 }}">{{ uv_all }} views</span>
                      {% endif %}
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </a>
        </div>
      {% endfor %}
      </div>
    </div>
    <hr>

{% endif %}

  <ul class="post-list">

    {% if page.pagination.enabled %}
      {% assign postlist = paginator.posts %}
    {% else %}
      {% assign postlist = site.posts %}
    {% endif %}

    {% for post in postlist %}

    {% if post.lang == 'zh' and post.en_url %}
      {% continue %}
    {% endif %}

    {% comment %}
      Find matching Chinese post for English posts with zh_url.
      Match by directly comparing candidate.en_url with post.url.
      Performance: O(n) lookup per English post, acceptable for small-medium blogs.
      The {% break %} tag exits the loop early once a match is found.
    {% endcomment %}
    {% assign zh_post = nil %}
    {% if post.lang == 'en' and post.zh_url %}
      {% for candidate in site.posts %}
        {% if candidate.lang == 'zh' and candidate.en_url == post.url %}
          {% assign zh_post = candidate %}
          {% break %}
        {% endif %}
      {% endfor %}
    {% endif %}

    <li>

{% if post.thumbnail %}

<div class="row">
          <div class="col-sm-9">
{% endif %}

    {% if zh_post %}
      {% assign post_slug = post.id | slugify %}
      <div class="lang-switcher">
        <ul class="nav" id="myTab-{{ post_slug }}" role="tablist">
          <li class="nav-item">
            <a class="nav-link active" id="en-tab-{{ post_slug }}" href="#en-{{ post_slug }}" role="tab" aria-controls="en-{{ post_slug }}" aria-selected="true" aria-label="View English version">English</a>
          </li>
          <li class="nav-item">
            <a class="nav-link" id="zh-tab-{{ post_slug }}" href="#zh-{{ post_slug }}" role="tab" aria-controls="zh-{{ post_slug }}" aria-selected="false" aria-label="查看简体中文版本">简体中文</a>
          </li>
        </ul>
      </div>

      <div class="tab-content" id="myTabContent-{{ post_slug }}">
        <div class="tab-pane fade show active" id="en-{{ post_slug }}" role="tabpanel" aria-labelledby="en-tab-{{ post_slug }}">
          {% include blog_post_body.html post=post uv_enabled=uv_enabled uv_has_modes=uv_has_modes %}
        </div>
        <div class="tab-pane fade" id="zh-{{ post_slug }}" role="tabpanel" aria-labelledby="zh-tab-{{ post_slug }}">
          {% include blog_post_body.html post=zh_post uv_enabled=uv_enabled uv_has_modes=uv_has_modes %}
        </div>
      </div>
    {% else %}
      {% include blog_post_body.html post=post uv_enabled=uv_enabled uv_has_modes=uv_has_modes %}
    {% endif %}

{% if post.thumbnail %}

</div>

  <div class="col-sm-3">
    <img class="card-img" src="{{ post.thumbnail | relative_url }}" style="object-fit: cover; height: 90%" alt="image">
  </div>
</div>
{% endif %}
    </li>

    {% endfor %}

  </ul>

{% if page.pagination.enabled %}
{% include pagination.html %}
{% endif %}

</div>

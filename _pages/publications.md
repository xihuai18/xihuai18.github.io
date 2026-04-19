---
layout: page
permalink: /publications/
title: A record of collaborative work.
display_title: 'A record of collaborative <span class="blog-title__dropcap">w</span>ork.'
tab_title: Publications
nav_title: Publications
eyebrow: Publications
lead: Peer-reviewed papers and preprints, organized by year. <strong>*</strong> denotes equal contribution. Full list also on <a href="https://scholar.google.com/citations?user=YEc4cq8AAAAJ" target="_blank" rel="noopener">Google Scholar</a>.
years: [2026, 2025, 2024, 2023, 2022, 2021]
topics:
  - {name: "LLM Reasoning & Agency", abbr: "LLM Reasoning and Agency"}
  - {name: "MARL Generalization",    abbr: "MARL Generalization"}
  - {name: "MARL Efficiency",        abbr: "MARL Efficiency"}
nav: true
nav_order: 1
---

<!-- _pages/publications.md -->
<div class="publications">

{%- comment -%} Topic filter chip bar — reuses `filterByVenue()` from venue_filter.js.
  Counts are computed by splitting jekyll-scholar's output on `</li>` closers. {%- endcomment -%}
{%- assign total_entries = 0 -%}
{%- for y in page.years -%}
  {%- capture _ye -%}{% bibliography -f papers -q @*[year={{y}}]* %}{%- endcapture -%}
  {%- assign _yc = _ye | split: '</li>' | size | minus: 1 -%}
  {%- assign total_entries = total_entries | plus: _yc -%}
{%- endfor -%}

<nav class="pub-topic-filter blog-controls" role="group" aria-label="Filter papers by topic" markdown="0">
  <button type="button" class="blog-pill pub-topic-pill is-active" data-topic-filter="" onclick="clearVenueFilter()">
    All papers <span class="blog-pill__count">{{ total_entries }}</span>
  </button>
  {%- for topic in page.topics -%}
    {%- capture _te -%}{% bibliography -f papers -q @*[abbr={{topic.abbr}}]* %}{%- endcapture -%}
    {%- assign _tc = _te | split: '</li>' | size | minus: 1 -%}
    {%- assign topic_dom = topic.abbr | replace: " and ", " & " -%}
    {%- if _tc > 0 -%}
    <button type="button" class="blog-pill pub-topic-pill" data-topic-filter="{{ topic_dom }}" onclick="filterByVenue('{{ topic_dom }}')">
      {{ topic.name }} <span class="blog-pill__count">{{ _tc }}</span>
    </button>
    {%- endif -%}
  {%- endfor -%}
</nav>

{%- for y in page.years %}
  {% capture year_entries %}{% bibliography -f papers -q @*[year={{y}}]* %}{% endcapture %}
  {% assign stripped = year_entries | strip_html | strip %}
  {%- if stripped != "" -%}
  {%- comment -%} Count bibliography items by counting `</li>` closers (robust to
    `<li>` / `<li class=...>` / `<li id=...>` variations emitted by jekyll-scholar). {%- endcomment -%}
  {%- assign year_item_count = year_entries | split: '</li>' | size | minus: 1 -%}
  <h2 class="year">
    <span class="year-num">{{y}}</span>
    <span class="year-rule" aria-hidden="true"></span>
    <span class="year-count">{{ year_item_count }} {% if year_item_count == 1 %}paper{% else %}papers{% endif %}</span>
  </h2>
  {{ year_entries }}
  {%- endif -%}
{% endfor %}

</div>

---
layout: page
title: Code & artifacts.
display_title: '<span class="blog-title__dropcap">C</span>ode &amp; artifacts.'
tab_title: Projects
nav_title: Projects
eyebrow: Projects
permalink: /projects/
description: Research code and toolkits I maintain or contributed to.
lead: Repositories tied to papers I've written or contributed meaningfully to. Shared here as research code and artifacts. Take them, break them, and let me know when they help.
nav: true
nav_order: 3
published: true
---

<!-- Projects page — follows the design system Projects prototype. -->

<div class="projects-page">
  <ul class="projects-grid">
    {%- for p in site.data.projects -%}
    {%- assign repo_href = nil -%}
    {%- assign repo_display = nil -%}
    {%- if p.repo -%}
      {%- assign repo_value = p.repo | append: '' -%}
      {%- assign repo_head = repo_value | split: '/' | first -%}
      {%- if repo_value contains '://' -%}
        {%- assign repo_href = repo_value -%}
        {%- assign repo_display = repo_value | remove: 'https://' | remove: 'http://' -%}
      {%- elsif repo_value contains '@' and repo_value contains ':' and repo_value contains '.' -%}
        {%- assign repo_host_path = repo_value | split: '@' | last | replace: ':', '/' -%}
        {%- assign repo_href = 'https://' | append: repo_host_path -%}
        {%- assign repo_display = repo_host_path -%}
      {%- elsif repo_head contains '.' and repo_value contains '/' -%}
        {%- assign repo_href = 'https://' | append: repo_value -%}
        {%- assign repo_display = repo_value -%}
      {%- else -%}
        {%- assign repo_href = 'https://github.com/' | append: repo_value -%}
        {%- assign repo_display = 'github.com/' | append: repo_value -%}
      {%- endif -%}
    {%- endif -%}
    <li class="ds-card ds-card-hover project-card">
      <div class="project-head">
        <span class="project-lang">
          <span class="project-lang-dot project-lang-dot--{{ p.language | downcase | replace: '+', 'p' | replace: ' ', '-' | default: 'other' }}"></span>
          {{ p.language | default: "—" }}
        </span>
        {%- if p.stars != blank or p.forks != blank -%}
        <span class="project-stats mono-meta">
          {%- if p.stars != blank -%}<span>★ {{ p.stars }}</span>{%- endif -%}
          {%- if p.forks != blank -%}<span>⑂ {{ p.forks }}</span>{%- endif -%}
        </span>
        {%- endif -%}
      </div>
      <h3 class="project-title">
        {%- if p.url -%}
        <a href="{{ p.url }}" target="_blank" rel="noopener">{{ p.name }}</a>
        {%- elsif repo_href -%}
        <a href="{{ repo_href }}" target="_blank" rel="noopener">{{ p.name }}</a>
        {%- else -%}
        {{ p.name }}
        {%- endif -%}
      </h3>
      <p class="project-blurb">{{ p.blurb }}</p>
      {%- if repo_href and repo_display -%}
      <div class="project-repo">
        <a href="{{ repo_href }}" target="_blank" rel="noopener">{{ repo_display }}</a>
      </div>
      {%- endif -%}
      {%- if p.tags and p.tags.size > 0 -%}
      <div class="project-tags">
        {%- for tag in p.tags -%}
        <span class="chip no-dot">{{ tag }}</span>
        {%- endfor -%}
      </div>
      {%- endif -%}
    </li>
    {%- endfor -%}
  </ul>
</div>

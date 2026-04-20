---
layout: page
title: Code & artifacts.
display_title: '<span class="blog-title__dropcap">C</span>ode &amp; artifacts.'
tab_title: Projects
nav_title: Projects
eyebrow: Projects
permalink: /projects/
description: Research code and toolkits I maintain or contributed to.
lead: Repositories tied to papers I've written or contributed meaningfully to. Everything is MIT or Apache. Take it, break it, ping me when it helps.
nav: true
nav_order: 3
published: true
---

<!-- Projects page — follows the design system Projects prototype. -->

<div class="projects-page">
  <ul class="projects-grid">
    {%- for p in site.data.projects -%}
    <li class="ds-card ds-card-hover project-card">
      <div class="project-head">
        <span class="project-lang">
          <span class="project-lang-dot project-lang-dot--{{ p.language | downcase | replace: '+', 'p' | replace: ' ', '-' | default: 'other' }}"></span>
          {{ p.language | default: "—" }}
        </span>
        {%- if p.stars or p.forks -%}
        <span class="project-stats mono-meta">
          {%- if p.stars -%}<span>★ {{ p.stars }}</span>{%- endif -%}
          {%- if p.forks -%}<span>⑂ {{ p.forks }}</span>{%- endif -%}
        </span>
        {%- endif -%}
      </div>
      <h3 class="project-title">
        {%- if p.url -%}
        <a href="{{ p.url }}" target="_blank" rel="noopener">{{ p.name }}</a>
        {%- else -%}
        <a href="https://{{ p.repo }}" target="_blank" rel="noopener">{{ p.name }}</a>
        {%- endif -%}
      </h3>
      <p class="project-blurb">{{ p.blurb }}</p>
      <div class="project-repo">
        <a href="https://{{ p.repo }}" target="_blank" rel="noopener">{{ p.repo }}</a>
      </div>
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

---
layout: page
title: Projects
permalink: /projects/
description: Research code and toolkits I maintain or contributed to.
nav: false
nav_order: 5
published: false
---

<!--
  HIDDEN PAGE — NOT LINKED FROM NAV, NOT INCLUDED IN SITE OUTPUT.
  To enable: set `published: true` above. The template below follows the
  design system Projects prototype (ui_kits/website/Pages.jsx § Projects).

  Data: each card reads from `site.data.projects` — edit `_data/projects.yml`.
  Expected schema per entry:
    - name: Project display name
      repo: github.com/org/repo
      url:  optional landing URL (defaults to https://<repo>)
      blurb: 1–2 sentence summary
      language: Python | TypeScript | Rust | C++  (drives the dot color)
      stars: integer (optional, display-only)
      tags: [rl, marl, benchmark, …]
-->

<div class="projects-page">
  <ul class="projects-grid">
    {%- for p in site.data.projects -%}
    <li class="ds-card ds-card-hover project-card">
      <div class="project-head">
        <span class="project-lang">
          <span class="project-lang-dot project-lang-dot--{{ p.language | downcase | replace: '+', 'p' | replace: ' ', '-' | default: 'other' }}"></span>
          {{ p.language | default: "—" }}
        </span>
        {%- if p.stars -%}
        <span class="project-stars">★ {{ p.stars }}</span>
        {%- endif -%}
      </div>
      <h3 class="project-title">
        <a href="{{ p.url | default: 'https://' | append: p.repo }}" target="_blank" rel="noopener">{{ p.name }}</a>
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

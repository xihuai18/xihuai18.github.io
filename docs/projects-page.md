# Projects page (hidden)

`/projects/` is implemented but not published. The route is currently gated
behind `published: false` in `_pages/projects.md`.

## What exists

- **Page**: `_pages/projects.md` — Liquid loop that renders `site.data.projects`
  into a 2-column card grid (1-column on mobile). `nav: false` + `published: false`
  keeps it off the navbar and out of the built site.
- **Data**: `_data/projects.yml` — seed entries for DPT-Agent, ZSC-Eval, PMAT.
  Edit to add / remove projects.
- **Styles**: `_sass/_design_system.scss` § 15b — card, language dot (Python / TS /
  Rust / etc.), stars count, repo line, tag chips. Follows the design kit
  prototype (`project/ui_kits/website/Pages.jsx` + `project/ui_kits/website/pages.css` §
  `.uk-proj*`).

## To enable

1. Flip `published: false` → `published: true` in `_pages/projects.md`.
2. Flip `nav: false` → `nav: true` (and pick a `nav_order`) if you want it in the
   top nav.
3. Populate `_data/projects.yml` with real `stars` counts or leave them blank.
4. Rebuild Jekyll.

## Schema

```yaml
- name: Display name                           # required
  repo: github.com/org/repo                    # required — renders under title
  url: https://optional.landing.page/          # optional — falls back to https://<repo>
  blurb: One or two factual sentences.         # required — runs in serif at 0.9rem
  language: Python | TypeScript | Rust | …     # drives the colored dot
  stars: 142                                   # optional, display-only
  tags: [rl, marl, benchmark]                  # optional — rendered as pill chips
```

## Notes

- Language-dot colors live in `_sass/_design_system.scss` § 15b.
  Add a new modifier class (e.g. `.project-lang-dot--julia`) if you extend the set.
- No automated GitHub-stars fetching yet. If wanted later, a small Liquid + data
  fetch in `_plugins/` (or a GitHub Action writing `_data/projects_stars.yml`)
  would close the loop.

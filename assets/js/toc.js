/* Auto-generate a table of contents for pages that include an element with
 * `data-toc`. Intended for CV + blog posts (or any page) and designed to be
 * no-op when the container doesn't exist.
 * 
 * Features:
 * - Auto-generated TOC from headings
 * - Collapsible sidebar toggle
 * - Collapsible sections within TOC
 * - Scroll spy for active section highlighting
 */

(() => {
  const STORAGE_KEY = "toc-collapsed";

  function onReady(fn) {
    if (document.readyState === "loading") {
      document.addEventListener("DOMContentLoaded", fn, { once: true });
    } else {
      fn();
    }
  }

  function slugify(text) {
    const raw = (text || "").trim().toLowerCase();
    if (!raw) return "";

    // Keep CJK and common unicode ranges while stripping punctuation.
    let slug = raw.normalize ? raw.normalize("NFKD") : raw;
    slug = slug.replace(/\s+/g, "-");
    slug = slug.replace(
      /[^a-z0-9_\-\u00c0-\u024f\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af]+/g,
      ""
    );
    slug = slug.replace(/-+/g, "-").replace(/^-|-$/g, "");
    return slug;
  }

  function ensureUniqueId(element, usedIds) {
    const existing = (element.getAttribute("id") || "").trim();
    if (existing) {
      usedIds.add(existing);
      return existing;
    }

    const base = slugify(element.textContent) || "section";
    let candidate = base;
    let counter = 2;
    while (usedIds.has(candidate) || document.getElementById(candidate)) {
      candidate = `${base}-${counter}`;
      counter += 1;
    }
    element.setAttribute("id", candidate);
    usedIds.add(candidate);
    return candidate;
  }

  function buildTocList(headings) {
    const rootUl = document.createElement("ul");
    let currentTopLi = null;

    for (const heading of headings) {
      const level = Number(heading.tagName.replace(/^H/i, ""));
      const li = document.createElement("li");
      const a = document.createElement("a");
      a.href = `#${heading.id}`;
      a.textContent = heading.textContent.trim();
      li.appendChild(a);

      const isSub = level >= 3 && currentTopLi;
      if (!isSub) {
        rootUl.appendChild(li);
        currentTopLi = li;
        continue;
      }

      let subUl = currentTopLi.querySelector("ul");
      if (!subUl) {
        subUl = document.createElement("ul");
        currentTopLi.appendChild(subUl);
        
        // Add section toggle button for collapsible sections
        currentTopLi.classList.add("has-children");
        const toggleBtn = document.createElement("button");
        toggleBtn.className = "toc-section-toggle";
        toggleBtn.setAttribute("aria-label", "Toggle section");
        // Use createElement instead of innerHTML for security
        const icon = document.createElement("i");
        icon.className = "fas fa-chevron-down";
        toggleBtn.appendChild(icon);
        toggleBtn.addEventListener("click", (e) => {
          e.preventDefault();
          e.stopPropagation();
          currentTopLi.classList.toggle("section-collapsed");
          toggleBtn.classList.toggle("is-collapsed");
        });
        currentTopLi.insertBefore(toggleBtn, currentTopLi.firstChild);
      }
      subUl.appendChild(li);
    }

    return rootUl;
  }

  function initCollapsibleToc() {
    const tocSidebar = document.querySelector("[data-toc-sidebar]");
    const toggleBtn = document.querySelector("[data-toc-toggle]");
    const expandBtn = document.querySelector("[data-toc-expand]");
    const tocLayout = document.querySelector(".toc-layout");

    if (!tocSidebar || !toggleBtn || !tocLayout) return;

    // Restore collapsed state from localStorage
    const isCollapsed = localStorage.getItem(STORAGE_KEY) === "true";
    if (isCollapsed) {
      setCollapsed(true);
    }

    function setCollapsed(collapsed) {
      if (collapsed) {
        tocSidebar.classList.add("is-collapsed");
        toggleBtn.classList.add("is-collapsed");
        tocLayout.classList.add("toc-collapsed");
        if (expandBtn) expandBtn.style.display = "flex";
      } else {
        tocSidebar.classList.remove("is-collapsed");
        toggleBtn.classList.remove("is-collapsed");
        tocLayout.classList.remove("toc-collapsed");
        if (expandBtn) expandBtn.style.display = "none";
      }
      localStorage.setItem(STORAGE_KEY, collapsed.toString());
    }

    toggleBtn.addEventListener("click", () => {
      const willCollapse = !tocSidebar.classList.contains("is-collapsed");
      setCollapsed(willCollapse);
    });

    if (expandBtn) {
      expandBtn.addEventListener("click", () => {
        setCollapsed(false);
      });
    }
  }

  function initOne(tocEl) {
    const wrapper = tocEl.closest(".toc-sidebar") || tocEl;
    const root = tocEl.closest(".toc-layout") || document;

    const contentSelector = tocEl.getAttribute("data-toc-content") || ".toc-content";
    const headingsSelector = tocEl.getAttribute("data-toc-headings") || "h2,h3";
    const minItems = Number(tocEl.getAttribute("data-toc-min-items") || "2");

    const content = root.querySelector(contentSelector) || root.querySelector("article") || root;
    const headings = Array.from(content.querySelectorAll(headingsSelector)).filter((h) => {
      const text = (h.textContent || "").trim();
      return text.length > 0;
    });

    if (headings.length < minItems) {
      wrapper.classList.add("is-empty");
      return;
    }

    const usedIds = new Set();
    for (const heading of headings) ensureUniqueId(heading, usedIds);

    const list = buildTocList(headings);
    tocEl.replaceChildren(list);

    // Smooth scrolling on click (respects the fixed navbar via scroll-margin-top in CSS).
    tocEl.addEventListener("click", (e) => {
      const link = e.target.closest("a");
      if (!link) return;
      const id = (link.getAttribute("href") || "").replace(/^#/, "");
      const target = id ? document.getElementById(id) : null;
      if (!target) return;
      e.preventDefault();
      target.scrollIntoView({ behavior: "smooth", block: "start" });
      history.pushState(null, "", `#${id}`);
    });

    // Lightweight scroll-spy.
    const linkById = new Map();
    tocEl.querySelectorAll("a[href^='#']").forEach((a) => {
      const id = a.getAttribute("href").slice(1);
      if (id) linkById.set(id, a);
    });

    let ticking = false;
    function updateActive() {
      const offset = 80; // fixed navbar + breathing room
      const y = window.scrollY + offset;

      let current = headings[0];
      for (const heading of headings) {
        const top = heading.getBoundingClientRect().top + window.scrollY;
        if (top <= y) current = heading;
        else break;
      }

      tocEl.querySelectorAll("a.is-active").forEach((a) => a.classList.remove("is-active"));
      const activeLink = current && linkById.get(current.id);
      if (activeLink) activeLink.classList.add("is-active");
    }

    function onScroll() {
      if (ticking) return;
      ticking = true;
      window.requestAnimationFrame(() => {
        updateActive();
        ticking = false;
      });
    }

    window.addEventListener("scroll", onScroll, { passive: true });
    window.addEventListener("resize", onScroll, { passive: true });
    updateActive();
  }

  function initAll() {
    document.querySelectorAll("[data-toc]").forEach(initOne);
    initCollapsibleToc();
  }

  onReady(initAll);
})();

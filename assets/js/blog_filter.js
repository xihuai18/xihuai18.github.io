/**
 * Blog filtering module for client-side year, category, and tag filtering.
 * Reads URL parameters and dynamically shows/hides blog posts without page reload.
 */
(function () {
  /**
   * Validates and normalizes a year value from URL parameters.
   * @param {string|null} rawYear - The raw year value from URL parameters
   * @returns {string|null} A valid 4-digit year string, or null if invalid
   */
  function normalizeYear(rawYear) {
    if (!rawYear) return null;
    const trimmed = String(rawYear).trim();
    return /^\d{4}$/.test(trimmed) ? trimmed : null;
  }

  /**
   * Validates and normalizes a category slug from URL parameters.
   * @param {string|null} rawCategory - The raw category value from URL parameters
   * @returns {string|null} A valid slugified category string, or null if invalid
   */
  function normalizeCategory(rawCategory) {
    if (!rawCategory) return null;
    const trimmed = String(rawCategory).trim().toLowerCase();
    // Slug format: lowercase letters/numbers separated by single hyphens
    return /^[a-z0-9]+(?:-[a-z0-9]+)*$/.test(trimmed) ? trimmed : null;
  }

  function normalizeTag(rawTag) {
    return normalizeCategory(rawTag);
  }

  function normalizeLang(rawLang) {
    if (!rawLang) return null;
    const normalized = String(rawLang).trim().toLowerCase();
    return normalized === "zh" ? "zh" : normalized === "en" ? "en" : null;
  }

  /**
   * Retrieves and validates current filter values from URL parameters.
   * @returns {{year: string|null, category: string|null, tag: string|null, lang: string|null}} Object containing validated filters
   */
  function getFilters() {
    const params = new URLSearchParams(window.location.search);
    const rawYear = params.get("year");
    const rawCategory = params.get("category");
    const rawTag = params.get("tag");
    const rawLang = params.get("lang");
    return {
      year: normalizeYear(rawYear),
      category: normalizeCategory(rawCategory),
      tag: normalizeTag(rawTag),
      lang: normalizeLang(rawLang),
    };
  }

  /**
   * Updates the browser URL with the provided filter parameters.
   * Supports year, category, and tag filters simultaneously.
   * Passing an empty object clears all active filters.
   * @param {Object} params - Object containing filter values to update
   * @param {string} [params.year] - Year filter value
   * @param {string} [params.category] - Category filter value
   * @param {string} [params.tag] - Tag filter value
   * @param {string} [params.lang] - Language preference for bilingual rows
   */
  function updateUrl(params) {
    const url = new URL(window.location.href);

    // If no filter keys are provided, clear all filters (used by clearFilters).
    const hasYearParam = Object.prototype.hasOwnProperty.call(params, "year");
    const hasCategoryParam = Object.prototype.hasOwnProperty.call(
      params,
      "category",
    );
    const hasTagParam = Object.prototype.hasOwnProperty.call(params, "tag");
    const hasLangParam = Object.prototype.hasOwnProperty.call(params, "lang");
    if (!hasYearParam && !hasCategoryParam && !hasTagParam && !hasLangParam) {
      url.searchParams.delete("year");
      url.searchParams.delete("category");
      url.searchParams.delete("tag");
      history.replaceState({}, "", url);
      return;
    }

    // Merge current filters with the provided ones so filters can coexist.
    const current = getFilters();
    const nextYear = hasYearParam ? params.year : current.year;
    const nextCategory = hasCategoryParam ? params.category : current.category;
    const nextTag = hasTagParam ? params.tag : current.tag;
    const nextLang = hasLangParam ? params.lang : current.lang;

    // Check if we are toggling off the current filter
    if (hasYearParam && params.year === current.year) {
      url.searchParams.delete("year");
    } else if (nextYear) {
      url.searchParams.set("year", nextYear);
    } else {
      url.searchParams.delete("year");
    }

    if (hasCategoryParam && params.category === current.category) {
      url.searchParams.delete("category");
    } else if (nextCategory) {
      url.searchParams.set("category", nextCategory);
    } else {
      url.searchParams.delete("category");
    }

    if (hasTagParam && params.tag === current.tag) {
      url.searchParams.delete("tag");
    } else if (nextTag) {
      url.searchParams.set("tag", nextTag);
    } else {
      url.searchParams.delete("tag");
    }

    if (hasLangParam && params.lang === current.lang) {
      if (nextLang) {
        url.searchParams.set("lang", nextLang);
      } else {
        url.searchParams.delete("lang");
      }
    } else if (nextLang) {
      url.searchParams.set("lang", nextLang);
    } else {
      url.searchParams.delete("lang");
    }

    history.replaceState({}, "", url);
  }

  function applyLanguagePreference(lang) {
    if (lang !== "zh" && lang !== "en") return;

    document.querySelectorAll(".post-row__langtab").forEach((tab) => {
      const active = tab.getAttribute("data-lang-target") === lang;
      tab.classList.toggle("is-active", active);
      tab.setAttribute("aria-selected", active ? "true" : "false");
    });

    document.querySelectorAll(".post-row__lang-pane").forEach((pane) => {
      const active = pane.getAttribute("data-lang") === lang;
      pane.hidden = !active;
      pane.classList.toggle("is-active", active);
    });
  }

  function updateYearCounts(yearGroups) {
    const preferredLang = getFilters().lang;
    yearGroups.forEach((group) => {
      const count = Array.from(group.querySelectorAll(".post-row")).filter(
        (post) => post.style.display !== "none",
      ).length;
      const countNode = group.querySelector(".blog-year-count");
      if (countNode) {
        countNode.textContent =
          preferredLang === "zh"
            ? `${count} 篇`
            : `${count} ${count === 1 ? "post" : "posts"}`;
      }
    });
  }

  function updateCategoryCounts(posts, year, tag) {
    const pills = document.querySelectorAll(
      '.blog-pill[data-filter-link][data-filter-type="category"]',
    );
    if (!pills.length) return;

    pills.forEach((pill) => {
      const value = pill.getAttribute("data-filter-value") || "";
      const countNode = pill.querySelector(".blog-pill__count");
      if (!countNode) return;

      const count = Array.from(posts).filter((post) => {
        const matchesYear = !year || post.dataset.year === year;
        const tags = (post.dataset.tags || "")
          .toLowerCase()
          .split(/\s+/)
          .filter(Boolean);
        const categories = (post.dataset.categories || "")
          .toLowerCase()
          .split(/\s+/)
          .filter(Boolean);
        const matchesTag = !tag || tags.includes(tag);
        const matchesCategory = !value || categories.includes(value);
        return matchesYear && matchesTag && matchesCategory;
      }).length;

      countNode.textContent = count;
    });
  }

  function updateTagCounts(posts, year, category) {
    const pills = document.querySelectorAll(
      '.blog-pill[data-filter-link][data-filter-type="tag"]',
    );
    if (!pills.length) return;

    pills.forEach((pill) => {
      const value = pill.getAttribute("data-filter-value") || "";
      const countNode = pill.querySelector(".blog-pill__count");
      if (!countNode) return;

      const count = Array.from(posts).filter((post) => {
        const matchesYear = !year || post.dataset.year === year;
        const tags = (post.dataset.tags || "")
          .toLowerCase()
          .split(/\s+/)
          .filter(Boolean);
        const categories = (post.dataset.categories || "")
          .toLowerCase()
          .split(/\s+/)
          .filter(Boolean);
        const matchesCategory = !category || categories.includes(category);
        const matchesTag = !value || tags.includes(value);
        return matchesYear && matchesCategory && matchesTag;
      }).length;

      countNode.textContent = count;
    });
  }

  /**
   * Applies the current filters to blog posts, showing or hiding them as appropriate.
   * Updates the filter status banner with the active filters and post count.
   * Also manages the border styling to ensure the last visible post has no bottom border.
   */
  function applyFilters() {
    const { year, category, tag, lang } = getFilters();
    const posts = document.querySelectorAll(".post-rows > .post-row");
    const yearGroups = document.querySelectorAll("[data-year-group]");
    let visibleCount = 0;
    const totalCount = posts.length;
    let lastVisiblePost = null;

    posts.forEach((post) => {
      post.style.borderBottom = "";
    });

    posts.forEach((post) => {
      const matchesYear = !year || post.dataset.year === year;
      const categories = (post.dataset.categories || "")
        .toLowerCase()
        .split(/\s+/)
        .filter(Boolean);
      const tags = (post.dataset.tags || "")
        .toLowerCase()
        .split(/\s+/)
        .filter(Boolean);
      const matchesCategory = !category || categories.includes(category);
      const matchesTag = !tag || tags.includes(tag);
      const isVisible = matchesYear && matchesCategory && matchesTag;
      post.style.display = isVisible ? "" : "none";
      if (isVisible) {
        visibleCount += 1;
        lastVisiblePost = post;
      }
    });

    const hasActiveFilters = year || category || tag;

    yearGroups.forEach((group) => {
      const hasVisiblePosts = Array.from(
        group.querySelectorAll(".post-row"),
      ).some((post) => post.style.display !== "none");
      group.style.display = hasVisiblePosts ? "" : "none";
    });

    if (lang) applyLanguagePreference(lang);
    updateYearCounts(yearGroups);
    updateCategoryCounts(posts, year, tag);
    updateTagCounts(posts, year, category);

    if (hasActiveFilters && lastVisiblePost) {
      lastVisiblePost.style.borderBottom = "none";
    }

    // Legacy: highlight text links with `active-filter` when they match.
    document.querySelectorAll("[data-filter-link]").forEach((link) => {
      const filterType = link.getAttribute("data-filter-type");
      const filterValue = link.getAttribute("data-filter-value");
      const active =
        (filterType === "year" && filterValue === year) ||
        (filterType === "category" && filterValue === category) ||
        (filterType === "tag" && filterValue === tag);
      link.classList.toggle("active-filter", active);
    });

    // Pill filter: toggle `.is-active` class. "All posts" (empty value) lights
    // up when neither category nor year is active.
    const pills = document.querySelectorAll(
      '.blog-pill[data-filter-link][data-filter-type="category"]',
    );
    if (pills.length) {
      pills.forEach((pill) => {
        const value = pill.getAttribute("data-filter-value") || "";
        const isAll = value === "";
        const isActive = isAll
          ? !category && !year && !tag
          : value === category;
        pill.classList.toggle("is-active", isActive);
      });
    }

    const tagPills = document.querySelectorAll(
      '.blog-pill[data-filter-link][data-filter-type="tag"]',
    );
    if (tagPills.length) {
      tagPills.forEach((pill) => {
        const value = pill.getAttribute("data-filter-value") || "";
        pill.classList.toggle("is-active", value === tag);
      });
    }

    updateFilterStatus(year, category, tag, visibleCount, totalCount);
  }

  /**
   * Updates or creates the filter status element with styling matching publications page.
   * @param {string|null} year - Active year filter
   * @param {string|null} category - Active category filter
   * @param {string|null} tag - Active tag filter
   * @param {number} visibleCount - Number of visible posts
   * @param {number} totalCount - Total number of posts
   */
  function updateFilterStatus(year, category, tag, visibleCount, totalCount) {
    let statusElement = document.getElementById("blog-filter-status");
    const preferredLang = getFilters().lang;

    if (!statusElement) {
      // Create status element with inline styles matching publications page
      statusElement = document.createElement("div");
      statusElement.id = "blog-filter-status";
      statusElement.setAttribute("role", "status");
      statusElement.setAttribute("aria-live", "polite");
      statusElement.style.cssText = `
        display: none;
        margin: 1rem 0;
        padding: 0.5rem 1rem;
        background-color: var(--global-code-bg-color);
        border-left: 3px solid var(--global-theme-color);
        border-radius: 0 4px 4px 0;
        font-size: 0.875rem;
        color: var(--global-text-color);
      `;

      // Create the message span
      const messageSpan = document.createElement("span");
      messageSpan.id = "blog-filter-message";
      statusElement.appendChild(messageSpan);

      // Create the clear button once with event listener
      const clearBtn = document.createElement("button");
      clearBtn.id = "blog-filter-clear-btn";
      clearBtn.textContent = "Clear Filter";
      clearBtn.setAttribute("aria-label", "Clear active blog filters");
      clearBtn.style.cssText = `
        margin-left: 1rem;
        padding: 0.2rem 0.5rem;
        background: var(--global-theme-color);
        color: white;
        border: none;
        border-radius: 3px;
        cursor: pointer;
        font-size: 0.75rem;
      `;
      clearBtn.addEventListener("click", clearFilters);
      statusElement.appendChild(clearBtn);

      // Insert before the first year group / post list
      const postSection = document.querySelector(".post");
      if (postSection) {
        const postList = postSection.querySelector(
          ".blog-year-group, .post-rows",
        );
        if (postList) {
          postList.parentNode.insertBefore(statusElement, postList);
        }
      }
    }

    const parts = [];
    const formatValue = (value) => String(value || "").replace(/-/g, " ");
    if (year)
      parts.push(preferredLang === "zh" ? `年份：${year}` : `Year: ${year}`);
    if (category)
      parts.push(
        preferredLang === "zh"
          ? `分类：${formatValue(category)}`
          : `Category: ${formatValue(category)}`,
      );
    if (tag)
      parts.push(
        preferredLang === "zh"
          ? `标签：${formatValue(tag)}`
          : `Tag: ${formatValue(tag)}`,
      );

    if (parts.length) {
      statusElement.style.display = "block";
      // Update only the message text, not the entire innerHTML
      const messageSpan = statusElement.querySelector("#blog-filter-message");
      const clearBtn = statusElement.querySelector("#blog-filter-clear-btn");
      if (clearBtn && preferredLang === "zh") {
        clearBtn.textContent = "清除筛选";
        clearBtn.setAttribute("aria-label", "清除当前博客筛选");
      } else if (clearBtn) {
        clearBtn.textContent = "Clear Filter";
        clearBtn.setAttribute("aria-label", "Clear active blog filters");
      }
      if (messageSpan) {
        const label = preferredLang === "zh" ? "筛选中" : "Filtering by";
        const separator = preferredLang === "zh" ? "，" : " and ";
        const showing =
          preferredLang === "zh"
            ? `：显示 ${visibleCount} / ${totalCount} 篇`
            : `: showing ${visibleCount} of ${totalCount} posts`;
        messageSpan.innerHTML = `${label} <strong>${parts.join(separator)}</strong>${showing}`;
      }
    } else {
      statusElement.style.display = "none";
    }
  }

  /**
   * Clears all active filters and updates the UI.
   * @param {Event} event - The click event from the clear button
   */
  function clearFilters(event) {
    if (event) event.preventDefault();
    updateUrl({});
    applyFilters();
  }

  /**
   * Handles clicks on filter links (year or category).
   * Prevents default navigation and applies the filter client-side.
   * @param {Event} event - The click event
   */
  function handleFilterLink(event) {
    const link = event.target.closest("[data-filter-link]");
    if (!link) return;

    event.preventDefault();
    const filterType = link.getAttribute("data-filter-type");
    const filterValue = link.getAttribute("data-filter-value");
    let nextLang = null;
    try {
      nextLang = normalizeLang(
        new URL(link.href, window.location.origin).searchParams.get("lang"),
      );
    } catch (e) {
      nextLang = null;
    }
    if (filterType === "category" && filterValue === "") {
      updateUrl({});
      applyFilters();
      link.blur();
      return;
    }

    // Check if we are about to cancel
    const current = getFilters();
    let isCancelling = false;
    if (filterType === "year" && current.year === filterValue)
      isCancelling = true;
    if (filterType === "category" && current.category === filterValue)
      isCancelling = true;
    if (filterType === "tag" && current.tag === filterValue)
      isCancelling = true;

    if (filterType === "year") {
      updateUrl({ year: filterValue, lang: nextLang || current.lang });
    } else if (filterType === "category") {
      updateUrl({ category: filterValue, lang: nextLang || current.lang });
    } else if (filterType === "tag") {
      updateUrl({ tag: filterValue, lang: nextLang || current.lang });
    }
    applyFilters();

    // Remove focus from the clicked link to remove hover/focus effects only when cancelling
    if (isCancelling) {
      link.blur();
    }
  }

  // Per-row EN/中 lang tab toggle. Swaps visibility of panes within the same
  // .post-row without navigating. Clicks elsewhere are unaffected.
  function handleLangTab(event) {
    const tab = event.target.closest(".post-row__langtab");
    if (!tab) return;
    event.preventDefault();
    const pair = tab.getAttribute("data-lang-pair");
    const target = tab.getAttribute("data-lang-target");
    if (!pair || !target) return;
    document
      .querySelectorAll('.post-row__langtab[data-lang-pair="' + pair + '"]')
      .forEach((t) => {
        const active = t.getAttribute("data-lang-target") === target;
        t.classList.toggle("is-active", active);
        t.setAttribute("aria-selected", active ? "true" : "false");
      });
    document
      .querySelectorAll('.post-row__lang-pane[data-lang-pair="' + pair + '"]')
      .forEach((p) => {
        p.hidden = p.getAttribute("data-lang") !== target;
        p.classList.toggle("is-active", p.getAttribute("data-lang") === target);
      });
  }

  document.addEventListener("DOMContentLoaded", () => {
    applyFilters();

    document.addEventListener("click", handleFilterLink);
    document.addEventListener("click", handleLangTab);
    window.addEventListener("popstate", applyFilters);
  });
})();

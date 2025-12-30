/**
 * Blog filtering module for client-side year and category filtering.
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

  /**
   * Retrieves and validates current filter values from URL parameters.
   * @returns {{year: string|null, category: string|null}} Object containing validated year and category filters
   */
  function getFilters() {
    const params = new URLSearchParams(window.location.search);
    const rawYear = params.get('year');
    const rawCategory = params.get('category');
    return {
      year: normalizeYear(rawYear),
      category: normalizeCategory(rawCategory),
    };
  }

  /**
   * Updates the browser URL with the provided filter parameters.
   * Supports both year and category filters simultaneously.
   * Passing an empty object clears all active filters.
   * @param {Object} params - Object containing filter values to update
   * @param {string} [params.year] - Year filter value
   * @param {string} [params.category] - Category filter value
   */
  function updateUrl(params) {
    const url = new URL(window.location.href);
    
    // If no filter keys are provided, clear both filters (used by clearFilters).
    const hasYearParam = Object.prototype.hasOwnProperty.call(params, 'year');
    const hasCategoryParam = Object.prototype.hasOwnProperty.call(params, 'category');
    if (!hasYearParam && !hasCategoryParam) {
      url.searchParams.delete('year');
      url.searchParams.delete('category');
      history.replaceState({}, '', url);
      return;
    }

    // Merge current filters with the provided ones so year and category can coexist.
    const current = getFilters();
    const nextYear = hasYearParam ? params.year : current.year;
    const nextCategory = hasCategoryParam ? params.category : current.category;

    if (nextYear) {
      url.searchParams.set('year', nextYear);
    } else {
      url.searchParams.delete('year');
    }

    if (nextCategory) {
      url.searchParams.set('category', nextCategory);
    } else {
      url.searchParams.delete('category');
    }

    history.replaceState({}, '', url);
  }

  /**
   * Applies the current filters to blog posts, showing or hiding them as appropriate.
   * Updates the filter status banner with the active filters and post count.
   * Also manages the border styling to ensure the last visible post has no bottom border.
   */
  function applyFilters() {
    const { year, category } = getFilters();
    const posts = document.querySelectorAll('.post-list > li.blog-post-list-item');
    let visibleCount = 0;
    const totalCount = posts.length;
    let lastVisiblePost = null;

    posts.forEach((post) => {
      const matchesYear = !year || post.dataset.year === year;
      const categories = (post.dataset.categories || '').toLowerCase().split(/\s+/).filter(Boolean);
      const matchesCategory = !category || categories.includes(category);
      const isVisible = matchesYear && matchesCategory;

      post.style.display = isVisible ? '' : 'none';
      // Reset border for all posts
      post.style.borderBottom = '';
      
      if (isVisible) {
        visibleCount += 1;
        lastVisiblePost = post;
      }
    });

    // Remove border from the last visible post
    if (lastVisiblePost) {
      lastVisiblePost.style.borderBottom = 'none';
    }

    updateFilterStatus(year, category, visibleCount, totalCount);
  }

  /**
   * Updates or creates the filter status element with styling matching publications page.
   * @param {string|null} year - Active year filter
   * @param {string|null} category - Active category filter
   * @param {number} visibleCount - Number of visible posts
   * @param {number} totalCount - Total number of posts
   */
  function updateFilterStatus(year, category, visibleCount, totalCount) {
    let statusElement = document.getElementById('blog-filter-status');

    if (!statusElement) {
      // Create status element with inline styles matching publications page
      statusElement = document.createElement('div');
      statusElement.id = 'blog-filter-status';
      statusElement.setAttribute('role', 'status');
      statusElement.setAttribute('aria-live', 'polite');
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
      const messageSpan = document.createElement('span');
      messageSpan.id = 'blog-filter-message';
      statusElement.appendChild(messageSpan);

      // Create the clear button once with event listener
      const clearBtn = document.createElement('button');
      clearBtn.id = 'blog-filter-clear-btn';
      clearBtn.textContent = 'Clear Filter';
      clearBtn.setAttribute('aria-label', 'Clear active blog filters');
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
      clearBtn.addEventListener('click', clearFilters);
      statusElement.appendChild(clearBtn);

      // Insert before the first post list or featured posts section
      const postSection = document.querySelector('.post');
      if (postSection) {
        const featuredPosts = postSection.querySelector('.featured-posts');
        const postList = postSection.querySelector('.post-list');
        const insertBefore = featuredPosts || postList;
        if (insertBefore) {
          insertBefore.parentNode.insertBefore(statusElement, insertBefore);
        }
      }
    }

    const parts = [];
    if (year) parts.push(`Year: ${year}`);
    if (category) parts.push(`Category: ${category}`);

    if (parts.length) {
      statusElement.style.display = 'block';
      // Update only the message text, not the entire innerHTML
      const messageSpan = statusElement.querySelector('#blog-filter-message');
      if (messageSpan) {
        messageSpan.innerHTML = `Filtering by <strong>${parts.join(' and ')}</strong>: showing ${visibleCount} of ${totalCount} posts`;
      }
    } else {
      statusElement.style.display = 'none';
    }
  }

  /**
   * Clears all active filters and updates the UI.
   * @param {Event} event - The click event from the clear button
   */
  function clearFilters(event) {
    event.preventDefault();
    updateUrl({});
    applyFilters();
  }

  /**
   * Handles clicks on filter links (year or category).
   * Prevents default navigation and applies the filter client-side.
   * @param {Event} event - The click event
   */
  function handleFilterLink(event) {
    const link = event.target.closest('[data-filter-link]');
    if (!link) return;

    event.preventDefault();
    const filterType = link.getAttribute('data-filter-type');
    const filterValue = link.getAttribute('data-filter-value');

    if (filterType === 'year') {
      updateUrl({ year: filterValue });
    } else if (filterType === 'category') {
      updateUrl({ category: filterValue });
    }
    applyFilters();
  }

  document.addEventListener('DOMContentLoaded', () => {
    applyFilters();

    document.addEventListener('click', handleFilterLink);
    window.addEventListener('popstate', applyFilters);
  });
})();

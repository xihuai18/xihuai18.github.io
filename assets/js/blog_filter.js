(function () {
  function getFilters() {
    const params = new URLSearchParams(window.location.search);
    return {
      year: params.get('year'),
      category: params.get('category'),
    };
  }

  function updateUrl(params) {
    const url = new URL(window.location.href);
    if (params.year) {
      url.searchParams.set('year', params.year);
      url.searchParams.delete('category');
    } else if (params.category) {
      url.searchParams.set('category', params.category);
      url.searchParams.delete('year');
    } else {
      url.searchParams.delete('year');
      url.searchParams.delete('category');
    }
    history.replaceState({}, '', url);
  }

  function applyFilters() {
    const { year, category } = getFilters();
    const posts = document.querySelectorAll('.post-list > li.blog-post-list-item');
    let visibleCount = 0;

    posts.forEach((post) => {
      const matchesYear = !year || post.dataset.year === year;
      const categories = (post.dataset.categories || '').toLowerCase().split(/\s+/).filter(Boolean);
      const matchesCategory = !category || categories.includes(category.toLowerCase());
      const isVisible = matchesYear && matchesCategory;

      post.style.display = isVisible ? '' : 'none';
      if (isVisible) visibleCount += 1;
    });

    const statusElement = document.getElementById('blog-filter-status');
    if (!statusElement) return;

    const label = statusElement.querySelector('.filter-label');
    const parts = [];
    if (year) parts.push(`Year: ${year}`);
    if (category) parts.push(`Category: ${category}`);

    if (parts.length) {
      label.textContent = `${parts.join(' · ')} — ${visibleCount} post${visibleCount === 1 ? '' : 's'}`;
      statusElement.classList.remove('d-none');
    } else {
      statusElement.classList.add('d-none');
    }
  }

  function clearFilters(event) {
    event.preventDefault();
    updateUrl({});
    applyFilters();
  }

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

    const clearButton = document.getElementById('blog-filter-clear');
    if (clearButton) {
      clearButton.addEventListener('click', clearFilters);
    }

    document.addEventListener('click', handleFilterLink);
    window.addEventListener('popstate', applyFilters);
  });
})();

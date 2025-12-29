// Blog post enhancements (tables, etc.)
document.addEventListener('DOMContentLoaded', () => {
  // Wrap markdown tables for horizontal scrolling on narrow screens.
  document.querySelectorAll('.post-content table').forEach((table) => {
    // Skip tables that are already in a responsive wrapper or are part of special layouts.
    if (table.closest('.table-responsive, .news-table')) return;

    // Bootstrap table styling (non-destructive if already present).
    table.classList.add('table', 'table-sm');

    const wrapper = document.createElement('div');
    wrapper.className = 'table-responsive';

    table.parentNode.insertBefore(wrapper, table);
    wrapper.appendChild(table);
  });

  // Hide empty references blocks (e.g., references enabled but no citations).
  document.querySelectorAll('.post-references').forEach((section) => {
    const hasEntries = section.querySelectorAll('li').length > 0;
    if (!hasEntries) section.remove();
  });
});

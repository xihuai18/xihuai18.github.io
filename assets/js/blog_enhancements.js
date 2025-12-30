// Blog post enhancements (tables, code blocks, math, etc.)
document.addEventListener('DOMContentLoaded', () => {
  // ============================================================================
  // Fix HTML blocks that were incorrectly rendered as code blocks
  // ============================================================================
  document.querySelectorAll('.post-content .language-plaintext.highlighter-rouge').forEach((block) => {
    const codeEl = block.querySelector('code');
    if (!codeEl) return;
    
    const text = codeEl.textContent.trim();
    
    // Check if this looks like HTML that should be rendered
    const htmlTags = ['<img', '<figure', '<figcaption', '<div', '<iframe', '<video', '<audio'];
    const isHTML = htmlTags.some(tag => text.startsWith(tag));
    
    if (isHTML) {
      // Replace the code block with actual HTML
      const temp = document.createElement('div');
      temp.innerHTML = text;
      block.replaceWith(...temp.childNodes);
    }
  });

  // ============================================================================
  // Add language labels to code blocks
  // ============================================================================
  document.querySelectorAll('.post-content .highlighter-rouge').forEach((block) => {
    // Extract language from class name
    const classes = block.className.split(' ');
    const langClass = classes.find(c => c.startsWith('language-'));
    
    if (langClass && langClass !== 'language-plaintext') {
      const lang = langClass.replace('language-', '');
      block.setAttribute('data-lang', lang);
    }
  });

  // ============================================================================
  // Wrap markdown tables for horizontal scrolling on narrow screens
  // ============================================================================
  let wrappedTables = false;
  document.querySelectorAll('.post-content table').forEach((table) => {
    // Skip tables that are already in a responsive wrapper or are part of special layouts.
    if (table.closest('.table-responsive, .news-table')) return;

    // Bootstrap table styling (non-destructive if already present).
    table.classList.add('table', 'table-sm');

    const wrapper = document.createElement('div');
    wrapper.className = 'table-responsive';

    table.parentNode.insertBefore(wrapper, table);
    wrapper.appendChild(table);
    wrappedTables = true;
  });
  if (wrappedTables) {
    window.requestMathTypeset?.(document.querySelector('.post-content'));
  }

  // ============================================================================
  // Hide empty references blocks (e.g., references enabled but no citations)
  // ============================================================================
  document.querySelectorAll('.post-references').forEach((section) => {
    const hasEntries = section.querySelectorAll('li').length > 0;
    if (!hasEntries) section.remove();
  });

  // ============================================================================
  // Fix math rendering issues caused by HTML processing
  // ============================================================================
  document.querySelectorAll('.post-content').forEach((content) => {
    // Find inline math that might have been corrupted
    const walker = document.createTreeWalker(content, NodeFilter.SHOW_TEXT, null, false);
    const nodesToFix = [];
    
    while (walker.nextNode()) {
      const text = walker.currentNode.textContent;
      // Check for common corruption patterns
      if (text.includes('<em>') || text.includes('</em>')) {
        nodesToFix.push(walker.currentNode);
      }
    }
    
    // Note: Most fixes are handled server-side by the Ruby plugins
    // This is just a fallback for edge cases
  });

  // ============================================================================
  // Blog index: bilingual tab switcher (English / 简体中文)
  // ============================================================================
  // No caching - always shows default English tab on page load.
  // Supports click and keyboard (arrow keys) navigation for accessibility.
  const activateTab = (tabLink) => {
    const postListItem = tabLink.closest('.post-list > li');
    if (!postListItem) return;

    const tabList = tabLink.closest('[role="tablist"]');
    const tabLinks = tabList
      ? Array.from(tabList.querySelectorAll('a[role="tab"]'))
      : Array.from(postListItem.querySelectorAll('.lang-switcher a[role="tab"]'));

    tabLinks.forEach((link) => {
      link.classList.remove('active');
      link.setAttribute('aria-selected', 'false');
      link.setAttribute('tabindex', '-1');
    });

    tabLink.classList.add('active');
    tabLink.setAttribute('aria-selected', 'true');
    tabLink.setAttribute('tabindex', '0');

    const tabContent = postListItem.querySelector('.tab-content');
    if (!tabContent) return;

    const targetSel = tabLink.getAttribute('href') || '';
    let targetPane = null;
    if (targetSel.startsWith('#')) {
      targetPane = tabContent.querySelector(targetSel);
    }
    if (!targetPane) {
      const controls = tabLink.getAttribute('aria-controls');
      if (controls) {
        const el = document.getElementById(controls);
        if (el && tabContent.contains(el)) targetPane = el;
      }
    }
    if (!targetPane) return;

    tabContent.querySelectorAll('.tab-pane').forEach((pane) => {
      pane.classList.remove('show', 'active');
      pane.style.display = 'none';
      pane.setAttribute('aria-hidden', 'true');
    });
    targetPane.classList.add('show', 'active');
    targetPane.style.display = '';
    targetPane.setAttribute('aria-hidden', 'false');
    window.requestMathTypeset?.(targetPane);
  };

  document.addEventListener(
    'click',
    (e) => {
      const tabLink = e.target.closest('.lang-switcher a[role="tab"]');
      if (!tabLink) return;
      e.preventDefault();
      e.stopPropagation();
      activateTab(tabLink);
    },
    true
  );

  // Ensure initial state is consistent (always English first).
  document.querySelectorAll('.lang-switcher [role="tablist"]').forEach((tabList) => {
    const active = tabList.querySelector('a[role="tab"].active') || tabList.querySelector('a[role="tab"]');
    if (active) activateTab(active);
  });

  document.addEventListener(
    'keydown',
    (e) => {
      const tabLink = e.target.closest('.lang-switcher a[role="tab"]');
      if (!tabLink) return;
      if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return;

      const tabList = tabLink.closest('[role="tablist"]');
      if (!tabList) return;
      const tabs = Array.from(tabList.querySelectorAll('a[role="tab"]'));
      const idx = tabs.indexOf(tabLink);
      if (idx < 0) return;

      e.preventDefault();
      e.stopPropagation();

      const nextIdx = e.key === 'ArrowLeft' ? idx - 1 : idx + 1;
      const next = tabs[(nextIdx + tabs.length) % tabs.length];
      next.focus();
      activateTab(next);
    },
    true
  );
});

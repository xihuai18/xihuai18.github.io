function getMermaidTheme() {
  const theme = document.documentElement.getAttribute('data-theme');
  return theme === 'dark' ? 'dark' : 'default';
}

function normalizeMermaidBlocks(root) {
  const codeBlocks = root.querySelectorAll(
    [
      'pre > code.language-mermaid',
      'pre > code.mermaid',
      'pre.language-mermaid > code',
      '.language-mermaid pre > code',
      '.language-mermaid code',
      'code[data-lang="mermaid"]',
    ].join(', ')
  );

  codeBlocks.forEach((code) => {
    // Avoid double-processing if the code block is already replaced.
    if (!code.isConnected) return;
    const definition = code.textContent.trim();
    if (!definition) return;

    const languageWrapper = code.closest('.language-mermaid');
    const highlighterWrapper = code.closest('.highlighter-rouge');
    const figureHighlight = code.closest('figure.highlight');
    const divHighlight = code.closest('div.highlight');
    const pre = code.closest('pre');

    const replacementTarget =
      (languageWrapper && (languageWrapper.tagName === 'DIV' || languageWrapper.tagName === 'FIGURE') && languageWrapper) ||
      (highlighterWrapper && (highlighterWrapper.tagName === 'DIV' || highlighterWrapper.tagName === 'FIGURE') && highlighterWrapper) ||
      figureHighlight ||
      divHighlight ||
      pre;
    if (!replacementTarget) return;

    const container = document.createElement('div');
    container.className = 'mermaid';
    container.dataset.mermaid = definition;
    container.textContent = definition;

    replacementTarget.parentNode.replaceChild(container, replacementTarget);
  });
}

async function renderMermaid(root) {
  if (!window.mermaid) return;

  normalizeMermaidBlocks(root);

  const theme = getMermaidTheme();
  mermaid.initialize({
    startOnLoad: false,
    theme,
    securityLevel: 'strict',
  });

  root.querySelectorAll('.mermaid').forEach((el) => {
    const definition = el.dataset.mermaid;
    if (!definition) return;
    el.textContent = definition;
    el.removeAttribute('data-processed');
  });

  try {
    await mermaid.run({ querySelector: '.mermaid' });
  } catch (e) {
    // Leave the source text in place if rendering fails.
  }
}

document.addEventListener('DOMContentLoaded', () => {
  renderMermaid(document);
});

document.addEventListener('theme:change', () => {
  renderMermaid(document);
});

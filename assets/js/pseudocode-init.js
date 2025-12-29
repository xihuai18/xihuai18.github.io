(() => {
  const selector = 'pre > code.language-pseudocode, pre > code.language-pseudo';

  const waitForMathJax = async () => {
    const mj = window.MathJax;
    if (!mj) return null;

    // MathJax v3 exposes `startup.promise` once the runtime script loads.
    if (mj.startup?.promise) {
      try {
        await mj.startup.promise;
      } catch {
        return null;
      }
    }

    if (typeof mj.typesetPromise === 'function') return mj;
    return null;
  };

  const typeset = async (elements) => {
    const mj = await waitForMathJax();
    if (!mj) return;
    try {
      await mj.typesetPromise(elements);
    } catch {
      // Non-fatal: pseudocode rendering should still succeed.
    }
  };

  const renderAll = async () => {
    if (!window.pseudocode || typeof window.pseudocode.renderElement !== 'function') return;

    const codeNodes = Array.from(document.querySelectorAll(selector));
    if (codeNodes.length === 0) return;

    const rendered = [];

    for (const codeNode of codeNodes) {
      const source = codeNode.textContent ?? '';
      const pre = codeNode.closest('pre');
      if (!pre) continue;

      const wrapper = pre.parentElement;
      if (!wrapper) continue;

      const container = document.createElement('pre');
      container.classList.add('pseudocode');
      container.appendChild(document.createTextNode(source));

      // Only remove the original code block after a successful render; otherwise
      // keep the original code block visible.
      wrapper.insertBefore(container, pre.nextSibling);
      try {
        window.pseudocode.renderElement(container);
        pre.remove();
        rendered.push(container);
      } catch (err) {
        container.remove();
        // eslint-disable-next-line no-console
        console.warn('[pseudocode] Failed to render a block:', err);
      }
    }

    await typeset(rendered);
  };

  if (document.readyState === 'complete' || document.readyState === 'interactive') {
    void renderAll();
  } else {
    document.addEventListener('DOMContentLoaded', () => void renderAll(), { once: true });
  }
})();

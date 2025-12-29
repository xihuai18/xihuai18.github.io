function createFootnotePreviewEl() {
  const el = document.createElement('div');
  el.className = 'footnote-preview';
  el.setAttribute('role', 'tooltip');
  el.hidden = true;
  document.body.appendChild(el);
  return el;
}

function sanitizeFootnoteHtml(html) {
  const tmp = document.createElement('div');
  tmp.innerHTML = html;
  tmp.querySelectorAll('a.reversefootnote, a.footnote-backlink').forEach((a) => a.remove());
  return tmp.innerHTML.trim();
}

function positionTooltip(tooltip, clientX, clientY) {
  const padding = 12;
  const offset = 14;

  tooltip.style.left = '0px';
  tooltip.style.top = '0px';
  tooltip.hidden = false;

  const rect = tooltip.getBoundingClientRect();
  const vw = window.innerWidth;
  const vh = window.innerHeight;

  let left = clientX + offset;
  let top = clientY + offset;

  if (left + rect.width + padding > vw) left = Math.max(padding, clientX - rect.width - offset);
  if (top + rect.height + padding > vh) top = Math.max(padding, clientY - rect.height - offset);

  tooltip.style.left = `${left}px`;
  tooltip.style.top = `${top}px`;
}

function initFootnotePreview(root = document) {
  const tooltip = createFootnotePreviewEl();
  let activeRef = null;

  const showForRef = (ref, event) => {
    const href = ref.getAttribute('href') || '';
    if (!href.startsWith('#')) return;
    const targetId = href.slice(1);
    const target = document.getElementById(targetId);
    if (!target) return;

    const html = sanitizeFootnoteHtml(target.innerHTML);
    if (!html) return;

    tooltip.innerHTML = html;
    activeRef = ref;

    const x = event?.clientX ?? ref.getBoundingClientRect().left;
    const y = event?.clientY ?? ref.getBoundingClientRect().bottom;
    positionTooltip(tooltip, x, y);
  };

  const hide = () => {
    tooltip.hidden = true;
    activeRef = null;
  };

  root.querySelectorAll('a.footnote[href^="#"]').forEach((ref) => {
    ref.addEventListener(
      'mouseenter',
      (e) => {
        showForRef(ref, e);
      },
      { passive: true }
    );

    ref.addEventListener(
      'mousemove',
      (e) => {
        if (activeRef !== ref || tooltip.hidden) return;
        positionTooltip(tooltip, e.clientX, e.clientY);
      },
      { passive: true }
    );

    ref.addEventListener('mouseleave', hide, { passive: true });

    ref.addEventListener('focusin', (e) => showForRef(ref, e), { passive: true });
    ref.addEventListener('focusout', hide, { passive: true });
  });

  window.addEventListener('scroll', hide, { passive: true });
  window.addEventListener('resize', hide, { passive: true });
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') hide();
  });
}

document.addEventListener('DOMContentLoaded', () => {
  initFootnotePreview(document);
});


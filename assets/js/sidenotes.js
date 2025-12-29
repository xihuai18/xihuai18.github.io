function buildSidenotes(post) {
  const content = post.querySelector('.post-content');
  const rail = post.querySelector('.sidenotes-rail');
  if (!content || !rail) return;

  const footnotesBlock = content.querySelector('.footnotes');
  if (!footnotesBlock) return;

  const listItems = footnotesBlock.querySelectorAll('ol > li[id]');
  if (!listItems.length) return;

  const notesById = new Map();
  listItems.forEach((li) => {
    const clone = li.cloneNode(true);
    clone.querySelectorAll('a.reversefootnote, a.footnote-backlink').forEach((a) => a.remove());
    notesById.set(li.id, clone.innerHTML.trim());
  });

  const refs = Array.from(content.querySelectorAll('a.footnote[href^=\"#\"]'));
  if (!refs.length) return;

  rail.innerHTML = '';

  const sidenotes = refs
    .map((ref) => {
      const targetId = (ref.getAttribute('href') || '').slice(1);
      const html = notesById.get(targetId);
      if (!html) return null;

      const note = document.createElement('aside');
      note.className = 'sidenote';
      note.dataset.footnoteTarget = targetId;
      note.innerHTML = html;
      rail.appendChild(note);

      ref.dataset.footnoteTarget = targetId;
      ref.classList.add('js-footnote-ref');

      return { ref, note };
    })
    .filter(Boolean);

  if (!sidenotes.length) return;

  const gapPx = 12;

  const position = () => {
    const railRect = rail.getBoundingClientRect();
    let lastBottom = -Infinity;

    sidenotes
      .map((pair) => {
        const rr = pair.ref.getBoundingClientRect();
        return { ...pair, top: rr.top - railRect.top };
      })
      .sort((a, b) => a.top - b.top)
      .forEach((pair) => {
        const top = Math.max(pair.top, lastBottom + gapPx);
        pair.note.style.top = `${Math.max(0, top)}px`;
        lastBottom = top + pair.note.offsetHeight;
      });
  };

  position();
  window.addEventListener('resize', position, { passive: true });
  window.addEventListener('load', position, { passive: true });

  content.addEventListener('click', (e) => {
    const link = e.target.closest('a.js-footnote-ref');
    if (!link) return;
    const targetId = link.dataset.footnoteTarget;
    const note = rail.querySelector(`.sidenote[data-footnote-target=\"${CSS.escape(targetId)}\"]`);
    if (!note) return;

    e.preventDefault();
    rail.querySelectorAll('.sidenote.is-active').forEach((el) => el.classList.remove('is-active'));
    note.classList.add('is-active');
    note.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
  });
}

document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('.post.post--sidenotes').forEach(buildSidenotes);
});

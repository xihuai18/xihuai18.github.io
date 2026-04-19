// Copy-BibTeX from the "Cite this post" footer. Attaches to `.post-cite__copy`;
// reads the paired `<pre>` by `data-copy-target` selector and drops its text
// into the clipboard. Keeps a timer ref on the button so rapid clicks don't
// leave the "is-copied" state stranded.

(function () {
  function copyFrom(trigger) {
    var sel = trigger.getAttribute("data-copy-target");
    if (!sel) return;
    var src = document.querySelector(sel);
    if (!src) return;
    var text = (src.textContent || "").replace(/^\s+|\s+$/g, "");
    if (!text) return;

    var label = trigger.querySelector(".post-cite__copy-label");
    var originalLabel = label ? label.textContent : null;
    var copiedLabel = trigger.getAttribute("data-copied-label") || "Copied";

    function flash(ok) {
      if (trigger.__citeFlashTimer) {
        clearTimeout(trigger.__citeFlashTimer);
        trigger.__citeFlashTimer = null;
      }
      trigger.classList.toggle("is-copied", !!ok);
      trigger.classList.toggle("is-failed", !ok);
      if (label) label.textContent = ok ? copiedLabel : (originalLabel || "");
      trigger.__citeFlashTimer = setTimeout(function () {
        trigger.classList.remove("is-copied", "is-failed");
        if (label && originalLabel !== null) label.textContent = originalLabel;
        trigger.__citeFlashTimer = null;
      }, 1400);
    }

    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(text).then(function () { flash(true); }, fallback);
    } else {
      fallback();
    }

    function fallback() {
      var ok = false;
      try {
        var ta = document.createElement("textarea");
        ta.value = text;
        ta.setAttribute("readonly", "");
        ta.style.position = "fixed";
        ta.style.top = "-9999px";
        document.body.appendChild(ta);
        ta.select();
        ok = document.execCommand("copy");
        document.body.removeChild(ta);
      } catch (e) {
        ok = false;
      }
      flash(ok);
    }
  }

  function onClick(e) {
    var btn = e.target.closest && e.target.closest(".post-cite__copy");
    if (!btn) return;
    e.preventDefault();
    copyFrom(btn);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", function () {
      document.addEventListener("click", onClick);
    });
  } else {
    document.addEventListener("click", onClick);
  }
})();

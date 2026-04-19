// Copy-BibTeX pill: clicking a .bibtex-copy button copies its paired
// .bibtex-source text to the clipboard and briefly confirms with an
// "is-copied" class + changed label. Falls back gracefully if the
// Clipboard API is unavailable (older browsers / non-HTTPS contexts).

(function () {
  function copyBibtex(trigger) {
    var key = trigger.getAttribute("data-entry-key");
    if (!key) return;
    // Find the matching source within the same bib entry
    var scope = trigger.closest("[id='" + key + "']") || trigger.closest(".row") || document;
    var source =
      scope.querySelector(".bibtex-source[data-entry-key='" + key + "']") ||
      document.querySelector(".bibtex-source[data-entry-key='" + key + "']");
    if (!source) return;
    var text = (source.textContent || "").replace(/^\s+|\s+$/g, "");
    if (!text) return;

    var label = trigger.querySelector(".bibtex-copy-label");
    var originalLabel = label ? label.textContent : null;

    var flashState = function (state) {
      // Cancel any pending revert from a prior rapid click so the new flash
      // holds its full 1400ms and the label can't snap back mid-animation.
      if (trigger.__bibFlashTimer) {
        clearTimeout(trigger.__bibFlashTimer);
        trigger.__bibFlashTimer = null;
      }
      trigger.classList.remove("is-copied", "is-failed");
      trigger.classList.add(state === "ok" ? "is-copied" : "is-failed");
      if (label) label.textContent = state === "ok" ? "Copied" : "Copy failed";
      trigger.__bibFlashTimer = setTimeout(function () {
        trigger.classList.remove("is-copied", "is-failed");
        if (label && originalLabel !== null) label.textContent = originalLabel;
        trigger.__bibFlashTimer = null;
      }, 1400);
    };

    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(text).then(
        function () { flashState("ok"); },
        function () { fallback(); }
      );
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
        // document.execCommand returns false if the copy fails (or is blocked
        // by the browser in insecure contexts).
        ok = document.execCommand("copy");
        document.body.removeChild(ta);
      } catch (e) {
        ok = false;
      }
      flashState(ok ? "ok" : "fail");
    }
  }

  function onClick(e) {
    var el = e.target.closest && e.target.closest(".bibtex-copy");
    if (!el) return;
    e.preventDefault();
    copyBibtex(el);
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", function () {
      document.addEventListener("click", onClick);
    });
  } else {
    document.addEventListener("click", onClick);
  }
})();

document.addEventListener("DOMContentLoaded", function () {
  // Note: legacy `.abstract` and `.bibtex` inline toggles were removed along
  // with the markup that produced them. See _layouts/bib.html + bibtex_copy.js.

  // --------------------------------------------------------------------------
  // Mobile navbar collapse (replaces Bootstrap JS; markup keeps BS4 classes)
  // --------------------------------------------------------------------------
  var toggler = document.querySelector(".navbar-toggler");
  if (toggler) {
    var targetSel = toggler.getAttribute("data-target") || "#navbarNav";
    var target = document.querySelector(targetSel);
    if (target) {
      toggler.addEventListener("click", function () {
        var open = target.classList.toggle("show");
        toggler.classList.toggle("collapsed", !open);
        toggler.setAttribute("aria-expanded", open ? "true" : "false");
      });
    }
  }

  // --------------------------------------------------------------------------
  // Blog views-mode toggle (all time / 30 days)
  // --------------------------------------------------------------------------
  function updateUvMode(mode) {
    var key = mode === "d30" ? "uvD30" : "uvAll";
    document.querySelectorAll(".post-uv").forEach(function (el) {
      var value = el.dataset[key];
      if (value === undefined || value === null || value === "") {
        value = 0;
      }
      // Allow per-element label override (e.g. "阅读" for ZH rows).
      var label = el.dataset.uvLabel || "views";
      el.textContent = value + " " + label;
    });

    document.querySelectorAll(".uv-toggle").forEach(function (el) {
      var active = el.dataset.uvMode === mode;
      el.classList.toggle("is-active", active);
      el.classList.remove("font-weight-bold");
      el.setAttribute("aria-pressed", active ? "true" : "false");
    });
  }

  var savedMode = null;
  try {
    savedMode = window.localStorage.getItem("uvMode");
  } catch (e) {
    savedMode = null;
  }
  if (savedMode !== "all" && savedMode !== "d30") {
    savedMode = "all";
  }
  updateUvMode(savedMode);

  document.querySelectorAll(".uv-toggle").forEach(function (el) {
    el.addEventListener("click", function (e) {
      e.preventDefault();
      var mode = el.dataset.uvMode;
      if (mode !== "all" && mode !== "d30") {
        return;
      }
      try {
        window.localStorage.setItem("uvMode", mode);
      } catch (e) {}
      updateUvMode(mode);
    });
  });
});

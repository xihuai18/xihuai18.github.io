$(document).ready(function () {
  // Note: legacy `.abstract` and `.bibtex` inline toggles were removed along
  // with the markup that produced them. See _layouts/bib.html + bibtex_copy.js.

  $("a").removeClass("waves-effect waves-light");

  function updateUvMode(mode) {
    var key = mode === "d30" ? "uvD30" : "uvAll";
    $(".post-uv").each(function () {
      var value = $(this).data(key);
      if (value === undefined || value === null || value === "") {
        value = 0;
      }
      // Allow per-element label override (e.g. "阅读" for ZH rows).
      var label = $(this).data("uvLabel") || "views";
      $(this).text(value + " " + label);
    });

    $(".uv-toggle").removeClass("font-weight-bold is-active");
    $('.uv-toggle[data-uv-mode="' + mode + '"]').addClass("is-active");
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

  $(".uv-toggle").click(function (e) {
    e.preventDefault();
    var mode = $(this).data("uvMode");
    if (mode !== "all" && mode !== "d30") {
      return;
    }
    try {
      window.localStorage.setItem("uvMode", mode);
    } catch (e) {}
    updateUvMode(mode);
  });
});

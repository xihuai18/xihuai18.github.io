function safeInsertRule(sheet, rule) {
  if (!sheet) return;
  try {
    sheet.insertRule(rule);
  } catch (e) {}
}

function overrideDistillFootnotes() {
  document.querySelectorAll("d-footnote").forEach((footnote) => {
    try {
      const root = footnote.shadowRoot;
      if (!root) return;

      const sup = root.querySelector("sup > span");
      if (sup) sup.setAttribute("style", "color: var(--global-theme-color);");

      const hoverStyle = root.querySelector("d-hover-box")?.shadowRoot?.querySelector("style");
      const sheet = hoverStyle?.sheet;
      safeInsertRule(sheet, ".panel {background-color: var(--global-bg-color) !important;}");
      safeInsertRule(sheet, ".panel {border-color: var(--global-divider-color) !important;}");
    } catch (e) {}
  });
}

function overrideDistillCitations() {
  document.querySelectorAll("d-cite").forEach((cite) => {
    try {
      const root = cite.shadowRoot;
      if (!root) return;

      const span = root.querySelector("div > span");
      if (span) span.setAttribute("style", "color: var(--global-theme-color);");

      const style = root.querySelector("style");
      const sheet = style?.sheet;
      safeInsertRule(sheet, "ul li a {color: var(--global-text-color) !important; text-decoration: none;}");
      safeInsertRule(sheet, "ul li a:hover {color: var(--global-theme-color) !important;}");

      const hoverStyle = root.querySelector("d-hover-box")?.shadowRoot?.querySelector("style");
      const hoverSheet = hoverStyle?.sheet;
      safeInsertRule(hoverSheet, ".panel {background-color: var(--global-bg-color) !important;}");
      safeInsertRule(hoverSheet, ".panel {border-color: var(--global-divider-color) !important;}");
    } catch (e) {}
  });
}

function applyDistillOverrides() {
  overrideDistillFootnotes();
  overrideDistillCitations();
}

document.addEventListener("DOMContentLoaded", () => {
  applyDistillOverrides();
  // Give web components a moment to upgrade / render.
  window.setTimeout(applyDistillOverrides, 50);
});

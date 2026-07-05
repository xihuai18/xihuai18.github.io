// Blog post enhancements (tables, code blocks, back-to-top, etc.)
document.addEventListener("DOMContentLoaded", () => {
  // ============================================================================
  // Back to Top Button
  // ============================================================================
  const createBackToTopButton = () => {
    // Only add on post pages or CV page
    if (
      !document.querySelector(".post-content") &&
      !document.querySelector(".cv")
    )
      return;
    const isZh = (document.documentElement.lang || "")
      .toLowerCase()
      .startsWith("zh");

    const btn = document.createElement("button");
    btn.className = "back-to-top";
    btn.setAttribute("aria-label", isZh ? "返回顶部" : "Back to top");
    // Use createElement instead of innerHTML for security
    const icon = document.createElement("i");
    icon.classList.add("fas", "fa-arrow-up");
    btn.appendChild(icon);
    document.body.appendChild(btn);

    const toggleVisibility = () => {
      if (window.scrollY > 300) {
        btn.classList.add("is-visible");
      } else {
        btn.classList.remove("is-visible");
      }
    };

    window.addEventListener("scroll", toggleVisibility, { passive: true });

    btn.addEventListener("click", () => {
      // Respect user's motion preferences
      const prefersReducedMotion = window.matchMedia
        ? window.matchMedia("(prefers-reduced-motion: reduce)").matches
        : false;
      window.scrollTo({
        top: 0,
        behavior: prefersReducedMotion ? "auto" : "smooth",
      });
    });
  };

  createBackToTopButton();

  // ============================================================================
  // Fix HTML blocks that were incorrectly rendered as code blocks
  // ============================================================================
  document
    .querySelectorAll(".post-content .language-plaintext.highlighter-rouge")
    .forEach((block) => {
      const codeEl = block.querySelector("code");
      if (!codeEl) return;

      const text = codeEl.textContent.trim();

      // Check if this looks like HTML that should be rendered
      const htmlTags = [
        "<img",
        "<figure",
        "<figcaption",
        "<div",
        "<iframe",
        "<video",
        "<audio",
      ];
      const isHTML = htmlTags.some((tag) => text.startsWith(tag));

      if (isHTML) {
        // Replace the code block with actual HTML
        const temp = document.createElement("div");
        temp.innerHTML = text;
        block.replaceWith(...temp.childNodes);
      }
    });

  // ============================================================================
  // Add language labels to code blocks
  // ============================================================================
  document
    .querySelectorAll(".post-content .highlighter-rouge")
    .forEach((block) => {
      // Extract language from class name
      const classes = block.className.split(" ");
      const langClass = classes.find((c) => c.startsWith("language-"));

      if (langClass && langClass !== "language-plaintext") {
        const lang = langClass.replace("language-", "");
        block.setAttribute("data-lang", lang);
      }
    });

  // ============================================================================
  // Wrap markdown tables for horizontal scrolling on narrow screens
  // ============================================================================
  document.querySelectorAll(".post-content table").forEach((table) => {
    // Skip tables that are already in a responsive wrapper or are part of special layouts.
    if (table.closest(".table-responsive, .news-table")) return;

    // Bootstrap table styling (non-destructive if already present).
    table.classList.add("table", "table-sm");

    const wrapper = document.createElement("div");
    wrapper.className = "table-responsive";

    table.parentNode.insertBefore(wrapper, table);
    wrapper.appendChild(table);
  });

  // ============================================================================
  // Hide empty references blocks (e.g., references enabled but no citations)
  // ============================================================================
  document.querySelectorAll(".post-references").forEach((section) => {
    const hasEntries = section.querySelectorAll("li").length > 0;
    if (!hasEntries) section.remove();
  });
});

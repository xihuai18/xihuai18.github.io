// Venue filter functionality for publications
let currentVenueFilter = null;

function getPublicationsPage() {
  return document.querySelector("[data-publications-page]");
}

function updateYearCounts(root) {
  if (!root) return;

  root.querySelectorAll("h2.year").forEach((header) => {
    const countElement = header.querySelector(".year-count");
    if (!countElement) return;

    const list = header.nextElementSibling;
    if (
      !list ||
      list.tagName !== "OL" ||
      !list.classList.contains("bibliography")
    )
      return;

    const visibleCount = getVisiblePublications(list).length;
    countElement.textContent =
      visibleCount + " " + (visibleCount === 1 ? "paper" : "papers");
  });
}

// Public: clear the active filter (used by the "All papers" topic chip).
// If no filter is active, this is a no-op.
function clearVenueFilter() {
  if (currentVenueFilter) filterByVenue(currentVenueFilter);
}

// Update topic pill active state to mirror the venue-tag state. Kept separate
// from filterByVenue's venue-tag loop so chip-bar styling stays self-contained.
function syncTopicPills(activeAbbr) {
  document.querySelectorAll(".pub-topic-pill").forEach((pill) => {
    const v = pill.getAttribute("data-topic-filter") || "";
    const isAll = v === "";
    const isActive = isAll ? !activeAbbr : v === activeAbbr;
    pill.classList.toggle("is-active", isActive);
    pill.setAttribute("aria-pressed", isActive ? "true" : "false");
  });
}

function filterByVenue(venue, options = {}) {
  const publicationsPage = getPublicationsPage();
  if (!publicationsPage) return;
  const skipHistoryUpdate = Boolean(options.skipHistoryUpdate);
  const skipScroll = Boolean(options.skipScroll);

  // Select all publication list items
  const publications = publicationsPage.querySelectorAll(
    "ol.bibliography > li",
  );

  if (publications.length === 0) {
    console.warn("No publications found to filter");
    return;
  }

  if (
    venue &&
    !Array.from(publications).some((pub) => {
      const venueElement = pub.querySelector("[data-venue]");
      return venueElement && venueElement.getAttribute("data-venue") === venue;
    })
  ) {
    currentVenueFilter = null;
    publications.forEach((pub) => {
      pub.style.display = "";
      pub.style.opacity = "1";
    });
    publicationsPage.querySelectorAll(".venue-tag").forEach((tag) => {
      tag.classList.remove("active-filter");
      tag.setAttribute("aria-pressed", "false");
    });
    publicationsPage.querySelectorAll("h2.year").forEach((header) => {
      header.style.display = "";
    });
    updateYearCounts(publicationsPage);
    updatePublicationDividers();
    updateFilterStatus(null, 0, publications.length);
    syncTopicPills(null);
    if (skipHistoryUpdate) {
      const url = new URL(window.location);
      url.searchParams.delete("venue");
      window.history.replaceState({}, "", url);
    } else {
      updateURL(null);
    }
    return;
  }

  publications.forEach((pub) => {
    if (pub.__venueHideTimer) {
      clearTimeout(pub.__venueHideTimer);
      pub.__venueHideTimer = null;
    }
  });

  // If clicking the same venue tag, reset filter
  if (currentVenueFilter === venue) {
    currentVenueFilter = null;
    publications.forEach((pub) => {
      pub.style.display = "";
      pub.style.opacity = "1";
    });

    // Remove active state from all tags
    publicationsPage.querySelectorAll(".venue-tag").forEach((tag) => {
      tag.classList.remove("active-filter");
      tag.setAttribute("aria-pressed", "false");
    });

    // Remove focus from active element to remove hover/focus effects
    if (document.activeElement) {
      document.activeElement.blur();
    }

    // Show all year headers
    publicationsPage.querySelectorAll("h2.year").forEach((header) => {
      header.style.display = "";
    });

    updateYearCounts(publicationsPage);

    // Reset all dividers
    updatePublicationDividers();

    // Hide filter status
    updateFilterStatus(null, 0, publications.length);

    // Sync topic-pill chip bar
    syncTopicPills(null);

    // Update URL
    if (!skipHistoryUpdate) updateURL(null);

    return;
  }

  // Set new filter
  currentVenueFilter = venue;

  // Filter publications
  let visibleCount = 0;
  publications.forEach((pub) => {
    // Find the data-venue attribute within the li element
    const venueElement = pub.querySelector("[data-venue]");
    const pubVenue = venueElement
      ? venueElement.getAttribute("data-venue")
      : null;

    if (pubVenue === venue) {
      pub.style.display = "";
      pub.style.opacity = "1";
      pub.style.transition = "opacity 0.3s ease";
      visibleCount++;
    } else {
      pub.style.opacity = "0";
      pub.__venueHideTimer = setTimeout(() => {
        pub.style.display = "none";
        pub.__venueHideTimer = null;
      }, 300);
    }
  });

  // Hide year headers that have no visible publications and update dividers
  setTimeout(() => {
    hideEmptyYearHeaders();
    updateYearCounts(publicationsPage);
    updatePublicationDividers();
  }, 350); // Wait for fade out animation to complete

  // Update active state of tags
  publicationsPage.querySelectorAll(".venue-tag").forEach((tag) => {
    if (tag.getAttribute("data-venue-filter") === venue) {
      tag.classList.add("active-filter");
      tag.setAttribute("aria-pressed", "true");
    } else {
      tag.classList.remove("active-filter");
      tag.setAttribute("aria-pressed", "false");
    }
  });

  // Show filter status
  updateFilterStatus(venue, visibleCount, publications.length);

  // Sync topic-pill chip bar
  syncTopicPills(venue);

  // Update URL
  if (!skipHistoryUpdate) updateURL(venue);

  // Scroll to top of publications section — respect reduced-motion preference.
  const pubSection = publicationsPage;
  if (pubSection && !skipScroll) {
    const prefersReduced =
      window.matchMedia &&
      window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    pubSection.scrollIntoView({
      behavior: prefersReduced ? "auto" : "smooth",
      block: "start",
    });
  }
}

function hideEmptyYearHeaders() {
  const publicationsPage = getPublicationsPage();
  if (!publicationsPage) return;

  const yearHeaders = publicationsPage.querySelectorAll("h2.year");

  yearHeaders.forEach((header) => {
    // Find the bibliography list following this year header
    let nextElement = header.nextElementSibling;
    let hasVisiblePubs = false;

    // Check if the next bibliography list has any visible publications
    if (
      nextElement &&
      nextElement.tagName === "OL" &&
      nextElement.classList.contains("bibliography")
    ) {
      const visiblePubs = Array.from(nextElement.querySelectorAll("li")).filter(
        (li) => {
          const computedStyle = window.getComputedStyle(li);
          return (
            computedStyle.display !== "none" && computedStyle.opacity !== "0"
          );
        },
      );
      hasVisiblePubs = visiblePubs.length > 0;
    }

    // Hide or show the year header based on whether it has visible publications
    if (hasVisiblePubs) {
      header.style.display = "";
    } else {
      header.style.display = "none";
    }
  });
}

function getVisiblePublications(listElement) {
  return Array.from(listElement.querySelectorAll("li")).filter((li) => {
    const computedStyle = window.getComputedStyle(li);
    return computedStyle.display !== "none" && computedStyle.opacity !== "0";
  });
}

function updatePublicationDividers() {
  const publicationsSection = getPublicationsPage();
  if (!publicationsSection) return;

  const yearHeaders = publicationsSection.querySelectorAll("h2.year");

  yearHeaders.forEach((header) => {
    const nextElement = header.nextElementSibling;

    if (
      !nextElement ||
      nextElement.tagName !== "OL" ||
      !nextElement.classList.contains("bibliography")
    ) {
      return;
    }

    const visiblePubs = getVisiblePublications(nextElement);

    nextElement
      .querySelectorAll("li")
      .forEach((li) => li.classList.remove("single-pub-in-year"));

    if (visiblePubs.length === 1) {
      visiblePubs[0].classList.add("single-pub-in-year");
    }
  });
}

function updateURL(venue) {
  if (!window.history || !window.history.pushState) return;

  try {
    const url = new URL(window.location);
    if (venue) {
      url.searchParams.set("venue", venue);
    } else {
      url.searchParams.delete("venue");
    }
    window.history.pushState({}, "", url);
  } catch (error) {
    console.warn("Failed to update URL:", error);
  }
}

function updateFilterStatus(venue, visibleCount, totalCount) {
  let statusElement = document.getElementById("venue-filter-status");

  if (!statusElement) {
    statusElement = document.createElement("div");
    statusElement.id = "venue-filter-status";
    statusElement.className = "venue-filter-status";
    statusElement.setAttribute("role", "status");
    statusElement.setAttribute("aria-live", "polite");

    const pubSection = getPublicationsPage();
    if (pubSection) {
      const firstH2 = pubSection.querySelector("h2.year, h2");
      if (firstH2) {
        firstH2.parentNode.insertBefore(statusElement, firstH2);
      } else {
        pubSection.insertBefore(statusElement, pubSection.firstChild);
      }
    }
  }

  if (venue) {
    statusElement.hidden = false;
    statusElement.replaceChildren();

    const label = document.createElement("span");
    label.className = "venue-filter-status__label";
    label.append("Filtering by ");

    const strong = document.createElement("strong");
    strong.textContent = venue;
    label.appendChild(strong);
    label.append(
      " — showing " + visibleCount + " of " + totalCount + " papers",
    );

    const clearBtn = document.createElement("button");
    clearBtn.type = "button";
    clearBtn.className = "venue-filter-status__clear";
    clearBtn.setAttribute("aria-label", "Clear filter");
    clearBtn.textContent = "Clear ×";
    clearBtn.addEventListener("click", function () {
      filterByVenue(venue);
    });

    statusElement.appendChild(label);
    statusElement.appendChild(clearBtn);
  } else {
    statusElement.hidden = true;
  }
}

// Initialize publication dividers on page load
function initPublicationDividers() {
  updatePublicationDividers();
}

// Check URL parameters on page load
document.addEventListener("DOMContentLoaded", function () {
  const publicationsPage = getPublicationsPage();
  if (!publicationsPage) return;

  initPublicationDividers();

  try {
    const urlParams = new URLSearchParams(window.location.search);
    const venueParam = urlParams.get("venue");

    if (venueParam) {
      filterByVenue(venueParam, {
        skipHistoryUpdate: true,
        skipScroll: true,
      });
    } else {
      // Double-check dividers after initial render completes
      // This handles edge cases where the initial call (line 256) runs before
      // all styles are computed, ensuring single publications get proper styling
      requestAnimationFrame(() => {
        updatePublicationDividers();
      });
    }
  } catch (error) {
    console.warn("Failed to parse URL parameters:", error);
  }
});

// Handle browser back/forward buttons
window.addEventListener("popstate", function () {
  if (!getPublicationsPage()) return;

  try {
    const urlParams = new URLSearchParams(window.location.search);
    const venueParam = urlParams.get("venue");

    if (venueParam && venueParam !== currentVenueFilter) {
      filterByVenue(venueParam, { skipHistoryUpdate: true, skipScroll: true });
    } else if (!venueParam && currentVenueFilter) {
      // Clear filter if URL has no venue parameter
      filterByVenue(currentVenueFilter, {
        skipHistoryUpdate: true,
        skipScroll: true,
      });
    }
  } catch (error) {
    console.warn("Failed to handle popstate:", error);
  }
});

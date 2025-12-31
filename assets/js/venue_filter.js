// Venue filter functionality for publications
let currentVenueFilter = null;

function filterByVenue(venue) {
  // Check if we're on the about page - if so, redirect to publications page with filter
  const isAboutPage = document.querySelector('.post') !== null && document.querySelector('.publications') === null;
  
  if (isAboutPage) {
    // Redirect to publications page with venue parameter
    const baseUrl = window.location.origin + (window.location.pathname.includes('/publications/') ? '/publications/' : '/publications/');
    window.location.href = baseUrl + '?venue=' + encodeURIComponent(venue);
    return;
  }
  
  // Select all publication list items
  const publications = document.querySelectorAll('.publications ol.bibliography > li');
  
  if (publications.length === 0) {
    console.warn('No publications found to filter');
    return;
  }
  
  // If clicking the same venue tag, reset filter
  if (currentVenueFilter === venue) {
    currentVenueFilter = null;
    publications.forEach(pub => {
      pub.style.display = '';
      pub.style.opacity = '1';
    });
    
    // Remove active state from all tags
    document.querySelectorAll('.venue-tag').forEach(tag => {
      tag.classList.remove('active-filter');
    });
    
    // Remove focus from active element to remove hover/focus effects
    if (document.activeElement) {
      document.activeElement.blur();
    }
    
    // Show all year headers
    document.querySelectorAll('.publications h2.year').forEach(header => {
      header.style.display = '';
    });
    
    // Reset all dividers
    updatePublicationDividers();
    
    // Hide filter status
    updateFilterStatus(null, 0, publications.length);
    
    // Update URL
    updateURL(null);
    
    return;
  }
  
  // Set new filter
  currentVenueFilter = venue;
  
  // Filter publications
  let visibleCount = 0;
  publications.forEach(pub => {
    // Find the data-venue attribute within the li element
    const venueElement = pub.querySelector('[data-venue]');
    const pubVenue = venueElement ? venueElement.getAttribute('data-venue') : null;
    
    if (pubVenue === venue) {
      pub.style.display = '';
      pub.style.opacity = '1';
      pub.style.transition = 'opacity 0.3s ease';
      visibleCount++;
    } else {
      pub.style.opacity = '0';
      setTimeout(() => {
        pub.style.display = 'none';
      }, 300);
    }
  });
  
  // Hide year headers that have no visible publications and update dividers
  setTimeout(() => {
    hideEmptyYearHeaders();
    updatePublicationDividers();
  }, 350); // Wait for fade out animation to complete
  
  // Update active state of tags
  document.querySelectorAll('.venue-tag').forEach(tag => {
    if (tag.getAttribute('data-venue-filter') === venue) {
      tag.classList.add('active-filter');
    } else {
      tag.classList.remove('active-filter');
    }
  });
  
  // Show filter status
  updateFilterStatus(venue, visibleCount, publications.length);
  
  // Update URL
  updateURL(venue);
  
  // Scroll to top of publications section
  const pubSection = document.querySelector('.publications');
  if (pubSection) {
    pubSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
  }
}

function hideEmptyYearHeaders() {
  const yearHeaders = document.querySelectorAll('.publications h2.year');
  
  yearHeaders.forEach(header => {
    // Find the bibliography list following this year header
    let nextElement = header.nextElementSibling;
    let hasVisiblePubs = false;
    
    // Check if the next bibliography list has any visible publications
    if (nextElement && nextElement.tagName === 'OL' && nextElement.classList.contains('bibliography')) {
      const visiblePubs = Array.from(nextElement.querySelectorAll('li')).filter(li => {
        const computedStyle = window.getComputedStyle(li);
        return computedStyle.display !== 'none' && computedStyle.opacity !== '0';
      });
      hasVisiblePubs = visiblePubs.length > 0;
    }
    
    // Hide or show the year header based on whether it has visible publications
    if (hasVisiblePubs) {
      header.style.display = '';
    } else {
      header.style.display = 'none';
    }
  });
}

function getVisiblePublications(listElement) {
  return Array.from(listElement.querySelectorAll('li')).filter(li => {
    const computedStyle = window.getComputedStyle(li);
    return computedStyle.display !== 'none' && computedStyle.opacity !== '0';
  });
}

function updatePublicationDividers() {
  // Handle Publications page (with year headers) vs About/other pages (single list)
  const publicationsSection = document.querySelector('.publications');
  if (!publicationsSection) return;

  const yearHeaders = publicationsSection.querySelectorAll('h2.year');

  if (yearHeaders.length > 0) {
    // Publications page: check per year
    yearHeaders.forEach(header => {
      // Find the bibliography list following this year header
      let nextElement = header.nextElementSibling;
      
      if (nextElement && nextElement.tagName === 'OL' && nextElement.classList.contains('bibliography')) {
        const visiblePubs = getVisiblePublications(nextElement);
        
        // Remove the class from all publications first
        nextElement.querySelectorAll('li').forEach(li => li.classList.remove('single-pub-in-year'));
        
        // If only one publication is visible in this year, add a class to hide dividers
        if (visiblePubs.length === 1) {
          visiblePubs.forEach(li => {
            li.classList.add('single-pub-in-year');
          });
        }
      }
    });
    return;
  }

  // About page or single publication list without year headers
  const bibliography = publicationsSection.querySelector('ol.bibliography');
  if (!bibliography) return;

  const visiblePubs = getVisiblePublications(bibliography);
  
  // Remove the class from all publications first
  bibliography.querySelectorAll('li').forEach(li => li.classList.remove('single-pub-in-year'));
  
  // If only one publication is visible, hide all dividers
  if (visiblePubs.length === 1) {
    visiblePubs.forEach(li => {
      li.classList.add('single-pub-in-year');
    });
  }
}

function updateURL(venue) {
  if (!window.history || !window.history.pushState) return;
  
  try {
    const url = new URL(window.location);
    if (venue) {
      url.searchParams.set('venue', venue);
    } else {
      url.searchParams.delete('venue');
    }
    window.history.pushState({}, '', url);
  } catch (error) {
    console.warn('Failed to update URL:', error);
  }
}

function updateFilterStatus(venue, visibleCount, totalCount) {
  let statusElement = document.getElementById('venue-filter-status');
  
  if (!statusElement) {
    // Create status element
    statusElement = document.createElement('div');
    statusElement.id = 'venue-filter-status';
    statusElement.style.cssText = `
      display: none;
      margin: 1rem 0;
      padding: 0.5rem 1rem;
      background-color: var(--global-code-bg-color);
      border-left: 3px solid var(--global-theme-color);
      border-radius: 0 4px 4px 0;
      font-size: 0.875rem;
      color: var(--global-text-color);
    `;
    
    const pubSection = document.querySelector('.publications, .post');
    if (pubSection) {
      const firstH2 = pubSection.querySelector('h2.year, h2');
      if (firstH2) {
        firstH2.parentNode.insertBefore(statusElement, firstH2);
      } else {
        pubSection.insertBefore(statusElement, pubSection.firstChild);
      }
    }
  }
  
  if (venue) {
    statusElement.style.display = 'block';
    statusElement.innerHTML = `
      Filtering by <strong>${venue}</strong>: showing ${visibleCount} of ${totalCount} publications
      <button onclick="filterByVenue('${venue}')" style="
        margin-left: 1rem;
        padding: 0.2rem 0.5rem;
        background: var(--global-theme-color);
        color: white;
        border: none;
        border-radius: 3px;
        cursor: pointer;
        font-size: 0.75rem;
      ">Clear Filter</button>
    `;
  } else {
    statusElement.style.display = 'none';
  }
}

// Initialize publication dividers on page load
function initPublicationDividers() {
  // Call updatePublicationDividers to handle both About and Publications pages
  updatePublicationDividers();
}

// Check URL parameters on page load
document.addEventListener('DOMContentLoaded', function() {
  // Initialize publication dividers for both About and Publications pages
  initPublicationDividers();
  
  // Only apply filter on publications page
  if (!document.querySelector('.publications')) return;
  
  try {
    const urlParams = new URLSearchParams(window.location.search);
    const venueParam = urlParams.get('venue');
    
    if (venueParam) {
      // Apply filter after a short delay to ensure DOM is ready
      setTimeout(() => {
        filterByVenue(venueParam);
      }, 100);
    } else {
      // Double-check dividers after initial render completes
      // This handles edge cases where the initial call (line 256) runs before
      // all styles are computed, ensuring single publications get proper styling
      requestAnimationFrame(() => {
        updatePublicationDividers();
      });
    }
  } catch (error) {
    console.warn('Failed to parse URL parameters:', error);
  }
});

// Handle browser back/forward buttons
window.addEventListener('popstate', function() {
  if (!document.querySelector('.publications')) return;
  
  try {
    const urlParams = new URLSearchParams(window.location.search);
    const venueParam = urlParams.get('venue');
    
    if (venueParam && venueParam !== currentVenueFilter) {
      filterByVenue(venueParam);
    } else if (!venueParam && currentVenueFilter) {
      // Clear filter if URL has no venue parameter
      filterByVenue(currentVenueFilter);
    }
  } catch (error) {
    console.warn('Failed to handle popstate:', error);
  }
});

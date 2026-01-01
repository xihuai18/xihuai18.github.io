// Initialize medium zoom.
$(document).ready(function() {
  // Get background color with transparency
  const bgColor = getComputedStyle(document.documentElement)
      .getPropertyValue('--global-bg-color') + 'ee';  // + 'ee' for transparency.

  // Initialize zoom for explicitly marked images
  medium_zoom = mediumZoom('[data-zoomable]', {
    background: bgColor,
  });

  // Auto-enable zoom for all images in blog post content
  // Exclude images that are already zoomable, avatars, icons, and very small images
  const postImages = document.querySelectorAll('.post-content img:not([data-zoomable]):not(.emoji):not([class*="avatar"]):not([class*="icon"])');
  
  postImages.forEach(function(img) {
    // Skip small images (likely icons or decorative elements)
    if (img.naturalWidth > 100 && img.naturalHeight > 100) {
      img.setAttribute('data-zoomable', '');
      medium_zoom.attach(img);
    } else {
      // For images not yet loaded, wait for load event
      img.addEventListener('load', function() {
        if (this.naturalWidth > 100 && this.naturalHeight > 100) {
          this.setAttribute('data-zoomable', '');
          medium_zoom.attach(this);
        }
      });
    }
  });

  // Also handle images in figures
  const figureImages = document.querySelectorAll('.post-content figure img:not([data-zoomable])');
  figureImages.forEach(function(img) {
    img.setAttribute('data-zoomable', '');
    medium_zoom.attach(img);
  });
});

# frozen_string_literal: true

# Add loading="lazy" and decoding="async" to content images after render,
# so markdown/HTML images in posts and news don't block first paint.
# Images that already declare a loading attribute are left untouched.
Jekyll::Hooks.register [:documents, :pages], :post_render do |doc|
  next unless doc.output_ext == '.html'

  doc.output = doc.output.gsub(/<img (?![^>]*\bloading=)/, '<img loading="lazy" decoding="async" ')
end

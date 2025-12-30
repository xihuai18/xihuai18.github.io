# frozen_string_literal: true

# HTML Block Protection Plugin for Jekyll
# ========================================
# This plugin ensures that raw HTML blocks (like <figure>, <div>, etc.)
# are properly preserved and not converted to code blocks by Kramdown.
#
# The main issue this solves:
# 1. HTML tags with tab indentation are treated as code blocks
# 2. HTML appearing after math blocks sometimes gets parsed incorrectly
# 3. HTML entities inside attributes can cause issues

module Jekyll
  module HtmlBlockProtection
    # HTML tags that should always be rendered as HTML, not code
    PROTECTED_TAGS = %w[
      figure figcaption
      div span
      img picture source
      video audio
      iframe embed object
      details summary
      aside article section
      nav header footer
      dl dt dd
      abbr cite dfn kbd mark q s small sub sup time
      i b u em strong
      a p br hr
      ul ol li
      h1 h2 h3 h4 h5 h6
      blockquote pre code
      table thead tbody tfoot tr th td
      form input button label select textarea
      svg path g rect circle
    ].freeze

    class << self
      # Pre-process content BEFORE Kramdown conversion
      def preprocess(content)
        return content if content.nil? || content.empty?

        processed = content.dup
        
        # CRITICAL FIX: Remove leading whitespace (tabs or 4+ spaces) from HTML blocks
        # Kramdown treats such indented content as code blocks
        
        # Pattern to match HTML tags with excessive leading whitespace
        # 4+ spaces or any tabs before an HTML tag
        PROTECTED_TAGS.each do |tag|
          # Opening tags with attributes
          processed = processed.gsub(/^([ ]{4,}|\t+)(<#{tag}[\s>])/im) do
            $2
          end
          
          # Closing tags
          processed = processed.gsub(/^([ ]{4,}|\t+)(<\/#{tag}>)/im) do
            $2
          end
        end
        
        # Special handling for self-closing and common inline tags
        # These are commonly indented inside other HTML blocks
        inline_tags = %w[img br hr input meta link]
        inline_tags.each do |tag|
          processed = processed.gsub(/^([ ]{4,}|\t+)(<#{tag}[\s>\/])/im) do
            $2
          end
        end

        processed
      end

      # Post-process content AFTER HTML generation
      def postprocess(content)
        return content if content.nil? || content.empty?

        processed = content.dup

        # Fix any HTML that was incorrectly escaped or turned into code blocks
        processed = fix_escaped_html_blocks(processed)

        processed
      end

      private

      def fix_escaped_html_blocks(content)
        # Find code blocks that actually contain HTML that should be rendered
        content.gsub(/<div class="language-plaintext highlighter-rouge">\s*<div class="highlight">\s*<pre class="highlight"><code>(.*?)<\/code><\/pre>\s*<\/div>\s*<\/div>/m) do |match|
          code_content = $1.strip
          
          # Check if this looks like HTML that should be rendered
          if looks_like_html_block?(code_content)
            # Unescape and return as raw HTML
            unescape_html(code_content)
          else
            match
          end
        end
      end

      def looks_like_html_block?(content)
        stripped = content.strip
        # Check if content starts with an HTML tag we want to render
        PROTECTED_TAGS.any? do |tag|
          stripped.match?(/^&lt;#{tag}[\s>]/i) || stripped.match?(/^<#{tag}[\s>]/i)
        end
      end

      def unescape_html(content)
        content
          .gsub('&lt;', '<')
          .gsub('&gt;', '>')
          .gsub('&amp;', '&')
          .gsub('&quot;', '"')
          .gsub('&#39;', "'")
          .gsub('&nbsp;', ' ')
      end
    end
  end
end

# Hook to PRE-process before Markdown conversion (lower priority than math)
Jekyll::Hooks.register :documents, :pre_render, priority: :normal do |doc|
  doc.content = Jekyll::HtmlBlockProtection.preprocess(doc.content)
end

Jekyll::Hooks.register :pages, :pre_render, priority: :normal do |page|
  if page.content
    page.content = Jekyll::HtmlBlockProtection.preprocess(page.content)
  end
end

# Hook to POST-process after HTML generation
Jekyll::Hooks.register :documents, :post_render, priority: :normal do |doc|
  next unless doc.output_ext == '.html'
  doc.output = Jekyll::HtmlBlockProtection.postprocess(doc.output)
end

Jekyll::Hooks.register :pages, :post_render, priority: :normal do |page|
  next unless page.output_ext == '.html'
  page.output = Jekyll::HtmlBlockProtection.postprocess(page.output)
end


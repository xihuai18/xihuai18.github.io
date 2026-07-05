# frozen_string_literal: true

require 'set'

# Heading ID Cleanup Plugin for Jekyll
# ====================================
# math_protection.rb swaps math for placeholders (⟦INLMATH42MATHEND⟧) before
# Kramdown runs, so Kramdown's auto_ids bakes lowercase placeholder fragments
# into heading ids, e.g.:
#
#   <h3 id="54-路由回放条件化-surrogate而不是直接缩小-inlmath256mathend">
#
# The math text itself is restored post-render, but the id attribute keeps the
# leaked fragment (and the fragment number changes between builds, breaking
# stable deep links). This hook rewrites those heading ids to drop the
# placeholder fragments and updates any in-document href="#..." references.
# The TOC (assets/js/toc.js) reads ids from the DOM at runtime, so it follows
# the cleaned ids automatically.

module Jekyll
  module HeadingIdCleanup
    # Lowercased placeholder fragment as it survives Kramdown's auto_id
    # (non-alphanumerics like ⟦⟧ are stripped, letters downcased).
    LEAK_PATTERN = /(?:inlmath|dispmath)\d+mathend/

    HEADING_ID = /(<h[1-6][^>]*\bid=")([^"]*)(")/

    class << self
      def cleanup(output)
        return output if output.nil? || output.empty?
        return output unless output =~ LEAK_PATTERN

        # All ids already present in the page, to keep cleaned ids unique.
        used_ids = output.scan(/\bid="([^"]*)"/).flatten.to_set
        renames = {}

        result = output.gsub(HEADING_ID) do
          pre, id, post = Regexp.last_match(1), Regexp.last_match(2), Regexp.last_match(3)
          if id =~ LEAK_PATTERN
            clean = unique_id(strip_leaks(id), used_ids)
            used_ids.add(clean)
            renames[id] = clean
            "#{pre}#{clean}#{post}"
          else
            "#{pre}#{id}#{post}"
          end
        end

        # Keep any anchors pointing at the old ids working.
        renames.each do |old_id, new_id|
          result = result.gsub("href=\"##{old_id}\"", "href=\"##{new_id}\"")
        end

        result
      end

      private

      def strip_leaks(id)
        cleaned = id.gsub(LEAK_PATTERN, '')
                    .gsub(/-{2,}/, '-')
                    .gsub(/\A-|-\z/, '')
        cleaned.empty? ? 'section' : cleaned
      end

      def unique_id(base, used_ids)
        return base unless used_ids.include?(base)

        counter = 2
        counter += 1 while used_ids.include?("#{base}-#{counter}")
        "#{base}-#{counter}"
      end
    end
  end
end

Jekyll::Hooks.register :documents, :post_render, priority: :low do |doc|
  next unless doc.output_ext == '.html'
  doc.output = Jekyll::HeadingIdCleanup.cleanup(doc.output)
end

Jekyll::Hooks.register :pages, :post_render, priority: :low do |page|
  next unless page.output_ext == '.html'
  page.output = Jekyll::HeadingIdCleanup.cleanup(page.output)
end

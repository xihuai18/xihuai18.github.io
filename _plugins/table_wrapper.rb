# frozen_string_literal: true

# Table Wrapper Plugin for Jekyll
# ===============================
# Wraps Markdown-generated tables in <div class="table-responsive"> at build
# time so wide tables scroll horizontally instead of being crushed into the
# column (previously assets/js/blog_enhancements.js did this client-side only,
# which meant a pre-JS layout flash and no wrapper without JS).
#
# Runs on :post_convert, i.e. after Kramdown turns the document body into HTML
# but before the layout is applied — layout/include tables (news, CV) are
# never touched. blog_enhancements.js skips tables already inside
# .table-responsive, so the client-side pass stays a harmless no-op.

module Jekyll
  module TableWrapper
    TABLE_BLOCK = %r{(<div[^>]*class="[^"]*table-responsive[^"]*"[^>]*>\s*)?<table(\s[^>]*)?>.*?</table>}m

    class << self
      def wrap_tables(content)
        return content if content.nil? || content.empty?
        return content unless content.include?('<table')

        content.gsub(TABLE_BLOCK) do |match|
          if Regexp.last_match(1)
            match # already wrapped by hand-written HTML
          else
            attrs = Regexp.last_match(2)
            table = with_table_classes(match, attrs)
            %(<div class="table-responsive">#{table}</div>)
          end
        end
      end

      private

      # Mirror the classes blog_enhancements.js used to add so the existing
      # .table / .table-sm styling keeps applying.
      def with_table_classes(table_html, attrs)
        if attrs && attrs =~ /\bclass="([^"]*)"/
          classes = Regexp.last_match(1).split(/\s+/)
          classes << 'table' unless classes.include?('table')
          classes << 'table-sm' unless classes.include?('table-sm')
          table_html.sub(/\bclass="[^"]*"/, %(class="#{classes.join(' ')}"))
        else
          table_html.sub(/\A<table/, '<table class="table table-sm"')
        end
      end
    end
  end
end

# Only touch Markdown sources: HTML pages (e.g. news.html) assemble their
# own table markup with dedicated scroll containers.
Jekyll::Hooks.register :documents, :post_convert do |doc|
  next unless %w[.md .markdown].include?(doc.extname)
  doc.content = Jekyll::TableWrapper.wrap_tables(doc.content)
end

Jekyll::Hooks.register :pages, :post_convert do |page|
  next unless %w[.md .markdown].include?(page.extname)
  page.content = Jekyll::TableWrapper.wrap_tables(page.content)
end

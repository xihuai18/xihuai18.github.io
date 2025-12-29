# frozen_string_literal: true

module Jekyll
  class DetailsBlock < Liquid::Block
    def initialize(tag_name, markup, tokens)
      super
      @markup = markup.to_s.strip
    end

    def render(context)
      site = context.registers[:site]
      markdown = site.find_converter_instance(::Jekyll::Converters::Markdown)

      summary = @markup
      open = false

      if summary.start_with?('open ')
        open = true
        summary = summary.sub(/^open\s+/, '')
      end

      summary = summary.gsub(/\A["']|["']\z/, '')
      summary = Liquid::Template.parse(summary).render(context).strip

      body_markdown = super.to_s.strip
      body_html = markdown.convert(body_markdown)

      open_attr = open ? ' open' : ''
      %(<details class="details-block"#{open_attr}><summary>#{summary}</summary><div class="details-content">#{body_html}</div></details>)
    end
  end
end

Liquid::Template.register_tag('details', Jekyll::DetailsBlock)


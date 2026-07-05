# frozen_string_literal: true

require 'kramdown/parser/html'

# kramdown Phrasing-Content Fix
# =============================
# kramdown classifies <summary> and <figcaption> as block-content containers
# (HTML_CONTENT_MODEL_BLOCK), which silently requires their open/close tags to
# sit on lines of their own. The natural one-line form
#
#   <summary>推导细节(点击展开)</summary>
#
# makes the block parser start mid-line: the line remainder becomes a
# paragraph, the in-line </summary> is "an invalidly used closing tag" (escaped
# into visible text), and everything after it — including </details> — is
# swallowed into the <summary> element. <figcaption> failed the same way,
# which is why figures used to require markdown="0" (see _pages/render-test.md
# §8 and the section-id fallback it caused).
#
# Per the HTML spec both elements hold phrasing content, so :span is the
# correct content model: kramdown then scans straight to the closing tag and
# still parses inline markdown (bold, code, math placeholders) inside.
Kramdown::Parser::Html::Constants::HTML_CONTENT_MODEL['summary'] = :span
Kramdown::Parser::Html::Constants::HTML_CONTENT_MODEL['figcaption'] = :span

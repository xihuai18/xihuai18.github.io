# frozen_string_literal: true

# Math Protection Plugin for Jekyll
# =================================
# This plugin protects LaTeX math expressions from Kramdown processing
# by replacing them with placeholders before Markdown conversion,
# then restoring them after.
#
# Key features:
# 1. Handles blockquote math (> $$ ... > $$) correctly
# 2. Preserves multiline math with \\ line breaks
# 3. Uses document-specific storage to avoid conflicts

module Jekyll
  module MathProtection
    # Use unique markers that won't appear in normal content
    DISPLAY_MARKER = "⟦DISPMATH"
    INLINE_MARKER = "⟦INLMATH"
    END_MARKER = "MATHEND⟧"
    # Special marker for \\ to prevent Kramdown from treating it as line break
    LINEBREAK_MARKER = "⟦LATEXBR⟧"
    # Markers for < and > to prevent HTML parsing
    LT_MARKER = "⟦LATEXLT⟧"
    GT_MARKER = "⟦LATEXGT⟧"
    
    # Store math per document path to avoid conflicts
    @stores = {}
    
    class << self
      def protect_math(content, doc_id = nil)
        return content if content.nil? || content.empty?
        
        doc_id ||= content.hash.to_s
        math_store = {}
        counter = 0
        
        result = content.dup
        
        # Step 1: Handle blockquote display math first
        # Pattern: > $$ (start) ... > $$ (end) with > on each line
        result, counter = process_blockquote_display_math(result, math_store, counter)
        
        # Step 2: Handle regular display math $$...$$
        result = result.gsub(/\$\$(.*?)\$\$/m) do |match|
          counter += 1
          key = "#{DISPLAY_MARKER}#{counter}#{END_MARKER}"
          # Protect special characters from being processed
          protected_match = protect_special_chars(match)
          math_store[key] = protected_match
          key
        end
        
        # Step 3: Handle display math \[...\]
        result = result.gsub(/\\\[(.*?)\\\]/m) do |match|
          counter += 1
          key = "#{DISPLAY_MARKER}#{counter}#{END_MARKER}"
          protected_match = protect_special_chars(match)
          math_store[key] = protected_match
          key
        end
        
        # Step 4: Handle inline math $...$ (not $$, single line)
        result = result.gsub(/(?<!\$)\$(?!\$)([^\$\n]+?)\$(?!\$)/) do |match|
          # Skip currency like $100
          if match =~ /^\$[\d,.]+$/
            match
          else
            counter += 1
            key = "#{INLINE_MARKER}#{counter}#{END_MARKER}"
            protected_match = protect_special_chars(match)
            math_store[key] = protected_match
            key
          end
        end
        
        # Step 5: Handle inline math \(...\)
        result = result.gsub(/\\\((.*?)\\\)/m) do |match|
          counter += 1
          key = "#{INLINE_MARKER}#{counter}#{END_MARKER}"
          protected_match = protect_special_chars(match)
          math_store[key] = protected_match
          key
        end
        
        # Store for later restoration
        @stores[doc_id] = math_store
        
        result
      end
      
      def restore_math(content, doc_id = nil)
        return content if content.nil? || content.empty?
        
        # Try to find the right store
        math_store = nil
        if doc_id && @stores[doc_id]
          math_store = @stores.delete(doc_id)
        elsif @stores.size == 1
          # If only one store, use it
          math_store = @stores.values.first
          @stores.clear
        else
          # Try to match by checking which store's keys are in the content
          @stores.each do |id, store|
            if store.keys.any? { |k| content.include?(k) }
              math_store = @stores.delete(id)
              break
            end
          end
        end
        
        return content if math_store.nil? || math_store.empty?
        
        result = content.dup
        
        # Restore all placeholders
        math_store.each do |key, math|
          escaped_key = Regexp.escape(key)
          # Restore special characters from markers
          restored_math = restore_special_chars(math)
          result = result.gsub(/#{escaped_key}/) { restored_math }
        end
        
        result
      end
      
      private
      
      # Protect special characters in math expressions
      def protect_special_chars(match)
        match.gsub('\\\\', LINEBREAK_MARKER)
             .gsub('<', LT_MARKER)
             .gsub('>', GT_MARKER)
      end
      
      # Restore special characters in math expressions
      def restore_special_chars(math)
        math.gsub(LINEBREAK_MARKER) { '\\\\' }
            .gsub(LT_MARKER) { '<' }
            .gsub(GT_MARKER) { '>' }
      end
      
      def process_blockquote_display_math(content, math_store, counter)
        lines = content.lines
        result = []
        in_math = false
        math_lines = []
        prefix = ""
        
        lines.each do |line|
          # Detect start of blockquote display math: > $$ or > $$
          if !in_math && line =~ /^(>\s*)\$\$\s*$/
            in_math = true
            prefix = $1
            math_lines = ["$$\n"]
            next
          end
          
          # Detect end of blockquote display math: > $$
          if in_math && line =~ /^>\s*\$\$\s*$/
            in_math = false
            math_lines << "$$"
            
            # Store the complete math block
            counter += 1
            key = "#{DISPLAY_MARKER}#{counter}#{END_MARKER}"
            # Join and protect special characters
            full_math = math_lines.join
            protected_math = protect_special_chars(full_math)
            math_store[key] = protected_math
            
            # Output placeholder with blockquote prefix
            result << "#{prefix}#{key}\n"
            math_lines = []
            next
          end
          
          if in_math
            # Inside blockquote math: strip the > prefix and keep content
            clean_line = line.sub(/^>\s?/, '')
            math_lines << clean_line
          else
            result << line
          end
        end
        
        # Handle unclosed math (shouldn't happen, but be safe)
        if in_math && !math_lines.empty?
          math_lines.each do |ml|
            result << "#{prefix}#{ml}"
          end
        end
        
        [result.join, counter]
      end
    end
  end
end

# Pre-render: protect math expressions
Jekyll::Hooks.register :documents, :pre_render, priority: :high do |doc|
  next unless doc.extname == '.md' || doc.extname == '.markdown'
  doc_id = doc.relative_path
  doc.content = Jekyll::MathProtection.protect_math(doc.content, doc_id)
end

# Post-render: restore math expressions
Jekyll::Hooks.register :documents, :post_render, priority: :low do |doc|
  next unless doc.output_ext == '.html'
  doc_id = doc.relative_path
  doc.output = Jekyll::MathProtection.restore_math(doc.output, doc_id)
end

# Also handle pages (like about.md)
Jekyll::Hooks.register :pages, :pre_render, priority: :high do |page|
  next unless page.extname == '.md' || page.extname == '.markdown'
  doc_id = page.relative_path || page.name
  page.content = Jekyll::MathProtection.protect_math(page.content, doc_id)
end

Jekyll::Hooks.register :pages, :post_render, priority: :low do |page|
  next unless page.output_ext == '.html'
  doc_id = page.relative_path || page.name
  page.output = Jekyll::MathProtection.restore_math(page.output, doc_id)
end

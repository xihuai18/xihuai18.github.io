// create element for copy button and wrap button in code blocks
var codeBlocks = document.querySelectorAll('pre');
codeBlocks.forEach(function (codeBlock) {
  if (codeBlock.querySelector('pre:not(.lineno)') || codeBlock.querySelector('code')) {
    // Set default wrap for code blocks
    codeBlock.classList.add('wrapped');
    
    // Create wrap button
    var wrapButton = document.createElement('button');
    wrapButton.className = 'wrap-toggle';
    wrapButton.type = 'button';
    wrapButton.ariaLabel = 'Toggle code wrap';
    wrapButton.innerHTML = '<i class="fas fa-align-left"></i>';
    codeBlock.append(wrapButton);
    
    // Create copy button
    var copyButton = document.createElement('button');
    copyButton.className = 'copy';
    copyButton.type = 'button';
    copyButton.ariaLabel = 'Copy code to clipboard';
    copyButton.innerText = 'Copy';
    copyButton.innerHTML = '<i class="fas fa-clipboard"></i>';
    codeBlock.append(copyButton);
    
    // Toggle wrap on/off
    wrapButton.addEventListener('click', function () {
      codeBlock.classList.toggle('wrapped');
      if (codeBlock.classList.contains('wrapped')) {
        wrapButton.innerHTML = '<i class="fas fa-align-left"></i>';
      } else {
        wrapButton.innerHTML = '<i class="fas fa-arrows-alt-h"></i>';
      }
    });

    // get code from code block and copy to clipboard
    copyButton.addEventListener('click', function () {
      // check if code block has line numbers
      // i.e. `kramdown.syntax_highlighter_opts.block.line_numbers` set to true in _config.yml
      // or using `jekyll highlight` liquid tag with `linenos` option
      if (codeBlock.querySelector('pre:not(.lineno)')) {
        // get code from code block ignoring line numbers
        var code = codeBlock.querySelector('pre:not(.lineno)').innerText.trim();
      } else { // if (codeBlock.querySelector('code')) {
        // get code from code block when line numbers are not displayed
        var code = codeBlock.querySelector('code').innerText.trim();
      }
      window.navigator.clipboard.writeText(code);
      copyButton.innerText = 'Copied';
      copyButton.innerHTML = '<i class="fas fa-clipboard-check"></i>';
      var waitFor = 3000;

      setTimeout(function () {
        copyButton.innerText = 'Copy';
        copyButton.innerHTML = '<i class="fas fa-clipboard"></i>';
      }, waitFor);
    });
  }
});

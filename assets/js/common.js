$(document).ready(function() {
    $('a.abstract').click(function() {
        $(this).parent().parent().find(".abstract.hidden").toggleClass('open');
        $(this).parent().parent().find(".bibtex.hidden.open").toggleClass('open');
    });
    $('a.bibtex').click(function() {
        $(this).parent().parent().find(".bibtex.hidden").toggleClass('open');
        $(this).parent().parent().find(".abstract.hidden.open").toggleClass('open');
    });
    $('a').removeClass('waves-effect waves-light');

    function updateUvMode(mode) {
        var key = mode === 'd30' ? 'uvD30' : 'uvAll';
        $('.post-uv').each(function() {
            var value = $(this).data(key);
            if (value === undefined || value === null || value === '') {
                value = 0;
            }
            $(this).text(value + ' views');
        });

        $('.uv-toggle').removeClass('font-weight-bold');
        $('.uv-toggle[data-uv-mode="' + mode + '"]').addClass('font-weight-bold');
    }

    var savedMode = null;
    try {
        savedMode = window.localStorage.getItem('uvMode');
    } catch (e) {
        savedMode = null;
    }
    if (savedMode !== 'all' && savedMode !== 'd30') {
        savedMode = 'all';
    }
    updateUvMode(savedMode);

    $('.uv-toggle').click(function(e) {
        e.preventDefault();
        var mode = $(this).data('uvMode');
        if (mode !== 'all' && mode !== 'd30') {
            return;
        }
        try {
            window.localStorage.setItem('uvMode', mode);
        } catch (e) {}
        updateUvMode(mode);
    });
});

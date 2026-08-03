// Maryland HS Sports Record Book — client-side enhancements.
// No external requests; progressive enhancement only.

// School index filter.
(function () {
  var input = document.getElementById('school-search');
  if (!input) return;
  var rows = document.querySelectorAll('.school-index tbody tr');
  var none = document.getElementById('no-results');
  function filter() {
    var q = input.value.trim().toLowerCase();
    var shown = 0;
    rows.forEach(function (tr) {
      var name = tr.cells[0].textContent.toLowerCase();
      var hit = !q || name.indexOf(q) !== -1;
      tr.style.display = hit ? '' : 'none';
      if (hit) shown++;
    });
    if (none) none.hidden = shown !== 0;
  }
  input.addEventListener('input', filter);
})();

// Copy-to-clipboard for fast-facts paragraphs.
(function () {
  document.querySelectorAll('.copy-btn').forEach(function (btn) {
    btn.addEventListener('click', function () {
      var text = btn.getAttribute('data-copy') || '';
      var done = function () { btn.classList.add('copied'); btn.textContent = 'Copied!'; setTimeout(reset, 1800); };
      var reset = function () { btn.classList.remove('copied'); btn.textContent = 'Copy paragraph'; };
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(done, fallback);
      } else { fallback(); }
      function fallback() {
        var ta = document.createElement('textarea');
        ta.value = text; document.body.appendChild(ta); ta.select();
        try { document.execCommand('copy'); done(); } catch (e) { reset(); }
        document.body.removeChild(ta);
      }
    });
  });
})();
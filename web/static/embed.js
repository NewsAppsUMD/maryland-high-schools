// Embed builder: generates an iframe snippet + live preview.
// No external requests; reads the inlined schools list.
(function () {
  var schoolsData = JSON.parse(document.getElementById('schools-data').textContent || '[]');
  var widgetSel = document.getElementById('widget');
  var schoolSel = document.getElementById('school');
  var themeSel = document.getElementById('theme');
  var snippet = document.getElementById('snippet');
  var preview = document.getElementById('preview');
  var copyBtn = document.getElementById('copy-snippet');
  var copyMsg = document.getElementById('copy-msg');

  // Default dimensions per widget.
  var SIZES = { timeline: { w: 760, h: 260 }, titles: { w: 360, h: 96 }, anniversaries: { w: 420, h: 360 } };

  function populateSchools() {
    schoolSel.innerHTML = '';
    schoolsData.forEach(function (s) {
      var o = document.createElement('option');
      o.value = s.slug; o.textContent = s.name;
      schoolSel.appendChild(o);
    });
  }

  function needsSchool() { return widgetSel.value !== 'anniversaries'; }

  function buildSrc() {
    var origin = window.location.origin;
    var base = origin + window.location.pathname.replace(/embed\/index\.html$/, '');
    var w = widgetSel.value, theme = themeSel.value;
    var path;
    if (w === 'anniversaries') {
      path = base + 'embed/anniversaries/index.html';
    } else {
      path = base + 'embed/' + w + '/' + schoolSel.value + '/index.html';
    }
    return path + '?theme=' + theme;
  }

  function render() {
    var showSchool = needsSchool();
    schoolSel.disabled = !showSchool;
    if (showSchool && !schoolSel.options.length) populateSchools();
    var src = buildSrc();
    var size = SIZES[widgetSel.value];
    snippet.value = '<iframe src="' + src + '" width="' + size.w +
      '" height="' + size.h + '" frameborder="0" scrolling="no" loading="lazy"></iframe>';
    preview.src = src;
    preview.width = size.w; preview.height = size.h;
  }

  [widgetSel, schoolSel, themeSel].forEach(function (el) {
    el.addEventListener('change', render);
  });

  copyBtn.addEventListener('click', function () {
    snippet.select();
    var done = function () { copyMsg.textContent = 'Copied!'; setTimeout(function () { copyMsg.textContent = ''; }, 1800); };
    try {
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(snippet.value).then(done, function () { document.execCommand('copy'); done(); });
      } else { document.execCommand('copy'); done(); }
    } catch (e) { copyMsg.textContent = 'Copy failed — select and copy manually.'; }
  });

  populateSchools();
  render();
})();
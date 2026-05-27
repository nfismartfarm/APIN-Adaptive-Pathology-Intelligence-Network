// Phase 8.F · Console nav active-link highlighter
// Strict `script-src 'self'` blocks the inline <script> that previously
// shipped inside _CONSOLE_NAV_BYTES. This file is loaded by every Console
// page via <script src="/static/console_nav.js"></script> (injected by the
// shared nav placeholder in apin_server.py::_CONSOLE_NAV_BYTES).
//
// Active-link rule: a link is active when the page's pathname equals its
// data-acn attribute, OR when the path begins with data-acn + '/' (so
// /account/api/keys/abc123 also highlights the Keys link).
(function () {
  var path = window.location.pathname;
  var anchors = document.querySelectorAll('.apin-console-nav a[data-acn]');
  for (var i = 0; i < anchors.length; i++) {
    var a = anchors[i];
    var match = a.getAttribute('data-acn');
    if (path === match || path.indexOf(match + '/') === 0) {
      a.classList.add('acn-active');
    }
  }
})();

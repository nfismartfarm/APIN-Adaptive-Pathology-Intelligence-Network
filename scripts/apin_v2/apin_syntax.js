// apin_syntax.js — Phase 9.N.7 · lightweight homegrown syntax highlighter
// Three lexers (bash, python, node) + one styler. No Prism.js. ~100 LOC.
//
// Public surface:
//   APIN.syntax.highlight(code, lang) → safe HTML string
//   APIN.syntax.applyTo(codeEl) → reads textContent, sets innerHTML
//
// Color tokens (resolved from CSS variables on the host page):
//   .syn-kw    keyword (import / const / let / await / curl / -X / -H)
//   .syn-str   string literal (single, double, template)
//   .syn-num   numeric literal
//   .syn-com   comment
//   .syn-url   URL literal
//   .syn-fn    function name / method call
//   .syn-tok   the redacted apin_<your_token> chip (special handling)
//
// The redacted token gets a wrapped chip with a lock icon prefix so it
// "pops" visually — see the original brief.

(function () {
  "use strict";
  if (!window.APIN) window.APIN = {};

  const KEYWORDS = {
    bash:   ['curl', 'wget', 'http', 'https', 'export'],
    python: ['import', 'from', 'as', 'with', 'await', 'async', 'def', 'return', 'class', 'if', 'else', 'try', 'except', 'finally', 'lambda', 'for', 'in', 'while', 'None', 'True', 'False'],
    node:   ['const', 'let', 'var', 'await', 'async', 'function', 'return', 'class', 'if', 'else', 'try', 'catch', 'finally', 'new', 'this', 'null', 'true', 'false', 'undefined', 'import', 'from', 'export', 'default'],
  };

  function _escape(s) {
    return String(s).replace(/[&<>"']/g,
      c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  }

  // Special token: apin_<your_token> placeholder gets a chip with lock icon
  function _replaceTokenPlaceholder(html) {
    return html.replace(
      /apin_&lt;your_token&gt;/g,
      '<span class="syn-tok" title="API token (redacted)">'
        + '<svg class="syn-tok-lock" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><use href="#i-lock"/></svg>'
        + '<span class="syn-tok-text">apin_&lt;your_token&gt;</span>'
        + '</span>'
    );
  }

  function _tokenize(src, lang) {
    const kw = KEYWORDS[lang] || [];
    const kwSet = new Set(kw);
    // We do a single regex-based pass so escaping happens once and we don't
    // double-color. Order matters: comments / strings first, then numbers,
    // keywords, URLs.
    // Capture groups (order):
    //   1 = string (single/double/triple)
    //   2 = comment
    //   3 = url
    //   4 = number
    //   5 = identifier
    //   6 = bash short flag (-X, -H, -F)
    //   7 = bash long flag (--header)
    const pat = new RegExp(
      "('''[\\s\\S]*?'''|\"\"\"[\\s\\S]*?\"\"\""    // python triple-strings
      + "|'(?:\\\\.|[^'\\\\])*'"                       // single-quote string
      + '|"(?:\\\\.|[^"\\\\])*"'                       // double-quote string
      + "|`(?:\\\\.|[^`\\\\])*`"                       // template literal
      + ")"
      + "|(#[^\\n]*|//[^\\n]*)"                        // comment line
      + "|((?:https?|file|ws|wss)://[^\\s\"'`)\\\\]+)"  // url
      + "|(\\b\\d+(?:\\.\\d+)?\\b)"                    // number
      + "|([A-Za-z_][A-Za-z0-9_]*)"                    // identifier
      + "|(-[A-Za-z])(?![A-Za-z])"                     // bash short flag
      + "|(--[a-z][a-z\\-]*)",                         // bash long flag
      'g'
    );
    let out = '';
    let last = 0;
    let m;
    while ((m = pat.exec(src)) !== null) {
      if (m.index > last) out += _escape(src.slice(last, m.index));
      if (m[1] !== undefined) {
        out += '<span class="syn-str">' + _escape(m[1]) + '</span>';
      } else if (m[2] !== undefined) {
        out += '<span class="syn-com">' + _escape(m[2]) + '</span>';
      } else if (m[3] !== undefined) {
        out += '<span class="syn-url">' + _escape(m[3]) + '</span>';
      } else if (m[4] !== undefined) {
        out += '<span class="syn-num">' + _escape(m[4]) + '</span>';
      } else if (m[5] !== undefined) {
        const id = m[5];
        // Distinguish keyword vs function-call (look-ahead for `(` or `.`)
        const after = src[pat.lastIndex];
        if (kwSet.has(id)) {
          out += '<span class="syn-kw">' + _escape(id) + '</span>';
        } else if (after === '(' || after === '.') {
          out += '<span class="syn-fn">' + _escape(id) + '</span>';
        } else {
          out += _escape(id);
        }
      } else if (m[6] !== undefined && (lang === 'bash' || lang === 'curl')) {
        out += '<span class="syn-kw">' + _escape(m[6]) + '</span>';
      } else if (m[7] !== undefined && (lang === 'bash' || lang === 'curl')) {
        out += '<span class="syn-kw">' + _escape(m[7]) + '</span>';
      } else {
        out += _escape(m[0]);
      }
      last = pat.lastIndex;
    }
    if (last < src.length) out += _escape(src.slice(last));
    return _replaceTokenPlaceholder(out);
  }

  function highlight(code, lang) {
    if (!code) return '';
    if (lang === 'curl') lang = 'bash';
    if (!KEYWORDS[lang]) lang = 'bash';
    return _tokenize(String(code), lang);
  }

  function applyTo(codeEl, lang) {
    if (!codeEl) return;
    const src = codeEl.textContent || '';
    codeEl.innerHTML = highlight(src, lang || codeEl.getAttribute('data-lang') || 'bash');
  }

  // Add the CSS once on first import
  function _installStyles() {
    if (document.getElementById('apin-syntax-style')) return;
    const css = document.createElement('style');
    css.id = 'apin-syntax-style';
    css.textContent =
      '.syn-kw   { color: var(--ink, #1a1612); font-weight: 600 }'
      + '.syn-str  { color: var(--c-amber, #d49620) }'
      + '.syn-num  { color: var(--c-info, #2d6a96) }'
      + '.syn-com  { color: var(--ink-soft, #6b6453); font-style: italic }'
      + '.syn-url  { color: var(--ink, #1a1612); text-decoration: underline; text-decoration-color: var(--ink-soft, #6b6453); text-underline-offset: 2px }'
      + '.syn-fn   { color: var(--ink, #1a1612); font-weight: 500 }'
      + '.syn-tok  { display: inline-flex; align-items: center; gap: 4px; '
      +              'padding: 0 6px 0 4px; margin: 0 1px; '
      +              'background: var(--paper-deep, #e9e2d1); '
      +              'border: 1px solid var(--paper-edge, #c7bca9); '
      +              'border-radius: 0; '
      +              'font-family: "JetBrains Mono", monospace; '
      +              'color: var(--ink, #1a1612); white-space: nowrap }'
      + '.syn-tok-lock { width: 11px; height: 11px; color: var(--ink-soft, #6b6453); flex-shrink: 0 }'
      + '.syn-tok-text { font-size: 0.92em }';
    document.head.appendChild(css);
  }
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', _installStyles);
  } else {
    _installStyles();
  }

  window.APIN.syntax = { highlight, applyTo };
})();

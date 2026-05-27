// apin_lightbox.js — Phase 9.N.3 · FLIP-grow lightbox shell
// One shared modal that any chart card can "grow into" on click of the
// expand-affordance. Uses Web Animations API + FLIP technique so the source
// card visually inflates to a centered modal without reflowing the grid.
//
// Public surface:
//   APIN.lightbox.open({
//     sourceCard:  HTMLElement,             // the card the click came from
//     title:       string,                  // "Status mix"
//     subtitle:    string,                  // "357 requests · 24h window"
//     hashKey:     string,                  // 'donut' (for #lightbox=donut)
//     build:       (panel) => Promise|void  // called to populate body
//   })
//   APIN.lightbox.close()
//   APIN.lightbox.isOpen()
//
// Animation choreography:
//   0–80ms   backdrop fades 0 → 0.6 (paper-deep noise overlay)
//   40–260ms source card clone FLIPs to centered position + scale
//   260–380ms sub-views inside the modal stagger-enter (translateY 16→0)
//
// Reverse (240ms):
//   0–120ms   sub-views fade and translateY down
//   80–240ms  card FLIPs back to source position
//   180–240ms backdrop fades

(function () {
  "use strict";
  if (!window.APIN || !window.APIN.fx) {
    console.warn("apin_lightbox.js requires APIN.fx — load apin_fx.js first");
  }

  // ── Singleton DOM ─────────────────────────────────────────────────────────
  let _root = null;          // overlay container (fixed, full screen)
  let _backdrop = null;
  let _modal = null;
  let _title = null;
  let _subtitle = null;
  let _body = null;
  let _closeBtn = null;
  let _sourceCard = null;
  let _hashKey = null;
  let _open = false;

  function _mount() {
    if (_root) return;
    _root = document.createElement("div");
    _root.id = "apin-lightbox";
    _root.setAttribute("aria-hidden", "true");
    _root.style.cssText = [
      "position:fixed",
      "inset:0",
      "z-index:9000",
      "pointer-events:none",
      "display:flex",
      "align-items:center",
      "justify-content:center",
    ].join(";");

    _backdrop = document.createElement("div");
    _backdrop.className = "apin-lightbox-backdrop";
    _backdrop.style.cssText = [
      "position:absolute",
      "inset:0",
      "background:rgba(20,16,12,0.55)",
      "opacity:0",
      "transition:opacity 200ms cubic-bezier(0.22, 1, 0.36, 1)",
      "pointer-events:none",
    ].join(";");

    _modal = document.createElement("div");
    _modal.className = "apin-lightbox-modal";
    _modal.setAttribute("role", "dialog");
    _modal.setAttribute("aria-modal", "true");
    _modal.style.cssText = [
      "position:relative",
      "background:var(--paper, #fbf9f3)",
      "border:1px solid var(--paper-edge, #c7bca9)",
      "box-shadow:0 0 0 4px var(--paper, #fbf9f3), 0 0 0 5px var(--paper-edge, #c7bca9), 0 24px 60px rgba(20,16,12,0.18)",
      "width:min(1080px, 92vw)",
      "max-height:90vh",
      "border-radius:0",
      "opacity:0",
      "transform:scale(0.92)",
      "pointer-events:auto",
      "display:flex",
      "flex-direction:column",
      "overflow:hidden",
    ].join(";");

    // Modal header
    const head = document.createElement("div");
    head.className = "apin-lightbox-head";
    head.style.cssText = [
      "display:flex",
      "align-items:flex-start",
      "justify-content:space-between",
      "padding:24px 28px 18px",
      "border-bottom:1px solid var(--paper-edge, #c7bca9)",
      "background:var(--paper-deep, #e9e2d1)",
      "gap:16px",
      "flex-shrink:0",
    ].join(";");

    const titlewrap = document.createElement("div");
    titlewrap.style.cssText = "flex:1;min-width:0";
    _title = document.createElement("h3");
    _title.className = "apin-lightbox-title";
    _title.style.cssText = [
      "font-family:'Fraunces',serif",
      "font-weight:500",
      "font-size:22px",
      "margin:0 0 4px 0",
      "color:var(--ink, #1a1612)",
      "letter-spacing:-0.01em",
    ].join(";");
    _subtitle = document.createElement("div");
    _subtitle.className = "apin-lightbox-sub";
    _subtitle.style.cssText = [
      "font-family:'JetBrains Mono',monospace",
      "font-size:11px",
      "color:var(--ink-soft, #6b6453)",
      "letter-spacing:0.02em",
    ].join(";");
    titlewrap.appendChild(_title);
    titlewrap.appendChild(_subtitle);

    _closeBtn = document.createElement("button");
    _closeBtn.className = "apin-lightbox-close";
    _closeBtn.setAttribute("aria-label", "Close");
    _closeBtn.style.cssText = [
      "background:transparent",
      "border:1px solid var(--paper-edge, #c7bca9)",
      "color:var(--ink, #1a1612)",
      "cursor:pointer",
      "width:32px",
      "height:32px",
      "display:inline-flex",
      "align-items:center",
      "justify-content:center",
      "padding:0",
      "transition:background 160ms cubic-bezier(0.22, 1, 0.36, 1)",
    ].join(";");
    _closeBtn.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" style="width:14px;height:14px"><path d="M5 5 L19 19 M19 5 L5 19"/></svg>';
    _closeBtn.addEventListener("mouseenter", () => {
      _closeBtn.style.background = "var(--paper, #fbf9f3)";
    });
    _closeBtn.addEventListener("mouseleave", () => {
      _closeBtn.style.background = "transparent";
    });
    _closeBtn.addEventListener("click", close);

    head.appendChild(titlewrap);
    head.appendChild(_closeBtn);

    _body = document.createElement("div");
    _body.className = "apin-lightbox-body";
    _body.style.cssText = [
      "flex:1",
      "overflow-y:auto",
      "padding:22px 28px 26px",
      "background:var(--paper, #fbf9f3)",
    ].join(";");

    _modal.appendChild(head);
    _modal.appendChild(_body);
    _root.appendChild(_backdrop);
    _root.appendChild(_modal);
    document.body.appendChild(_root);

    _backdrop.addEventListener("click", close);
    window.addEventListener("keydown", _onKey);
    window.addEventListener("hashchange", _onHashChange);
  }

  function _onKey(e) {
    if (!_open) return;
    if (e.key === "Escape") {
      e.preventDefault();
      close();
    }
  }

  function _readHash() {
    const m = (location.hash || "").match(/(?:^|[#&])lightbox=([a-z0-9_\-]+)/i);
    return m ? m[1] : null;
  }
  function _writeHash(key) {
    const h = (location.hash || "").replace(/^#/, "");
    const parts = h ? h.split("&") : [];
    const filtered = parts.filter(p => !p.startsWith("lightbox="));
    if (key) filtered.unshift("lightbox=" + key);
    history.replaceState(null, "", "#" + filtered.join("&"));
  }

  function _onHashChange() {
    const cur = _readHash();
    if (_open && !cur) close({ skipHash: true });
  }

  // ── Open ──────────────────────────────────────────────────────────────────
  function open(opts) {
    _mount();
    opts = opts || {};
    if (_open) close({ skipAnim: true });
    _open = true;
    _sourceCard = opts.sourceCard || null;
    _hashKey = opts.hashKey || null;

    _root.setAttribute("aria-hidden", "false");
    _root.style.pointerEvents = "auto";
    _title.textContent = opts.title || "";
    _subtitle.textContent = opts.subtitle || "";
    _body.innerHTML = "";

    // Backdrop fade-in
    requestAnimationFrame(() => {
      _backdrop.style.opacity = "1";
      _backdrop.style.pointerEvents = "auto";
    });

    // FLIP from source card (if provided)
    if (_sourceCard) {
      const srcRect = _sourceCard.getBoundingClientRect();
      const dstRect = _modal.getBoundingClientRect();
      const dx = srcRect.left + srcRect.width / 2
               - (dstRect.left + dstRect.width / 2);
      const dy = srcRect.top + srcRect.height / 2
               - (dstRect.top + dstRect.height / 2);
      const sx = srcRect.width  / dstRect.width;
      const sy = srcRect.height / dstRect.height;
      _modal.style.transformOrigin = "center center";
      _modal.style.opacity = "0";
      _modal.style.transform = `translate(${dx}px, ${dy}px) scale(${Math.max(sx, 0.2)}, ${Math.max(sy, 0.2)})`;
      // Force reflow before animating
      void _modal.offsetWidth;
      _modal.animate([
        { opacity: 0, transform: `translate(${dx}px, ${dy}px) scale(${Math.max(sx, 0.2)}, ${Math.max(sy, 0.2)})` },
        { opacity: 1, transform: "translate(0, 0) scale(1, 1)" }
      ], {
        duration: 380,
        easing: "cubic-bezier(0.22, 1, 0.36, 1)",
        fill: "both",
      }).finished.then(() => {
        _modal.style.opacity = "1";
        _modal.style.transform = "none";
      }).catch(()=>{});
    } else {
      // No source card — fade modal in
      _modal.animate([
        { opacity: 0, transform: "scale(0.92)" },
        { opacity: 1, transform: "scale(1)" }
      ], {
        duration: 380,
        easing: "cubic-bezier(0.22, 1, 0.36, 1)",
        fill: "both",
      });
    }

    // Build body content (deferred to after grow phase)
    setTimeout(async () => {
      try {
        if (typeof opts.build === "function") {
          await opts.build(_body);
        }
        // Stagger-in any direct children
        const children = [..._body.children];
        children.forEach((c, i) => {
          c.style.opacity = "0";
          c.style.transform = "translateY(16px)";
          c.animate([
            { opacity: 0, transform: "translateY(16px)" },
            { opacity: 1, transform: "translateY(0)" }
          ], {
            duration: 320,
            delay: 60 * i,
            easing: "cubic-bezier(0.22, 1, 0.36, 1)",
            fill: "forwards",
          });
        });
      } catch (e) {
        console.error("lightbox build error", e);
      }
    }, 220);

    if (_hashKey) _writeHash(_hashKey);
    if (_closeBtn) try { _closeBtn.focus(); } catch (e) {}
  }

  // ── Close ─────────────────────────────────────────────────────────────────
  function close(opts) {
    opts = opts || {};
    if (!_open) return;
    _open = false;

    const finish = () => {
      _root.setAttribute("aria-hidden", "true");
      _root.style.pointerEvents = "none";
      _backdrop.style.pointerEvents = "none";
      _body.innerHTML = "";
      _sourceCard = null;
      _hashKey = null;
      if (!opts.skipHash) _writeHash(null);
    };

    if (opts.skipAnim) { finish(); return; }

    // Sub-views fade down
    const children = [..._body.children];
    children.forEach((c, i) => {
      c.animate([
        { opacity: 1, transform: "translateY(0)" },
        { opacity: 0, transform: "translateY(16px)" }
      ], { duration: 120, easing: "cubic-bezier(0.22, 1, 0.36, 1)", fill: "forwards" });
    });

    // Backdrop fade
    setTimeout(() => { _backdrop.style.opacity = "0"; }, 180);

    // FLIP back
    if (_sourceCard) {
      const srcRect = _sourceCard.getBoundingClientRect();
      const dstRect = _modal.getBoundingClientRect();
      const dx = srcRect.left + srcRect.width / 2
               - (dstRect.left + dstRect.width / 2);
      const dy = srcRect.top + srcRect.height / 2
               - (dstRect.top + dstRect.height / 2);
      const sx = srcRect.width  / dstRect.width;
      const sy = srcRect.height / dstRect.height;
      _modal.animate([
        { opacity: 1, transform: "translate(0, 0) scale(1, 1)" },
        { opacity: 0, transform: `translate(${dx}px, ${dy}px) scale(${Math.max(sx, 0.2)}, ${Math.max(sy, 0.2)})` }
      ], {
        duration: 240,
        delay: 80,
        easing: "cubic-bezier(0.22, 1, 0.36, 1)",
        fill: "forwards",
      }).finished.then(finish).catch(finish);
    } else {
      _modal.animate([
        { opacity: 1, transform: "scale(1)" },
        { opacity: 0, transform: "scale(0.92)" }
      ], {
        duration: 240,
        easing: "cubic-bezier(0.22, 1, 0.36, 1)",
        fill: "forwards",
      }).finished.then(finish).catch(finish);
    }
  }

  // ── Section helper for body builders ──────────────────────────────────────
  // APIN.lightbox.section(title) → returns an <section> element with header,
  // appended to the body. Callers append content into it.
  function section(title) {
    const sec = document.createElement("section");
    sec.className = "apin-lb-section";
    sec.style.cssText = "margin-bottom:22px";
    if (title) {
      const h = document.createElement("h4");
      h.style.cssText = [
        "font-family:'Fraunces',serif",
        "font-weight:500",
        "font-style:italic",
        "font-size:11.5px",
        "letter-spacing:0.12em",
        "text-transform:uppercase",
        "color:var(--ink-soft, #6b6453)",
        "margin:0 0 10px 0",
        "padding-bottom:6px",
        "border-bottom:1px solid var(--paper-edge, #c7bca9)",
      ].join(";");
      h.textContent = title;
      sec.appendChild(h);
    }
    return sec;
  }

  // ── Public ────────────────────────────────────────────────────────────────
  window.APIN = window.APIN || {};
  window.APIN.lightbox = {
    open, close,
    isOpen: () => _open,
    section,
  };
})();

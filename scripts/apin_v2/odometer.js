// Shared digit-odometer (Phase 8.H) — lifted from pipeline.html lines
// 10026-10168 so the alerts pill count + Console nav badge can reuse the
// same animation the inference-site KPI tiles use.
//
// Builds 10-row digit columns inside an element and animates translateY
// when the value changes. Speed scales with the magnitude of the delta.
//
// Usage:
//   APIN.odometer.roll(el, 42)          // first call seeds, subsequent animate
//   APIN.odometer.set(el, 42)           // force-seed without animation
//
// CSS contract (must be present on the page hosting the element):
//   .apin-odometer            { display:inline-flex; line-height:1; ... }
//   .apin-odometer-digit      { width:.6em; height:1em; overflow:hidden; ... }
//   .apin-odometer-digit-col  { display:flex; flex-direction:column; ... }
//   .apin-odometer-digit-col>span { height:1em; line-height:1; ... }
//   .apin-odometer-static     { display:inline-block; ... }
//   .apin-odometer-digit.is-spinning .apin-odometer-digit-col {
//       filter: blur(0.6px);                       /* during fast roll */
//   }

(function () {
  "use strict";

  function seedOdo(el, str) {
    el.innerHTML = "";
    const cols = [];
    for (let i = 0; i < str.length; i++) {
      const ch = str[i];
      if (/[0-9]/.test(ch)) {
        const cell = document.createElement("span");
        cell.className = "apin-odometer-digit";
        const col = document.createElement("span");
        col.className = "apin-odometer-digit-col";
        for (let d = 0; d < 10; d++) {
          const s = document.createElement("span");
          s.textContent = String(d);
          col.appendChild(s);
        }
        cell.appendChild(col);
        el.appendChild(cell);
        const digit = Number(ch);
        col.style.transition = "none";
        col.style.transform = "translateY(-" + digit + "em)";
        void col.offsetHeight;  // force reflow so subsequent transitions apply
        col.style.transition = "";
        cols.push({ col: col, digit: digit });
      } else {
        const stat = document.createElement("span");
        stat.className = "apin-odometer-static";
        stat.textContent = ch;
        el.appendChild(stat);
        cols.push(null);
      }
    }
    el._odo = { lastStr: str, cols: cols };
  }

  function animateOdo(el, newStr, durMs, staggerMs, withBlur) {
    const prev = el._odo;
    if (!prev) return;
    // Iterate right→left so the fastest-changing (rightmost) digit kicks off
    // first; the eye reads this as "odometer rolling forward."
    let idx = 0;
    for (let i = newStr.length - 1; i >= 0; i--) {
      const ch = newStr[i];
      const slot = prev.cols[i];
      if (!slot) { idx++; continue; }
      const newDigit = /[0-9]/.test(ch) ? Number(ch) : 0;
      if (slot.digit === newDigit) { idx++; continue; }
      const col = slot.col;
      const cell = col.parentElement;
      const delay = idx * staggerMs;
      col.style.transition = "transform " + durMs +
        "ms cubic-bezier(0.22, 1, 0.36, 1) " + delay + "ms";
      col.style.transform = "translateY(-" + newDigit + "em)";
      if (withBlur) {
        cell.classList.add("is-spinning");
        setTimeout(function () { cell.classList.remove("is-spinning"); },
                   delay + durMs + 40);
      }
      slot.digit = newDigit;
      idx++;
    }
    prev.lastStr = newStr;
  }

  function rollOdometer(el, newValueRaw) {
    if (!el) return;
    const newStr = (typeof newValueRaw === "number")
      ? newValueRaw.toLocaleString()
      : String(newValueRaw);
    const prev = el._odo;
    if (!prev) {
      seedOdo(el, newStr);
      return;
    }
    if (prev.lastStr === newStr) return;
    const oldNum = Number(prev.lastStr.replace(/[^0-9.\-]/g, ""));
    const newNum = Number(newStr.replace(/[^0-9.\-]/g, ""));
    const delta = Math.abs(newNum - oldNum);
    if (!Number.isFinite(delta) || Number.isNaN(delta)) {
      seedOdo(el, newStr);
      return;
    }
    if (prev.lastStr.length !== newStr.length) {
      seedOdo(el, newStr);
      return;
    }
    // Speed-adaptive timing — bigger deltas spin faster (and apply a small
    // blur for the "rolling" cinematic feel at 100+).
    let durMs, stagger, blur;
    if      (delta < 10)    { durMs = 280;  stagger = 0;   blur = false; }
    else if (delta < 100)   { durMs = 420;  stagger = 50;  blur = false; }
    else if (delta < 1000)  { durMs = 600;  stagger = 70;  blur = true;  }
    else if (delta < 10000) { durMs = 800;  stagger = 90;  blur = true;  }
    else                    { durMs = 1000; stagger = 110; blur = true;  }
    animateOdo(el, newStr, durMs, stagger, blur);
  }

  // Public surface
  window.APIN = window.APIN || {};
  window.APIN.odometer = {
    roll: rollOdometer,
    set: seedOdo,
  };
})();

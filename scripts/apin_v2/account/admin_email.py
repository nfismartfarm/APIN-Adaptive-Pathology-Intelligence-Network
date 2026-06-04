"""Premium admin-OTP email via Resend (HTTP API — no extra dependency).

Sends a branded, table-based (email-client-safe), accessible OTP email with a
plain-text twin. Brand palette is APIN's paper-ink + green so it reads as a
first-party security mail. The logo image is referenced from the public base
URL when set (it loads in production; locally it gracefully degrades to the
wordmark). The 6-digit code is the hero; security metadata (when/where/device)
lets the recipient answer 'was this me?' at a glance.
"""
from __future__ import annotations

import logging
import os

import requests

log = logging.getLogger("apin_v2.account.admin_email")

RESEND_ENDPOINT = "https://api.resend.com/emails"


def _key() -> str:
    return os.environ.get("RESEND_KEY") or os.environ.get("RESEND_API_KEY") or ""


def _from() -> str:
    # Resend's onboarding sender works without a verified domain (delivers to
    # the Resend account's own email). Override with RESEND_FROM once a domain
    # is verified (e.g. "APIN Security <security@yourdomain>").
    return os.environ.get("RESEND_FROM", "APIN Security <onboarding@resend.dev>")


def _logo_url() -> str:
    base = (os.environ.get("APIN_PUBLIC_BASE_URL") or "").rstrip("/")
    return (base + "/logo.png") if base else ""


def send_admin_otp(*, to_email: str, code: str, expires_minutes: int = 10,
                   ip=None, location=None, device=None, when=None, ref=None):
    """Send the OTP email. Returns (ok: bool, reason: str|None).

    ``ref`` is a short per-send reference appended to the subject. Gmail (and
    most clients) collapse messages with an identical subject into one thread;
    a unique ref makes every code arrive as its OWN message instead of stacking
    in a growing thread — better UX for "the newest code is the newest email".
    """
    key = _key()
    if not key:
        log.error("RESEND_KEY not set; cannot send admin OTP email.")
        return False, "no_key"
    subject = "Verify your APIN admin sign-in"
    if ref:
        subject += " · #%s" % ref
    payload = {
        "from": _from(),
        "to": [to_email],
        "subject": subject,
        "html": _render_html(code, expires_minutes, ip, location, device, when),
        "text": _render_text(code, expires_minutes, ip, location, device, when),
        # Belt-and-suspenders: a unique entity ref discourages thread grouping.
        "headers": {"X-Entity-Ref-ID": (ref or code)},
    }
    try:
        r = requests.post(
            RESEND_ENDPOINT, timeout=15,
            headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
            json=payload)
        if r.status_code in (200, 201):
            return True, None
        log.error("Resend send failed %s: %s", r.status_code, (r.text or "")[:300])
        return False, "http_%s" % r.status_code
    except Exception as e:  # noqa: BLE001
        log.error("Resend send exception: %s", e)
        return False, "exception"


# ── Templates ───────────────────────────────────────────────────────────────
def _meta_rows(ip, location, device, when) -> str:
    rows = []
    if when:
        rows.append(("Requested", when))
    if location:
        rows.append(("Location", location))
    if device:
        rows.append(("Device", device))
    if ip:
        rows.append(("IP", ip))
    if not rows:
        return ""
    inner = "".join(
        "<tr>"
        "<td style='padding:3px 0;color:#8a8273;font:12px Menlo,monospace;width:90px'>%s</td>"
        "<td style='padding:3px 0;color:#3a342a;font:12px Menlo,monospace'>%s</td></tr>" % (k, v)
        for k, v in rows)
    return ("<table role='presentation' cellpadding='0' cellspacing='0' "
            "style='margin:18px auto 0;'>%s</table>" % inner)


def _render_html(code, mins, ip, location, device, when) -> str:
    logo = _logo_url()
    brand = (("<img src='%s' alt='APIN' width='34' height='34' "
              "style='display:block;border-radius:8px'>" % logo) if logo else "")
    spaced = " ".join(list(str(code)))
    return """<!doctype html><html><body style="margin:0;background:#0e0e10;padding:32px 12px;
font-family:-apple-system,Segoe UI,Inter,Helvetica,Arial,sans-serif">
<table role="presentation" width="100%" cellpadding="0" cellspacing="0"><tr><td align="center">
<table role="presentation" width="460" cellpadding="0" cellspacing="0"
 style="max-width:460px;background:#fbf9f3;border-radius:18px;overflow:hidden;
 box-shadow:0 14px 50px rgba(0,0,0,.5)">
 <tr><td style="background:#1b3326;padding:22px 28px">
   <table role="presentation" cellpadding="0" cellspacing="0"><tr>
     <td style="padding-right:11px">""" + brand + """</td>
     <td style="color:#eaf3ec;font-weight:700;letter-spacing:.14em;font-size:15px">APIN
       <div style="color:#8fb89c;font-weight:500;font-size:9px;letter-spacing:.22em">ADMIN&nbsp;SECURITY</div></td>
   </tr></table>
 </td></tr>
 <tr><td style="padding:30px 30px 8px">
   <div style="font-size:19px;font-weight:700;color:#1a1612">Verify your admin sign-in</div>
   <div style="font-size:13.5px;color:#5a5246;margin-top:8px;line-height:1.6">
     Enter this code to open the APIN admin console.</div>
 </td></tr>
 <tr><td align="center" style="padding:8px 30px 4px">
   <div style="display:inline-block;background:#eef3ec;border:1px solid #cfe0d4;border-radius:12px;
    padding:16px 26px;font:700 30px Menlo,Consolas,monospace;letter-spacing:.34em;color:#1b3326">""" + spaced + """</div>
   <div style="font-size:12px;color:#8a8273;margin-top:12px">This code expires in """ + str(mins) + """ minutes.</div>
   """ + _meta_rows(ip, location, device, when) + """
 </td></tr>
 <tr><td style="padding:22px 30px 6px">
   <div style="background:#f4efe6;border-radius:10px;padding:13px 16px;font-size:12.5px;color:#5a5246;line-height:1.6">
     <b style="color:#1a1612">Didn't try to sign in?</b> Your password is still safe — just ignore this email.
     If this keeps happening, change your password.</div>
 </td></tr>
 <tr><td style="padding:18px 30px 26px">
   <div style="border-top:1px solid #e7dfcf;padding-top:14px;font-size:11px;color:#a59c88;line-height:1.6">
     APIN · Adaptive Pathological Intelligence Network<br>This is an automated security message; please don't reply.</div>
 </td></tr>
</table></td></tr></table></body></html>"""


def _render_text(code, mins, ip, location, device, when) -> str:
    lines = [
        "APIN — Verify your admin sign-in",
        "",
        "Your admin verification code is: %s" % code,
        "It expires in %s minutes." % mins,
        "",
    ]
    if when:
        lines.append("Requested: %s" % when)
    if location:
        lines.append("Location:  %s" % location)
    if device:
        lines.append("Device:    %s" % device)
    if ip:
        lines.append("IP:        %s" % ip)
    lines += [
        "",
        "Didn't try to sign in? Your password is safe — ignore this email.",
        "",
        "APIN · automated security message — do not reply.",
    ]
    return "\n".join(lines)

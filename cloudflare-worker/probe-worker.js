// 9.N.8i · Cloudflare Worker — external uptime probe (1-min cron).
//
// Parallel to the GitHub Action probe. Runs from a Cloudflare edge node
// (closer to most users than us-east-1), and writes results to the same
// external_probes Turso table with probe_source='cloudflare-worker'.
//
// The /status page's "External Availability" section reads both sources
// and shows them as two lines on the same chart — disagreements reveal
// interesting things (e.g. GitHub→HF route slow but CF→HF fast).
//
// Secrets (set via `wrangler secret put`):
//   · APIN_PROBE_TOKEN     — shared with HF Space env var
//   · TURSO_DATABASE_URL   — libsql://...
//   · TURSO_AUTH_TOKEN     — JWT
//
// CF Workers don't have curl, so we implement the timing breakdown
// using performance.now() at each Fetch lifecycle hook. The breakdown
// is less precise than curl's (no DNS/TCP/TLS split) but the total
// latency from the probe's perspective is what matters for uptime.

const PROBE_TARGET_URL = "https://dxv-404-apin.hf.space/api/probe/external";
const PROBE_VERSION    = "probe-worker.v1.0.0";
const SCHEMA_VERSION   = "probe.v1";
const SLO_LATENCY_MS   = 5000;
const REGION_TAG       = "cf-worker";   // refined to colo at request time
const RETRY_ATTEMPTS   = 3;

export default {
  async scheduled(event, env, ctx) {
    ctx.waitUntil(runProbe(env, "shallow"));
  },
  // Allow manual GET for testing: `curl https://your-worker.workers.dev/`
  async fetch(request, env, ctx) {
    const url = new URL(request.url);
    if (url.pathname === "/probe") {
      const result = await runProbe(env, "shallow");
      return new Response(JSON.stringify(result, null, 2), {
        headers: { "Content-Type": "application/json" },
      });
    }
    return new Response("apin-uptime-probe — see /probe", { status: 200 });
  },
};

async function runProbe(env, depth) {
  // Hourly deep probe — same logic as the GH Action
  const utcMin = new Date().getUTCMinutes();
  if (utcMin === 0 || utcMin === 1) depth = "deep";

  const probeId = crypto.randomUUID();
  const issuedAt = isoUtc();
  const userAgent = `apin-probe/${PROBE_VERSION} (cloudflare-worker)`;
  const target = `${PROBE_TARGET_URL}?depth=${depth}`;

  // ── Issue the probe ──────────────────────────────────────────────
  const t0 = Date.now();
  let response = null, bodyText = "", curlExit = 0, httpStatus = 0;
  let errorClass = "", errorDetail = "";
  try {
    response = await fetch(target, {
      method: "GET",
      headers: {
        "Authorization": `Bearer ${env.APIN_PROBE_TOKEN}`,
        "Accept":        "application/json",
        "User-Agent":    userAgent,
      },
      cf: { cacheTtl: 0, cacheEverything: false },  // never cache probe
    });
    httpStatus = response.status;
    bodyText = await response.text();
  } catch (e) {
    errorClass = classifyFetchError(e);
    errorDetail = String(e?.message || e).slice(0, 200);
  }
  const totalMs = Date.now() - t0;

  // ── Parse + validate body ────────────────────────────────────────
  let parsed = null;
  let parseError = null;
  try { parsed = JSON.parse(bodyText); }
  catch (e) { parseError = String(e?.message || e).slice(0, 200); }

  const gates = {
    http_2xx:           httpStatus >= 200 && httpStatus < 300 ? 1 : 0,
    json_parseable:     parsed !== null ? 1 : 0,
    schema_match:       0,
    overall_ok:         0,
    all_components_up:  0,
    latency_under_slo:  totalMs < SLO_LATENCY_MS ? 1 : 0,
    no_resource_alerts: 1,   // default ok; we'll downgrade below
  };

  let overall = "down";
  let serverRecvAt = null, serverSendAt = null;
  let componentsJson = "{}", componentsUp = null, componentsTotal = null;
  let componentFailures = [];
  let resources = {}, stats5m = {}, build = {};
  let processUptimeS = null;

  if (parsed) {
    if (parsed.schema_version && parsed.overall && parsed.components &&
        parsed.server_ts_utc) {
      gates.schema_match = 1;
      overall = parsed.overall || "down";
      serverRecvAt    = parsed.server_recv_at_utc || null;
      serverSendAt    = parsed.server_send_at_utc || null;
      processUptimeS  = parsed.process_uptime_s ?? null;
      componentsJson  = JSON.stringify(parsed.components || {});
      componentsTotal = Object.keys(parsed.components || {}).length;
      componentsUp    = Object.values(parsed.components || {})
                          .filter(c => c.status === "up").length;
      componentFailures = Object.entries(parsed.components || {})
                            .filter(([_, c]) => c.status !== "up")
                            .map(([k, _]) => k);
      resources = parsed.resources || {};
      stats5m   = parsed.internal_stats_last_5min || {};
      build     = parsed.build || {};
      if (overall === "operational") gates.overall_ok = 1;
      if (componentsUp === componentsTotal && componentsTotal > 0)
        gates.all_components_up = 1;
      // Resource alert check — uses per-process memory_rss_mb, NOT the
      // system-wide memory_pct (which on HF Space's shared host reflects
      // OTHER tenants' memory usage, not ours).  HF free-tier hard-kills
      // around 2 GB, so 1800 MB is a sensible per-process ceiling.
      const memOk = resources.memory_rss_mb == null || resources.memory_rss_mb < 1800;
      const fdsOk = resources.open_fds      == null || resources.open_fds      < 1000;
      const dskOk = resources.disk_free_gb  == null || resources.disk_free_gb  > 0.5;
      gates.no_resource_alerts = (memOk && fdsOk && dskOk) ? 1 : 0;
    }
  }

  // ── Determine error_class if any gate failed ─────────────────────
  if (errorClass) {
    // already set by fetch exception
  } else if (!gates.http_2xx) {
    if (httpStatus >= 500) errorClass = "http_5xx";
    else if (httpStatus >= 400) errorClass = "http_4xx";
    else errorClass = "http_3xx_unexpected";
    errorDetail = `HTTP ${httpStatus}`;
  } else if (!gates.json_parseable) {
    errorClass = "body_not_json";
    errorDetail = parseError || "JSON parse failed";
  } else if (!gates.schema_match) {
    errorClass = "body_schema_mismatch";
    errorDetail = "probe.v1 required fields missing";
  } else if (bodyText.length < 100) {
    errorClass = "body_too_small";
    errorDetail = `body was ${bodyText.length} bytes`;
  } else if (!gates.overall_ok) {
    errorClass = "body_overall_not_ok";
    errorDetail = `overall = ${overall}`;
  } else if (!gates.all_components_up) {
    errorClass = "component_down";
    errorDetail = componentFailures.join(",");
  } else if (!gates.latency_under_slo) {
    errorClass = "latency_sla_breach";
    errorDetail = `total_ms=${totalMs} exceeded slo=${SLO_LATENCY_MS}`;
  } else if (!gates.no_resource_alerts) {
    errorClass = "resource_alert";
    errorDetail = `mem_rss=${resources.memory_rss_mb}MB fds=${resources.open_fds} disk=${resources.disk_free_gb}GB`;
  }

  const success = (gates.http_2xx && gates.json_parseable && gates.schema_match &&
                   gates.overall_ok && gates.all_components_up &&
                   gates.latency_under_slo && gates.no_resource_alerts) ? 1 : 0;
  const slaBreach = gates.latency_under_slo ? 0 : 1;

  // ── Compute body checksum ────────────────────────────────────────
  let bodyChecksum = "";
  if (bodyText) {
    const hashBuf = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(bodyText));
    bodyChecksum = [...new Uint8Array(hashBuf)].slice(0, 8)
                      .map(b => b.toString(16).padStart(2, "0")).join("");
  }

  // ── Build INSERT row payload ─────────────────────────────────────
  const recordedAt = isoUtc();
  const row = {
    probe_id:                 probeId,
    schema_version:           SCHEMA_VERSION,
    issued_at_utc:            issuedAt,
    server_recv_at_utc:       serverRecvAt,
    server_send_at_utc:       serverSendAt,
    recorded_at_utc:          recordedAt,
    success:                  success,
    overall:                  overall,
    error_class:              errorClass || null,
    error_detail:             errorDetail || null,
    sla_breach:               slaBreach,
    http_status:              httpStatus,
    target_url:               target,
    response_bytes:           bodyText.length,
    response_checksum_sha256: bodyChecksum || null,
    // CF Workers' fetch doesn't expose dns/tcp/tls timing breakdown like
    // curl -w. We record total_ms; the per-phase fields stay NULL.
    dns_ms:                   null,
    tcp_connect_ms:           null,
    tls_handshake_ms:         null,
    ttfb_ms:                  null,
    download_ms:              null,
    total_ms:                 totalMs,
    components_json:          componentsJson,
    components_up_count:      componentsUp,
    components_total_count:   componentsTotal,
    component_failures_json:  JSON.stringify(componentFailures),
    process_uptime_s:         processUptimeS,
    memory_rss_mb:            resources.memory_rss_mb ?? null,
    memory_pct:               resources.memory_pct ?? null,
    cpu_pct_1m:               resources.cpu_pct_1m ?? null,
    gpu_vram_used_mb:         resources.gpu_vram_used_mb ?? null,
    gpu_vram_total_mb:        resources.gpu_vram_total_mb ?? null,
    disk_free_gb:             resources.disk_free_gb ?? null,
    open_fds:                 resources.open_fds ?? null,
    event_loop_lag_ms:        resources.event_loop_lag_ms ?? null,
    request_count_5m:         stats5m.request_count_5m ?? null,
    error_count_5m:           stats5m.error_count_5m ?? null,
    error_rate_5m_pct:        stats5m.error_rate_5m_pct ?? null,
    p50_latency_5m_ms:        stats5m.p50_latency_5m_ms ?? null,
    p95_latency_5m_ms:        stats5m.p95_latency_5m_ms ?? null,
    build_version:            build.version || null,
    build_git_sha:            build.git_sha || null,
    build_deployed_at_utc:    build.deployed_at_utc || null,
    probe_source:             "cloudflare-worker",
    probe_runner:             "cf-worker",
    probe_region:             REGION_TAG,
    probe_version:            PROBE_VERSION,
    probe_depth:              depth,
    probe_user_agent:         userAgent,
    gate_http_2xx:            gates.http_2xx,
    gate_json_parseable:      gates.json_parseable,
    gate_schema_match:        gates.schema_match,
    gate_overall_ok:          gates.overall_ok,
    gate_all_components_up:   gates.all_components_up,
    gate_latency_under_slo:   gates.latency_under_slo,
    gate_no_resource_alerts:  gates.no_resource_alerts,
  };

  // ── Write to Turso (Hrana-HTTP /v2/pipeline) with retries ────────
  await writeProbeToTurso(env, row);

  return { probe_id: probeId, success, overall, http_status: httpStatus,
           total_ms: totalMs, error_class: errorClass || null };
}

function classifyFetchError(e) {
  const msg = String(e?.message || e || "").toLowerCase();
  if (msg.includes("timeout") || msg.includes("timed out")) return "timeout";
  if (msg.includes("dns") || msg.includes("resolve"))       return "dns_error";
  if (msg.includes("ssl") || msg.includes("tls") || msg.includes("certificate"))
                                                            return "tls_error";
  if (msg.includes("refused"))                              return "tcp_refused";
  return "unknown";
}

function isoUtc() {
  // Microsecond-precision UTC ISO timestamp matching the Python format
  const d = new Date();
  return d.toISOString().replace("Z", "000Z");
}

async function writeProbeToTurso(env, row) {
  const httpUrl = env.TURSO_DATABASE_URL.replace(/^libsql:\/\//, "https://");
  const url = `${httpUrl}/v2/pipeline`;

  // Build the args array in the same column order as the INSERT
  const argFor = (v, typeHint) => {
    if (v === null || v === undefined) return { type: "null", value: null };
    if (typeHint === "integer")        return { type: "integer", value: String(v) };
    if (typeHint === "float")          return { type: "float", value: Number(v) };
    return { type: "text", value: String(v) };
  };

  const sql = "INSERT OR IGNORE INTO external_probes (" +
    "probe_id, schema_version, issued_at_utc, server_recv_at_utc, server_send_at_utc, " +
    "recorded_at_utc, success, overall, error_class, error_detail, sla_breach, " +
    "http_status, target_url, response_bytes, response_checksum_sha256, " +
    "dns_ms, tcp_connect_ms, tls_handshake_ms, ttfb_ms, download_ms, total_ms, " +
    "components_json, components_up_count, components_total_count, component_failures_json, " +
    "process_uptime_s, memory_rss_mb, memory_pct, cpu_pct_1m, " +
    "gpu_vram_used_mb, gpu_vram_total_mb, disk_free_gb, open_fds, event_loop_lag_ms, " +
    "request_count_5m, error_count_5m, error_rate_5m_pct, p50_latency_5m_ms, p95_latency_5m_ms, " +
    "build_version, build_git_sha, build_deployed_at_utc, " +
    "probe_source, probe_runner, probe_region, probe_version, probe_depth, probe_user_agent, " +
    "gate_http_2xx, gate_json_parseable, gate_schema_match, gate_overall_ok, " +
    "gate_all_components_up, gate_latency_under_slo, gate_no_resource_alerts" +
    ") VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)";

  const args = [
    argFor(row.probe_id),
    argFor(row.schema_version),
    argFor(row.issued_at_utc),
    argFor(row.server_recv_at_utc),
    argFor(row.server_send_at_utc),
    argFor(row.recorded_at_utc),
    argFor(row.success, "integer"),
    argFor(row.overall),
    argFor(row.error_class),
    argFor(row.error_detail),
    argFor(row.sla_breach, "integer"),
    argFor(row.http_status, "integer"),
    argFor(row.target_url),
    argFor(row.response_bytes, "integer"),
    argFor(row.response_checksum_sha256),
    argFor(row.dns_ms, "integer"),
    argFor(row.tcp_connect_ms, "integer"),
    argFor(row.tls_handshake_ms, "integer"),
    argFor(row.ttfb_ms, "integer"),
    argFor(row.download_ms, "integer"),
    argFor(row.total_ms, "integer"),
    argFor(row.components_json),
    argFor(row.components_up_count, "integer"),
    argFor(row.components_total_count, "integer"),
    argFor(row.component_failures_json),
    argFor(row.process_uptime_s, "integer"),
    argFor(row.memory_rss_mb, "integer"),
    argFor(row.memory_pct, "float"),
    argFor(row.cpu_pct_1m, "float"),
    argFor(row.gpu_vram_used_mb, "integer"),
    argFor(row.gpu_vram_total_mb, "integer"),
    argFor(row.disk_free_gb, "float"),
    argFor(row.open_fds, "integer"),
    argFor(row.event_loop_lag_ms, "integer"),
    argFor(row.request_count_5m, "integer"),
    argFor(row.error_count_5m, "integer"),
    argFor(row.error_rate_5m_pct, "float"),
    argFor(row.p50_latency_5m_ms, "integer"),
    argFor(row.p95_latency_5m_ms, "integer"),
    argFor(row.build_version),
    argFor(row.build_git_sha),
    argFor(row.build_deployed_at_utc),
    argFor(row.probe_source),
    argFor(row.probe_runner),
    argFor(row.probe_region),
    argFor(row.probe_version),
    argFor(row.probe_depth),
    argFor(row.probe_user_agent),
    argFor(row.gate_http_2xx, "integer"),
    argFor(row.gate_json_parseable, "integer"),
    argFor(row.gate_schema_match, "integer"),
    argFor(row.gate_overall_ok, "integer"),
    argFor(row.gate_all_components_up, "integer"),
    argFor(row.gate_latency_under_slo, "integer"),
    argFor(row.gate_no_resource_alerts, "integer"),
  ];

  const payload = JSON.stringify({
    requests: [{ type: "execute", stmt: { sql, args } }],
  });

  for (let attempt = 1; attempt <= RETRY_ATTEMPTS; attempt++) {
    try {
      const r = await fetch(url, {
        method: "POST",
        headers: {
          "Authorization": `Bearer ${env.TURSO_AUTH_TOKEN}`,
          "Content-Type":  "application/json",
        },
        body: payload,
      });
      if (r.ok) {
        const j = await r.json();
        if (j?.results?.[0]?.type === "ok") return true;
      }
    } catch (e) {
      // network error — fall through to retry
    }
    if (attempt < RETRY_ATTEMPTS) {
      await new Promise(res => setTimeout(res, attempt * 1500));
    }
  }
  return false;
}

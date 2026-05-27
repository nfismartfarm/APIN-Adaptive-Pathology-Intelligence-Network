#!/usr/bin/env bash
# 9.N.8i · External availability probe — single execution.
#
# Sequence:
#   1. Generate probe_id (UUIDv4) for dedup-on-retry.
#   2. curl /api/probe/external with bearer auth + timing-breakdown -w.
#   3. Validate JSON shape against probe.v1 schema.
#   4. Compute 7 validation gates (HTTP 2xx, JSON parseable, schema match,
#      overall ok, all components up, latency under SLO, no resource alerts).
#   5. Map outcome to error_class enum if any gate failed.
#   6. INSERT a row into external_probes via Turso Hrana-HTTP /v2/pipeline.
#
# Idempotent: probe_id is UNIQUE-indexed in the DB, so re-runs with the
# same UUID are no-ops. We never retry an UUID — each run generates fresh.
#
# Exit code is ALWAYS 0 — we want the failure recorded in the DB, not
# the GH workflow itself failing (which would suppress the next probe).

set -uo pipefail

# ── Inputs (from workflow env) ────────────────────────────────────────
: "${PROBE_TARGET_URL:?missing PROBE_TARGET_URL}"
: "${PROBE_TOKEN:?missing PROBE_TOKEN}"
: "${TURSO_DATABASE_URL:?missing TURSO_DATABASE_URL}"
: "${TURSO_AUTH_TOKEN:?missing TURSO_AUTH_TOKEN}"
PROBE_SOURCE="${PROBE_SOURCE:-github-actions}"
PROBE_RUNNER="${PROBE_RUNNER:-ubuntu-latest}"
PROBE_REGION="${PROBE_REGION:-gh-actions-us}"
PROBE_VERSION="${PROBE_VERSION:-probe-script.v1.0.0}"
PROBE_DEPTH="${PROBE_DEPTH:-shallow}"
SLO_LATENCY_MS="${SLO_LATENCY_MS:-5000}"

# Hourly deep-probe escalation — every 30th 2-min slot ≈ hourly.
# UTC minute that is a multiple of 60 → deep. Cheap and self-contained.
UTC_MIN=$(date -u +%M)
if [ "${UTC_MIN}" = "00" ] || [ "${UTC_MIN}" = "01" ]; then
  PROBE_DEPTH="deep"
fi

# ── 1. Generate probe_id (RFC 4122 v4 UUID) ───────────────────────────
PROBE_ID=$(cat /proc/sys/kernel/random/uuid)
ISSUED_AT=$(date -u +"%Y-%m-%dT%H:%M:%S.%6NZ")
USER_AGENT="apin-probe/${PROBE_VERSION} (${PROBE_SOURCE}; ${PROBE_RUNNER})"

# Build target URL with depth query
TARGET="${PROBE_TARGET_URL}?depth=${PROBE_DEPTH}"

echo "==> probe_id=${PROBE_ID}"
echo "==> target=${TARGET}"
echo "==> depth=${PROBE_DEPTH}"

# ── 2. curl with timing breakdown ─────────────────────────────────────
# Output body to /tmp/body and timing format to /tmp/timing
TIMING_FORMAT='%{http_code}|%{time_namelookup}|%{time_connect}|%{time_appconnect}|%{time_starttransfer}|%{time_total}|%{size_download}'
CURL_OUT=$(curl -sS \
  --max-time 30 \
  -H "Authorization: Bearer ${PROBE_TOKEN}" \
  -H "Accept: application/json" \
  -H "User-Agent: ${USER_AGENT}" \
  -o /tmp/probe_body \
  -w "${TIMING_FORMAT}" \
  "${TARGET}" 2>/tmp/probe_curl_err) || CURL_EXIT=$?
CURL_EXIT=${CURL_EXIT:-0}

# Parse timing output (only if curl produced it; on network failure it's empty)
HTTP_STATUS=0
T_DNS_S=0; T_CONNECT_S=0; T_APPCONNECT_S=0; T_STARTTRANSFER_S=0; T_TOTAL_S=0; SIZE_DOWNLOAD=0
if [ -n "${CURL_OUT}" ]; then
  IFS='|' read -r HTTP_STATUS T_DNS_S T_CONNECT_S T_APPCONNECT_S T_STARTTRANSFER_S T_TOTAL_S SIZE_DOWNLOAD <<< "${CURL_OUT}"
fi

# Convert curl's seconds-with-decimals to integer milliseconds
to_ms() { awk -v v="$1" 'BEGIN{printf "%d", v*1000}'; }
DNS_MS=$(to_ms "${T_DNS_S:-0}")
TCP_MS=$(to_ms "${T_CONNECT_S:-0}")
TLS_MS_RAW=$(to_ms "${T_APPCONNECT_S:-0}")
TLS_MS=$(( TLS_MS_RAW - TCP_MS ))         # TLS = appconnect - connect
[ "${TLS_MS}" -lt 0 ] && TLS_MS=0
TTFB_MS_RAW=$(to_ms "${T_STARTTRANSFER_S:-0}")
TTFB_MS=$(( TTFB_MS_RAW - TLS_MS_RAW ))   # TTFB = starttransfer - appconnect (server work)
[ "${TTFB_MS}" -lt 0 ] && TTFB_MS=0
TOTAL_MS=$(to_ms "${T_TOTAL_S:-0}")
DL_MS=$(( TOTAL_MS - TTFB_MS_RAW ))
[ "${DL_MS}" -lt 0 ] && DL_MS=0

# Response body checksum (first 16 hex chars of sha256)
BODY_SHA=""
if [ -s /tmp/probe_body ]; then
  BODY_SHA=$(sha256sum /tmp/probe_body | cut -c1-16)
fi

# ── 3. Validate JSON shape + extract metrics via jq ───────────────────
# We use jq for safe extraction. Any extraction failure → null.
SUCCESS=0
OVERALL="down"
ERROR_CLASS=""
ERROR_DETAIL=""
GATE_HTTP_2XX=0
GATE_JSON_PARSEABLE=0
GATE_SCHEMA_MATCH=0
GATE_OVERALL_OK=0
GATE_ALL_COMPONENTS_UP=0
GATE_LATENCY_UNDER_SLO=0
GATE_NO_RESOURCE_ALERTS=1

# Server-side fields default to null/empty
SERVER_RECV_AT="NULL"
SERVER_SEND_AT="NULL"
PROCESS_UPTIME_S="NULL"
MEMORY_RSS_MB="NULL"
MEMORY_PCT="NULL"
CPU_PCT_1M="NULL"
GPU_VRAM_USED_MB="NULL"
GPU_VRAM_TOTAL_MB="NULL"
DISK_FREE_GB="NULL"
OPEN_FDS="NULL"
EVENT_LOOP_LAG_MS="NULL"
REQ_COUNT_5M="NULL"
ERR_COUNT_5M="NULL"
ERR_RATE_5M="NULL"
P50_5M="NULL"
P95_5M="NULL"
BUILD_VERSION="NULL"
BUILD_GIT_SHA="NULL"
BUILD_DEPLOYED_AT="NULL"
COMPONENTS_JSON="{}"
COMPONENTS_UP_COUNT="NULL"
COMPONENTS_TOTAL_COUNT="NULL"
COMPONENT_FAILURES_JSON="[]"

# Gate 1: HTTP 2xx?
if [ "${HTTP_STATUS}" -ge 200 ] && [ "${HTTP_STATUS}" -lt 300 ]; then
  GATE_HTTP_2XX=1
fi

# Gate 2-7: only meaningful if body is parseable JSON
if [ -s /tmp/probe_body ] && jq -e . /tmp/probe_body >/dev/null 2>&1; then
  GATE_JSON_PARSEABLE=1
  # Gate 3: schema match — required top-level fields present
  if jq -e '.schema_version and .overall and .components and .server_ts_utc' /tmp/probe_body >/dev/null 2>&1; then
    GATE_SCHEMA_MATCH=1
    OVERALL=$(jq -r '.overall // "down"' /tmp/probe_body)
    SERVER_RECV_AT="\"$(jq -r '.server_recv_at_utc // ""' /tmp/probe_body)\""
    SERVER_SEND_AT="\"$(jq -r '.server_send_at_utc // ""' /tmp/probe_body)\""
    PROCESS_UPTIME_S=$(jq -r '.process_uptime_s // "null"' /tmp/probe_body)
    # Resources
    MEMORY_RSS_MB=$(jq -r '.resources.memory_rss_mb // "null"' /tmp/probe_body)
    MEMORY_PCT=$(jq -r '.resources.memory_pct // "null"' /tmp/probe_body)
    CPU_PCT_1M=$(jq -r '.resources.cpu_pct_1m // "null"' /tmp/probe_body)
    GPU_VRAM_USED_MB=$(jq -r '.resources.gpu_vram_used_mb // "null"' /tmp/probe_body)
    GPU_VRAM_TOTAL_MB=$(jq -r '.resources.gpu_vram_total_mb // "null"' /tmp/probe_body)
    DISK_FREE_GB=$(jq -r '.resources.disk_free_gb // "null"' /tmp/probe_body)
    OPEN_FDS=$(jq -r '.resources.open_fds // "null"' /tmp/probe_body)
    EVENT_LOOP_LAG_MS=$(jq -r '.resources.event_loop_lag_ms // "null"' /tmp/probe_body)
    # Internal stats
    REQ_COUNT_5M=$(jq -r '.internal_stats_last_5min.request_count_5m // "null"' /tmp/probe_body)
    ERR_COUNT_5M=$(jq -r '.internal_stats_last_5min.error_count_5m // "null"' /tmp/probe_body)
    ERR_RATE_5M=$(jq -r '.internal_stats_last_5min.error_rate_5m_pct // "null"' /tmp/probe_body)
    P50_5M=$(jq -r '.internal_stats_last_5min.p50_latency_5m_ms // "null"' /tmp/probe_body)
    P95_5M=$(jq -r '.internal_stats_last_5min.p95_latency_5m_ms // "null"' /tmp/probe_body)
    # Build
    BUILD_VERSION="\"$(jq -r '.build.version // ""' /tmp/probe_body)\""
    BUILD_GIT_SHA="\"$(jq -r '.build.git_sha // ""' /tmp/probe_body)\""
    BUILD_DEPLOYED_AT="\"$(jq -r '.build.deployed_at_utc // ""' /tmp/probe_body)\""
    # Components — serialize whole subtree as JSON string
    COMPONENTS_JSON=$(jq -c '.components // {}' /tmp/probe_body)
    COMPONENTS_UP_COUNT=$(jq -r '[.components[] | select(.status=="up")] | length' /tmp/probe_body 2>/dev/null || echo "null")
    COMPONENTS_TOTAL_COUNT=$(jq -r '.components | length' /tmp/probe_body 2>/dev/null || echo "null")
    COMPONENT_FAILURES_JSON=$(jq -c '[.components | to_entries[] | select(.value.status != "up") | .key] // []' /tmp/probe_body 2>/dev/null || echo "[]")
    # Gate 4: overall == "operational"
    if [ "${OVERALL}" = "operational" ]; then
      GATE_OVERALL_OK=1
    fi
    # Gate 5: all components up?
    if [ "${COMPONENTS_UP_COUNT}" = "${COMPONENTS_TOTAL_COUNT}" ] && [ "${COMPONENTS_UP_COUNT}" != "null" ]; then
      GATE_ALL_COMPONENTS_UP=1
    fi
    # Gate 7: no resource alerts.
    # Per-process memory (memory_rss_mb) < 1800 MB on HF free-tier (hard
    # limit is ~2 GB), open_fds < 1000, disk > 0.5 GB.
    # memory_pct was REMOVED from this gate — on a shared HF Space it
    # measures the WHOLE NODE's memory (including other tenants), which
    # is environmental noise our app can do nothing about.  See
    # _qa_tmp/investigate_resource_alerts.py for the diagnosis (memory_pct
    # ranged 71-96% while RSS stayed flat at ~1360 MB).
    if { [ "${MEMORY_RSS_MB}" = "null" ] || [ "${MEMORY_RSS_MB}" -lt 1800 ]; } && \
       { [ "${OPEN_FDS}" = "null" ] || [ "${OPEN_FDS}" -lt 1000 ]; } && \
       { [ "${DISK_FREE_GB}" = "null" ] || awk -v v="${DISK_FREE_GB}" 'BEGIN{exit !(v > 0.5)}'; }; then
      GATE_NO_RESOURCE_ALERTS=1
    else
      GATE_NO_RESOURCE_ALERTS=0
    fi
  fi
fi

# Gate 6: latency under SLO
if [ "${TOTAL_MS}" -lt "${SLO_LATENCY_MS}" ]; then
  GATE_LATENCY_UNDER_SLO=1
fi

# Pick error_class based on which gate failed first (in priority order)
if [ "${CURL_EXIT}" -ne 0 ]; then
  case "${CURL_EXIT}" in
    6)  ERROR_CLASS="dns_error" ;;
    7)  ERROR_CLASS="tcp_refused" ;;
    28) ERROR_CLASS="timeout" ;;
    35|51|58|59|60|77|82|83) ERROR_CLASS="tls_error" ;;
    *)  ERROR_CLASS="unknown" ;;
  esac
  ERROR_DETAIL="curl exit ${CURL_EXIT}: $(head -c 200 /tmp/probe_curl_err 2>/dev/null | tr -d '\n' | sed 's/"/\\"/g')"
elif [ "${GATE_HTTP_2XX}" = "0" ]; then
  if [ "${HTTP_STATUS}" -ge 500 ]; then
    ERROR_CLASS="http_5xx"
  elif [ "${HTTP_STATUS}" -ge 400 ]; then
    ERROR_CLASS="http_4xx"
  else
    ERROR_CLASS="http_3xx_unexpected"
  fi
  ERROR_DETAIL="HTTP ${HTTP_STATUS}"
elif [ "${GATE_JSON_PARSEABLE}" = "0" ]; then
  ERROR_CLASS="body_not_json"
  ERROR_DETAIL="response body was not valid JSON"
elif [ "${GATE_SCHEMA_MATCH}" = "0" ]; then
  ERROR_CLASS="body_schema_mismatch"
  ERROR_DETAIL="probe.v1 required fields missing"
elif [ "${SIZE_DOWNLOAD}" -lt 100 ]; then
  ERROR_CLASS="body_too_small"
  ERROR_DETAIL="response body was only ${SIZE_DOWNLOAD} bytes"
elif [ "${GATE_OVERALL_OK}" = "0" ]; then
  ERROR_CLASS="body_overall_not_ok"
  ERROR_DETAIL="overall = ${OVERALL}"
elif [ "${GATE_ALL_COMPONENTS_UP}" = "0" ]; then
  ERROR_CLASS="component_down"
  ERROR_DETAIL="$(echo "${COMPONENT_FAILURES_JSON}" | jq -r 'join(",")')"
elif [ "${GATE_LATENCY_UNDER_SLO}" = "0" ]; then
  ERROR_CLASS="latency_sla_breach"
  ERROR_DETAIL="total_ms=${TOTAL_MS} exceeded slo=${SLO_LATENCY_MS}"
elif [ "${GATE_NO_RESOURCE_ALERTS}" = "0" ]; then
  ERROR_CLASS="resource_alert"
  ERROR_DETAIL="memory_rss_mb=${MEMORY_RSS_MB} open_fds=${OPEN_FDS} disk_free_gb=${DISK_FREE_GB}"
fi

# Final success = all gates passed
if [ "${GATE_HTTP_2XX}" = "1" ] && [ "${GATE_JSON_PARSEABLE}" = "1" ] && \
   [ "${GATE_SCHEMA_MATCH}" = "1" ] && [ "${GATE_OVERALL_OK}" = "1" ] && \
   [ "${GATE_ALL_COMPONENTS_UP}" = "1" ] && [ "${GATE_LATENCY_UNDER_SLO}" = "1" ] && \
   [ "${GATE_NO_RESOURCE_ALERTS}" = "1" ]; then
  SUCCESS=1
  ERROR_CLASS=""
  ERROR_DETAIL=""
fi
SLA_BREACH=0
[ "${GATE_LATENCY_UNDER_SLO}" = "0" ] && SLA_BREACH=1

echo "==> http_status=${HTTP_STATUS} total_ms=${TOTAL_MS} success=${SUCCESS} error_class=${ERROR_CLASS:-none}"

# ── 6. INSERT into external_probes via Turso /v2/pipeline ─────────────
# Build the Hrana-HTTP pipeline request as JSON, then POST.
TURSO_HTTP_URL=$(echo "${TURSO_DATABASE_URL}" | sed 's|^libsql://|https://|')
RECORDED_AT=$(date -u +"%Y-%m-%dT%H:%M:%S.%6NZ")

# Helper: turn a value into a Hrana-HTTP arg object {type,value}
arg_text()    { printf '{"type":"text","value":%s}' "$(jq -Rn --arg v "$1" '$v')"; }
arg_int()     { if [ "$1" = "null" ] || [ -z "$1" ]; then printf '{"type":"null","value":null}'; else printf '{"type":"integer","value":"%s"}' "$1"; fi; }
arg_real()    { if [ "$1" = "null" ] || [ -z "$1" ]; then printf '{"type":"null","value":null}'; else printf '{"type":"float","value":%s}' "$1"; fi; }
arg_text_n()  { if [ -z "$1" ] || [ "$1" = "\"\"" ]; then printf '{"type":"null","value":null}'; else printf '{"type":"text","value":%s}' "$(jq -Rn --arg v "${1//\"/}" '$v')"; fi; }

# Build the SQL + args array.
PIPELINE_JSON=$(jq -n \
  --arg sql 'INSERT OR IGNORE INTO external_probes (probe_id, schema_version, issued_at_utc, server_recv_at_utc, server_send_at_utc, recorded_at_utc, success, overall, error_class, error_detail, sla_breach, http_status, target_url, response_bytes, response_checksum_sha256, dns_ms, tcp_connect_ms, tls_handshake_ms, ttfb_ms, download_ms, total_ms, components_json, components_up_count, components_total_count, component_failures_json, process_uptime_s, memory_rss_mb, memory_pct, cpu_pct_1m, gpu_vram_used_mb, gpu_vram_total_mb, disk_free_gb, open_fds, event_loop_lag_ms, request_count_5m, error_count_5m, error_rate_5m_pct, p50_latency_5m_ms, p95_latency_5m_ms, build_version, build_git_sha, build_deployed_at_utc, probe_source, probe_runner, probe_region, probe_version, probe_depth, probe_user_agent, gate_http_2xx, gate_json_parseable, gate_schema_match, gate_overall_ok, gate_all_components_up, gate_latency_under_slo, gate_no_resource_alerts) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)' \
  --arg probe_id "${PROBE_ID}" \
  --arg schema_version "probe.v1" \
  --arg issued_at "${ISSUED_AT}" \
  --arg server_recv "${SERVER_RECV_AT//\"/}" \
  --arg server_send "${SERVER_SEND_AT//\"/}" \
  --arg recorded_at "${RECORDED_AT}" \
  --argjson success "${SUCCESS}" \
  --arg overall "${OVERALL}" \
  --arg error_class "${ERROR_CLASS}" \
  --arg error_detail "${ERROR_DETAIL}" \
  --argjson sla_breach "${SLA_BREACH}" \
  --argjson http_status "${HTTP_STATUS}" \
  --arg target_url "${TARGET}" \
  --argjson response_bytes "${SIZE_DOWNLOAD}" \
  --arg body_sha "${BODY_SHA}" \
  --argjson dns_ms "${DNS_MS}" \
  --argjson tcp_ms "${TCP_MS}" \
  --argjson tls_ms "${TLS_MS}" \
  --argjson ttfb_ms "${TTFB_MS}" \
  --argjson dl_ms "${DL_MS}" \
  --argjson total_ms "${TOTAL_MS}" \
  --arg components_json "${COMPONENTS_JSON}" \
  --arg comp_up "${COMPONENTS_UP_COUNT}" \
  --arg comp_total "${COMPONENTS_TOTAL_COUNT}" \
  --arg comp_failures "${COMPONENT_FAILURES_JSON}" \
  --arg process_uptime "${PROCESS_UPTIME_S}" \
  --arg memory_rss "${MEMORY_RSS_MB}" \
  --arg memory_pct "${MEMORY_PCT}" \
  --arg cpu_pct "${CPU_PCT_1M}" \
  --arg gpu_used "${GPU_VRAM_USED_MB}" \
  --arg gpu_total "${GPU_VRAM_TOTAL_MB}" \
  --arg disk_free "${DISK_FREE_GB}" \
  --arg fds "${OPEN_FDS}" \
  --arg loop_lag "${EVENT_LOOP_LAG_MS}" \
  --arg req_5m "${REQ_COUNT_5M}" \
  --arg err_5m "${ERR_COUNT_5M}" \
  --arg err_rate "${ERR_RATE_5M}" \
  --arg p50 "${P50_5M}" \
  --arg p95 "${P95_5M}" \
  --arg build_version "${BUILD_VERSION//\"/}" \
  --arg build_sha "${BUILD_GIT_SHA//\"/}" \
  --arg build_deployed "${BUILD_DEPLOYED_AT//\"/}" \
  --arg probe_source "${PROBE_SOURCE}" \
  --arg probe_runner "${PROBE_RUNNER}" \
  --arg probe_region "${PROBE_REGION}" \
  --arg probe_version "${PROBE_VERSION}" \
  --arg probe_depth "${PROBE_DEPTH}" \
  --arg user_agent "${USER_AGENT}" \
  --argjson g1 "${GATE_HTTP_2XX}" \
  --argjson g2 "${GATE_JSON_PARSEABLE}" \
  --argjson g3 "${GATE_SCHEMA_MATCH}" \
  --argjson g4 "${GATE_OVERALL_OK}" \
  --argjson g5 "${GATE_ALL_COMPONENTS_UP}" \
  --argjson g6 "${GATE_LATENCY_UNDER_SLO}" \
  --argjson g7 "${GATE_NO_RESOURCE_ALERTS}" \
  '{
    requests: [{
      type: "execute",
      stmt: {
        sql: $sql,
        args: [
          {type:"text",value:$probe_id},
          {type:"text",value:$schema_version},
          {type:"text",value:$issued_at},
          (if $server_recv == "" then {type:"null",value:null} else {type:"text",value:$server_recv} end),
          (if $server_send == "" then {type:"null",value:null} else {type:"text",value:$server_send} end),
          {type:"text",value:$recorded_at},
          {type:"integer",value:($success|tostring)},
          {type:"text",value:$overall},
          (if $error_class == "" then {type:"null",value:null} else {type:"text",value:$error_class} end),
          (if $error_detail == "" then {type:"null",value:null} else {type:"text",value:$error_detail} end),
          {type:"integer",value:($sla_breach|tostring)},
          {type:"integer",value:($http_status|tostring)},
          {type:"text",value:$target_url},
          {type:"integer",value:($response_bytes|tostring)},
          (if $body_sha == "" then {type:"null",value:null} else {type:"text",value:$body_sha} end),
          {type:"integer",value:($dns_ms|tostring)},
          {type:"integer",value:($tcp_ms|tostring)},
          {type:"integer",value:($tls_ms|tostring)},
          {type:"integer",value:($ttfb_ms|tostring)},
          {type:"integer",value:($dl_ms|tostring)},
          {type:"integer",value:($total_ms|tostring)},
          {type:"text",value:$components_json},
          (if $comp_up == "null" or $comp_up == "" then {type:"null",value:null} else {type:"integer",value:$comp_up} end),
          (if $comp_total == "null" or $comp_total == "" then {type:"null",value:null} else {type:"integer",value:$comp_total} end),
          {type:"text",value:$comp_failures},
          (if $process_uptime == "null" or $process_uptime == "" then {type:"null",value:null} else {type:"integer",value:$process_uptime} end),
          (if $memory_rss == "null" or $memory_rss == "" then {type:"null",value:null} else {type:"integer",value:$memory_rss} end),
          (if $memory_pct == "null" or $memory_pct == "" then {type:"null",value:null} else {type:"float",value:($memory_pct|tonumber)} end),
          (if $cpu_pct == "null" or $cpu_pct == "" then {type:"null",value:null} else {type:"float",value:($cpu_pct|tonumber)} end),
          (if $gpu_used == "null" or $gpu_used == "" then {type:"null",value:null} else {type:"integer",value:$gpu_used} end),
          (if $gpu_total == "null" or $gpu_total == "" then {type:"null",value:null} else {type:"integer",value:$gpu_total} end),
          (if $disk_free == "null" or $disk_free == "" then {type:"null",value:null} else {type:"float",value:($disk_free|tonumber)} end),
          (if $fds == "null" or $fds == "" then {type:"null",value:null} else {type:"integer",value:$fds} end),
          (if $loop_lag == "null" or $loop_lag == "" then {type:"null",value:null} else {type:"integer",value:$loop_lag} end),
          (if $req_5m == "null" or $req_5m == "" then {type:"null",value:null} else {type:"integer",value:$req_5m} end),
          (if $err_5m == "null" or $err_5m == "" then {type:"null",value:null} else {type:"integer",value:$err_5m} end),
          (if $err_rate == "null" or $err_rate == "" then {type:"null",value:null} else {type:"float",value:($err_rate|tonumber)} end),
          (if $p50 == "null" or $p50 == "" then {type:"null",value:null} else {type:"integer",value:$p50} end),
          (if $p95 == "null" or $p95 == "" then {type:"null",value:null} else {type:"integer",value:$p95} end),
          (if $build_version == "" then {type:"null",value:null} else {type:"text",value:$build_version} end),
          (if $build_sha == "" then {type:"null",value:null} else {type:"text",value:$build_sha} end),
          (if $build_deployed == "" then {type:"null",value:null} else {type:"text",value:$build_deployed} end),
          {type:"text",value:$probe_source},
          {type:"text",value:$probe_runner},
          {type:"text",value:$probe_region},
          {type:"text",value:$probe_version},
          {type:"text",value:$probe_depth},
          {type:"text",value:$user_agent},
          {type:"integer",value:($g1|tostring)},
          {type:"integer",value:($g2|tostring)},
          {type:"integer",value:($g3|tostring)},
          {type:"integer",value:($g4|tostring)},
          {type:"integer",value:($g5|tostring)},
          {type:"integer",value:($g6|tostring)},
          {type:"integer",value:($g7|tostring)}
        ]
      }
    }]
  }')

# POST to Turso. Retry up to 3 times with exponential backoff for transient
# 5xx / network errors. Probe failure recording is critical — losing it would
# corrupt the uptime stats.
TURSO_RESP=""
for attempt in 1 2 3; do
  TURSO_RESP=$(curl -sS \
    --max-time 15 \
    -X POST \
    -H "Authorization: Bearer ${TURSO_AUTH_TOKEN}" \
    -H "Content-Type: application/json" \
    -d "${PIPELINE_JSON}" \
    "${TURSO_HTTP_URL}/v2/pipeline" 2>&1) && break
  echo "==> Turso INSERT attempt ${attempt} failed, retrying..."
  sleep $((attempt * 2))
done

if [ -n "${TURSO_RESP}" ]; then
  # Hrana returns {"results":[{"type":"ok",...}]} on success
  if echo "${TURSO_RESP}" | jq -e '.results[0].type == "ok"' >/dev/null 2>&1; then
    echo "==> recorded probe ${PROBE_ID} (success=${SUCCESS})"
  else
    echo "==> Turso reported an error:"
    echo "${TURSO_RESP}" | jq -C . 2>/dev/null || echo "${TURSO_RESP}"
  fi
else
  echo "==> Turso request failed after 3 attempts"
fi

# Always exit 0 — we want the next probe to run regardless of this one's fate
exit 0

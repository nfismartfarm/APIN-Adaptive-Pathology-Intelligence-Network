"""Poll the dxv-404/Apin Space build/run state and surface logs.

Usage:
    python deploy/watch_space.py           # poll until terminal state / timeout
    python deploy/watch_space.py logs build   # dump the build log and exit
    python deploy/watch_space.py logs run     # dump the run log and exit
"""
import sys
import time

import requests
from huggingface_hub import HfApi, get_token

REPO = "dxv-404/Apin"
TERMINAL_OK = {"RUNNING"}
TERMINAL_BAD = {"BUILD_ERROR", "RUNTIME_ERROR", "CONFIG_ERROR", "PAUSED",
                "SLEEPING", "DELETING"}


def fetch_logs(kind):
    """kind in {'build','run'} — returns the log text (best-effort)."""
    tok = get_token()
    url = f"https://huggingface.co/api/spaces/{REPO}/logs/{kind}"
    try:
        r = requests.get(url, headers={"Authorization": f"Bearer {tok}"},
                         timeout=30, stream=True)
        if r.status_code != 200:
            return f"(logs/{kind} HTTP {r.status_code})"
        out = []
        for line in r.iter_lines(decode_unicode=True):
            if line:
                # SSE-style "data: {...}" or plain text
                out.append(line)
            if len(out) > 4000:
                break
        return "\n".join(out) if out else f"(logs/{kind} empty)"
    except Exception as e:
        return f"(logs/{kind} fetch failed: {e})"


def poll(max_minutes=12):
    api = HfApi()
    last = None
    deadline = time.time() + max_minutes * 60
    while time.time() < deadline:
        rt = api.get_space_runtime(REPO)
        stage = rt.stage
        if stage != last:
            print(f"[{time.strftime('%H:%M:%S')}] stage -> {stage}", flush=True)
            last = stage
        if stage in TERMINAL_OK:
            print("BUILD+BOOT OK — Space is RUNNING", flush=True)
            return 0
        if stage in TERMINAL_BAD:
            print(f"TERMINAL BAD STATE: {stage}", flush=True)
            print("=" * 64)
            print("BUILD LOG (tail):")
            print("=" * 64)
            log = fetch_logs("build")
            print("\n".join(log.splitlines()[-120:]), flush=True)
            if stage == "RUNTIME_ERROR":
                print("=" * 64)
                print("RUN LOG (tail):")
                print("=" * 64)
                print("\n".join(fetch_logs("run").splitlines()[-120:]),
                      flush=True)
            return 1
        time.sleep(20)
    print(f"(still {last} after {max_minutes} min — poll again)", flush=True)
    return 2


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "logs":
        print(fetch_logs(sys.argv[2]))
        sys.exit(0)
    sys.exit(poll())

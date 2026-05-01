# Phase 3 Task 6 — Pre-commit Hook Verification

**Date:** 2026-05-01
**Authority:** master prompt Section 4 Phase 3 task 6 (lines 218-229)

## Hook installed at `.git/hooks/pre-commit`

```bash
#!/usr/bin/env bash
# Block modifications to Section 15 tests after Phase 3
if git diff --cached --name-only | grep -E '^tomato_sandbox/tests/integration/test_section15_.*\.py$' > /dev/null; then
  echo "ERROR: Section 15 test files are immutable. See tomato_master_prompt.md section 5 Rule A."
  echo "Files attempting modification:"
  git diff --cached --name-only | grep -E '^tomato_sandbox/tests/integration/test_section15_.*\.py$'
  exit 1
fi
```

Mode: `-rwxr-xr-x` (executable, verified by `ls -la .git/hooks/pre-commit`).

## Verification — dummy modification attempt

**Sequence:**

1. Append single newline to `tomato_sandbox/tests/integration/test_section15_tier1.py`. File grew 14431 → 14432 bytes.
2. Note: `tomato*/` is in `.gitignore`, so normal `git add` would skip this file. Used `git add -f` to force-stage (this is the actual attack surface — anyone could `git add -f` to override the ignore).
3. Stage confirmed: `git diff --cached --name-only` shows the file.
4. Attempted `git commit -m "DUMMY VERIFY HOOK BLOCKS - DO NOT KEEP"`.

**Hook fired with exact expected message:**

```
ERROR: Section 15 test files are immutable. See tomato_master_prompt.md section 5 Rule A.
Files attempting modification:
tomato_sandbox/tests/integration/test_section15_tier1.py
```

Commit was blocked. No commit was created.

## Cleanup

- `git reset HEAD tomato_sandbox/tests/integration/test_section15_tier1.py` — unstaged.
- `git checkout` skipped because the file is gitignored (not tracked, so no committed version to restore from).
- Manually trimmed the trailing newline via Python: file restored to 14431 bytes.

## Status

**Hook verified PASS:**
- Installed at `.git/hooks/pre-commit` with executable permissions.
- Fires correctly on staged Section 15 test file modification.
- Exits with code 1 (blocking the commit).
- Prints exact error message specified in master prompt sample script.

This satisfies master prompt Section 4 Phase 3 task 6 ("Make the hook executable: `chmod +x .git/hooks/pre-commit`. Verify by attempting a dummy modification and confirming the commit is blocked.").

Pre-commit framework registration (Phase 3 task 7) is N/A — the bash hook above is the chosen enforcement mechanism per DEC-020.

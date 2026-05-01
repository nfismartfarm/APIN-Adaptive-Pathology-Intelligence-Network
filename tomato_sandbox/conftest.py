"""
Phase 3 task 3 per master prompt Section 4 Phase 3 (lines 215).

This conftest.py is intentionally minimal. The 135 Section 15 integration tests
in `tomato_sandbox/tests/integration/test_section15_*.py` construct their
synthetic input dicts inline via per-file `_make_signal` / `_make_psv` /
`_make_classifier` / `_make_conformal` helper functions (per Section 15.2
conventions). They do NOT need shared fixtures at this stage.

Phase 4 implementer adds fixtures here as needed (e.g., real model loaders,
GPU lock fixtures, IQA sample images). Each Phase 4 task that needs a fixture
must:

  1. Add the fixture here with a docstring citing the spec section that motivates it.
  2. Use it from the relevant test file via parameter injection (`def test_x(fixture_name)`).
  3. Note the fixture in `tomato_decisions.md` if its scope or behavior is non-trivial.

Authority: master prompt Section 4 Phase 3 task 3.
"""
"""See module docstring; this file is intentionally a docstring-only stub at Phase 3 close."""

# `tests/`

`unittest` coverage for the reusable libraries (not the CLIs). Run with `pixi run test`
(`pixi run test-quiet` for the terse summary). Every external boundary is mocked so the suite needs
no camera, network, model download, display, or GPU; ~480 tests run in a few seconds. Discovery is
`python -m unittest discover -s tests -t .`, run from the repo root; `tests/` is a package
(`__init__.py`) so `from tests.fakes import …` resolves.

**Per-file notes live with the package each file exercises** (folded into that package's `CLAUDE.md`
so the production code and its tests are documented together). When editing a test file, read:

- `tests/fakes.py` — shared duck-typed stand-ins (see root `CLAUDE.md` → Tests).
- `tests/test_motor_metrics.py`, `tests/test_live.py` → `motor_metrics/CLAUDE.md`
- `tests/test_annotate.py` → `annotate/CLAUDE.md`
- `tests/test_feather.py` → `adafruit_feather_sense/CLAUDE.md`
- `tests/test_viewer.py` → `rerun_viewer/CLAUDE.md`

Shared harness conventions (what's patched, what runs unmocked) are in the root `CLAUDE.md` Tests
section.

# Final Whole-Branch Review Fix Report

## Finding 1: Interactive retraining recovery

- Files changed: `src/hmm_segmenter/cli.py`, `tests/test_cli.py`
- Change: Option `3` in `interactive_menu` now wraps `train_hmm` in a
  `try/except` block for `FileNotFoundError`, `OSError`, `UnicodeError`,
  `ValueError`, and `ModelError`. It prints `训练失败：...` and continues the
  menu loop instead of propagating the exception to `main`.
- Regression test:
  `test_interactive_retraining_failure_keeps_menu_running` patches `train_hmm`
  to raise `OSError("disk full")`, supplies `3` then `0` as user input, and
  asserts both that the failure message is printed and that the menu is shown
  again before exit.

## Finding 2: Normalized model validation

- Files changed: `src/hmm_segmenter/model.py`, `tests/test_model.py`
- Change: Probability validation now rejects values greater than `1` in `pi`,
  `A`, and `B`. `pi` must sum to `1`, every `A` row must sum to `1`, and every
  non-empty `B` row must sum to `1`; all sums use `math.isclose` with absolute
  and relative tolerances of `1e-9`. Empty `B` rows remain valid for sparse
  models.
- Fixture review: `tests/test_cli.py` and `tests/test_web.py` already used
  normalized non-empty emission rows. `tests/test_web_page.py` uses empty
  emission rows, which remain valid. `tests/test_model.py` was updated so its
  emission rows sum to `1` while preserving the Chinese and English boundary
  behavior.
- Regression tests: values greater than `1`, non-normalized `pi`,
  non-normalized `A`, and non-normalized non-empty `B`.

## Finding 3: Error logging

- Files changed: `web/app.py`, `tests/test_web.py`
- Change: The unexpected exception handler now calls
  `app.logger.exception("Unhandled application error")` before returning the
  existing `INTERNAL_ERROR` JSON response.
- Regression test:
  `test_unexpected_api_exception_logs_and_returns_internal_error` injects an
  API route that raises `RuntimeError`, asserts the response is status `500`
  with error code `INTERNAL_ERROR`, and verifies `logger.exception` is called.

## Verification commands and observed results

1. Focused tests:

   ```sh
   PYTHONPATH=src:. python3 -m unittest tests.test_model tests.test_cli tests.test_web -v
   ```

   Result: `Ran 31 tests ... OK`

2. Full suite with `ResourceWarning` promoted to an error:

   ```sh
   PYTHONPATH=src:. python3 -W error::ResourceWarning -m unittest discover -s tests -v
   ```

   Result: `Ran 49 tests ... OK`

3. Browser script syntax check:

   ```sh
   node --check web/static/app.js
   ```

   Result: exit code `0`, no output.

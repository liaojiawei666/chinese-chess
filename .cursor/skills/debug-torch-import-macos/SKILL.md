---
name: debug-torch-import-macos
description: >-
  Diagnose Python `import torch` failures on macOS in this repo: segfault,
  "symbol not found in flat namespace '__PyCode_GetExtra'", or "module
  'torch._C' has no attribute ...". Use whenever torch import crashes, pytest
  fails on torch, or before changing the Python/torch version to fix an import.
---

# Debug torch import failures (macOS, this repo)

This repo now uses **one** libtorch: both the Python trainer and the Rust datagen
side (`tch` via `LIBTORCH_USE_PYTORCH=1`) link the libtorch bundled in the venv's
pip `torch` wheel (currently torch 2.11 / libtorch 2.11; `tch` 0.24). Because it
is a single shared copy, the old "two copies version-skew" crash is gone — but a
stray `DYLD_LIBRARY_PATH`/`LIBTORCH` left over from an old setup, or pointing at a
*different* libtorch, can still break `import torch` in confusing ways.

## Check this FIRST (before touching Python/torch versions)

The leak is almost always the cause. Symptoms it produces, depending on torch version:
- `ImportError: symbol not found in flat namespace '__PyCode_GetExtra'`
- `AttributeError: module 'torch._C' has no attribute 'AcceleratorError' / '_dlpack_exchange_api'`
- plain segfault (exit 139) during `import torch`

Diagnose in one shot — if import works with the vars unset, the leak was the cause:

```bash
echo "DYLD_LIBRARY_PATH=[$DYLD_LIBRARY_PATH] LIBTORCH=[$LIBTORCH]"
env -u DYLD_LIBRARY_PATH -u DYLD_FALLBACK_LIBRARY_PATH -u LIBTORCH \
  trainer/.venv/bin/python -c "import torch; print('OK', torch.__version__)"
```

Fix: `unset DYLD_LIBRARY_PATH DYLD_FALLBACK_LIBRARY_PATH LIBTORCH` in that shell.
Also confirm it is not persisted: `grep -nE 'DYLD_LIBRARY_PATH|LIBTORCH' ~/.zshrc ~/.zprofile ~/.zshenv`.

## Root rule

The justfile sets `DYLD_LIBRARY_PATH`/`LIBTORCH_USE_PYTORCH` **inline per-recipe**
(now pointing at the venv's `torch/lib`), never globally — keep it that way. Since
Rust and Python share the same venv libtorch, a leak no longer version-skews, but
keeping env vars unset in your normal shell avoids any surprise.

## Do NOT immediately

- bump/downgrade the Python version, change the torch pin, `uv cache clean`, or
  rebuild the venv. Those were dead ends here; the import error was purely the
  DYLD leak. Only consider version changes after the leak is ruled out.

## Cursor-sandbox-only noise (ignore for the user's real terminal)

When running tools inside the Cursor agent sandbox:
- `uv run` → "No environment file found at: .env": pass `--no-env-file` (the
  sandbox injects `UV_ENV_FILE=.env`). Or call `trainer/.venv/bin/python` directly.
- `OMP: Error #179: ... Can't open SHM2 ... Operation not permitted`: OpenMP
  shared-memory is blocked by the sandbox; rerun the command outside the sandbox.

These do not affect the user's normal shell.

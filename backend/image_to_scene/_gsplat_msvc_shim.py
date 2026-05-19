"""Shim that strips the GCC-only `-Wno-attributes` flag from extra_cflags before
gsplat's JIT-compile pass invokes torch.utils.cpp_extension.load. MSVC rejects
this flag with error D8021.

Importing this module patches `torch.utils.cpp_extension.load` in place on
Windows. Safe no-op on non-Windows. Must be imported BEFORE any `import gsplat`
so the patch is in effect when gsplat's `_backend.py` runs at module load.
"""
from __future__ import annotations
import sys


def _strip(flags):
    if not flags:
        return flags
    return [f for f in flags if f != "-Wno-attributes"]


def _patch() -> None:
    if sys.platform != "win32":
        return
    try:
        from torch.utils import cpp_extension as _ce
    except ImportError:
        return

    if getattr(_ce, "_gsplat_msvc_patched", False):
        return

    # gsplat uses torch.utils.cpp_extension._jit_compile directly.
    _orig_jit = _ce._jit_compile

    def _patched_jit(name, sources, extra_cflags=None, extra_cuda_cflags=None,
                    *args, **kwargs):
        return _orig_jit(
            name, sources,
            _strip(extra_cflags), extra_cuda_cflags, *args, **kwargs
        )

    _ce._jit_compile = _patched_jit

    # Belt-and-suspenders: also patch the public .load in case other code paths use it.
    _orig_load = _ce.load

    def _patched_load(*args, **kwargs):
        flags = kwargs.get("extra_cflags")
        if flags is not None:
            kwargs["extra_cflags"] = _strip(flags)
        return _orig_load(*args, **kwargs)

    _ce.load = _patched_load
    _ce._gsplat_msvc_patched = True


_patch()

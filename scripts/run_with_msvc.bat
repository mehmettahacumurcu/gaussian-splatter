@echo off
REM Wrapper that sources Visual Studio 2022 BuildTools vcvars64 so gsplat's
REM CUDA extension can JIT-compile, then forwards all arguments to py -3.
REM
REM Usage:
REM   scripts\run_with_msvc.bat scripts\run_b_fixture.py --scene fixture-a-render --profile fast
REM
REM Also re-exports DISTUTILS_USE_SDK=1 so torch.utils.cpp_extension respects
REM the vcvars-configured MSVC.

call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
if errorlevel 1 (
    echo Failed to source vcvars64.bat
    exit /b 1
)
set DISTUTILS_USE_SDK=1
REM Force UTF-8 stdout/stderr so unicode arrows etc. in pipeline progress prints
REM don't crash with charmap encode errors on Windows.
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
py -3 %*

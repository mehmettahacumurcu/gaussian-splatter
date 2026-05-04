@echo off
REM One-shot launcher for scripts/reeval_nvs.py with conda + vcvars setup.
cd /d "%~dp0.."
set "VCVARS="
for %%E in (BuildTools Community Professional Enterprise) do (
  if not defined VCVARS if exist "C:\Program Files\Microsoft Visual Studio\2022\%%E\VC\Auxiliary\Build\vcvars64.bat" set "VCVARS=C:\Program Files\Microsoft Visual Studio\2022\%%E\VC\Auxiliary\Build\vcvars64.bat"
)
if not defined VCVARS (
  for %%E in (BuildTools Community Professional Enterprise) do (
    if not defined VCVARS if exist "C:\Program Files (x86)\Microsoft Visual Studio\2022\%%E\VC\Auxiliary\Build\vcvars64.bat" set "VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\2022\%%E\VC\Auxiliary\Build\vcvars64.bat"
  )
)
call "%VCVARS%" >nul
call conda activate gs4d
set "PYTHONUTF8=1"
set "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512"
python scripts\reeval_nvs.py %*

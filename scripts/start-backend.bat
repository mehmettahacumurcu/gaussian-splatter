@echo off
REM 4DGS Backend - FastAPI + gsplat + CUDA

cd /d "%~dp0.."

echo ========================================================
echo   4DGS Backend  -  http://127.0.0.1:8000  (Swagger: /docs)
echo ========================================================
echo.

REM --- 1/3: conda ---
echo [1/3] conda activate gs4d
call conda activate gs4d
if errorlevel 1 (
    echo.
    echo [HATA] conda activate gs4d basarisiz.
    echo        conda env create -f environment.yml ile olustur.
    pause
    exit /b 1
)

REM --- 2/3: vcvars64 ---
REM %ProgramFiles(x86)%'i PATH'te parantez icerdigi icin if blogu bozuluyor.
REM Ondan kacinmak icin direkt VS 2022 fixed pathlere bakiyoruz.
echo [2/3] VS Build Tools
set "VCVARS="
for %%E in (BuildTools Community Professional Enterprise) do (
    if not defined VCVARS if exist "C:\Program Files\Microsoft Visual Studio\2022\%%E\VC\Auxiliary\Build\vcvars64.bat" set "VCVARS=C:\Program Files\Microsoft Visual Studio\2022\%%E\VC\Auxiliary\Build\vcvars64.bat"
)
if not defined VCVARS (
    for %%E in (BuildTools Community Professional Enterprise) do (
        if not defined VCVARS if exist "C:\Program Files (x86)\Microsoft Visual Studio\2022\%%E\VC\Auxiliary\Build\vcvars64.bat" set "VCVARS=C:\Program Files (x86)\Microsoft Visual Studio\2022\%%E\VC\Auxiliary\Build\vcvars64.bat"
    )
)

if defined VCVARS (
    echo       bulundu: %VCVARS%
    call "%VCVARS%" >nul
    where cl >nul 2>&1
    if errorlevel 1 (
        echo       [UYARI] vcvars64 calisti ama 'cl' hala PATH'te degil.
    ) else (
        echo       [OK] cl.exe PATH'te
    )
) else (
    echo       [UYARI] VS 2022 vcvars64 bulunamadi.
    echo               VS 2022 Build Tools kurulu mu?
    echo               gsplat cache varsa sorun yok; yoksa compile fail eder.
)

REM --- 3/3: uvicorn ---
echo [3/3] UTF-8 stdout + uvicorn
set "PYTHONIOENCODING=utf-8"
set "PYTHONUTF8=1"
echo.
echo --------------------------------------------------------
echo   Baslatiliyor: uvicorn backend.api:app
echo --------------------------------------------------------
echo.

uvicorn backend.api:app --host 127.0.0.1 --port 8000

echo.
echo ========================================================
echo   Backend durdu.
echo ========================================================
pause

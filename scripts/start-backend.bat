@echo off
REM 4DGS Backend - FastAPI + gsplat + CUDA

cd /d "%~dp0.."

echo ========================================================
echo   4DGS Backend  -  http://127.0.0.1:8000  (Swagger: /docs)
echo ========================================================
echo.

REM vcvars64 bul (vswhere ile)
set "VCVARS="
if exist "%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" (
    for /f "usebackq tokens=*" %%i in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -find "VC\Auxiliary\Build\vcvars64.bat" 2^>nul`) do set "VCVARS=%%i"
)

if defined VCVARS (
    echo [OK] vcvars64 bulundu
    call "%VCVARS%" >nul 2>&1
) else (
    echo [UYARI] vcvars64.bat bulunamadi - VS Build Tools yok ya da vswhere yok.
    echo         gsplat cache varsa sorun yok, devam ediliyor...
)
echo.

call conda activate gs4d
if errorlevel 1 (
    echo.
    echo [HATA] conda activate gs4d basarisiz.
    echo        conda env create -f environment.yml ile olustur.
    pause
    exit /b 1
)

set "PYTHONIOENCODING=utf-8"
set "PYTHONUTF8=1"

echo [OK] Conda env gs4d aktif, UTF-8 modu aktif.
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

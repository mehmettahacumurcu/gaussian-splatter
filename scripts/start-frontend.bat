@echo off
REM 4DGS Frontend - Tauri + React dev mode

cd /d "%~dp0..\frontend"

echo ========================================================
echo   4DGS Frontend  -  Tauri + React + Vite
echo ========================================================
echo.

if not exist "node_modules" (
    echo [UYARI] node_modules yok - 'npm install' calistiriliyor...
    echo         Ilk kurulum 2-5 dk surebilir.
    echo.
    call npm install
    if errorlevel 1 (
        echo [HATA] npm install basarisiz. Node.js kurulu mu?
        pause
        exit /b 1
    )
    echo.
)

echo --------------------------------------------------------
echo   Baslatiliyor: npm run tauri dev
echo   Ilk sefer Rust compile ~1-2 dk surebilir.
echo --------------------------------------------------------
echo.

npm run tauri dev

echo.
echo ========================================================
echo   Frontend durdu.
echo ========================================================
pause

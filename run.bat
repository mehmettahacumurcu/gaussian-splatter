@echo off
REM 4DGS Studio launcher - backend + frontend penceresi
setlocal

echo.
echo ========================================================
echo   4DGS Studio - baslatiliyor
echo ========================================================
echo.

start "4DGS Backend" "%~dp0scripts\start-backend.bat"

REM 2 saniye bekle (Unix timeout'a dusmemek icin ping trick)
ping -n 3 127.0.0.1 >nul 2>&1

start "4DGS Frontend" "%~dp0scripts\start-frontend.bat"

echo Iki pencere acildi:
echo   [1] Backend   http://127.0.0.1:8000  (Swagger: /docs)
echo   [2] Frontend  Tauri pencere (ilk sefer Rust compile ~1-2 dk)
echo.
echo Bu pencereyi kapatabilirsin.
ping -n 4 127.0.0.1 >nul 2>&1
endlocal

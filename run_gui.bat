@echo off
title SoilML Toolbox Launcher
setlocal

rem === Change to this BAT directory ===
cd /d "%~dp0" || (
  echo [Error] Cannot change to script directory.
  pause
  exit /b 1
)

rem === Init Conda ===
if exist "C:\ProgramData\Anaconda3\Scripts\activate.bat" (
  call "C:\ProgramData\Anaconda3\Scripts\activate.bat"
) else if exist "%USERPROFILE%\anaconda3\Scripts\activate.bat" (
  call "%USERPROFILE%\anaconda3\Scripts\activate.bat"
) else (
  echo [Warning] Could not find conda activate script.
)

rem === Activate env (gpu by default) ===
call conda activate gpu 2>nul || call conda activate base

rem === Run GUI ===
python SoilML_Toolbox_GUI.py || (
  echo.
  echo [Error] Failed to launch GUI.
  echo Check file name and dependencies.
  pause
  exit /b 1
)

echo.
echo [Done] SoilML Toolbox closed.
pause

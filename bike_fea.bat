@echo off
cd /d "%~dp0"
".venv\Scripts\python.exe" -m bike_fea
if errorlevel 1 pause

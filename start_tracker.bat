@echo off
chcp 65001 >nul
title Fortnite Chest Tracker (OBS Widget)
echo Запуск Fortnite Chest Tracker...
echo Панель управления: http://localhost:5050/dashboard
echo Ссылка для OBS:    http://localhost:5050/widget
echo.
python app.py
pause

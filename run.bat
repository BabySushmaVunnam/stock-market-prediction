@echo off
title Stock Market Prediction Dashboard
cd /d "%~dp0"

echo.
echo  Starting Stock Market Prediction Dashboard...
echo  Open your browser at: http://localhost:8501
echo  Press Ctrl+C to stop.
echo.

"C:\Users\sushma vunnam\AppData\Local\Programs\Python\Python310\python.exe" -m streamlit run dashboard.py --server.port 8501
pause

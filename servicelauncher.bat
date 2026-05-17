@echo off
cd /d "%~dp0"

echo Launching CineIQ Project Stack...
echo ---------------------------------

echo Starting Backend API (Uvicorn)...
start "CineIQ Backend API" powershell -NoExit -Command "python -m uvicorn api:app --reload --port 8000"

timeout /t 3 /nobreak >nul
echo Starting Frontend Dashboard (Streamlit)...
start "CineIQ Streamlit Dashboard" powershell -NoExit -Command "python -m streamlit run app.py --server.port 8501"

echo ---------------------------------
echo Both services have been requested to start! 
echo You can close this main launcher window.
timeout /t 5
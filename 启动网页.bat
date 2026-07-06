@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo 正在启动排气风阀扭矩曲线分析网页...
echo 浏览器将自动打开 http://localhost:8501
echo 按 Ctrl+C 可停止服务
echo.
python -m streamlit run app.py --server.address 0.0.0.0 --server.port 8501
pause

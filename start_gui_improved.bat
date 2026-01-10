@echo off
REM CodeAssistant GUI 启动脚本 - 改进版
REM ========================================
REM 支持正确的进程终止

echo.
echo ========================================
echo  CodeAssistant GUI 启动器 (改进版)
echo ========================================
echo.

REM 检查虚拟环境
if not exist ".venv\Scripts\activate.bat" (
    echo [错误] 虚拟环境不存在
    echo 请先运行: python -m venv .venv
    echo 然后安装依赖: .venv\Scripts\pip install -r requirements.txt
    pause
    exit /b 1
)

REM 激活虚拟环境
echo [1/3] 激活虚拟环境...
call .venv\Scripts\activate.bat

REM 检查 Streamlit
echo [2/3] 检查依赖...
python -c "import streamlit" 2>nul
if errorlevel 1 (
    echo [警告] Streamlit 未安装，正在安装...
    pip install streamlit plotly
)

REM 启动 GUI (使用改进的启动脚本)
echo [3/3] 启动 GUI...
echo.
echo ========================================
echo  GUI 将在浏览器中自动打开
echo  地址: http://localhost:8501
echo  
echo  [重要] 停止方法:
echo  1. 按 Ctrl+C 然后输入 Y
echo  2. 或直接关闭此窗口
echo  3. 如卡住，打开任务管理器结束 streamlit.exe
echo ========================================
echo.

REM 使用 Python 启动脚本以获得更好的进程控制
python start_streamlit.py

pause

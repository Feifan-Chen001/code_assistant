"""
Streamlit 启动脚本 - 支持正确的信号处理
在 Windows 上可以正确响应 Ctrl+C 停止命令
"""
import sys
import signal
import subprocess
from pathlib import Path


def signal_handler(sig, frame):
    """处理 Ctrl+C 信号"""
    print("\n正在停止 Streamlit 应用...")
    sys.exit(0)


def main():
    # 注册信号处理器
    signal.signal(signal.SIGINT, signal_handler)
    if sys.platform == "win32":
        signal.signal(signal.SIGBREAK, signal_handler)
    
    # 获取当前目录
    current_dir = Path(__file__).parent
    app_path = current_dir / "app.py"
    
    print(f"启动 Streamlit 应用: {app_path}")
    print("按 Ctrl+C 停止应用")
    print("-" * 50)
    
    try:
        # 使用 subprocess 启动 streamlit，这样可以更好地控制进程
        process = subprocess.Popen(
            [sys.executable, "-m", "streamlit", "run", str(app_path)],
            cwd=str(current_dir)
        )
        
        # 等待进程结束
        process.wait()
        
    except KeyboardInterrupt:
        print("\n收到停止信号...")
        if process:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                print("强制终止进程...")
                process.kill()
        print("应用已停止")
    except Exception as e:
        print(f"错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

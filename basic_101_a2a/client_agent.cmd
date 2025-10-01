@echo off

@title A2A Client Agent

REM Change to project directory
cd /d C:\Carlo\projects\a2a

REM Create venv if it doesn't exist
if not exist a2a-env (
    python -m venv a2a-env
)

REM Activate venv
call a2a-env\Scripts\activate.bat

REM Run the client Agent
cd /d C:\Carlo\projects\a2a\basic_101_a2a
python client_agent.py

pause
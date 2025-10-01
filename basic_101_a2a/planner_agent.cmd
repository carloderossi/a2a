@echo off

@title A2A Planner Agent

REM Change to project directory
cd /d C:\Carlo\projects\a2a

REM Create venv if it doesn't exist
if not exist a2a-env (
    python -m venv a2a-env
)

REM Activate venv
call a2a-env\Scripts\activate.bat

REM Run the palnner agent
cd /d C:\Carlo\projects\a2a\basic_101_a2a
python planner_agent.py

pause
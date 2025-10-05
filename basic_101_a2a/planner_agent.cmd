@echo off

@title A2A Planner Agent

REM Change to project directory
cd /d C:\Carlo\projects\a2a

REM Run the Planner Agent inside the uv-managed .venv
cd /d C:\Carlo\projects\a2a\basic_101_a2a
uv run python planner_agent.py

pause
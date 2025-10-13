@echo off

@title A2A Planner Agent

REM Change to project directory
cd /d C:\Carlo\projects\

REM Run the Planner Agent inside the uv-managed .venv
cd /d C:\Carlo\projects\a2a\mcp_a2a_simple
uv run python planner_agent_mcp.py

pause
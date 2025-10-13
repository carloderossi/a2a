@echo off

@title A2A Research Agent

REM Change to project directory
cd /d C:\Carlo\projects\a2a

REM Run the Research Agent inside the uv-managed .venv
cd /d C:\Carlo\projects\a2a\mcp_a2a_simple
uv run python research_agent_mcp.py

pause
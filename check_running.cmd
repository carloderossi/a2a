@echo off
setlocal

set URL1=http://localhost:9002/.well-known/agent-card.json
set URL2=http://localhost:9001/.well-known/agent-card.json
set URL3=http://127.0.0.1:11434/

echo Checking URL: %URL1%
curl -s -o nul -w "Status: %%{http_code}\n" %URL1%
echo.

echo Checking URL: %URL2%
curl -s -o nul -w "Status: %%{http_code}\n" %URL2%
echo.

echo Checking URL: %URL3%
curl -s "Status: %%{http_code}\n" %URL3%
echo.
curl -s -L "Status: %%{http_code}\n" %URL3%/api/tags
echo.

endlocal
pause
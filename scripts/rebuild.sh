#!/bin/zsh
cd /Users/abebraham/Desktop/resume/edge
python3 scripts/parse_candles.py && python3 scripts/weather_core.py 2>&1 | grep -E "events|consistency" && python3 scripts/weather_snap.py 2>&1 | head -3

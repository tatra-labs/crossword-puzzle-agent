#!/usr/bin/env bash
# Repeatable demo sequence for the one-minute video.
#
# Run it once to warm the clue cache (so the recorded take is fast and identical),
# then record the second run. Shot numbers match docs/VIDEO_SCRIPT.md.
#
#   ./scripts/demo.sh warm      # shot rehearsal + cache warming, don't record this
#   ./scripts/demo.sh record    # the take
#
set -euo pipefail
cd "$(dirname "$0")/.."

export PYTHONIOENCODING=utf-8
PUZZLE="${XWORD_DEMO_PUZZLE:-data/puzzles/bundled/midi-01.json}"

pause() { [ "${MODE:-record}" = "record" ] && sleep "${1:-2}" || true; }

MODE="${1:-record}"

if [ "$MODE" = "warm" ]; then
    echo "Warming the clue cache so the recorded run is fast and deterministic..."
    xword solve "$PUZZLE" --quiet >/dev/null
    xword eval run --suite bundled --systems full,greedy-llm --seed 0 >/dev/null
    echo "Ready. Now run: ./scripts/demo.sh record"
    exit 0
fi

clear

# --- Shots C/D/E: the solve, the self-critique, the result -------------------
xword solve "$PUZZLE" --clues
pause 3

# --- Shot F: the evaluation report ------------------------------------------
clear
echo "$ xword eval run --suite bundled --systems full,greedy-llm"
echo
sed -n '1,40p' reports/report.md
pause 3

echo
echo "Full report: reports/report.html"

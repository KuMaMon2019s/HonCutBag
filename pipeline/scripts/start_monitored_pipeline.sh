#!/bin/bash
# Start the HonCut pipeline with phase-level monitoring.

set -e

if [ "$#" -lt 3 ]; then
    echo "Usage: $0 INPUT DURATION OUTPUT_DIR [MEDIA_PROFILE]" >&2
    exit 2
fi

INPUT="$1"
DURATION="$2"
OUTPUT_DIR="$3"
MEDIA_PROFILE="${4:-720p}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PIPELINE_DIR="$(dirname "$SCRIPT_DIR")"
CONFIG_FILE="$(mktemp "${TMPDIR:-/tmp}/honcut_config.XXXXXX.json")"

cleanup() {
    rm -f "$CONFIG_FILE"
}
trap cleanup EXIT

python - "$CONFIG_FILE" "$INPUT" "$DURATION" "$OUTPUT_DIR" "$MEDIA_PROFILE" <<'PY'
import json
import sys
from pathlib import Path

config_file, input_file, duration, output_dir, media_profile = sys.argv[1:]
with open(config_file, "w", encoding="utf-8") as handle:
    json.dump(
        {
            "input": str(Path(input_file).expanduser().resolve()),
            "duration": int(duration),
            "output_dir": str(Path(output_dir).expanduser().resolve()),
            "media_profile": media_profile,
            "transition": "crossfade",
            "auto_approve": True,
        },
        handle,
        ensure_ascii=False,
        indent=2,
    )
PY

echo "🐼 Starting HonCut Pipeline with Phase Monitoring"
echo "Config: $CONFIG_FILE"
echo "Output: $OUTPUT_DIR"

cd "$PIPELINE_DIR"
python scripts/phase_orchestrator.py --config "$CONFIG_FILE" &
ORCHESTRATOR_PID=$!

echo "Orchestrator PID: $ORCHESTRATOR_PID"
echo ""
echo "To monitor progress, run:"
echo "  python $SCRIPT_DIR/cron_monitor.py --output-dir '$OUTPUT_DIR' --process-id $ORCHESTRATOR_PID"
echo ""
echo "To create a Hermes Cron job for monitoring:"
echo "  cronjob create --name 'HonCut Phase Monitor' \\"
echo "    --schedule 'every 2m' \\"
echo "    --prompt \"Check HonCut pipeline progress: python $SCRIPT_DIR/cron_monitor.py --output-dir '$OUTPUT_DIR' --process-id $ORCHESTRATOR_PID\""

set +e
wait "$ORCHESTRATOR_PID"
EXIT_CODE=$?
set -e

if [ "$EXIT_CODE" -eq 0 ]; then
    echo ""
    echo "🎉 Pipeline completed successfully!"
else
    echo ""
    echo "❌ Pipeline failed with exit code $EXIT_CODE"
fi

exit "$EXIT_CODE"

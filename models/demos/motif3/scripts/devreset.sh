#!/usr/bin/env bash
# Reset the Blackhole Galaxy chips after a hung/killed device job. Takes the same device lock as devrun.sh.
#   scripts/devreset.sh            -> tt-smi -r   (reset all PCIe chips)
#   (glx tray reset is unavailable: it needs sudo ipmitool)
# Only use after a job hung or was SIGKILLed; a normal failed test does not need a reset.
set -uo pipefail
ROOT=/home/ttuser/hchang/experiments/motif-3
LOCK=$ROOT/.device.lock
MODE=${1:-r}
exec 9>"$LOCK"
flock 9
mkdir -p "$ROOT/logs/dev"
LOG="$ROOT/logs/dev/$(date +%Y%m%d_%H%M%S)_reset_${MODE}.log"
if [[ $MODE == glx ]]; then
  # -glx_reset* drives `sudo ipmitool` and fails without a password on this host (2026-10-01); `tt-smi -r` works.
  echo "[devreset] glx reset needs sudo ipmitool (unavailable); use plain 'scripts/devreset.sh' (tt-smi -r)" >&2
  exit 2
fi
ARGS=(-r)
timeout 900 uv tool run --from git+https://github.com/tenstorrent/tt-smi tt-smi "${ARGS[@]}" > "$LOG" 2>&1
RC=$?
echo "[devreset] tt-smi ${ARGS[*]} exit=$RC log=$LOG"
tail -n 15 "$LOG"
exit $RC

#!/usr/bin/env bash
# Run ONE command on the Blackhole Galaxy with exclusive access to the 32 chips.
#
#   scripts/devrun.sh [-t TIMEOUT_S] [-n NAME] -- <command...>
#
# - Serializes every device job through an flock on $LOCK (other callers wait).
# - Sets up the tt-metal env (venv, TT_METAL_HOME, PYTHONPATH) and runs from the tt-metal root.
# - Logs to logs/dev/<timestamp>_<name>.log and prints the log path + exit code.
# - On timeout sends SIGTERM first (SIGKILL only after a grace period): killing a device job
#   mid-op can wedge the next open.
set -uo pipefail

ROOT=/home/ttuser/hchang/experiments/motif-3
METAL=$ROOT/tt-metal
LOCK=$ROOT/.device.lock
TIMEOUT_S=1800
NAME=job

while [[ $# -gt 0 ]]; do
  case "$1" in
    -t) TIMEOUT_S=$2; shift 2 ;;
    -n) NAME=$2; shift 2 ;;
    --) shift; break ;;
    *) break ;;
  esac
done
if [[ $# -eq 0 ]]; then
  echo "usage: $0 [-t TIMEOUT_S] [-n NAME] -- <command...>" >&2
  exit 2
fi

mkdir -p "$ROOT/logs/dev"
TS=$(date +%Y%m%d_%H%M%S)
LOG="$ROOT/logs/dev/${TS}_${NAME//[^A-Za-z0-9_.-]/_}.log"

exec 9>"$LOCK"
echo "[devrun] waiting for device lock ($LOCK) ..." >&2
flock 9
echo "[devrun] lock acquired; log: $LOG" >&2

(
  cd "$METAL" || exit 97
  # shellcheck disable=SC1091
  source "$METAL/python_env/bin/activate"
  export TT_METAL_HOME="$METAL"
  export PYTHONPATH="$METAL${PYTHONPATH:+:$PYTHONPATH}"
  export ARCH_NAME=blackhole
  echo "[devrun] $(date -Is) cmd: $*"
  timeout --signal=TERM --kill-after=120 "$TIMEOUT_S" "$@"
) > "$LOG" 2>&1
RC=$?

echo "[devrun] exit=$RC log=$LOG"
if [[ $RC -eq 124 || $RC -eq 137 ]]; then
  echo "[devrun] TIMEOUT after ${TIMEOUT_S}s — the next device open may hang; consider scripts/devreset.sh" >&2
fi
# show the tail for convenience
tail -n 25 "$LOG"
exit $RC

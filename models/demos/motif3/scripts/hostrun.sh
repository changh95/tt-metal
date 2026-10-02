#!/usr/bin/env bash
# Run ONE host-only command with the Tenstorrent devices HIDDEN (CPU tests, vLLM host tests, CPU goldens).
#
#   scripts/hostrun.sh [-t TIMEOUT_S] [-n NAME] -- <command...>
#
#   scripts/hostrun.sh -- python -m pytest --noconftest -p no:cacheprovider -o addopts="" --import-mode=importlib -q \
#       models/demos/motif3/tests/unit/test_infra_config.py
#   scripts/hostrun.sh -n bridge -- python -m pytest -p no:cacheprovider -q models/demos/motif3/tests/test_generator_vllm_host.py
#
# - Enters a private user + mount namespace (unshare -Urm --propagation private, unprivileged on this host) and mounts
#   an empty tmpfs over /dev/tenstorrent. Nothing inside -- tt-metal's root conftest.py (which opens the UMD cluster
#   even for --collect-only), ttnn, UMD, vLLM's TT platform -- can reach the 32 shared chips: a stray device open fails
#   with "No chips detected" instead of disturbing the device job another agent is running under scripts/devrun.sh.
#   Never run plain pytest / python on this host for host-only work; use this wrapper.
# - Same environment as scripts/devrun.sh (tt-metal venv, TT_METAL_HOME, PYTHONPATH=<tt-metal root>, ARCH_NAME) and
#   runs from the tt-metal root, but takes NO device lock: host jobs run while a device job holds the chips.
# - The namespace is private to this command: the mount and the uid mapping vanish when it exits (files it writes are
#   owned by the calling user as usual).
# - -n NAME also tees the output to logs/host/<timestamp>_<NAME>.log. -t TIMEOUT_S (default 3600) SIGTERMs the command
#   (SIGKILL 60 s later). The exit code is the command's (124 on timeout).
set -uo pipefail

ROOT=/home/ttuser/hchang/experiments/motif-3
METAL=$ROOT/tt-metal
TIMEOUT_S=3600
NAME=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    -t) TIMEOUT_S=$2; shift 2 ;;
    -n) NAME=$2; shift 2 ;;
    -h|--help) sed -n '2,24p' "$0"; exit 0 ;;
    --) shift; break ;;
    *) break ;;
  esac
done
if [[ $# -eq 0 ]]; then
  echo "usage: $0 [-t TIMEOUT_S] [-n NAME] -- <command...>" >&2
  exit 2
fi
if [[ ! -d /dev/tenstorrent ]]; then
  echo "[hostrun] /dev/tenstorrent does not exist on this host; nothing to hide" >&2
fi

# The inner script runs as (mapped) root inside the new namespaces; "$@" is passed through untouched.
INNER='
set -uo pipefail
if [[ -d /dev/tenstorrent ]]; then
  mount -t tmpfs -o size=1m,mode=0755 none /dev/tenstorrent || { echo "[hostrun] cannot hide /dev/tenstorrent" >&2; exit 98; }
  if [[ -n "$(ls -A /dev/tenstorrent)" ]]; then echo "[hostrun] /dev/tenstorrent still populated; refusing" >&2; exit 98; fi
fi
cd "$HOSTRUN_METAL" || exit 97
# shellcheck disable=SC1091
source "$HOSTRUN_METAL/python_env/bin/activate"
export TT_METAL_HOME="$HOSTRUN_METAL"
export PYTHONPATH="$HOSTRUN_METAL${PYTHONPATH:+:$PYTHONPATH}"
export ARCH_NAME=blackhole
echo "[hostrun] $(date -Is) devices hidden (/dev/tenstorrent: $(ls -A /dev/tenstorrent 2>/dev/null | wc -l) entries); cmd: $*" >&2
exec timeout --signal=TERM --kill-after=60 "$HOSTRUN_TIMEOUT" "$@"
'

run() {
  HOSTRUN_METAL="$METAL" HOSTRUN_TIMEOUT="$TIMEOUT_S" \
    unshare -Urm --propagation private bash -c "$INNER" hostrun "$@"
}

if [[ -n "$NAME" ]]; then
  mkdir -p "$ROOT/logs/host"
  LOG="$ROOT/logs/host/$(date +%Y%m%d_%H%M%S)_${NAME//[^A-Za-z0-9_.-]/_}.log"
  run "$@" 2>&1 | tee "$LOG"
  RC=${PIPESTATUS[0]}
  echo "[hostrun] exit=$RC log=$LOG" >&2
else
  run "$@"
  RC=$?
fi
if [[ $RC -eq 124 ]]; then
  echo "[hostrun] TIMEOUT after ${TIMEOUT_S}s" >&2
fi
exit $RC

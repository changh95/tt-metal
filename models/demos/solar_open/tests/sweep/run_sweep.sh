#!/usr/bin/env bash
# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0
#
# ISL/OSL x batch sweep driver for models/demos/solar_open/tests/test_multi_user_regression.py.
#
# One pytest process per batch size, selected by its exact node id (`-k batch1` would also match batch16), each under
# `timeout` (a hung device run must not block the sweep), with a per-batch log, a ledger line per attempt and a board
# reset + cool-down after any failure. A batch that hit its timeout is retried once after the reset (a hang on a
# throttled board is the known failure mode of this box); an assertion failure is recorded and the sweep moves on.
#
#   source env.sh   # python_env, TT_METAL_HOME, HF_MODEL, TT_CACHE_PATH, MESH_DEVICE=P150x8
#   models/demos/solar_open/tests/sweep/run_sweep.sh                                 # batches 1 2 4 8 16 32
#   BATCHES="1 32" TAG=_rerun PAIRS="128:128,8192:1024" models/demos/solar_open/tests/sweep/run_sweep.sh
#
# Knobs (environment): BATCHES ("1 2 4 8 16 32"), TMO (pytest timeout per batch, s, 3600), TAG (results file suffix,
# SOLAR_OPEN_REGRESSION_TAG), PAIRS (SOLAR_OPEN_REGRESSION_PAIRS subset), COOLDOWN_C (78: the harness waits for every
# board to cool below it before each timed prefill and the first decode step; 0 disables), COOLDOWN_TIMEOUT_S (120),
# START_BELOW_C (60: wait for the boards before starting a batch, perf consistency; 0 disables), START_WAIT_S (900),
# RESET_BELOW_C (60: temperature to wait for after a `tt-smi -r`), RETRY_TIMEOUTS (1: rerun a timed-out batch once),
# LOG_DIR (generated/solar_open_multi_user_regression/logs; the ledger runs.txt and the per-batch logs go there),
# DRY_RUN (1: print the commands, touch no device).
set -u

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$SCRIPT_DIR/../../../../.." && pwd)
TEST=models/demos/solar_open/tests/test_multi_user_regression.py
NODE_PREFIX="$TEST::test_multi_user_regression[blackhole-1x8-batch"

BATCHES=${BATCHES:-"1 2 4 8 16 32"}
TMO=${TMO:-3600}
TAG=${TAG:-${SOLAR_OPEN_REGRESSION_TAG:-}}
PAIRS=${PAIRS:-${SOLAR_OPEN_REGRESSION_PAIRS:-}}
COOLDOWN_C=${COOLDOWN_C:-78}
COOLDOWN_TIMEOUT_S=${COOLDOWN_TIMEOUT_S:-120}
START_BELOW_C=${START_BELOW_C:-60}
START_WAIT_S=${START_WAIT_S:-900}
RESET_BELOW_C=${RESET_BELOW_C:-60}
RETRY_TIMEOUTS=${RETRY_TIMEOUTS:-1}
LOG_DIR=${LOG_DIR:-$REPO/generated/solar_open_multi_user_regression/logs}
DRY_RUN=${DRY_RUN:-0}
LEDGER=$LOG_DIR/runs.txt
JSONL=$REPO/generated/solar_open_multi_user_regression/Solar-Open-100B_1x8${TAG}.jsonl

cd "$REPO" || exit 2
mkdir -p "$LOG_DIR"
if [ "$DRY_RUN" != "1" ]; then
    : "${HF_MODEL:?source env.sh first (HF_MODEL, TT_CACHE_PATH, MESH_DEVICE, python_env)}"
    command -v tt-smi > /dev/null || { echo "tt-smi not on PATH"; exit 2; }
fi
export SOLAR_OPEN_REGRESSION_TAG="$TAG"
export SOLAR_OPEN_REGRESSION_TIMEOUT_S="$TMO"  # the harness's pytest-timeout marker (it overrides --timeout)
export SOLAR_OPEN_REGRESSION_COOLDOWN_C="$COOLDOWN_C"
export SOLAR_OPEN_REGRESSION_COOLDOWN_TIMEOUT_S="$COOLDOWN_TIMEOUT_S"
if [ -n "$PAIRS" ]; then export SOLAR_OPEN_REGRESSION_PAIRS="$PAIRS"; else unset SOLAR_OPEN_REGRESSION_PAIRS; fi

log() { echo "=== $(date +%FT%T) $*"; }

# Max ASIC temperature over the boards (tt-smi snapshot), "nan" when telemetry is unavailable.
max_temp() {
    timeout 60 tt-smi -s --snapshot_no_tty 2> /dev/null | python3 -c '
import json, sys
out = sys.stdin.read()
try:
    data = json.loads(out[out.find("{"):])
    temps = [float(str(d.get("telemetry", {}).get("asic_temperature", "nan")).strip()) for d in data.get("device_info", [])]
    print(max(temps) if temps else "nan")
except Exception:
    print("nan")
'
}

# wait_below <temperature C> <timeout s> <stage>: block until every board is below the temperature.
wait_below() {
    local limit=$1 budget=$2 stage=$3 t0 t
    [ "$limit" = "0" ] || [ "$DRY_RUN" = "1" ] && return 0
    t0=$(date +%s)
    while :; do
        t=$(max_temp)
        if [ "$t" = "nan" ]; then log "$stage: tt-smi telemetry unavailable, not waiting"; return 0; fi
        if python3 -c "import sys; sys.exit(0 if float('$t') < float('$limit') else 1)"; then
            log "$stage: max board temperature $t C < $limit C"
            return 0
        fi
        if [ $(( $(date +%s) - t0 )) -ge "$budget" ]; then
            log "$stage: max board temperature still $t C after ${budget}s (limit $limit C), continuing"
            return 1
        fi
        sleep 10
    done
}

# Kill python processes still holding a device (a hung run past its timeout), then reset the boards and cool down.
reset_boards() {
    local pids
    pids=$(fuser /dev/tenstorrent/* 2> /dev/null | tr ' ' '\n' | sort -u)
    for pid in $pids; do
        if readlink "/proc/$pid/exe" 2> /dev/null | grep -q python; then
            log "killing leftover device process $pid ($(tr '\0' ' ' < "/proc/$pid/cmdline" 2> /dev/null | cut -c1-120))"
            kill -9 "$pid" 2> /dev/null
        fi
    done
    sleep 5
    log "resetting boards (tt-smi -r)"
    timeout 300 tt-smi -r > "$LOG_DIR/tt-smi-reset_$(date +%s).log" 2>&1
    log "reset exit $?"
    wait_below "$RESET_BELOW_C" 900 "after reset"
}

log "SWEEP start: batches [$BATCHES] pairs ${SOLAR_OPEN_REGRESSION_PAIRS:-all} tag '${TAG}' TMO ${TMO}s cooldown ${COOLDOWN_C}C git $(git rev-parse --short HEAD 2> /dev/null)"
log "results $JSONL, logs + ledger under $LOG_DIR"
echo "# $(date +%FT%T) sweep start batches [$BATCHES] pairs ${SOLAR_OPEN_REGRESSION_PAIRS:-all} tag '${TAG}' TMO ${TMO}s cooldown ${COOLDOWN_C}C git $(git rev-parse --short HEAD 2> /dev/null)" >> "$LEDGER"

overall=0
for b in $BATCHES; do
    attempt=1
    while :; do
        logfile=$LOG_DIR/sweep_b${b}_a${attempt}.log
        node="${NODE_PREFIX}${b}]"
        cmd=(timeout -k 60 $((TMO + 180)) python -m pytest "$node" --timeout "$TMO" --timeout-method thread -p no:cacheprovider)
        wait_below "$START_BELOW_C" "$START_WAIT_S" "before batch $b"
        log "BATCH $b attempt $attempt START -> $logfile"
        echo "+ ${cmd[*]}"
        t0=$(date +%s)
        if [ "$DRY_RUN" = "1" ]; then
            rc=0
            echo "dry run: $node" > "$logfile"
        else
            "${cmd[@]}" > "$logfile" 2>&1
            rc=$?
        fi
        wall=$(( $(date +%s) - t0 ))
        summary=$(grep -E '=+ .*(passed|failed|error|skipped).* in [0-9.]+s' "$logfile" | tail -1 | tr -d '=' | sed 's/^ *//;s/ *$//')
        timed_out=0
        if [ $rc -eq 124 ] || [ $rc -eq 137 ] || grep -qE "Timeout \(>|\+\+\+ Timeout|Failed: Timeout" "$logfile"; then timed_out=1; fi
        rows=$( [ -f "$JSONL" ] && python3 -c "
import json, sys
n = sum(1 for l in open('$JSONL') if l.strip() and json.loads(l).get('batch') == $b)
print(n)" 2> /dev/null || echo 0)
        log "BATCH $b attempt $attempt EXIT rc=$rc wall=${wall}s timed_out=$timed_out :: ${summary:-no pytest summary} (rows for batch $b in jsonl: $rows)"
        echo "$(date +%FT%T) batch=$b attempt=$attempt rc=$rc timed_out=$timed_out wall_s=$wall summary='${summary:-none}' rows_b$b=$rows log=$logfile" >> "$LEDGER"
        if [ $rc -eq 0 ]; then break; fi
        overall=1
        grep -E "Regression failures|ISL [0-9]+ OSL [0-9]+:|Timeout|TT_FATAL|TT_THROW|Error" "$logfile" | grep -v "TT_FATAL: Only TILE" | head -8
        if [ "$DRY_RUN" != "1" ]; then reset_boards; fi
        if [ $timed_out -eq 1 ] && [ "$RETRY_TIMEOUTS" = "1" ] && [ $attempt -lt 2 ]; then
            log "BATCH $b timed out: retrying once after the reset"
            attempt=$((attempt + 1))
            continue
        fi
        break
    done
done

log "SWEEP done (overall rc $overall)"
echo "# $(date +%FT%T) sweep done overall_rc=$overall" >> "$LEDGER"
if [ -f "$JSONL" ]; then
    python3 "$SCRIPT_DIR/report.py" --matrix --out "$LOG_DIR/REPORT${TAG}.md" "$JSONL" > /dev/null && log "report: $LOG_DIR/REPORT${TAG}.md"
fi
exit $overall

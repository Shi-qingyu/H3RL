#!/usr/bin/env bash
# Collect live driver/rollout stacks and local Ray logs on the training host.
set -uo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd -- "$SCRIPT_DIR/../../.." && pwd)
cd -- "$REPO_ROOT"

watch=0
system_snapshot=0
case ${1:-} in
    --watch) watch=1; shift ;;
    --system) system_snapshot=1; shift ;;
esac
driver_pid=${1:-}
if [[ -n "$driver_pid" && ! "$driver_pid" =~ ^[1-9][0-9]*$ ]]; then
    echo "Usage: bash $0 [--watch] [training-driver-pid] | --system" >&2
    exit 2
fi

report_dir=${RAY_DIAGNOSTICS_DIR:-$REPO_ROOT/outputs/ray_diagnostics}
mkdir -p -- "$report_dir" || exit 1
report_path="$report_dir/$(date +%Y%m%d_%H%M%S)_$(hostname)_$$.log"
exec > >(tee -- "$report_path") 2>&1

capture() {
    printf '\nCommand:'
    printf ' %q' "$@"
    printf '\n'
    timeout --kill-after=1s "${command_timeout:-20}s" "$@"
    printf '[exit=%s; 124 means timeout]\n' "$?"
}

find_training_drivers() {
    if [[ -n "${RAY_DIAGNOSTICS_RAY_ROOT:-}" ]]; then
        local driver_log candidate_pid
        for driver_log in "$RAY_DIAGNOSTICS_RAY_ROOT"/session_*/logs/python-core-driver-*.log; do
            [[ -f "$driver_log" ]] || continue
            candidate_pid=${driver_log##*_}
            candidate_pid=${candidate_pid%.log}
            if [[ "$(ps -p "$candidate_pid" -o args=)" == *"-m verl_omni.trainer.main_diffusion "* ]]; then
                printf '%s\n' "$candidate_pid"
            fi
        done | sort -u
        return
    fi
    ps -eo pid=,comm=,args= | \
        awk '$2 ~ /^python/ && /-m verl_omni[.]trainer[.]main_diffusion([[:space:]]|$)/ {print $1}'
}

find_driver_log_session() {
    local driver_log canonical_log candidate_session matched_session=""
    for driver_log in "${RAY_DIAGNOSTICS_RAY_ROOT:-/tmp/ray}"/session_*/logs/python-core-driver-*_"$driver_pid".log; do
        [[ -f "$driver_log" ]] || continue
        canonical_log=$(readlink -f -- "$driver_log")
        candidate_session=${canonical_log%/*}
        if [[ -n "$matched_session" && "$matched_session" != "$candidate_session" ]]; then
            return 1
        fi
        matched_session=$candidate_session
    done
    [[ -n "$matched_session" ]] || return 1
    printf '%s\n' "$matched_session"
}

printf 'Host: %s\nTime: %s\nReport: %s\n' "$(hostname)" "$(date -Is)" "$report_path"
shopt -s nullglob
if (( system_snapshot )); then
    command_timeout=5
    capture python3 --version
    capture cat /proc/meminfo /proc/self/cgroup
    capture awk '/ - cgroup2? /' /proc/self/mountinfo
    cgroup_dirs=(/sys/fs/cgroup /sys/fs/cgroup/memory)
    while IFS=: read -r hierarchy controllers cgroup_path; do
        if [[ "$hierarchy" == 0 ]]; then
            cgroup_dirs+=("/sys/fs/cgroup${cgroup_path%/}")
        elif [[ ",$controllers," == *,memory,* ]]; then
            cgroup_dirs+=("/sys/fs/cgroup/memory${cgroup_path%/}")
        fi
    done < /proc/self/cgroup
    for cgroup_dir in "${cgroup_dirs[@]}"; do
        for metric in memory.current memory.peak memory.max memory.events memory.events.local \
            memory.usage_in_bytes memory.max_usage_in_bytes memory.limit_in_bytes memory.failcnt memory.oom_control; do
            [[ -r "$cgroup_dir/$metric" ]] || continue
            capture cat "$cgroup_dir/$metric"
        done
    done
    # Read after exit as well: kernel OOM/crash records do not need a live worker.
    capture bash -o pipefail -c 'dmesg --ctime 2>&1 | tail -n 250'
    if command -v journalctl >/dev/null 2>&1; then
        capture journalctl --no-pager -k --since '-20 min' -n 200
    fi
    capture nvidia-smi --query-gpu=index,uuid,memory.total,memory.used,utilization.gpu --format=csv
    capture nvidia-smi --query-compute-apps=pid,process_name,used_gpu_memory --format=csv
    capture bash -o pipefail -c 'ps -eo pid,ppid,stat,rss,comm --sort=-rss | head -n 25'
    exit 0
fi
if (( watch )); then
    watch_dir=${report_path%.log}
    mkdir -p -- "$watch_dir" || exit 1
    deadline=$((SECONDS + 3600))
    echo "Waiting up to one hour for the training driver and rollout server readiness."
    while [[ -z "$driver_pid" ]] && (( SECONDS < deadline )); do
        mapfile -t driver_pids < <(find_training_drivers)
        if (( ${#driver_pids[@]} > 1 )); then
            echo "Multiple training drivers found; restart --watch with the intended driver PID." >&2
            exit 2
        elif (( ${#driver_pids[@]} == 1 )); then
            driver_pid=${driver_pids[0]}
        else
            sleep 2
        fi
    done
    if [[ -z "$driver_pid" ]]; then
        echo "No training driver appeared before the deadline."
        exit 1
    fi
    driver_started=$(ps -p "$driver_pid" -o lstart=)
    driver_alive() {
        [[ -n "$driver_started" && "$(ps -p "$driver_pid" -o lstart=)" == "$driver_started" \
            && "$(ps -p "$driver_pid" -o stat=)" != *Z* ]]
    }
    printf 'Watching training driver PID: %s\n' "$driver_pid"
    ray_log_dir=""
    rollout_ready=0
    while driver_alive && (( SECONDS < deadline )); do
        if [[ -z "$ray_log_dir" ]]; then
            ray_log_dir=$(find_driver_log_session) || ray_log_dir=""
        fi
        if [[ -n "$ray_log_dir" ]]; then
            if [[ -n "${VERL_OMNI_DIAGNOSTICS_DIR:-}" ]]; then
                event_files=("$VERL_OMNI_DIAGNOSTICS_DIR"/*.events.jsonl)
                if (( ${#event_files[@]} )) && \
                    awk '/"event": "rollout_server.ready"/ {found=1; exit} END {exit !found}' "${event_files[@]}"; then
                    rollout_ready=1
                    break
                fi
            else
                worker_logs=("$ray_log_dir"/worker-*.err)
                if (( ${#worker_logs[@]} )) && \
                    awk '/No model-specific Continuous Token builder/ {found=1; exit} END {exit !found}' \
                        "${worker_logs[@]}"; then
                    rollout_ready=1
                    break
                fi
            fi
        fi
        sleep 1
    done
    if (( rollout_ready )); then
        echo "Rollout startup trigger detected; capturing for up to ten minutes."
        capture_deadline=$((SECONDS + 600))
        sample=0
        native_captured=0
        while driver_alive && (( SECONDS < capture_deadline )); do
            sample=$((sample + 1))
            printf '\nSnapshot %s at %s\n' "$sample" "$(date -Is)"
            capture_native=0
            if (( native_captured == 0 )) && [[ -n "${VERL_OMNI_DIAGNOSTICS_DIR:-}" ]]; then
                event_files=("$VERL_OMNI_DIAGNOSTICS_DIR"/*.events.jsonl)
                if (( ${#event_files[@]} )) && \
                    awk '/"event": "rollout.rpc.start"/ {found=1; exit} END {exit !found}' "${event_files[@]}"; then
                    capture_native=1
                    native_captured=1
                fi
            fi
            # Keep each snapshot bounded and avoid a Ray status RPC in this window.
            RAY_DIAGNOSTICS_DIR="$watch_dir" RAY_LOG_DIR="$ray_log_dir" \
                RAY_DIAGNOSTICS_FAST=1 RAY_DIAGNOSTICS_GDB="$capture_native" \
                bash "$SCRIPT_DIR/diagnose_video_edit_ray.sh" "$driver_pid" \
                > "$watch_dir/snapshot_${sample}.console.log" 2>&1
            printf 'Snapshot exit status: %s\n' "$?"
            sleep 2
        done
    else
        echo "Driver exited or the deadline elapsed before the rollout startup trigger."
    fi
    if [[ -n "$ray_log_dir" && -d "$ray_log_dir" ]]; then
        mkdir -p -- "$watch_dir/ray_logs"
        capture cp -a -- "$ray_log_dir/." "$watch_dir/ray_logs/"
    fi
    printf '\nAutomatic capture finished: %s\n' "$watch_dir"
    exit 0
fi

fast=${RAY_DIAGNOSTICS_FAST:-0}
if [[ "$fast" == 1 ]]; then
    command_timeout=5
fi
if [[ "$fast" != 1 ]]; then
    ps -eo pid,ppid,stat,etime,pcpu,args | \
        awk '/[m]ain_diffusion|[g]cs_server|[r]aylet|[T]askRunner|[d]efault_worker|[s]erve_video_edit_reward/'
fi

if [[ -z "$driver_pid" ]]; then
    mapfile -t driver_pids < <(find_training_drivers)
    if (( ${#driver_pids[@]} == 1 )); then
        driver_pid=${driver_pids[0]}
    else
        printf '\nFound %s training drivers; pass the intended PID as the first argument for a stack dump.\n' \
            "${#driver_pids[@]}"
    fi
fi

if [[ -n "$driver_pid" && "$fast" != 1 ]]; then
    printf '\nTraining driver PID: %s\n' "$driver_pid"
    if [[ -x "$REPO_ROOT/.venv/bin/py-spy" ]]; then
        capture "$REPO_ROOT/.venv/bin/py-spy" dump --pid "$driver_pid"
    else
        echo "No .venv/bin/py-spy; stack dump skipped."
    fi
fi

if command -v nvidia-smi >/dev/null 2>&1; then
    capture nvidia-smi
fi
ray_log_dir=${RAY_LOG_DIR:-/tmp/ray/session_latest/logs}
ray_session_verified=0
if [[ -n "${RAY_LOG_DIR:-}" ]]; then
    ray_session_verified=1
fi
# The driver filename ties the log session to this process on multi-job hosts.
if [[ -z "${RAY_LOG_DIR:-}" && -n "$driver_pid" ]]; then
    if driver_log_session=$(find_driver_log_session); then
        ray_log_dir=$driver_log_session
        ray_session_verified=1
        echo "Ray log session matched to the training driver PID."
    else
        echo "No unique log session matched this PID; session_latest below is unverified. Set RAY_LOG_DIR if needed."
    fi
elif [[ -z "${RAY_LOG_DIR:-}" ]]; then
    echo "No driver PID selected; session_latest below is unverified."
fi
printf '\nRay logs: %s\n' "$ray_log_dir"
if [[ -d "$ray_log_dir" ]]; then
    printf '\nRecent log files:\n'
    ls -lt -- "$ray_log_dir" | head -n 25
    log_files=(
        "$ray_log_dir"/python-core-driver-*_${driver_pid:-*}.log
        "$ray_log_dir"/raylet.out "$ray_log_dir"/raylet.err
        "$ray_log_dir"/gcs_server.out "$ray_log_dir"/gcs_server.err
        "$ray_log_dir"/runtime_env_agent.log "$ray_log_dir"/runtime_env_agent.err
    )
    rollout_pids=()
    if (( ray_session_verified )); then
        for worker_log in "$ray_log_dir"/worker-*.err; do
            if ! awk '/^:actor_name:vLLMOmniHttpServer$/ {found=1; exit} END {exit !found}' "$worker_log"; then
                continue
            fi
            rollout_pid=${worker_log##*-}
            rollout_pid=${rollout_pid%.err}
            [[ "$rollout_pid" =~ ^[1-9][0-9]*$ ]] || continue
            log_files+=(
                "$worker_log" "${worker_log%.err}.out"
                "$ray_log_dir"/python-core-worker-*_"$rollout_pid".log
            )
            # Old session files can outlive a process; check its current title.
            if [[ "$(ps -p "$rollout_pid" -o args=)" == *"ray::vLLMOmniHttpServer"* ]]; then
                rollout_pids+=("$rollout_pid")
            fi
        done
    fi
    if (( ${#rollout_pids[@]} == 0 )); then
        echo "No live rollout servers matched this session. Capture again while the first rollout is stalled."
    fi
    dump_rollout_server() {
        local rollout_pid=$1
        printf '\nRollout server PID: %s\n' "$rollout_pid"
        capture ps -p "$rollout_pid" -o pid,ppid,pgid,sid,tpgid,tty,stat,wchan:32,comm
        capture readlink -- "/proc/$rollout_pid/fd/0"
        capture ps -L -p "$rollout_pid" -o pid,tid,stat,pcpu,wchan:32,comm
        capture ps --ppid "$rollout_pid" -o pid,ppid,stat,pcpu,rss,args
        if [[ "$(ps -p "$rollout_pid" -o stat=)" == *T* ]]; then
            echo "Worker is signal-stopped (T), not waiting for model computation; skipping debugger attachment."
            return
        fi
        if [[ -x "$REPO_ROOT/.venv/bin/py-spy" ]]; then
            capture "$REPO_ROOT/.venv/bin/py-spy" dump --native --pid "$rollout_pid"
        else
            echo "No .venv/bin/py-spy; rollout stack dump skipped."
        fi
        # py-spy only unwinds threads with Python frames; include native gRPC threads.
        if [[ "${RAY_DIAGNOSTICS_GDB:-0}" == 1 ]]; then
            if command -v gdb >/dev/null 2>&1; then
                capture gdb --nx --nh --batch \
                    -iex 'set auto-load off' -iex 'set debuginfod enabled off' \
                    -ex 'set pagination off' -ex 'thread apply all bt 16' -ex detach -p "$rollout_pid"
            else
                echo "No gdb; native-only RPC thread stacks skipped."
            fi
        fi
    }
    child_pids=()
    dump_pids=()
    for rollout_pid in "${rollout_pids[@]}"; do
        if [[ "$fast" == 1 ]]; then
            dump_rollout_server "$rollout_pid" > "${report_path%.log}_server_${rollout_pid}.log" 2>&1 &
            dump_pids+=("$!")
        else
            dump_rollout_server "$rollout_pid"
            while read -r child_pid; do
                child_pids+=("$child_pid")
            done < <(ps --ppid "$rollout_pid" -o pid=,args= | awk '/spawn_main|DiffusionWorker|Orchestrator/ {print $1}')
        fi
    done
    for dump_pid in "${dump_pids[@]}"; do
        wait "$dump_pid"
    done
    if [[ "$fast" == 1 ]]; then
        for rollout_pid in "${rollout_pids[@]}"; do
            printf '\nRollout stack file: %s_server_%s.log\n' "${report_path%.log}" "$rollout_pid"
        done
    fi
    # Capture servers first so a slow native unwind cannot hide all RPC stacks.
    for child_pid in "${child_pids[@]}"; do
        if [[ -d "/proc/$child_pid" && -x "$REPO_ROOT/.venv/bin/py-spy" ]]; then
            printf '\nRollout child PID: %s\n' "$child_pid"
            capture "$REPO_ROOT/.venv/bin/py-spy" dump --native --pid "$child_pid"
        fi
    done
    for log_file in "${log_files[@]}"; do
        if [[ -f "$log_file" ]]; then
            printf '\nFile: %s\n' "$log_file"
            capture tail -n 80 -- "$log_file"
        fi
    done
else
    echo "Ray log directory is absent; set RAY_LOG_DIR to this job's session logs if it uses a custom temp directory."
fi

if [[ "$fast" != 1 && -x "$REPO_ROOT/.venv/bin/ray" ]]; then
    echo "Ray status below describes the default local cluster (--address=auto)."
    capture "$REPO_ROOT/.venv/bin/ray" status --address=auto
fi

printf '\nDiagnostic report saved to %s\n' "$report_path"

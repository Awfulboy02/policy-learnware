#!/usr/bin/env bash
# Run on the selected training host: start_pool_node.sh RUN_ROOT coordinator|GPU ...
set -eu
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 RUN_ROOT coordinator|GPU ..." >&2
    exit 2
fi
root=$(cd "$1" && pwd)
shift
code=$(cd "$(dirname "$0")/.." && pwd)
python_bin=${POOL_PYTHON:-/share/songyf/RL_Learnware/.kmepoa_envs/e01_matrix_py312/bin/python}
host=$(hostname -s)
mkdir -p "$root/launch"
for slot in "$@"; do
    case "$slot" in
        coordinator)
            role=(coordinator)
            if [ -n "${POOL_DEADLINE:-}" ]; then role+=(--deadline "$POOL_DEADLINE"); fi
            ;;
        [0-7]) role=(worker --gpu "$slot");;
        *) echo "Unknown role: $slot" >&2; exit 2;;
    esac
    session="pl600-$slot"
    if tmux has-session -t "=$session" 2>/dev/null; then
        echo "$host $session already active"
        continue
    fi
    log="$root/launch/$host-$slot.log"
    printf -v command '%q ' env PYTHONUNBUFFERED=1 PYTHONPATH="$code/src" "$python_bin" -u "$code/scripts/pool_queue.py" --root "$root" "${role[@]}"
    printf -v quoted_log '%q' "$log"
    tmux new-session -d -s "$session" -c "$root" "exec $command >> $quoted_log 2>&1"
    echo "$host $session $log"
done

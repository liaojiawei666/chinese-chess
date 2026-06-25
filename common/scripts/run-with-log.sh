#!/usr/bin/env bash
# 将 stdout/stderr 追加写入 repo 根目录 log/<name>.log，同时仍输出到终端。
set -euo pipefail

if [[ $# -lt 2 || "$2" != "--" ]]; then
  echo "usage: run-with-log.sh <name> -- <command...>" >&2
  exit 2
fi

name=$1
shift 2
root=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$root/log"
log="$root/log/${name}.log"

exec > >(tee -a "$log") 2>&1
printf '\n=== %s %s CHESS_PROFILE=%s pid=%s ===\n' \
  "$(date -Iseconds)" "$name" "${CHESS_PROFILE:-local}" "$$"
exec "$@"

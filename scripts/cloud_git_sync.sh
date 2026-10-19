#!/usr/bin/env bash
# cloud_git_sync.sh — 云端 git 同步辅助脚本
#
# 作用：
# - 如果目标目录已经是 git 仓库，则尝试从 origin 拉取最新代码
# - 如果目标目录还不是 git 仓库，但提供了 remote URL，则可初始化并绑定远端
#
# 用法：
#   bash scripts/cloud_git_sync.sh
#   GIT_REMOTE_URL=<url> GIT_BRANCH=main bash scripts/cloud_git_sync.sh --mode init

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${MODE:-auto}"              # auto | pull | init | off
GIT_REMOTE_URL="${GIT_REMOTE_URL:-}"
GIT_BRANCH="${GIT_BRANCH:-main}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --mode) MODE="$2"; shift 2 ;;
        --remote) GIT_REMOTE_URL="$2"; shift 2 ;;
        --branch) GIT_BRANCH="$2"; shift 2 ;;
        -h|--help)
            cat <<'EOF'
Usage:
  bash scripts/cloud_git_sync.sh [--mode auto|pull|init|off] [--remote URL] [--branch BRANCH]
EOF
            exit 0
            ;;
        *) echo "Unknown arg: $1" >&2; exit 2 ;;
    esac
done

if [[ "$MODE" == "off" ]]; then
    echo "Git sync disabled."
    exit 0
fi

if ! command -v git >/dev/null 2>&1; then
    echo "Warning: git not found; skip sync." >&2
    exit 0
fi

cd "$ROOT_DIR"

if [[ -d .git ]]; then
    echo "Git repo detected at ${ROOT_DIR}"
    if git remote get-url origin >/dev/null 2>&1; then
        echo "Remote origin: $(git remote get-url origin)"
        if [[ "$MODE" == "init" ]]; then
            echo "Mode=init and repo already exists; skip re-init."
            exit 0
        fi
        if [[ "$(git status --porcelain)" != "" ]]; then
            echo "Warning: working tree is dirty; pull may fail if local changes conflict." >&2
        fi
        git fetch origin "$GIT_BRANCH"
        git pull --ff-only origin "$GIT_BRANCH"
    elif [[ -n "$GIT_REMOTE_URL" ]]; then
        echo "Adding missing origin remote: ${GIT_REMOTE_URL}"
        git remote add origin "$GIT_REMOTE_URL"
        git fetch origin "$GIT_BRANCH"
        git pull --ff-only origin "$GIT_BRANCH"
    else
        echo "Warning: no origin remote configured and no GIT_REMOTE_URL provided; skip sync." >&2
    fi
    exit 0
fi

if [[ "$MODE" == "pull" ]]; then
    echo "Warning: no .git directory found; pull mode cannot proceed." >&2
    exit 0
fi

if [[ -z "$GIT_REMOTE_URL" ]]; then
    echo "Warning: no .git directory and no GIT_REMOTE_URL provided; skip init." >&2
    exit 0
fi

echo "Initializing git repo from ${GIT_REMOTE_URL} (branch=${GIT_BRANCH})"
git init
git remote add origin "$GIT_REMOTE_URL"
git fetch origin "$GIT_BRANCH"
git checkout -B "$GIT_BRANCH" FETCH_HEAD

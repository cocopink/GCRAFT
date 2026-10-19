#!/usr/bin/env bash
# package_cloud_bundle.sh — 打包当前仓库为云端部署压缩包
#
# 目标：
# - 保留当前代码、脚本、数据与配置
# - 排除本地运行产物（logs/results/cache/__pycache__ 等）
# - 生成一个可直接搬运到云主机的 tar.gz
#
# 用法：
#   bash scripts/package_cloud_bundle.sh
#   OUT_DIR=/tmp bash scripts/package_cloud_bundle.sh

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAMP="$(date +%Y%m%d-%H%M%S)"
BUNDLE_NAME="table-critic-cloud-${STAMP}"
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/dist}"
STAGE_DIR="$(mktemp -d "${TMPDIR:-/tmp}/table-critic-cloud.XXXXXX")"
WORK_DIR="${STAGE_DIR}/${BUNDLE_NAME}"
ARCHIVE_PATH="${OUT_DIR}/${BUNDLE_NAME}.tar.gz"

mkdir -p "$OUT_DIR" "$WORK_DIR"

echo "==> Staging repository into ${WORK_DIR}"
rsync -a \
    --exclude '.git/' \
    --exclude '.pytest_cache/' \
    --exclude '__pycache__/' \
    --exclude '*.pyc' \
    --exclude '*.pyo' \
    --exclude '*.cache' \
    --exclude '*.log' \
    --exclude 'logs/' \
    --exclude 'results/' \
    --exclude 'test/results/' \
    --exclude 'dist/' \
    "${ROOT_DIR}/" \
    "${WORK_DIR}/"

GIT_SHA="$(git -C "$ROOT_DIR" rev-parse --short HEAD 2>/dev/null || echo unknown)"
GIT_BRANCH="$(git -C "$ROOT_DIR" branch --show-current 2>/dev/null || echo unknown)"
GIT_REMOTE_URL="$(git -C "$ROOT_DIR" remote get-url origin 2>/dev/null || echo unknown)"
cat > "${WORK_DIR}/CLOUD_BUNDLE_INFO.txt" <<EOF
Bundle: ${BUNDLE_NAME}
Created: ${STAMP}
Git SHA: ${GIT_SHA}
Git Branch: ${GIT_BRANCH}
Git Remote: ${GIT_REMOTE_URL}
Root: ${ROOT_DIR}
Notes:
- Run scripts/cloud_mainline.sh on the target machine after extracting this archive.
- Use BACKEND=vllm and BASE_URL=http://127.0.0.1:8000/v1 for the suggested 5090 setup.
- If the target machine has network access, run scripts/cloud_git_sync.sh first to pull the latest code from origin.
EOF

cat > "${WORK_DIR}/.cloud_bundle.env" <<EOF
export GIT_SHA="${GIT_SHA}"
export GIT_BRANCH="${GIT_BRANCH}"
export GIT_REMOTE_URL="${GIT_REMOTE_URL}"
EOF

echo "==> Creating archive ${ARCHIVE_PATH}"
tar -czf "$ARCHIVE_PATH" -C "$STAGE_DIR" "$BUNDLE_NAME"

if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$ARCHIVE_PATH" > "${ARCHIVE_PATH}.sha256"
fi

echo "==> Done"
echo "Archive: ${ARCHIVE_PATH}"
if [[ -f "${ARCHIVE_PATH}.sha256" ]]; then
    echo "Checksum: ${ARCHIVE_PATH}.sha256"
fi

rm -rf "$STAGE_DIR"

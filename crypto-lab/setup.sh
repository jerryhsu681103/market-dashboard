#!/usr/bin/env bash
# 在全新環境重建實驗室：虛擬環境、套件、資料。可重複執行。
set -euo pipefail
cd "$(dirname "$0")"
# 套件清單需要 Python 3.12 以上；自動挑選可用的版本。
py_ok() { "$1" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 12) else 1)' >/dev/null 2>&1; }

PY=""
for cand in python3.13 python3.12 python3; do
  if command -v "$cand" >/dev/null 2>&1 && py_ok "$cand"; then
    PY="$cand"
    break
  fi
done
if [ -z "$PY" ]; then
  echo "錯誤：找不到 Python 3.12 以上版本（已嘗試 python3.13、python3.12、python3）。請先安裝後再執行 setup.sh。" >&2
  exit 1
fi

# 既有的 .venv 如果壞掉或 Python 版本低於 3.12 就重建。
if [ -d .venv ] && ! py_ok .venv/bin/python; then
  echo "既有的 .venv 不是 Python 3.12 以上，重新建立。"
  rm -rf .venv
fi
[ -x .venv/bin/python ] || "$PY" -m venv .venv
.venv/bin/pip install -q -r requirements.txt
.venv/bin/python -m lab.download_data --daily
.venv/bin/python -m lab.download_extra --sets metrics spot premium
.venv/bin/python -m pytest -q

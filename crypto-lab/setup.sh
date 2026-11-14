#!/usr/bin/env bash
# 在全新環境重建實驗室：虛擬環境、套件、資料。可重複執行。
set -euo pipefail
cd "$(dirname "$0")"
[ -x .venv/bin/python ] || python3 -m venv .venv
.venv/bin/pip install -q -r requirements.txt
.venv/bin/python -m lab.download_data --daily
.venv/bin/python -m lab.download_extra --sets metrics spot premium
.venv/bin/python -m pytest -q

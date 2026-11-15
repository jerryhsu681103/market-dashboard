#!/usr/bin/env bash
# 值守＋紙賬的例行動作：更新公開行情 → 接著記紙面帳 → 值守檢查。只讀公開資料，不連任何下單介面。
# 回傳碼 0＝正常；2＝已停記（看 paper/desk/HALT）。可重複執行。
set -uo pipefail
cd "$(dirname "$0")/.."
[ -x .venv/bin/python ] || python3 -m venv .venv
.venv/bin/pip install -q -r requirements.txt || exit 1
mkdir -p paper/desk
for g in "BTCUSDT ETHUSDT BNBUSDT" "XRPUSDT DOGEUSDT ADAUSDT" "MATICUSDT DOTUSDT LTCUSDT" "SOLUSDT AVAXUSDT LINKUSDT"; do
  .venv/bin/python -m lab.download_data --daily --symbols $g > "paper/desk/.dl_${g%% *}.log" 2>&1 &
done
wait
grep -l "Traceback" paper/desk/.dl_*.log 2>/dev/null && echo "⚠ 有幣種下載失敗，見上列日誌（值守會以行情停更/落後處理）"
.venv/bin/python -W ignore -m desk.ledger

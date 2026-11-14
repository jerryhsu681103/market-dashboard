（已完成，保留作為紀錄與重建步驟）

在 crypto-lab/ 重建實驗室：
1. python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
2. .venv/bin/python -m lab.download_data
   下載 BTCUSDT、ETHUSDT 永續合約 1 小時 K 線與資金費率，存成 data/{symbol}_1h.parquet、data/{symbol}_funding.parquet。
   檢查結果在 data/quality_{symbol}.txt：時間是否連續、有沒有重複行、有沒有缺失的小時。把問題列出來。
3. .venv/bin/python -m pytest
   必須全部通過，包含：同一組參數連跑 3 次結果完全一致、成本未填會報錯、前視偏差會被抓到。
4. 手續費和滑點在 config/costs.json，預設 null，由我自己填。

規矩：所有數字只能來自 lab/engine.py 的執行結果。不准估算，不准憑記憶寫任何回測數字。

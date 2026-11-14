每晚自動研究（無人值守，由排程觸發）。

開工：
1. 切到 `crypto-lab-research` 分支（遠端沒有就從 main 建立），之後所有 commit 都推送到這個分支。不開 PR，不動 main。
2. 進入 crypto-lab/，執行 `bash setup.sh`（建環境、下載資料、跑測試）。測試沒有全過就停工，把原因寫進 reports/日期_nightly.md 後推送。
3. 讀 FACTORY_RULES.md、research_log.csv、reports/ 裡最近的點子清單。

研究（規則一律照 FACTORY_RULES.md，不得放寬任何門檻、不得修改 config/costs.json 與 lab/engine.py）：
4. 依 prompts/03-ideas.md 產生 10 個新點子，寫進 reports/ideas_日期.md。
   每個點子標示與死亡點子的重疊程度；重疊「高」的不跑。可跑的點子不足 3 個就只跑可跑的，一個都沒有就照實寫並收工。
5. 先把點子清單、信號函式、scripts/stage1.py 的參數網格 commit（看到結果之前固定），再跑第 1 關：
   `python -m scripts.stage1 --timeframe 4h <點子...>`
6. 通過第 1 關的點子，接著跑：
   - 第 2 關：`python -m scripts.stage2 <點子> --timeframe 4h`
   - 第 3 關：先 `--dry-run` 彩排，再正式執行 `python -m scripts.stage3 <點子> --timeframe 4h`
   - 第 4 關：先 `--dry-run --n-perm 8` 彩排，再正式執行 `python -m scripts.stage4 <點子> --timeframe 4h`
   樣本外資料每個點子每一關只能讀一次，腳本報錯就停，不得繞過。
7. 4 關全過的點子，依 prompts/07-paper-trading.md 加進 watchlist.csv。

收工：
8. 寫 reports/日期_nightly.md：測了哪些點子、各死在哪一關、死因、每個數字的來源命令、不確定的地方。
9. commit 並推送到 `crypto-lab-research`。

永遠不連接交易所下單介面，不讀 API 金鑰，不寫買賣建議。

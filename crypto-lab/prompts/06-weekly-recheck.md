讀 watchlist.csv 裡所有在紙面觀察名單的策略。名單是空的就直接收工，不需要寫報告。
（排程執行時：先切到 `crypto-lab-research` 分支，沒有就從 main 建立；在 crypto-lab/ 執行 `bash setup.sh`；結果 commit 並推送到同一分支。）
先執行 python -m lab.download_data --daily 更新到昨天。
用截止到昨天的最新 K 線（lab.data.load_full(..., stage=6)），對每個策略重跑前推置換檢驗（permutation_test(kind="walkforward")）。
p 值變差（超過晉級時的門檻）、或紙面表現（python -m lab.paper）明顯偏離回測預期的，標記為「退回」，寫清楚原因。
「明顯偏離」定義：晉級後 Sharpe 低於回測前推 Sharpe 的一半，或晉級後最大回撤超過回測最大回撤的 1.5 倍。
不要為了讓它通過去改參數。

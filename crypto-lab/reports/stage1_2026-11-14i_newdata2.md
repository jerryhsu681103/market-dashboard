# 第 1 關結果：第五批（新資料集第二輪）前 3 名（2026-11-14）

- 資料：BTCUSDT 永續 4 小時線樣本內（2020-01-01 → 2023-12-31），併入現貨／metrics 資料
- 成本：手續費 0.0005、滑點 0.0002（單邊），資金費率按結算時點收付
- 命令：`python -m scripts.stage1 --timeframe 4h spot_perp_flow_divergence oi_buildup_breakout covering_move_fade`
- 理由與參數網格在執行前已 commit（reports/ideas_2026-11-14e_newdata2.md）

| 點子 | 最優參數 | Sharpe | PF | 總報酬 | 最大回撤 | 交易數 | Sharpe>0 組數 | 結果 | 死因 |
|---|---|---|---|---|---|---|---|---|---|
| covering_move_fade | k=1.0, hold=18 | 0.758 | 1.374 | 138.2% | -56.4% | 234 | 18/25 | 淘汰 | Sharpe < 1.0；參數尖峰 |
| oi_buildup_breakout | oi_z=0.0, channel=30 | 0.408 | 1.263 | 39.1% | -52.4% | 108 | 10/25 | 淘汰 | Sharpe < 1.0 |
| spot_perp_flow_divergence | threshold=1.0, window=18 | 0.266 | 1.139 | -1.5% | -53.2% | 178 | 3/25 | 淘汰 | Sharpe、PF 不足；參數尖峰 |

補充（不影響判定）：不限交易次數時，三個點子全部參數組合的最高 Sharpe 為 0.92（covering_move_fade，51 筆）、
0.68（oi_buildup_breakout，88 筆）、0.27，沒有任何一組達到 1.0。
oi_buildup_breakout 的最優組合 oi_z=0.0 表示未平倉量條件幾乎沒有過濾作用，成績主要來自突破本身。

維護紙面交易觀察名單：
1. 點子 4 關全過時，在 watchlist.csv 新增一行：
   name, module（例如 strategies.funding_revert）, params_json（前推最後一個窗口的參數）, symbol,
   promoted_date（晉級當天 UTC 日期）, backtest_sharpe, backtest_pf（第 3 關前推成績）, notes
2. 晉級後參數凍結，不准再優化。
3. 每天：python -m lab.download_data --daily && python -m lab.paper
   輸出 paper/summary.csv 與 paper/{name}_equity.csv，供 05 日報與 06 週檢使用。
4. 被 06 標記「退回」的策略，從 watchlist.csv 移除，並在 research_log.csv 補記退回原因。
永遠不連接交易所下單介面。

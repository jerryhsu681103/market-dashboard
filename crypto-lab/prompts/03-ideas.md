讀 FACTORY_RULES.md 和 research_log.csv。
給我 10 個新的 BTC/ETH 永續合約策略點子，每個寫清楚：
1. 它賺的是誰的錢，背後的市場行為是什麼（比如資金費率過高後的回落、週末流動性變差、大級別突破後的跟隨）
2. 什麼情況下它會失效
3. 信號函式大概怎麼寫，有幾個參數，每個參數打算掃多寬
4. 持倉期間的資金費率對它是助力還是成本
沒有理由的點子不要給。日誌裡死過的點子不要再出現（換名字也不行）。
最後按「理由的可信度」排序，挑前 3 個開始跑第 1 關：
- 信號函式寫在 strategies/{點子名稱}.py，函式名稱 signal(df, **params)
- 用 lab.data.load_insample() 讀資料，用 lab.research.grid_search() 掃參數
- 用 plateau_check() 判斷高原、save_heatmap() 存熱力圖到 reports/heatmaps/
- 依 FACTORY_RULES.md 第 1 關的數字門檻判生死，寫進 research_log.csv

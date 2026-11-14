對 [點子名稱] 跑第 2 關置換檢驗：
- 用 BTCUSDT 1 小時線樣本內資料（lab.data.load_insample）
- 呼叫 lab.research.permutation_test(kind="insample", n_perm=1000, seed=<固定種子>)
  打亂方式：保持起點和終點價格不變，打亂每根 K 線的相對漲跌順序，開高低收的相對關係一起保留；時間索引不動
- 每條假行情用和真實資料完全相同的參數掃描流程（同一個 grid、同一個 valid）重新優化
- 多幣種點子改用 permute_many，所有幣種同一組打亂順序
輸出：真實成績、假行情成績的分佈圖（save_permutation_plot 存到 reports/permutation/）、p 值、結論。
結果寫進 research_log.csv（含種子與命令），圖存進 reports/。

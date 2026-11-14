# 策略點子清單：第五批，新資料集第二輪（2026-11-14）

已排除日誌中全部死亡點子及其換名變體（含第四批：spot_led_move、trapped_positioning、whale_retail_divergence）。
週期 4 小時線，標的 BTCUSDT。理由與參數網格在執行第 1 關之前寫定。

已知資訊的揭露：whale_retail_divergence 的第 3 關讓我們知道 2025 年對「多空比」訊號不利。
本批點子的設計沒有參考樣本外的價格走勢，但這項認知無法抹去，照實記錄。

| 排名 | 點子 | 資料 | 賺誰的錢／市場行為 | 失效情境 | 參數（4h 根） | 與死亡點子重疊 |
|---|---|---|---|---|---|---|
| 1 | spot_perp_flow_divergence 現貨與永續主動買賣流分歧 | 現貨 | 永續上的槓桿投機方：現貨主動淨買、永續主動淨賣時，現貨的真實需求在吸收槓桿賣壓，空方後續被迫回補；反之亦然 | 現貨流量被做市商對沖扭曲 | threshold 0.5–2.5、window 3–30 | 低至中：spot_led_move 看成交量占比，taker_imbalance 只看永續單邊；本點子看兩個市場主動流向的差 |
| 2 | oi_buildup_breakout 槓桿堆積後突破 | metrics | 突破時站錯邊的槓桿方：未平倉量在過去 7 天明顯堆積後，突破方向的反向部位被迫止損／強平，推動行情延續 | 堆積的部位主要是對沖盤 | oi_z 0–2.0、channel 6–42 | 中：結構同 vol_squeeze_breakout，但條件變數是未平倉量堆積而非波動壓縮 |
| 3 | covering_move_fade 平倉驅動行情的反向 | metrics | 追逐「空頭回補上漲／多頭平倉下跌」的晚到資金：價格移動伴隨未平倉量下降，代表是既有部位平倉推動而非新資金進場，推力耗盡後回吐 | 平倉後緊接新資金進場 | k 0.5–2.5、hold 3–30 | 中：trapped_positioning 看未平倉量上升＋多空比；本點子看未平倉量下降，不用多空比。與 liq_reversal 同屬反轉 |
| 4 | leverage_ratio_regime 槓桿率狀態 | metrics＋現貨 | 未平倉量／現貨成交額偏高時，順勢部位擁擠，逆短期趨勢 | 槓桿率結構性上升 | 分位數門檻、持有 | 中：與 spot_led_move 的「槓桿主導則反向」部分重疊 |
| 5 | top_account_position_gap 大戶內部分歧 | metrics | 大戶帳戶數多空比 vs 大戶持倉多空比 | — | — | **高**：與 whale_retail_divergence 同資料同機制，不跑 |
| 6 | retail_crowd_fade 散戶極端反向 | metrics | 散戶 | — | — | **高**：whale_retail_divergence 的一半，不跑 |
| 7 | oi_flush_rebound 去槓桿後反彈 | metrics | 被強平方 | — | — | **高**：liq_reversal、multiday_dip_buy，不跑 |
| 8 | basis_extreme 現貨永續基差極端 | 現貨／溢價 | 槓桿擁擠方 | — | — | **高**：與資金費率同資訊，不跑 |
| 9 | spot_volume_regime_trend 現貨量能擴張趨勢 | 現貨 | 趨勢 | — | — | **高**：趨勢家族加過濾，不跑 |
| 10 | depth_imbalance 訂單簿失衡 | depth | 掛單薄的一方 | — | — | 樣本內只有 1 年，規則不允許 |

挑前 3 名跑第 1 關。

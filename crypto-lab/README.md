# crypto-lab：加密貨幣策略工廠

研究與紙面交易專用的回測實驗室。**不連接任何交易所下單介面，不讀 API 金鑰。**
標的：Binance USDT 本位永續合約（BTCUSDT、ETHUSDT）1 小時線，成本含手續費、滑點、資金費率。

## 快速開始

```bash
cd crypto-lab
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m lab.download_data        # 下載 K 線與資金費率（data.binance.vision）
.venv/bin/python -m lab.download_extra       # 下載未平倉量／多空比、現貨、溢價指數、訂單簿深度
.venv/bin/python -m pytest                   # 引擎測試
# 先在 config/costs.json 填入 fee、slippage（單邊比例）
.venv/bin/python smoke_test.py               # 02 煙霧測試
```

## 目錄

| 路徑 | 用途 |
|---|---|
| `FACTORY_RULES.md` | 研究規矩（每次開工前先讀） |
| `prompts/01–07` | 各階段的工作指令 |
| `lab/engine.py` | 回測引擎，所有數字的唯一來源：`run(...)`（單一幣種）、`run_spread(...)`（多腳價差） |
| `lab/data.py` | 樣本內／樣本外切分；樣本外讀取記錄在 `oos_access_log.csv` |
| `lab/permutation.py` | K 線置換（假行情） |
| `lab/research.py` | 參數掃描、參數高原、滾動前推、置換檢驗、第 4 關門檻 |
| `lab/paper.py` | 紙面交易追蹤 |
| `strategies/` | 信號函式，一個點子一個檔案 |
| `config/costs.json` | 手續費、滑點（預設 null，由你填） |
| `research_log.csv` / `watchlist.csv` | 研究日誌／紙面觀察名單 |
| `data/quality_*.txt` | 資料品質報告 |

## 引擎規則摘要

- 第 t 根信號在第 t+1 根開盤成交；每次執行都做截斷檢查，用到未來資料直接報錯。
- 成本：部位變動 ×（fee + slippage）；資金費率在結算時點按持倉收付。
- fee、slippage 為 null 或未提供 funding 時拒絕執行。
- Sharpe 年化係數 √8760（24/7 市場）。

## 執行時間參考（4 核心）

| 作業 | 時間 |
|---|---|
| 單次回測（4 年樣本內） | 約 0.3 秒 |
| 第 2 關置換 1000 次（6 組參數） | 約 5–10 分鐘 |
| 單次滾動前推（6 組參數） | 約 17 秒 |
| 第 4 關前推置換 1000 次 | 約 1.5 小時（隨參數組數線性增加） |

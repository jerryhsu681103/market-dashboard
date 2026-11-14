"""診斷用：同一組參數在零手續費、零滑點下的成績（資金費照扣）。不作為任何關卡的判定依據。

用法：python -m scripts.diag_zero_cost <idea> '<params_json>' [--symbol ETHUSDT] [--attach BTCUSDT]
"""
import argparse
import importlib
import json

from lab import engine
from lab.data import attach_funding, load_insample

ap = argparse.ArgumentParser()
ap.add_argument("idea")
ap.add_argument("params")
ap.add_argument("--symbol", default="BTCUSDT")
ap.add_argument("--attach", nargs="*", default=[])
a = ap.parse_args()
params = json.loads(a.params)
df, funding = load_insample(a.symbol)
df = attach_funding(df, funding)
for other in a.attach:
    df[f"{other}_close"] = load_insample(other)[0]["close"]
fn = importlib.import_module(f"strategies.{a.idea}").signal
r = engine.run(fn, params, df, 0.0, 0.0, funding)
print(a.idea, a.symbol, params, {k: round(r[k], 4) for k in ("sharpe", "profit_factor", "total_return", "max_drawdown")},
      "trades", r["trade_count"])

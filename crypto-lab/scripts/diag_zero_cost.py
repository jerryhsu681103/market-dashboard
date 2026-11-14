"""診斷用：同一組參數在零手續費、零滑點下的成績（資金費照扣）。不作為任何關卡的判定依據。

用法：python -m scripts.diag_zero_cost <idea> '<params_json>'
"""
import importlib
import json
import sys

from lab import engine
from lab.data import attach_funding, load_insample

name, params = sys.argv[1], json.loads(sys.argv[2])
df, funding = load_insample("BTCUSDT")
df = attach_funding(df, funding)
fn = importlib.import_module(f"strategies.{name}").signal
r = engine.run(fn, params, df, 0.0, 0.0, funding)
print(name, params, {k: round(r[k], 4) for k in ("sharpe", "profit_factor", "total_return", "max_drawdown")},
      "trades", r["trade_count"])

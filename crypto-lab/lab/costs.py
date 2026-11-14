"""讀取 config/costs.json 的手續費與滑點。值為 null 時保留 None，交給 engine 報錯。"""
import json
from pathlib import Path

PATH = Path(__file__).resolve().parents[1] / "config" / "costs.json"


def load_costs() -> tuple[float | None, float | None]:
    c = json.loads(PATH.read_text(encoding="utf-8"))
    return c.get("fee"), c.get("slippage")

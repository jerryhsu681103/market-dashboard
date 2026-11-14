import numpy as np
import pandas as pd


def hold_after(event: pd.Series, hold: int) -> pd.Series:
    """事件發生後（含當根）連續持有 hold 根。只用當根及之前的事件，不含未來。"""
    return event.astype(float).rolling(hold, min_periods=1).max().fillna(0).astype(bool)


def combine(long_flag: pd.Series, short_flag: pd.Series) -> pd.Series:
    """多空同時成立時空倉。"""
    return long_flag.astype(int) - short_flag.astype(int)


def step_hours(df: pd.DataFrame) -> float:
    """K 線週期（小時），由時間索引推得。"""
    d = pd.Series(df.index[: min(len(df), 101)]).diff().dropna()
    return d.median().total_seconds() / 3600


def bars(df: pd.DataFrame, hours: float) -> int:
    """把以小時定義的長度換算成根數（至少 1 根）。1 小時線時等於原值。"""
    return max(1, int(round(hours / step_hours(df))))

"""分析区间截取（手动指定起止行号）"""

import pandas as pd


def validate_row_range(
    start_row: int | None,
    end_row: int | None,
    total_rows: int,
    label: str,
) -> tuple[int, int]:
    """校验行号区间，返回起止行号"""
    if start_row is None or end_row is None:
        raise ValueError(
            f"请在 config.py 中设置 {label}（1-based 数据行号，"
            "第 1 行指 CSV 中第一条数据，不含表头）"
        )
    if start_row < 1 or end_row > total_rows or start_row > end_row:
        raise ValueError(
            f"无效{label}: {start_row}~{end_row} "
            f"(数据共 {total_rows} 行，有效范围 1~{total_rows})"
        )
    return start_row, end_row


def slice_by_row_range(df: pd.DataFrame, start_row: int, end_row: int) -> pd.DataFrame:
    """按 1-based 行号截取数据"""
    return df.iloc[start_row - 1 : end_row].copy()

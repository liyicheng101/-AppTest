"""排气风阀扭矩曲线 — 工业数据分析程序"""

import os
import sys

import matplotlib.pyplot as plt
import pandas as pd

import config
from torque_analysis import run_torque_analysis


def setup_chinese_font():
    """配置 matplotlib 中文字体显示"""
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def read_csv_auto_encoding(filepath: str) -> tuple[pd.DataFrame, str]:
    """自动识别 UTF-8 / GBK 编码读取 CSV"""
    for encoding in ("utf-8-sig", "utf-8", "gbk"):
        try:
            df = pd.read_csv(filepath, encoding=encoding)
            return df, encoding
        except UnicodeDecodeError:
            continue
    raise ValueError(f"无法识别文件编码，已尝试: utf-8-sig, utf-8, gbk — {filepath}")


def read_uploaded_csv(file_bytes: bytes, filename: str = "") -> tuple[pd.DataFrame, str]:
    """从上传字节流自动识别编码读取 CSV"""
    from io import BytesIO

    for encoding in ("utf-8-sig", "utf-8", "gbk"):
        try:
            df = pd.read_csv(BytesIO(file_bytes), encoding=encoding)
            return df, encoding
        except UnicodeDecodeError:
            continue
    label = filename or "上传文件"
    raise ValueError(f"无法识别文件编码，已尝试: utf-8-sig, utf-8, gbk — {label}")


def print_data_info(df: pd.DataFrame, columns: list[str]) -> None:
    """输出数据基本信息"""
    print("=" * 50)
    print("数据基本信息")
    print("=" * 50)
    print(f"数据总行数: {len(df)}")
    print()

    print("每列数据类型:")
    for col in columns:
        if col in df.columns:
            print(f"  {col}: {df[col].dtype}")
        else:
            print(f"  {col}: [列不存在]")
    print()

    print("空值情况:")
    for col in columns:
        if col in df.columns:
            null_count = df[col].isna().sum()
            print(f"  {col}: {null_count} 个空值")
        else:
            print(f"  {col}: [列不存在]")
    has_null = df[columns].isna().any().any() if all(c in df.columns for c in columns) else df.isna().any().any()
    print(f"  是否存在空值: {'是' if has_null else '否'}")
    print()

    dup_count = df.duplicated().sum()
    print(f"重复行数: {dup_count}")
    print(f"是否存在重复数据: {'是' if dup_count > 0 else '否'}")
    print("=" * 50)


def parse_time_column(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    """将时间列解析为 datetime"""
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
    invalid = df[time_col].isna().sum()
    if invalid > 0:
        print(f"警告: 时间列有 {invalid} 行无法解析，将在绘图中忽略")
    return df


def validate_columns(df: pd.DataFrame, columns: list[str]) -> None:
    """检查必需列是否存在，缺失时抛出 ValueError"""
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"CSV 缺少以下列: {missing}；当前列名: {list(df.columns)}")


def validate_columns_or_exit(df: pd.DataFrame, columns: list[str]) -> None:
    """检查必需列是否存在，缺失时退出程序（命令行用）"""
    try:
        validate_columns(df, columns)
    except ValueError as e:
        print(f"错误: {e}")
        sys.exit(1)


def main():
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")

    setup_chinese_font()

    csv_path = config.CSV_FILE
    if not os.path.exists(csv_path):
        print(f"错误: 找不到数据文件 {csv_path}")
        sys.exit(1)

    columns = [config.COL_TIME, config.COL_FEEDBACK, config.COL_COMMAND, config.COL_TORQUE]

    df, encoding = read_csv_auto_encoding(csv_path)
    print(f"文件编码: {encoding}")
    print()

    validate_columns_or_exit(df, columns)
    print_data_info(df, columns)

    df = parse_time_column(df, config.COL_TIME)
    plot_df = df.dropna(subset=[config.COL_TIME])

    try:
        run_torque_analysis(plot_df, config.TORQUE_ANALYSIS_OUTPUT_DIR)
    except ValueError:
        sys.exit(1)

    print()
    print("分析完成。")


if __name__ == "__main__":
    main()

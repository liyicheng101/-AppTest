"""扭矩分箱分析：以阀门开度输出为横轴、Torque 为纵轴"""

import os
from dataclasses import dataclass
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import config
from motion_analysis import slice_by_row_range, validate_row_range

PHASE_OPENING = "Opening"
PHASE_CLOSING = "Closing"
COMMAND_VALUES = list(range(101))
OPENING_X_RANGE = (0, 100)
CLOSING_X_RANGE = (100, 0)
PNG_XLABEL = "Valve Command (%)"


@dataclass(frozen=True)
class TorqueMetric:
    """曲线指标配置（峰值 / 平均值）"""

    key: str
    column: str
    output_dir: str
    ylabel: str
    hover_label: str
    title_word: str
    ylabel_en: str
    title_word_en: str
    opening_x_desc: str = "0~100"
    closing_x_desc: str = "100~0"


METRIC_PEAK = TorqueMetric(
    key="peak",
    column="峰值扭矩",
    output_dir="Peak",
    ylabel=f"分箱峰值{config.COL_TORQUE}",
    hover_label=f"峰值{config.COL_TORQUE}",
    title_word="峰值",
    ylabel_en="Binned Peak Torque",
    title_word_en="Peak",
)
METRIC_MEAN = TorqueMetric(
    key="mean",
    column="平均值",
    output_dir="Mean",
    ylabel=f"分箱平均{config.COL_TORQUE}",
    hover_label=f"平均{config.COL_TORQUE}",
    title_word="平均",
    ylabel_en="Binned Mean Torque",
    title_word_en="Mean",
)
TORQUE_METRICS = (METRIC_PEAK, METRIC_MEAN)


def _png_single_title(metric: TorqueMetric, x_ascending: bool) -> str:
    """PNG 单图英文标题（避免 Linux 服务器缺中文字体）"""
    phase = "Opening" if x_ascending else "Closing"
    x_desc = metric.opening_x_desc if x_ascending else metric.closing_x_desc
    return f"{phase} Binned {metric.title_word_en} Torque (Command {x_desc}%)"


def _png_combined_title(metric: TorqueMetric) -> str:
    """PNG 合并图英文总标题"""
    return f"Opening / Closing Binned {metric.title_word_en} Torque"


@dataclass(frozen=True)
class RowRangeConfig:
    """开阀 / 关阀区间（1-based 数据行号，不含表头）"""

    opening_start: int
    opening_end: int
    closing_start: int
    closing_end: int

    @classmethod
    def from_config(cls) -> "RowRangeConfig":
        return cls(
            opening_start=config.OPENING_START_ROW,
            opening_end=config.OPENING_END_ROW,
            closing_start=config.CLOSING_START_ROW,
            closing_end=config.CLOSING_END_ROW,
        )


@dataclass
class WebAnalysisResult:
    """Web 分析结果：统计表 + Plotly 图 + PNG 字节"""

    phase_stats: dict[str, pd.DataFrame]
    plotly_figures: dict[str, go.Figure]
    png_bytes: dict[str, bytes]


def bin_by_command(df: pd.DataFrame) -> pd.DataFrame:
    """按阀门开度输出整数值 (0~100) 保留样本"""
    result = df[[config.COL_COMMAND, config.COL_TORQUE]].copy()
    result[config.COL_COMMAND] = pd.to_numeric(result[config.COL_COMMAND], errors="coerce")
    result[config.COL_TORQUE] = pd.to_numeric(result[config.COL_TORQUE], errors="coerce")
    result = result.dropna(subset=[config.COL_COMMAND, config.COL_TORQUE])
    result[config.COL_COMMAND] = result[config.COL_COMMAND].round().astype(int)
    return result[(result[config.COL_COMMAND] >= 0) & (result[config.COL_COMMAND] <= 100)]


def _peak_torque(values: pd.Series) -> float:
    """分箱内绝对值最大的扭矩（保留原符号；全为负时取绝对值更大者）"""
    if values.empty:
        return np.nan
    return float(values.iloc[values.abs().argmax()])


def compute_torque_stats(grouped_df: pd.DataFrame) -> pd.DataFrame:
    """按阀门开度输出 (0~100) 计算扭矩统计量，共 101 个分箱"""
    empty = _empty_stats()
    if grouped_df.empty:
        return empty

    stats = (
        grouped_df.groupby(config.COL_COMMAND, observed=True)[config.COL_TORQUE]
        .agg(
            平均值="mean",
            最大值="max",
            最小值="min",
            标准差="std",
            样本数量="count",
            峰值扭矩=_peak_torque,
        )
        .reset_index()
        .rename(columns={config.COL_COMMAND: "阀门开度输出"})
    )
    stats = stats.set_index("阀门开度输出").reindex(COMMAND_VALUES).reset_index()
    stats["样本数量"] = stats["样本数量"].fillna(0).astype(int)
    return stats


def _empty_stats() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "阀门开度输出": COMMAND_VALUES,
            "平均值": np.nan,
            "最大值": np.nan,
            "最小值": np.nan,
            "标准差": np.nan,
            "样本数量": 0,
            "峰值扭矩": np.nan,
        }
    )


def get_phase_dataframes(
    df: pd.DataFrame,
    row_range: RowRangeConfig | None = None,
) -> dict[str, pd.DataFrame]:
    """按手动行号截取开阀 / 关阀数据"""
    if row_range is None:
        row_range = RowRangeConfig.from_config()

    n = len(df)
    opening_start, opening_end = validate_row_range(
        row_range.opening_start,
        row_range.opening_end,
        n,
        "OPENING_START_ROW 和 OPENING_END_ROW",
    )
    closing_start, closing_end = validate_row_range(
        row_range.closing_start,
        row_range.closing_end,
        n,
        "CLOSING_START_ROW 和 CLOSING_END_ROW",
    )

    print()
    print("=" * 50)
    print("扭矩分析区间（手动配置）")
    print("=" * 50)
    print(f"  开阀 OPENING: 行 {opening_start} ~ {opening_end}  ({opening_end - opening_start + 1} 行)")
    print(f"  关阀 CLOSING: 行 {closing_start} ~ {closing_end}  ({closing_end - closing_start + 1} 行)")
    print("=" * 50)

    return {
        PHASE_OPENING: slice_by_row_range(df, opening_start, opening_end),
        PHASE_CLOSING: slice_by_row_range(df, closing_start, closing_end),
    }


def compute_phase_torque_stats(phase_dfs: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """分别计算开阀 / 关阀分箱扭矩统计"""
    return {
        phase: compute_torque_stats(bin_by_command(phase_df))
        for phase, phase_df in phase_dfs.items()
    }


def print_torque_stats(stats: pd.DataFrame, title: str) -> None:
    """打印分箱统计结果（仅显示有样本的分箱）"""
    active = stats[stats["样本数量"] > 0]
    print("=" * 50)
    print(title)
    print("=" * 50)
    print(f"分箱方式: {config.COL_COMMAND} 整数值 0~100（共 101 个分箱）")
    print(f"有样本分箱数: {len(active)} / {len(stats)}")
    print()

    if active.empty:
        print("警告: 无有效分箱数据")
        print("=" * 50)
        return

    display = active.copy()
    for col in ("平均值", "最大值", "最小值", "标准差", "峰值扭矩"):
        display[col] = display[col].map(lambda x: f"{x:.4f}")
    display["样本数量"] = display["样本数量"].astype(int)
    print(display.to_string(index=False))
    print("=" * 50)


def _ordered_stats(
    stats: pd.DataFrame,
    ascending: bool,
    max_command: int | None = None,
) -> pd.DataFrame:
    """按阀门开度输出排序，用于绘图（仅含样本数 > 0 的分箱）"""
    active = stats[stats["样本数量"] > 0].copy()
    if max_command is not None:
        active = active[active["阀门开度输出"] <= max_command]
    return active.sort_values("阀门开度输出", ascending=ascending)


def _y_limits(metric: TorqueMetric, *stats_list: pd.DataFrame) -> tuple[float, float] | None:
    """根据曲线数据计算 Y 轴范围（与 matplotlib 默认边距一致）"""
    values: list[float] = []
    for stats in stats_list:
        active = stats[stats["样本数量"] > 0]
        if not active.empty:
            values.extend(active[metric.column].dropna().tolist())
    if not values:
        return None

    y_min, y_max = min(values), max(values)
    if y_min == y_max:
        margin = abs(y_max) * 0.05 or 1.0
    else:
        margin = (y_max - y_min) * 0.05
    return y_min - margin, y_max + margin


def _build_torque_curve_figure(
    stats: pd.DataFrame,
    metric: TorqueMetric,
    title: str,
    color: str,
    x_ascending: bool,
    y_limits: tuple[float, float] | None = None,
    max_command: int | None = None,
    x_range: tuple[int, int] | None = None,
) -> go.Figure | None:
    """构建单条分箱扭矩曲线（Plotly）"""
    active = _ordered_stats(stats, ascending=x_ascending, max_command=max_command)
    if active.empty:
        return None

    if x_range is None:
        x_range = OPENING_X_RANGE if x_ascending else CLOSING_X_RANGE

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=active["阀门开度输出"],
            y=active[metric.column],
            mode="lines+markers",
            marker=dict(size=8, color=color),
            line=dict(width=1.2, color=color),
            hovertemplate=(
                f"{config.COL_COMMAND}: %{{x}}%<br>"
                f"{metric.hover_label}: %{{y:.4f}}"
                "<extra></extra>"
            ),
        )
    )
    yaxis = dict(title=metric.ylabel, fixedrange=True)
    if y_limits:
        yaxis["range"] = list(y_limits)

    fig.update_layout(
        title=title,
        xaxis=dict(
            title=f"{config.COL_COMMAND} (%)",
            range=list(x_range),
            fixedrange=True,
            dtick=10,
        ),
        yaxis=yaxis,
        hovermode="closest",
        template="plotly_white",
        width=int(config.FIG_SIZE[0] * 80),
        height=int(config.FIG_SIZE[1] * 80),
        margin=dict(l=60, r=30, t=60, b=50),
    )
    return fig


def _build_torque_curve_png_bytes(
    stats: pd.DataFrame,
    metric: TorqueMetric,
    title: str,
    color: str,
    marker: str,
    x_ascending: bool,
    y_limits: tuple[float, float] | None = None,
    max_command: int | None = None,
    x_range: tuple[int, int] | None = None,
    *,
    english_labels: bool = False,
) -> bytes | None:
    """绘制单条分箱扭矩曲线（PNG 字节）"""
    active = _ordered_stats(stats, ascending=x_ascending, max_command=max_command)
    if active.empty:
        return None

    if x_range is None:
        x_range = OPENING_X_RANGE if x_ascending else CLOSING_X_RANGE

    if english_labels:
        png_title = _png_single_title(metric, x_ascending)
        xlabel = PNG_XLABEL
        ylabel = metric.ylabel_en
    else:
        png_title = title
        xlabel = f"{config.COL_COMMAND} (%)"
        ylabel = metric.ylabel

    fig, ax = plt.subplots(figsize=config.FIG_SIZE)
    ax.plot(
        active["阀门开度输出"],
        active[metric.column],
        marker=marker,
        markersize=4,
        linewidth=1.2,
        color=color,
    )
    ax.set_title(png_title, fontsize=14)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xlim(x_range)
    if y_limits:
        ax.set_ylim(y_limits)
    ax.grid(True, linestyle="--", alpha=0.6)
    fig.tight_layout()
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=config.DPI, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def _save_torque_curve_png(
    stats: pd.DataFrame,
    metric: TorqueMetric,
    output_path: str,
    title: str,
    color: str,
    marker: str,
    x_ascending: bool,
    y_limits: tuple[float, float] | None = None,
    max_command: int | None = None,
    x_range: tuple[int, int] | None = None,
) -> bool:
    """绘制单条分箱扭矩曲线（静态 PNG）"""
    png_bytes = _build_torque_curve_png_bytes(
        stats,
        metric,
        title,
        color,
        marker,
        x_ascending,
        y_limits,
        max_command,
        x_range,
    )
    if png_bytes is None:
        return False
    with open(output_path, "wb") as f:
        f.write(png_bytes)
    return True


def _save_torque_curve_html(
    stats: pd.DataFrame,
    metric: TorqueMetric,
    output_path: str,
    title: str,
    color: str,
    x_ascending: bool,
    y_limits: tuple[float, float] | None = None,
    max_command: int | None = None,
    x_range: tuple[int, int] | None = None,
) -> bool:
    """绘制单条分箱扭矩曲线（交互 HTML）"""
    fig = _build_torque_curve_figure(
        stats, metric, title, color, x_ascending, y_limits, max_command, x_range
    )
    if fig is None:
        return False
    fig.write_html(output_path, include_plotlyjs="cdn")
    return True


def _build_combined_torque_curves_figure(
    opening: pd.DataFrame,
    closing: pd.DataFrame,
    metric: TorqueMetric,
    y_limits: tuple[float, float] | None = None,
) -> go.Figure | None:
    """构建开/关阀合并对比图（Plotly）"""
    opening_active = _ordered_stats(opening, ascending=True)
    closing_active = _ordered_stats(closing, ascending=False)
    if opening_active.empty and closing_active.empty:
        return None

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Opening", "Closing"),
        shared_yaxes=True,
        horizontal_spacing=0.08,
    )

    hovertemplate = (
        f"{config.COL_COMMAND}: %{{x}}%<br>"
        f"{metric.hover_label}: %{{y:.4f}}"
        "<extra></extra>"
    )

    if not opening_active.empty:
        fig.add_trace(
            go.Scatter(
                x=opening_active["阀门开度输出"],
                y=opening_active[metric.column],
                mode="lines+markers",
                marker=dict(size=7, color="#2ca02c"),
                line=dict(width=1.2, color="#2ca02c"),
                hovertemplate=hovertemplate,
                showlegend=False,
            ),
            row=1,
            col=1,
        )

    if not closing_active.empty:
        fig.add_trace(
            go.Scatter(
                x=closing_active["阀门开度输出"],
                y=closing_active[metric.column],
                mode="lines+markers",
                marker=dict(size=7, color="#ff7f0e"),
                line=dict(width=1.2, color="#ff7f0e"),
                hovertemplate=hovertemplate,
                showlegend=False,
            ),
            row=1,
            col=2,
        )

    yaxis_kwargs = dict(title=metric.ylabel, fixedrange=True)
    if y_limits:
        yaxis_kwargs["range"] = list(y_limits)

    fig.update_xaxes(
        title_text=f"{config.COL_COMMAND} (%)",
        range=list(OPENING_X_RANGE),
        dtick=10,
        fixedrange=True,
        row=1,
        col=1,
    )
    fig.update_xaxes(
        title_text=f"{config.COL_COMMAND} (%)",
        range=list(CLOSING_X_RANGE),
        dtick=10,
        fixedrange=True,
        row=1,
        col=2,
    )
    fig.update_yaxes(**yaxis_kwargs, row=1, col=1)
    fig.update_layout(
        title=f"Opening / Closing 分箱{metric.title_word}扭矩曲线",
        hovermode="closest",
        template="plotly_white",
        width=int(config.FIG_SIZE[0] * 1.6 * 80),
        height=int(config.FIG_SIZE[1] * 80),
        margin=dict(l=60, r=30, t=80, b=50),
    )
    return fig


def _save_combined_torque_curves_html(
    opening: pd.DataFrame,
    closing: pd.DataFrame,
    metric: TorqueMetric,
    output_path: str,
    y_limits: tuple[float, float] | None = None,
) -> bool:
    """绘制开/关阀合并对比图（交互 HTML）"""
    fig = _build_combined_torque_curves_figure(opening, closing, metric, y_limits)
    if fig is None:
        return False
    fig.write_html(output_path, include_plotlyjs="cdn")
    return True


def _build_combined_torque_curves_png_bytes(
    opening: pd.DataFrame,
    closing: pd.DataFrame,
    metric: TorqueMetric,
    y_limits: tuple[float, float] | None = None,
    *,
    english_labels: bool = False,
) -> bytes | None:
    """绘制开/关阀合并对比图（PNG 字节）"""
    opening_active = _ordered_stats(opening, ascending=True)
    closing_active = _ordered_stats(closing, ascending=False)
    if opening_active.empty and closing_active.empty:
        return None

    if english_labels:
        xlabel = PNG_XLABEL
        ylabel = metric.ylabel_en
        suptitle = _png_combined_title(metric)
    else:
        xlabel = f"{config.COL_COMMAND} (%)"
        ylabel = metric.ylabel
        suptitle = f"Opening / Closing 分箱{metric.title_word}扭矩曲线"

    fig, axes = plt.subplots(1, 2, figsize=(config.FIG_SIZE[0] * 1.6, config.FIG_SIZE[1]), sharey=True)

    if not opening_active.empty:
        axes[0].plot(
            opening_active["阀门开度输出"],
            opening_active[metric.column],
            marker="o",
            markersize=4,
            linewidth=1.2,
            color="#2ca02c",
        )
    axes[0].set_title("Opening", fontsize=13)
    axes[0].set_xlabel(xlabel, fontsize=12)
    axes[0].set_ylabel(ylabel, fontsize=12)
    axes[0].set_xlim(OPENING_X_RANGE)
    axes[0].grid(True, linestyle="--", alpha=0.6)

    if not closing_active.empty:
        axes[1].plot(
            closing_active["阀门开度输出"],
            closing_active[metric.column],
            marker="s",
            markersize=4,
            linewidth=1.2,
            color="#ff7f0e",
        )
    axes[1].set_title("Closing", fontsize=13)
    axes[1].set_xlabel(xlabel, fontsize=12)
    axes[1].set_xlim(CLOSING_X_RANGE)
    axes[1].grid(True, linestyle="--", alpha=0.6)

    if y_limits:
        axes[0].set_ylim(y_limits)
        axes[1].set_ylim(y_limits)

    fig.suptitle(suptitle, fontsize=14)
    fig.tight_layout()
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=config.DPI, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def _save_combined_torque_curves_png(
    opening: pd.DataFrame,
    closing: pd.DataFrame,
    metric: TorqueMetric,
    output_path: str,
    y_limits: tuple[float, float] | None = None,
) -> bool:
    """绘制开/关阀合并对比图（静态 PNG）"""
    png_bytes = _build_combined_torque_curves_png_bytes(opening, closing, metric, y_limits)
    if png_bytes is None:
        return False
    with open(output_path, "wb") as f:
        f.write(png_bytes)
    return True


def save_phase_torque_curves(
    phase_stats: dict[str, pd.DataFrame],
    base_output_dir: str,
) -> None:
    """保存峰值 / 平均值两套曲线（各含 PNG + HTML）"""
    opening = phase_stats[PHASE_OPENING]
    closing = phase_stats[PHASE_CLOSING]

    for metric in TORQUE_METRICS:
        output_dir = os.path.join(base_output_dir, metric.output_dir)
        os.makedirs(output_dir, exist_ok=True)
        combined_y_limits = _y_limits(metric, opening, closing)

        print(f"绘制{metric.title_word}扭矩曲线 ({metric.output_dir}/):")
        opening_title = (
            f"Opening 分箱{metric.title_word}扭矩曲线 "
            f"({config.COL_COMMAND} {metric.opening_x_desc})"
        )
        opening_png = os.path.join(output_dir, "opening_torque_vs_command.png")
        opening_html = os.path.join(output_dir, "opening_torque_vs_command.html")
        opening_y_limits = _y_limits(metric, opening)
        if _save_torque_curve_png(
            opening,
            metric,
            opening_png,
            opening_title,
            "#2ca02c",
            "o",
            True,
            opening_y_limits,
        ):
            print(f"  已保存: {opening_png}")
        else:
            print(f"  警告: 开阀无有效数据，跳过 {opening_png}")

        if _save_torque_curve_html(
            opening, metric, opening_html, opening_title, "#2ca02c", True, opening_y_limits
        ):
            print(f"  已保存: {opening_html}")

        closing_title = (
            f"Closing 分箱{metric.title_word}扭矩曲线 "
            f"({config.COL_COMMAND} {metric.closing_x_desc})"
        )
        closing_png = os.path.join(output_dir, "closing_torque_vs_command.png")
        closing_html = os.path.join(output_dir, "closing_torque_vs_command.html")
        closing_y_limits = _y_limits(metric, closing)
        if _save_torque_curve_png(
            closing,
            metric,
            closing_png,
            closing_title,
            "#ff7f0e",
            "s",
            False,
            closing_y_limits,
        ):
            print(f"  已保存: {closing_png}")
        else:
            print(f"  警告: 关阀无有效数据，跳过 {closing_png}")

        if _save_torque_curve_html(
            closing,
            metric,
            closing_html,
            closing_title,
            "#ff7f0e",
            False,
            closing_y_limits,
        ):
            print(f"  已保存: {closing_html}")

        combined_png = os.path.join(output_dir, "opening_closing_torque_vs_command.png")
        if _save_combined_torque_curves_png(opening, closing, metric, combined_png, combined_y_limits):
            print(f"  已保存: {combined_png}")

        combined_html = os.path.join(output_dir, "opening_closing_torque_vs_command.html")
        if _save_combined_torque_curves_html(
            opening, closing, metric, combined_html, combined_y_limits
        ):
            print(f"  已保存: {combined_html}")
        print()


def _build_web_figures(phase_stats: dict[str, pd.DataFrame]) -> tuple[dict[str, go.Figure], dict[str, bytes]]:
    """构建 Web 展示用的 Plotly 图与 PNG 字节"""
    opening = phase_stats[PHASE_OPENING]
    closing = phase_stats[PHASE_CLOSING]
    plotly_figures: dict[str, go.Figure] = {}
    png_bytes: dict[str, bytes] = {}

    for metric in TORQUE_METRICS:
        combined_y_limits = _y_limits(metric, opening, closing)
        opening_y_limits = _y_limits(metric, opening)
        closing_y_limits = _y_limits(metric, closing)

        opening_title = (
            f"Opening 分箱{metric.title_word}扭矩曲线 "
            f"({config.COL_COMMAND} {metric.opening_x_desc})"
        )
        closing_title = (
            f"Closing 分箱{metric.title_word}扭矩曲线 "
            f"({config.COL_COMMAND} {metric.closing_x_desc})"
        )

        prefix = metric.key
        opening_fig = _build_torque_curve_figure(
            opening, metric, opening_title, "#2ca02c", True, opening_y_limits
        )
        if opening_fig is not None:
            plotly_figures[f"{prefix}_opening"] = opening_fig

        closing_fig = _build_torque_curve_figure(
            closing, metric, closing_title, "#ff7f0e", False, closing_y_limits
        )
        if closing_fig is not None:
            plotly_figures[f"{prefix}_closing"] = closing_fig

        opening_png = _build_torque_curve_png_bytes(
            opening,
            metric,
            opening_title,
            "#2ca02c",
            "o",
            True,
            opening_y_limits,
            english_labels=True,
        )
        if opening_png is not None:
            png_bytes[f"{prefix}_opening"] = opening_png

        closing_png = _build_torque_curve_png_bytes(
            closing,
            metric,
            closing_title,
            "#ff7f0e",
            "s",
            False,
            closing_y_limits,
            english_labels=True,
        )
        if closing_png is not None:
            png_bytes[f"{prefix}_closing"] = closing_png

        combined_png = _build_combined_torque_curves_png_bytes(
            opening, closing, metric, combined_y_limits, english_labels=True
        )
        if combined_png is not None:
            png_bytes[f"{prefix}_combined"] = combined_png

    return plotly_figures, png_bytes


def analyze_torque_for_web(df: pd.DataFrame, row_range: RowRangeConfig) -> WebAnalysisResult:
    """Web 端扭矩分析：返回统计表、Plotly 图与 PNG 字节"""
    phase_dfs = get_phase_dataframes(df, row_range)
    phase_stats = compute_phase_torque_stats(phase_dfs)
    plotly_figures, png_bytes = _build_web_figures(phase_stats)
    return WebAnalysisResult(
        phase_stats=phase_stats,
        plotly_figures=plotly_figures,
        png_bytes=png_bytes,
    )


def run_torque_analysis(df: pd.DataFrame, output_dir: str) -> dict[str, pd.DataFrame]:
    """执行扭矩分箱分析：按手动开/关阀区间分别统计、绘图、导出 CSV"""
    try:
        phase_dfs = get_phase_dataframes(df)
    except ValueError as e:
        print(f"错误: {e}")
        raise

    phase_stats = compute_phase_torque_stats(phase_dfs)
    os.makedirs(output_dir, exist_ok=True)

    print()
    print_torque_stats(phase_stats[PHASE_OPENING], title="开阀 (Opening) 扭矩分箱分析")
    opening_path = os.path.join(output_dir, "torque_binned_stats_opening.csv")
    phase_stats[PHASE_OPENING].to_csv(opening_path, index=False, encoding="utf-8-sig")
    print(f"已保存统计表: {opening_path}")

    print()
    print_torque_stats(phase_stats[PHASE_CLOSING], title="关阀 (Closing) 扭矩分箱分析")
    closing_path = os.path.join(output_dir, "torque_binned_stats_closing.csv")
    phase_stats[PHASE_CLOSING].to_csv(closing_path, index=False, encoding="utf-8-sig")
    print(f"已保存统计表: {closing_path}")

    print()
    save_phase_torque_curves(phase_stats, output_dir)

    return phase_stats
    
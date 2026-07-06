"""排气风阀扭矩曲线 — Streamlit 网页版（内部分享用）"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import config
from analyze import parse_time_column, read_uploaded_csv, setup_chinese_font, validate_columns
from torque_analysis import (
    PHASE_CLOSING,
    PHASE_OPENING,
    RowRangeConfig,
    analyze_torque_for_web,
)

REQUIRED_COLUMNS = [config.COL_TIME, config.COL_FEEDBACK, config.COL_COMMAND, config.COL_TORQUE]

st.set_page_config(
    page_title="排气风阀扭矩曲线分析",
    page_icon="📈",
    layout="wide",
)

setup_chinese_font()


def build_preview_figure(df: pd.DataFrame) -> go.Figure:
    """开度输出随行号变化预览，便于确定开/关阀区间"""
    preview = df[[config.COL_COMMAND]].copy()
    preview[config.COL_COMMAND] = pd.to_numeric(preview[config.COL_COMMAND], errors="coerce")
    preview = preview.dropna()
    preview["行号"] = preview.index + 1

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=preview["行号"],
            y=preview[config.COL_COMMAND],
            mode="lines",
            line=dict(width=1, color="#1f77b4"),
            hovertemplate="行号: %{x}<br>开度输出: %{y:.1f}%<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"{config.COL_COMMAND} 随行号变化（用于辅助确定开/关阀区间）",
        xaxis_title="行号（1-based，不含表头）",
        yaxis_title=f"{config.COL_COMMAND} (%)",
        hovermode="x unified",
        template="plotly_white",
        height=360,
        margin=dict(l=50, r=20, t=60, b=40),
    )
    return fig


def stats_to_csv_bytes(stats: pd.DataFrame) -> bytes:
    return stats.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")


def render_downloads(result, metric_key: str, metric_label: str) -> None:
    """渲染 PNG / CSV 下载按钮"""
    cols = st.columns(4)
    png_map = {
        f"{metric_key}_opening": "开阀曲线.png",
        f"{metric_key}_closing": "关阀曲线.png",
        f"{metric_key}_combined": "开关阀对比.png",
    }
    for idx, (key, filename) in enumerate(png_map.items()):
        png = result.png_bytes.get(key)
        if png:
            cols[idx].download_button(
                f"下载 {filename}",
                data=png,
                file_name=filename,
                mime="image/png",
                key=f"dl_{metric_key}_{key}",
            )

    opening_csv = stats_to_csv_bytes(result.phase_stats[PHASE_OPENING])
    closing_csv = stats_to_csv_bytes(result.phase_stats[PHASE_CLOSING])
    cols[3].download_button(
        "下载开阀统计.csv",
        data=opening_csv,
        file_name="torque_binned_stats_opening.csv",
        mime="text/csv",
        key=f"dl_{metric_key}_opening_csv",
    )
    st.download_button(
        "下载关阀统计.csv",
        data=closing_csv,
        file_name="torque_binned_stats_closing.csv",
        mime="text/csv",
        key=f"dl_{metric_key}_closing_csv",
    )


def render_metric_tab(result, metric_key: str, metric_label: str) -> None:
    """展示某一指标（峰值 / 平均）的图表与下载"""
    render_downloads(result, metric_key, metric_label)

    opening = result.plotly_figures.get(f"{metric_key}_opening")
    closing = result.plotly_figures.get(f"{metric_key}_closing")

    if opening:
        st.plotly_chart(opening, use_container_width=True, key=f"chart_{metric_key}_opening")
    else:
        st.warning("开阀区间无有效分箱数据")

    if closing:
        st.plotly_chart(closing, use_container_width=True, key=f"chart_{metric_key}_closing")
    else:
        st.warning("关阀区间无有效分箱数据")

    with st.expander("查看分箱统计表"):
        st.markdown("**开阀 (Opening)**")
        st.dataframe(result.phase_stats[PHASE_OPENING], use_container_width=True)
        st.markdown("**关阀 (Closing)**")
        st.dataframe(result.phase_stats[PHASE_CLOSING], use_container_width=True)


st.title("排气风阀扭矩曲线分析")
st.caption("上传厂家 CSV，设置开/关阀行号区间，在线查看扭矩曲线并下载图片。")

uploaded = st.file_uploader("拖拽或选择 CSV 文件", type=["csv"])

if uploaded is None:
    st.info("请先上传 CSV 数据文件。")
    st.stop()

try:
    raw_df, encoding = read_uploaded_csv(uploaded.getvalue(), uploaded.name)
    validate_columns(raw_df, REQUIRED_COLUMNS)
except ValueError as e:
    st.error(str(e))
    st.stop()

df = parse_time_column(raw_df, config.COL_TIME)
total_rows = len(df)

st.success(f"已加载 **{uploaded.name}**（编码: {encoding}，共 **{total_rows}** 行数据）")

with st.expander("数据预览与区间辅助图", expanded=True):
    st.dataframe(df.head(10), use_container_width=True)
    st.plotly_chart(build_preview_figure(df), use_container_width=True)
    st.caption("提示：在上图中观察开度从 0→100（开阀）和 100→0（关阀）对应的行号范围。")

st.subheader("开/关阀区间设置")
st.caption("行号为 1-based，第 1 行指 CSV 中第一条数据，不含表头。")

col_a, col_b, col_c, col_d = st.columns(4)
default = RowRangeConfig.from_config()
opening_start = col_a.number_input(
    "开阀起始行",
    min_value=1,
    max_value=total_rows,
    value=min(default.opening_start, total_rows),
    step=1,
)
opening_end = col_b.number_input(
    "开阀结束行",
    min_value=1,
    max_value=total_rows,
    value=min(default.opening_end, total_rows),
    step=1,
)
closing_start = col_c.number_input(
    "关阀起始行",
    min_value=1,
    max_value=total_rows,
    value=min(default.closing_start, total_rows),
    step=1,
)
closing_end = col_d.number_input(
    "关阀结束行",
    min_value=1,
    max_value=total_rows,
    value=min(default.closing_end, total_rows),
    step=1,
)

analyze_clicked = st.button("开始分析", type="primary", use_container_width=True)

if analyze_clicked:
    row_range = RowRangeConfig(
        opening_start=int(opening_start),
        opening_end=int(opening_end),
        closing_start=int(closing_start),
        closing_end=int(closing_end),
    )
    try:
        with st.spinner("正在分析..."):
            result = analyze_torque_for_web(df, row_range)
    except ValueError as e:
        st.error(str(e))
        st.stop()

    st.subheader("分析结果")
    st.info(
        f"开阀: 行 {row_range.opening_start} ~ {row_range.opening_end} "
        f"（{row_range.opening_end - row_range.opening_start + 1} 行）｜"
        f"关阀: 行 {row_range.closing_start} ~ {row_range.closing_end} "
        f"（{row_range.closing_end - row_range.closing_start + 1} 行）"
    )

    tab_peak, tab_mean = st.tabs(["峰值扭矩", "平均扭矩"])
    with tab_peak:
        render_metric_tab(result, "peak", "峰值")
    with tab_mean:
        render_metric_tab(result, "mean", "平均")

    st.caption("鼠标悬停图表上的点可查看具体数值；需要打印时请先下载 PNG。")

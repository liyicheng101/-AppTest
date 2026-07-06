"""列名与绘图配置（列名变更时只需修改此处）"""

# CSV 列名
COL_TIME = "时间"
COL_FEEDBACK = "阀门开度反馈"
COL_COMMAND = "阀门开度输出"
COL_TORQUE = "扭矩传感器反馈"

# 输入输出路径
CSV_FILE = "开度反馈数据_exp.csv"

# 绘图参数
DPI = 300
FIG_SIZE = (12, 5)

# 开阀 / 关阀区间（扭矩分箱分析专用，1-based）
OPENING_START_ROW = 1822
OPENING_END_ROW = 12015
CLOSING_START_ROW = 12016
CLOSING_END_ROW = 21973

# 扭矩分箱分析（按阀门开度输出整数值 0~100，共 101 个分箱）
TORQUE_ANALYSIS_OUTPUT_DIR = "output/torque"

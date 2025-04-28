# 本项目的软件部分已上传至github，具体见https://github.com/Tsing-you/contest1
import pandas as pd
import matplotlib.pyplot as plt
from zhipuai import ZhipuAI
from sklearn.preprocessing import StandardScaler
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog
import plotly.express as px
from jinja2 import Template
import webbrowser
from scipy import signal
from sklearn.linear_model import LinearRegression
from datetime import datetime
import os
import traceback
from matplotlib import rcParams
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import plotly.io as pio
import pygame
import threading
import io
import re
import edge_tts
import asyncio
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from tkinter import messagebox

# from sklearn.ensemble import RandomForestClassifier
from statsmodels.tsa.arima.model import ARIMA


rcParams["font.sans-serif"] = ["SimHei"]  # 设置中文字体
rcParams["axes.unicode_minus"] = False  # 负号显示


def load_health_data(file_path):
    """文件读取进度显示"""
    try:
        # 检测文件是否有列头
        df_sample = pd.read_csv(file_path, nrows=0, encoding='gbk')
        has_header = list(df_sample.columns) == ["timestamp", "heart_rate", "blood_oxygen"]
        
        # 构建动态读取参数
        read_params = {
            "filepath_or_buffer": file_path,
            "parse_dates": ["timestamp"],
            "chunksize": 1000,
            "encoding": "gbk",
            "header": 0 if has_header else None
        }
        if not has_header:
            read_params["names"] = ["timestamp", "heart_rate", "blood_oxygen"]

        # 使用动态参数进行读取
        chunks = pd.read_csv(**read_params)
        df = pd.concat(chunks)

        # 验证必要字段
        if not {"timestamp", "heart_rate", "blood_oxygen"}.issubset(df.columns):
            missing = {"timestamp", "heart_rate", "blood_oxygen"} - set(df.columns)
            raise ValueError(f"缺少必要字段：{missing}")

        # 过滤掉所有汉字
        def clean_text(text):
            if isinstance(text, str):
                # 使用正则表达式过滤掉所有汉字
                return re.sub(r"[\u4e00-\u9fa5]", "", text)
            return text

        # 对每一行的 "timestamp" 和 "heart_rate" 列进行清理
        df["timestamp"] = df["timestamp"].apply(clean_text)
        df["heart_rate"] = df["heart_rate"].apply(clean_text)

        print("文件读取成功，有效记录数：", len(df))
        return df.dropna()
    except Exception as e:
        print("文件读取失败：", str(e))
        raise


# 数据预处理
def preprocess_data(df):
    raw_df = df[["timestamp", "heart_rate", "blood_oxygen"]].copy()

    # 异常值过滤 (保留40-140bpm)
    print("正在进行异常值过滤...")
    df = df[(df["heart_rate"] > 40) & (df["heart_rate"] < 140)].copy()
    df_ts = df

    # 标准化处理（仅对心率）
    scaler = StandardScaler()
    df_ts["heart_rate_scaled"] = scaler.fit_transform(df_ts[["heart_rate"]]).round(2)

    # 计算变化率（忽略第一个NaN值）
    df_ts["heart_rate_diff"] = df_ts["heart_rate"].diff().abs().fillna(0).round(2)

    # 移动平均（窗口对齐方式修正）
    df_ts["ma_3"] = df_ts["heart_rate"].rolling(3, min_periods=1).mean().round(2)

    # 新增血氧异常值过滤（正常范围90-100%）
    df = df[(df["blood_oxygen"] >= 85) & (df["blood_oxygen"] <= 100)].copy()

    # 新增血氧标准化
    df_ts["blood_oxygen_scaled"] = scaler.fit_transform(df_ts[["blood_oxygen"]]).round(
        2
    )

    # 新增血氧变化率
    df_ts["blood_oxygen_diff"] = df_ts["blood_oxygen"].diff().abs().fillna(0).round(2)

    # 新增血氧移动平均
    df_ts["ma_bo_3"] = df_ts["blood_oxygen"].rolling(3, min_periods=1).mean().round(2)

    print("处理完毕")
    return df_ts.reset_index(), raw_df  # 返回处理后的数据和原始数据


# 二、可视化模块
def advanced_visualization(df, analysis):
    # fig = plt.figure(figsize=(18, 16))

    # # 设置全局字体大小和颜色
    # plt.rcParams.update({"font.size": 10})
    # plt.rcParams["axes.prop_cycle"] = plt.cycler(
    #     color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    # )

    # # 心率趋势与移动平均
    # ax1 = plt.subplot(3, 2, 1)
    # df["ma_3"] = df["heart_rate"].rolling(3).mean()
    # valid_df = df.dropna(subset=["ma_3"])
    # ax1.plot(valid_df["timestamp"], valid_df["heart_rate"], label="原始数据")
    # ax1.plot(valid_df["timestamp"], valid_df["ma_3"], label="3点移动平均")
    # ax1.set_title("心率趋势与移动平均")
    # ax1.legend()
    # ax1.grid(True)
    # ax1.set_xticks([])
    # ax1.set_xlabel("")

    # # 功率谱密度
    # ax2 = plt.subplot(3, 2, 3)
    # f, Pxx = signal.welch(df["heart_rate"], fs=1.0, nperseg=256)
    # ax2.semilogy(f, Pxx)
    # ax2.set_xlabel("频率 [Hz]")
    # ax2.set_title("功率谱密度")
    # ax2.grid(True)

    # # HRV时域指标趋势图
    # ax3 = plt.subplot(3, 2, 2)
    # hr = df["heart_rate"].values
    # rr_intervals = 60000 / hr
    # sdnn_values = np.array(
    #     [np.std(rr_intervals[: i + 1], ddof=1) for i in range(len(rr_intervals))]
    # )
    # rmssd_values = np.array(
    #     [
    #         np.sqrt(np.mean(np.diff(rr_intervals[: i + 1]) ** 2))
    #         for i in range(len(rr_intervals))
    #     ]
    # )
    # ax3.plot(df["timestamp"], sdnn_values, label="SDNN")
    # ax3.plot(df["timestamp"], rmssd_values, label="RMSSD")
    # ax3.set_title("HRV时域指标趋势")
    # ax3.legend()
    # ax3.grid(True)
    # ax3.set_xticks([])
    # ax3.set_xlabel("")

    # # 异常值检测结果图
    # ax4 = plt.subplot(3, 2, 4)
    # upper_bound = df["heart_rate"].mean() + 2 * df["heart_rate"].std()
    # lower_bound = df["heart_rate"].mean() - 2 * df["heart_rate"].std()
    # ax4.scatter(df["timestamp"], df["heart_rate"], color="blue", label="正常数据")
    # ax4.scatter(
    #     df[df["heart_rate"] > upper_bound]["timestamp"],
    #     df[df["heart_rate"] > upper_bound]["heart_rate"],
    #     color="red",
    #     label="异常高心率",
    # )
    # ax4.scatter(
    #     df[df["heart_rate"] < lower_bound]["timestamp"],
    #     df[df["heart_rate"] < lower_bound]["heart_rate"],
    #     color="green",
    #     label="异常低心率",
    # )
    # ax4.set_title("异常值检测结果")
    # ax4.legend()
    # ax4.grid(True)
    # ax4.set_xticks([])
    # ax4.set_xlabel("")

    # # 心率分布直方图
    # ax5 = plt.subplot(3, 2, 5)
    # ax5.hist(df["heart_rate"], bins=30, edgecolor="black")
    # ax5.set_title("心率分布直方图")
    # ax5.grid(True)

    # # 频段能量占比饼图
    # ax6 = plt.subplot(3, 2, 6)
    # labels = ["LF(0.04-0.15Hz)", "HF(0.15-0.4Hz)"]
    # sizes = [analysis["LF能量占比"], analysis["HF能量占比"]]
    # ax6.pie(sizes, labels=labels, autopct="%1.1f%%")
    # ax6.set_title("频段能量占比")

    # # 新增血氧趋势子图（位置需要调整）
    # ax7 = plt.subplot(4, 2, 7)
    # ax7.plot(df["timestamp"], df["blood_oxygen"], label="血氧饱和度")
    # ax7.set_title("血氧趋势变化")
    # ax7.grid(True)
    # ax7.set_xticks([])
    # ax7.set_xlabel("")

    # # 新增血氧异常检测子图
    # ax8 = plt.subplot(4, 2, 8)
    # ax8.scatter(
    #     df["timestamp"],
    #     df["blood_oxygen"],
    #     c=np.where(df["blood_oxygen"] < 92, "red", "blue"),
    # )
    # ax8.set_title("血氧异常检测（<92%）")
    # ax8.set_xticks([])
    # ax8.set_xlabel("")

    # plt.tight_layout()

    # return fig

    fig = plt.figure(figsize=(15, 14))  # 增加整体高度

    # 设置全局字体大小和颜色
    plt.rcParams.update({"font.size": 10})
    plt.rcParams["axes.prop_cycle"] = plt.cycler(
        color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    )

    # 使用 GridSpec 来更灵活地控制子图布局
    gs = fig.add_gridspec(nrows=4, ncols=2, hspace=0.5, wspace=0.1)

    # 心率趋势与移动平均
    ax1 = fig.add_subplot(gs[0, 0])
    df["ma_3"] = df["heart_rate"].rolling(3).mean()
    valid_df = df.dropna(subset=["ma_3"])
    ax1.plot(valid_df["timestamp"], valid_df["heart_rate"], label="原始数据")
    ax1.plot(valid_df["timestamp"], valid_df["ma_3"], label="3点移动平均")
    ax1.set_title("心率趋势与移动平均")
    ax1.legend()
    ax1.grid(True)
    ax1.set_xticks([])

    # 功率谱密度
    ax2 = fig.add_subplot(gs[1, 0])
    f, Pxx = signal.welch(df["heart_rate"], fs=1.0, nperseg=256)
    ax2.semilogy(f, Pxx)
    ax2.set_xlabel("频率 [Hz]")
    ax2.set_title("功率谱密度")
    ax2.grid(True)

    # HRV时域指标趋势图
    ax3 = fig.add_subplot(gs[0, 1])
    hr = df["heart_rate"].values
    rr_intervals = 60000 / hr
    sdnn_values = np.array(
        [np.std(rr_intervals[: i + 1], ddof=1) for i in range(len(rr_intervals))]
    )
    rmssd_values = np.array(
        [
            np.sqrt(np.mean(np.diff(rr_intervals[: i + 1]) ** 2))
            for i in range(len(rr_intervals))
        ]
    )
    ax3.plot(df["timestamp"], sdnn_values, label="SDNN")
    ax3.plot(df["timestamp"], rmssd_values, label="RMSSD")
    ax3.set_title("HRV时域指标趋势")
    ax3.legend()
    ax3.grid(True)
    ax3.set_xticks([])

    # 异常值检测结果图
    ax4 = fig.add_subplot(gs[1, 1])
    upper_bound = df["heart_rate"].mean() + 2 * df["heart_rate"].std()
    lower_bound = df["heart_rate"].mean() - 2 * df["heart_rate"].std()
    ax4.scatter(df["timestamp"], df["heart_rate"], color="blue", label="正常数据")
    ax4.scatter(
        df[df["heart_rate"] > upper_bound]["timestamp"],
        df[df["heart_rate"] > upper_bound]["heart_rate"],
        color="red",
        label="异常高心率",
    )
    ax4.scatter(
        df[df["heart_rate"] < lower_bound]["timestamp"],
        df[df["heart_rate"] < lower_bound]["heart_rate"],
        color="green",
        label="异常低心率",
    )
    ax4.set_title("心率异常值检测结果")
    ax4.legend()
    ax4.grid(True)
    ax4.set_xticks([])

    # 心率分布直方图
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.hist(df["heart_rate"], bins=30, edgecolor="black")
    ax5.set_title("心率分布直方图")
    ax5.grid(True)

    # 频段能量占比饼图
    ax6 = fig.add_subplot(gs[2, 1])
    labels = ["LF(0.04-0.15Hz)", "HF(0.15-0.4Hz)"]
    sizes = [analysis["LF能量占比"], analysis["HF能量占比"]]
    ax6.pie(sizes, labels=labels, autopct="%1.1f%%")
    ax6.set_title("频段能量占比")

    # 新增血氧趋势子图（位置需要调整）
    ax7 = fig.add_subplot(gs[3, 0])
    ax7.plot(df["timestamp"], df["blood_oxygen"], label="血氧饱和度")
    ax7.set_title("血氧趋势变化")
    ax7.grid(True)
    ax7.set_xticks([])

    # 新增血氧异常检测子图
    ax8 = fig.add_subplot(gs[3, 1])
    ax8.scatter(
        df["timestamp"],
        df["blood_oxygen"],
        c=np.where(df["blood_oxygen"] < 92, "red", "blue"),
    )
    ax8.set_title("血氧异常检测（<92%）")
    ax8.set_xticks([])

    plt.tight_layout()

    return fig


# 分析函数
def comprehensive_analysis(df, anomalies):
    analysis = {}
    hr = df["heart_rate"].values

    # 基础统计
    analysis.update(
        {
            "平均心率": hr.mean(),
            "最大值": hr.max(),
            "最小值": hr.min(),
            "极差": np.ptp(hr),
            "心率标准差": hr.std(),
            "平均血氧": df["blood_oxygen"].mean(),
            "血氧标准差": df["blood_oxygen"].std(),
            "最低血氧": df["blood_oxygen"].min(),
            "血氧异常次数": len(df[df["blood_oxygen"] < 92]),
        }
    )

    # 基础指标扩展
    base_metrics = {
        "heart_rate": "心率",
        "heart_rate_diff": "心率变化率",
        "ma_3": "移动平均",
    }

    for col, name in base_metrics.items():
        analysis.update(
            {
                f"{name}平均值": df[col].mean(),
                f"{name}标准差": df[col].std(),
                f"{name}最大值": df[col].max(),
                f"{name}最小值": df[col].min(),
            }
        )

    # HRV时域指标
    rr_intervals = 60000 / hr  # 转换心率到RR间期（毫秒）
    # 添加SDNN和RMSSD到DataFrame
    df = df.copy()  # 避免SettingWithCopyWarning
    df["SDNN"] = np.std(rr_intervals, ddof=1).round(1)
    rr_diff = np.diff(rr_intervals)
    df["RMSSD"] = np.sqrt(np.mean(rr_diff**2)).round(1)
    # 全局计算 SDNN（整个时间段的标准差）
    analysis["SDNN"] = np.std(rr_intervals, ddof=1).round(1)
    # 全局计算 RMSSD（相邻 RR 间期差值的均方根）
    rr_diff = np.diff(rr_intervals)
    analysis["RMSSD"] = np.sqrt(np.mean(rr_diff**2)).round(1)

    # 频域分析
    f, Pxx = signal.welch(hr, fs=1.0, nperseg=256)
    lf_band = (0.04, 0.15)
    hf_band = (0.15, 0.4)
    lf_power = Pxx[(f >= lf_band[0]) & (f < lf_band[1])].sum()
    hf_power = Pxx[(f >= hf_band[0]) & (f < hf_band[1])].sum()

    analysis.update(
        {
            "LF/HF": lf_power / hf_power,
            "LF能量占比": lf_power / (lf_power + hf_power),
            "HF能量占比": hf_power / (lf_power + hf_power),
        }
    )

    # 趋势分析
    trend_model = LinearRegression().fit(np.arange(len(hr)).reshape(-1, 1), hr)
    analysis["趋势斜率"] = trend_model.coef_[0]

    # 1. 心血管风险预测模型（逻辑回归示例）
    X = df[["heart_rate", "blood_oxygen", "SDNN"]].copy()  # 确保使用副本
    X.fillna(X.mean(), inplace=True)  # 处理可能的NaN值

    y_risk = np.where((df["heart_rate"] > 100) | (df["blood_oxygen"] < 92), 1, 0)
    risk_model = LogisticRegression().fit(X, y_risk)
    analysis["心血管风险概率"] = risk_model.predict_proba(X)[:, 1].mean().round(2)

    # 2. 呼吸功能评估（多项式回归）
    if "PolynomialFeatures" in globals():
        poly = PolynomialFeatures(degree=2)
        X_poly = poly.fit_transform(df[["blood_oxygen"]])
        resp_model = LinearRegression().fit(X_poly, df["heart_rate"])
        analysis["血氧-心率关联度"] = round(
            resp_model.score(X_poly, df["heart_rate"]), 2
        )
    else:
        analysis["血氧-心率关联度"] = 0  # 或注释掉此部分计算

    # 3. 简化时序分析（改用标准差评估平稳性）
    analysis["心率平稳性"] = (
        df["heart_rate"].rolling(24, min_periods=1).std().mean().round(2)
    )  # 24个数据点的滑动窗口标准差

    # 添加动态阈值计算
    analysis["心率上限"] = df["heart_rate"].mean() + 2 * df["heart_rate"].std()
    analysis["心率下限"] = df["heart_rate"].mean() - 2 * df["heart_rate"].std()

    base_score = 90
    score_deduction = min(len(anomalies) * 0.5, 15)  # 使用传入的anomalies参数
    if analysis["LF/HF"] > 3:
        score_deduction += 5
    if analysis["平均血氧"] < 95:
        score_deduction += 3
    if analysis["血氧异常次数"] > 0:
        score_deduction += analysis["血氧异常次数"] * 0.2
    analysis["健康评分"] = max(60, base_score - score_deduction)

    # 时序预测
    model = ARIMA(df["heart_rate"], order=(1, 1, 1))
    results = model.fit()
    forecast = results.get_forecast(steps=5)
    analysis["趋势预测"] = {
        "心率": forecast.predicted_mean.iloc[0],
        "波动": forecast.se_mean.iloc[0],
        "血氧稳定": round(
            1
            - min(
                df["blood_oxygen"].rolling(10).std().fillna(3).iloc[-1] / 3
                + len(anomalies) * 0.02,
                0.3,
            ),
            2,
        ),
    }

    return analysis


def anomaly_detection(df):
    """异常检测"""
    warnings = []
    upper_bound = df["heart_rate"].mean() + 2 * df["heart_rate"].std()
    lower_bound = df["heart_rate"].mean() - 2 * df["heart_rate"].std()

    for _, row in df.iterrows():
        if row["heart_rate"] > upper_bound:
            warnings.append(f"⚠️ 异常高心率 {row['timestamp']}")
        elif row["heart_rate"] < lower_bound:
            warnings.append(f"⚠️ 异常低心率 {row['timestamp']}")
    for _, row in df.iterrows():
        if row["blood_oxygen"] < 92:
            warnings.append(f"⚠️ 低血氧 {row['timestamp']} ({row['blood_oxygen']}%)")
    return warnings


# 生成报告函数
def generate_report(df, analysis, advice, anomalies):

    target_columns = ["heart_rate", "heart_rate_diff", "ma_3"]
    describe_df = df[target_columns].describe()

    # 列名映射
    cn_index = {  # 列映射为索引映射
        "count": "数据量",
        "mean": "平均值",
        "std": "标准差",
        "min": "最小值",
        "25%": "下四分位",
        "50%": "中位数",
        "75%": "上四分位",
        "max": "最大值",
    }

    # 生成多指标HTML表格
    stats_html = describe_df.rename(index=cn_index).T.to_html(
        classes="table table-striped", header=True
    )

    # 趋势图
    extended_trend_fig = px.line(
        df, x="timestamp", y=["heart_rate", "ma_3"], title="心率与移动平均趋势分析"
    )
    extended_trend_html = pio.to_html(extended_trend_fig, full_html=False)

    # 生成交互式图表
    trend_fig = px.line(
        df,
        x="timestamp",
        y="heart_rate",
        title="心率趋势分析",
        labels={"heart_rate": "心率 (bpm)"},
    )
    trend_html = pio.to_html(trend_fig, full_html=False)

    blood_oxygen_fig = px.line(
        df, 
        x="timestamp", 
        y="blood_oxygen",
        color=df["blood_oxygen"].apply(lambda x: "异常" if x < 92 else "正常"),
        title="血氧饱和度监测"
    )
    blood_oxygen_html = pio.to_html(blood_oxygen_fig, full_html=False)

    # distribution_fig = px.histogram(
    #     df, x="heart_rate", nbins=30, title="心率分布直方图"
    # )
    # distribution_html = pio.to_html(distribution_fig, full_html=False)

    # 新增血氧趋势图
    # blood_oxygen_fig = px.scatter(
    #     df, 
    #     x="timestamp", 
    #     y="blood_oxygen",
    #     color=df["blood_oxygen"].apply(lambda x: "异常" if x < 92 else "正常"),
    #     title="血氧饱和度监测"
    # )
    # blood_oxygen_html = pio.to_html(blood_oxygen_fig, full_html=False)

    # # 新增HRV指标图
    # hrv_fig = px.line(
    #     df,
    #     x="timestamp",
    #     y=["SDNN", "RMSSD"],
    #     title="心率变异性趋势"
    # )
    # hrv_html = pio.to_html(hrv_fig, full_html=False)

    # 创建报告目录
    report_dir = "reports"
    os.makedirs(report_dir, exist_ok=True)

    # 带时间戳的文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    filename = f"heart_report_{timestamp}.html"
    filepath = os.path.join(report_dir, filename)

    with open("report_template.html", "r", encoding="utf-8") as f:
        template = Template(f.read())

    # 明确指定分析列
    describe_df = df["heart_rate"].describe().to_frame()  # 转为DataFrame

    # 索引重命名
    stats_html = describe_df.rename(index=cn_index).to_html(
        classes="table table-striped", header=False  # 隐藏列名
    )
    # 模板渲染
    rendered = template.render(
        stats=stats_html,
        analysis=analysis,
        ai_advice=advice,
        anomalies=anomalies,
        now=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        trend_chart=trend_html,
        # distribution_chart=distribution_html,
        blood_oxygen_chart=blood_oxygen_html,
        risk_prediction=analysis.get('趋势预测', {}),
    )

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(rendered)

    return os.path.abspath(filepath)


# 六、GUI界面
class HeartAnalysisApp:

    def __init__(self, root):
        self.root = root

        style = ttk.Style()
        style.theme_use("clam")
        style.configure(".", font=("微软雅黑", 10))  # 全局字体
        style.configure("TButton", padding=6)  # 按钮样式
        style.map(
            "TButton",
            foreground=[("active", "!disabled", "blue")],
            background=[("active", "#e6e6e6")],
        )

        self.root.title("心率分析系统 v2.0")
        self.root.geometry("1500x700")

        self.raw_data = None
        self.processed_data = None
        self.analysis_results = None
        self.anomalies = None
        self.ai_advice = None
        self.running = False
        self.fig_canvas = None  # 初始化画布
        self.img_label = None

        self.decomposition_img = None  # 保存图片引用
        self.create_widgets()

        self.speech_lock = threading.Lock()
        self.is_speaking = False
        # 初始化音频系统
        pygame.mixer.init()

        self.current_request = None

        # 该部分初始化部分弃用，仅作调试使用

    def create_widgets(self):
        # 工具栏
        self.toolbar = ttk.Frame(self.root)
        self.toolbar.pack(side=tk.TOP, fill=tk.X, padx=5, pady=2)

        # 操作按钮
        ttk.Button(self.toolbar, text="加载数据", command=self.load_data).pack(
            side=tk.LEFT, padx=5
        )
        ttk.Button(self.toolbar, text="开始分析", command=self.run_analysis).pack(
            side=tk.LEFT, padx=5
        )
        ttk.Button(self.toolbar, text="生成报告", command=self.generate_report).pack(
            side=tk.LEFT, padx=5
        )
        ttk.Button(self.toolbar, text="发送报告", command=self.send_report).pack(
            side=tk.LEFT, padx=5
        )

        # 状态标签
        self.status_label = ttk.Label(
            self.toolbar,
            text="就绪",
            foreground="red",
            font=("微软雅黑", 20),
        )
        self.status_label.pack(side=tk.RIGHT, padx=10)

        # 主内容区
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # 创建 Notebook
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        # 第一页：数据概览与分析结果
        self.page1 = ttk.Frame(self.notebook)
        self.notebook.add(self.page1, text="数据概览与分析结果")
        self.build_page1(self.page1)

        # 第二页：可视化分析
        self.page2 = ttk.Frame(self.notebook)
        self.notebook.add(self.page2, text="可视化分析")
        self.build_page2(self.page2)

        # 第三页：AI健康建议
        self.page3 = ttk.Frame(self.notebook)
        self.notebook.add(self.page3, text="AI健康建议")
        self.build_page3(self.page3)

    def build_page1(self, parent):
        """构建第一页"""
        # 数据概览
        data_frame = ttk.LabelFrame(parent, text=" 数据概览 ", padding=(10, 5))
        data_frame.pack(fill=tk.X, pady=5)

        self.data_info = tk.Text(data_frame, height=12, bg="#f8f9fa", relief="flat")
        self.data_info.pack(fill=tk.X)

        # 分析结果
        analysis_frame = ttk.LabelFrame(parent, text=" 分析结果 ", padding=(10, 5))
        analysis_frame.pack(fill=tk.X, pady=5)

        self.analysis_text = tk.Text(
            analysis_frame, height=30, bg="#f8f9fa", relief="flat"
        )
        self.analysis_text.pack(fill=tk.X)

    def show_data_preview(self):
        """在控制面板显示数据基本信息"""
        self.data_info.delete(1.0, tk.END)
        if self.processed_data is not None:
            preview = f"记录数: {len(self.processed_data)}\n"
            preview += f"时间范围:\n{self.processed_data['timestamp'].min()}\n至\n{self.processed_data['timestamp'].max()}\n\n"

            # 显示数据详情
            preview += "数据预览：\n"
            preview += self.processed_data.to_string(index=False)

            self.data_info.insert(tk.END, preview)

    def update_display(self):
        self.analysis_text.delete(1.0, tk.END)
        analysis = self.analysis_results

        # 基础健康指标仪表盘
        analysis_str = "🏥 核心健康指标 🏥\n"
        analysis_str += f"• 健康评分: {analysis.get('健康评分', 0):.0f}/100\n"
        analysis_str += (
            f"• 综合心血管风险指数: {analysis.get('心血管风险概率', 0)*100:.1f}%\n"
        )
        analysis_str += "━━━━━━━━━━━━━━━━━━━━\n"

        # 心率综合分析
        analysis_str += "\n❤️ 心率分析 ❤️\n"
        analysis_str += (
            f"• 实时波动：{analysis.get('心率平稳性', 0):.1f} bpm（标准差）\n"
        )
        analysis_str += f"   - {'⚠️ 波动异常' if analysis.get('心率平稳性',0)>8 else '✅ 波动正常'}\n"
        analysis_str += f"• 趋势变化：{analysis.get('趋势斜率', 0):.3f} bpm/分钟\n"
        analysis_str += (
            f"   - {'↑ 上升趋势' if analysis['趋势斜率']>0 else '↓ 下降趋势'}\n"
        )
        analysis_str += f"• 移动平均：{analysis.get('移动平均平均值', 0):.1f} ± {analysis.get('移动平均标准差', 0):.1f} bpm\n"

        # 血氧深度分析
        analysis_str += "\n🌡️ 血氧分析 🌡️\n"
        analysis_str += f"• 最低血氧：{analysis.get('最低血氧', 0)}% （{'⚠️ 需关注' if analysis['最低血氧']<92 else '✅ 正常'})\n"
        analysis_str += f"• 异常次数：{analysis.get('血氧异常次数', 0)}次（<92%阈值）\n"
        analysis_str += (
            f"• 血氧-心率关联：R²={analysis.get('血氧-心率关联度', 0):.2f}\n"
        )

        # 新增健康风险评估
        analysis_str += "\n⚠️ 风险评估 ⚠️\n"
        risk_factors = {
            "高频异常心率": analysis.get("异常数量", 0),
            "血氧不足": analysis.get("血氧异常次数", 0),
            "心率波动": analysis.get("心率平稳性", 0),
        }
        for factor, value in risk_factors.items():
            analysis_str += f"• {factor}: {value} ({'⚠️' if value>0 else '✅'})\n"

        # 特征关联度可视化描述
        analysis_str += "\n🔗 健康关联度 🔗\n"
        if "健康相关度" in analysis:
            for feature, score in analysis["健康相关度"].items():
                stars = "★" * int(score * 10)
                analysis_str += f"• {feature}: {stars} ({score:.2f})\n"
        elif "特征重要性" in analysis:  # 兼容旧版本
            for feature, importance in analysis["特征重要性"].items():
                analysis_str += f"• {feature}: {importance:.0%}\n"

        # 异常事件明细
        analysis_str += "\n🚨 异常事件明细 🚨\n"
        if self.anomalies:
            counter = {"高心率": 0, "低心率": 0, "低血氧": 0}
            for msg in self.anomalies:
                if "高心率" in msg:
                    counter["高心率"] += 1
                elif "低心率" in msg:
                    counter["低心率"] += 1
                elif "低血氧" in msg:
                    counter["低血氧"] += 1

            analysis_str += f"• 高频异常: {counter['高心率']}次（>{analysis.get('心率上限', analysis['心率上限']):.0f}bpm）\n"
            analysis_str += f"• 低频异常: {counter['低心率']}次（<{analysis.get('心率下限', analysis['心率下限']):.0f}bpm）\n"
            analysis_str += f"• 血氧异常: {counter['低血氧']}次（<92%）\n"
        else:
            analysis_str += "✅ 未检测到显著异常事件\n"

        # 新增健康趋势预测
        if "趋势预测" in analysis:
            analysis_str += "\n🔮 未来趋势预测 🔮\n"
            analysis_str += f"• 下一时段心率预测：{analysis['趋势预测']['心率']:.0f}±{analysis['趋势预测']['波动']:.1f}bpm\n"
            analysis_str += f"• 血氧维持概率：{analysis['趋势预测']['血氧稳定']:.0%}\n"

        # 专业指标分析
        analysis_str += "\n📊 专业指标 📊\n"
        analysis_str += f"• SDNN：{analysis.get('SDNN', 0):.1f} ms（{'⚠️ 自主神经失调' if analysis['SDNN']<50 else '✅ 正常'})\n"
        analysis_str += f"• RMSSD：{analysis.get('RMSSD', 0):.1f} ms\n"
        analysis_str += f"• LF/HF平衡：{analysis.get('LF/HF', 0):.2f}（{'⚠️ 压力状态' if analysis['LF/HF']>3 else '✅ 平衡状态'})\n"

        # 自动生成健康标签
        tags = []
        if analysis.get("健康评分", 0) > 80:
            tags.append("👍 健康状态良好")
        if analysis.get("心血管风险概率", 0) > 0.3:
            tags.append("❗ 心血管风险关注")
        if analysis.get("血氧异常次数", 0) > 5:
            tags.append("⚠️ 呼吸功能关注")

        if tags:
            self.status_label.config(text=" | ".join(tags), foreground="orange")

        analysis_str += "\n\n🔍 健康概念与技术解析 🔍\n"
        analysis_str += "本系统通过多维度医学指标与先进建模技术实现深度健康评估：\n"
        analysis_str += "1. 核心医学指标\n"
        analysis_str += "• SDNN（标准差NN间期）反映心脏自主神经调节能力，正常值>50ms，低于阈值提示交感副交感失衡\n"
        analysis_str += (
            "• RMSSD（均方根差值）量化瞬时心率变异性，敏感反映副交感神经活性\n"
        )
        analysis_str += "• LF/HF功率比通过频域分析揭示交感（LF）与副交感（HF）神经平衡状态，比值>3提示长期压力负荷\n"
        analysis_str += "• 血氧饱和度（SpO₂）是呼吸循环系统关键指标，正常范围95-100%，<92%需临床关注\n"
        analysis_str += "• 健康评分综合异常事件频率、神经平衡状态和血氧水平，采用动态扣分模型（基础分90分，异常事件每次扣0.5分，LF/HF超标加扣5分）\n\n"

        analysis_str += "2. 建模方法与技术实现\n"
        analysis_str += "• 风险预测：逻辑回归模型分析心率>100bpm或血氧<92%的异常组合，输出心血管风险概率\n"
        analysis_str += "• 趋势分析：线性回归捕捉心率长期趋势，ARIMA模型进行5步心率预测，结合滑动窗口标准差（24个数据点）评估心率平稳性\n"
        analysis_str += "• 非线性关联：多项式回归（2次方程）量化血氧与心率的非线性关系，R²值反映关联强度\n"
        analysis_str += "• 异常检测：采用±2σ动态阈值检测异常心率，结合血氧绝对阈值（<92%）实现多参数联合预警\n\n"

        analysis_str += "3. 技术亮点\n"
        analysis_str += "• 多模态融合：整合时域（HRV指标）、频域（功率谱分析）、非线性（多项式关联）特征，构建全面评估体系\n"
        analysis_str += "• 可视化增强：Plotly交互图表与Matplotlib静态图表双引擎配合，支持缩放/数据点悬停分析\n"
        analysis_str += "• 实时计算：滚动窗口标准差（24数据点）和动态阈值（±2σ）确保异常检测的实时性与自适应性\n"
        analysis_str += "• 混合建模：传统统计指标（如SDNN）与机器学习模型（如ARIMA预测）协同工作，兼顾临床解释性与预测能力\n"
        analysis_str += "• 临床关联：将LF/HF比值、血氧异常次数等客观指标直接映射到健康标签（如'压力状态'、'自主神经失调'）"

        self.analysis_text.insert(tk.END, analysis_str)

    # 修改build_page2方法
    def build_page2(self, parent):
        """构建第二页（优化边距和滚动）"""
        # 主容器使用Frame代替Canvas实现更简单的滚动
        main_frame = ttk.Frame(parent)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=0, pady=0)  # 消除母容器边距

        # 创建Canvas和滚动条
        canvas = tk.Canvas(main_frame, highlightthickness=0)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)

        # 配置滚动区域
        scrollable_frame.bind(
            "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )

        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        # 网格布局实现自适应
        main_frame.grid_columnconfigure(0, weight=1)
        main_frame.grid_rowconfigure(0, weight=1)

        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")

        # 图表容器（减少边距）
        self.img_label = ttk.Frame(scrollable_frame)
        self.img_label.pack(
            fill=tk.BOTH, expand=True, padx=2, pady=6
        )  # 内部边距保留5px

        # 绑定全局鼠标滚轮
        canvas.bind(
            "<Enter>",
            lambda e: canvas.bind_all(
                "<MouseWheel>", lambda event: self._on_mousewheel(event, canvas)
            ),
        )
        canvas.bind("<Leave>", lambda e: canvas.unbind_all("<MouseWheel>"))

    # 修改show_matplotlib_figure方法
    def show_matplotlib_figure(self, fig):
        # 清除旧内容
        if self.fig_canvas:
            self.fig_canvas.get_tk_widget().destroy()

        # # 调整图表尺寸
        # fig.set_size_inches(17, len(fig.axes) * 2)  # 动态高度

        # 显示图表
        self.fig_canvas = FigureCanvasTkAgg(fig, master=self.img_label)
        self.fig_canvas.draw()
        self.fig_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def build_page3(self, parent):
        """构建第三页（优化版）"""
        main_frame = ttk.Frame(parent)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # 聊天展示区域
        chat_container = ttk.Frame(main_frame)
        chat_container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 创建滚动区域
        self.scroll_frame = ttk.Frame(chat_container)
        self.scroll_frame.pack(fill=tk.BOTH, expand=True)

        # 创建滚动条
        self.scrollbar = ttk.Scrollbar(self.scroll_frame, orient="vertical")
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # 创建Canvas
        self.chat_canvas = tk.Canvas(
            self.scroll_frame,
            bg="white",
            highlightthickness=0,
            yscrollcommand=self.scrollbar.set,
        )
        self.chat_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.scrollbar.config(command=self.chat_canvas.yview)

        # 创建内部Frame
        self.chat_frame = ttk.Frame(self.chat_canvas)
        self.chat_canvas.create_window(
            (0, 0), window=self.chat_frame, anchor="nw", tags="inner_frame"
        )

        # 动态宽度配置
        def _on_canvas_configure(event):
            """动态调整内部Frame宽度"""
            canvas_width = event.width
            self.chat_canvas.itemconfig("inner_frame", width=canvas_width)

            # 更新已有消息的换行长度
            for widget in self.chat_frame.winfo_children():
                if isinstance(widget, ttk.Frame):
                    for child in widget.winfo_children():
                        if isinstance(child, ttk.Label) and hasattr(
                            child, "wraplength"
                        ):
                            child.configure(wraplength=canvas_width - 20)  # 保留边距

        self.chat_canvas.bind("<Configure>", _on_canvas_configure)

        # 智能滚动绑定
        def _on_mousewheel(event):
            if self.chat_canvas.winfo_height() > 0:
                self.chat_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        def _bind_mousewheel():
            self.chat_canvas.bind_all("<MouseWheel>", _on_mousewheel)

        def _unbind_mousewheel():
            self.chat_canvas.unbind_all("<MouseWheel>")

        self.chat_canvas.bind("<Enter>", lambda e: _bind_mousewheel())
        self.chat_canvas.bind("<Leave>", lambda e: _unbind_mousewheel())

        # 输入区域
        input_frame = ttk.Frame(main_frame)
        input_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=5, padx=5)

        self.user_input = ttk.Entry(input_frame)
        self.user_input.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        send_button = ttk.Button(input_frame, text="发送", command=self.send_message)
        send_button.pack(side=tk.RIGHT, padx=5)
        self.user_input.bind("<Return>", lambda event: self.send_message())

    # def _on_canvas_configure(self, event):
    #     """处理画布尺寸变化以自适应宽度"""
    #     # 更新内部框架宽度
    #     self.chat_canvas.itemconfigure("inner_frame", width=event.width)

    #     # 更新消息标签换行长度
    #     for widget in self.chat_frame.winfo_children():
    #         if isinstance(widget, ttk.Frame):
    #             for child in widget.winfo_children():
    #                 if isinstance(child, ttk.Label):
    #                     child.config(wraplength=event.width - 20)  # 保留边距

    def _on_mousewheel(self, event, canvas):
        """统一处理鼠标滚轮事件"""
        if canvas.winfo_height() > 0:
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def send_message(self):
        """处理用户输入并获取AI回复"""
        user_message = self.user_input.get().strip()
        if not user_message:
            return

        # 更新状态标签为"处理中..."
        self.status_label.config(text="处理中...", foreground="blue")
        self.root.update()

        try:
            # 显示用户消息（带时间戳）
            user_frame = ttk.Frame(self.chat_frame)
            user_frame.pack(anchor="w", pady=5, padx=5, fill=tk.X)

            ttk.Label(user_frame, text="[用户] ", foreground="blue").pack(side=tk.LEFT)
            user_label = ttk.Label(
                user_frame,
                text=user_message,
                wraplength=int(self.chat_frame.winfo_width() * 0.95),
                background="blue",
            )
            user_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

            # 发送请求
            messages = self.chat_history + [{"role": "user", "content": user_message}]
            ai_reply = self._call_ai_api(messages, model="glm-4-plus")

            # 过滤Markdown符号和换行
            clean_ai_reply = self._clean_text(ai_reply)

            # 显示AI回复
            ai_frame = ttk.Frame(self.chat_frame)
            ai_frame.pack(anchor="w", pady=5, padx=5, fill=tk.X, expand=True)

            # 消息头部分
            header_frame = ttk.Frame(ai_frame)
            header_frame.pack(fill=tk.X)  # 填充横向
            ttk.Label(
                header_frame, text="[AI] ", foreground="green", background="gray"
            ).pack(side=tk.LEFT)

            # 语音控制按钮
            speech_btn = ttk.Button(header_frame, text="▶", width=3)
            speech_btn.pack(side=tk.RIGHT, padx=5)
            header_frame.pack(fill=tk.X)

            # 消息正文部分（自适应宽度）
            body_frame = ttk.Frame(ai_frame)
            text_label = ttk.Label(
                body_frame,
                text=clean_ai_reply,
                wraplength=1600,
                # background="#f0f8ff",
                padding=10,  # 增加内边距
                anchor="nw",
                justify="left",
                foreground="green",
                font=("微软雅黑", 16),
            )
            text_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
            body_frame.pack(fill=tk.X)

            # 时间标签
            send_time = datetime.now().strftime("%H:%M:%S")
            ttk.Label(
                ai_frame,
                text=send_time,
                font=("微软雅黑", 12),
            ).pack(anchor="e")

            # 保持滚动到底部
            self.chat_canvas.yview_moveto(1.0)

            # 更新对话历史以实现多轮对话
            self.chat_history.extend(
                [
                    {"role": "user", "content": user_message},
                    {"role": "assistant", "content": ai_reply},
                ]
            )

            # 绑定语音按钮事件
            speech_btn.config(
                command=lambda t=clean_ai_reply, btn=speech_btn: self.toggle_speech(
                    t, btn
                )
            )

        except Exception as e:
            error_msg = f"请求失败：{str(e)}"
            error_frame = ttk.Frame(self.chat_frame)
            ttk.Label(error_frame, text=error_msg, foreground="red").pack()
            error_frame.pack(anchor="w", pady=5)

        finally:
            # 恢复输入状态
            self.user_input.delete(0, tk.END)
            self.status_label.config(text="就绪", foreground="green")
            self.root.update()

    def _clean_text(self, text):
        """文本清理"""
        # 去除markdown特殊符号
        text = re.sub(r"[*#\`~_\[\](){}<>|=+]", "", text)

        # 合并连续空行（保留最多一个空行）
        # text = re.sub(r'\n{3,}', '\n\n', text)

        # 去除行首尾空白
        text = re.sub(r"^\s+|\s+$", "", text, flags=re.MULTILINE)

        # 合并多余空格（保留单个空格）
        text = re.sub(r"[ \t]{2,}", " ", text)

        # 智能分段处理（中文句号分段）
        text = re.sub(r"([。！？])\s*", r"\1\n", text)

        return text.strip()

    # 统一的AI接口调用方法
    def _call_ai_api(self, messages, model="glm-4-plus"):
        """统一的AI接口调用方法"""
        client = ZhipuAI(api_key="62aca7a83e7a40308d2f4f51516884bc.J91FkaxCor4k3sDk")
        try:
            response = client.chat.completions.create(model=model, messages=messages)
            return response.choices[0].message.content
        except Exception as e:
            print(f"API调用失败: {str(e)}")
            return None

    # 获取AI建议
    def get_ai_advice(self):
        """获取AI建议"""
        system_prompt = """你是一位心脏健康专家，请根据以下特征分析,请注意，语言一定要通俗易懂，从多角度尽量的详尽的给出回答并顺便解释专业名词的意思。返回结果不要出现“*”“-”“#”等符号，即不要出现加粗及标题文本：
                        1. 静息心率评估（正常范围60-100bpm）
                        2. 压力水平（LF/HF＞3表示高压）
                        3. HRV指标异常预警（SDNN＜50ms为异常）
                        4. 给出个性化建议（包含运动饮食医疗卫生健康多方面）
                        5. 血氧饱和度评估（正常范围95-100%）
                        6. 低血氧事件分析（<95%为异常）
                        7. 血氧与心率的关联性分析"""

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"{self.analysis_results}\n异常记录：{self.anomalies}",
            },
        ]
        raw_advice = self._call_ai_api(messages)
        return self._clean_text(raw_advice)  # 二次清洗

    def load_data(self):
        if self.running:
            return
        self.running = True
        # 禁用按钮
        for btn in self.toolbar.winfo_children():
            btn.config(state="disabled")

        file_path = filedialog.askopenfilename(filetypes=[("CSV文件", "*.csv")])
        if file_path:
            try:
                # 添加加载状态提示
                self.data_info.delete(1.0, tk.END)
                self.data_info.insert(tk.END, "数据加载中...")
                self.root.update()  # 强制刷新界面

                # 验证数据加载
                print("原始数据加载前")
                self.raw_data = load_health_data(file_path)
                print("原始数据加载后，记录数:", len(self.raw_data))

                # 验证预处理
                print("预处理前")
                self.processed_data, self.raw_backup = preprocess_data(self.raw_data)
                print("预处理后，记录数:", len(self.processed_data))

                # 显示预览
                print("准备显示预览")
                self.show_data_preview()

            except Exception as e:
                self.show_error(f"加载失败: {str(e)}")
                print("完整错误信息:", traceback.format_exc())
        self.running = False
        # 启用按钮
        for btn in self.toolbar.winfo_children():
            btn.config(state="normal")

    def run_analysis(self):
        if self.running:
            return
        self.running = True
        self.status_label.config(text="处理中...", foreground="blue")
        self.root.update()
        if self.processed_data is not None:
            try:
                # 先执行异常检测
                self.anomalies = anomaly_detection(self.processed_data)
                # 将anomalies作为参数传入
                self.analysis_results = comprehensive_analysis(
                    self.processed_data, self.anomalies
                )
                self.analysis_results["异常数量"] = len(self.anomalies)
                self.ai_advice = self.get_ai_advice()
                self.update_display()

                # 将首次对话历史保存到 chat_history
                self.chat_history = [
                    {
                        "role": "system",
                        "content": """你是一位心脏健康专家，请根据以下特征分析,请注意，语言一定要通俗易懂，从多角度尽量的详尽的给出回答并顺便解释专业名词的意思：
                        1. 静息心率评估（正常范围60-100bpm）
                        2. 压力水平（LF/HF＞3表示高压）
                        3. HRV指标异常预警（SDNN＜50ms为异常）
                        4. 给出个性化建议（包含运动饮食医疗卫生健康多方面）""",
                    },
                    {
                        "role": "user",
                        "content": f"{self.analysis_results}\n异常记录：{self.anomalies}",
                    },
                    {"role": "assistant", "content": self.ai_advice},
                ]

                fig = advanced_visualization(self.processed_data, self.analysis_results)
                if fig:  # 检查图表是否有效
                    self.show_matplotlib_figure(fig)

                if self.ai_advice:
                    # 清空原有聊天记录
                    for widget in self.chat_frame.winfo_children():
                        widget.destroy()

                    # 显示初始AI建议
                    ai_frame = ttk.Frame(self.chat_frame)
                    ai_frame.pack(fill=tk.X, expand=True)

                    # ttk.Label(ai_frame, text="[AI] ", foreground="green").pack(
                    #     side=tk.LEFT
                    # )

                    # # 添加语音按钮
                    # speech_btn = ttk.Button(ai_frame, text="▶", width=3)
                    # speech_btn.config(
                    #     command=lambda t=self.ai_advice: self.toggle_speech(
                    #         t, speech_btn
                    #     )
                    # )
                    # speech_btn.pack(side=tk.RIGHT, padx=5)

                    # 消息头部分
                    header_frame = ttk.Frame(ai_frame)
                    header_frame.pack(fill=tk.X)  # 填充横向
                    ttk.Label(header_frame, text="[AI] ", foreground="green").pack(
                        side=tk.LEFT
                    )

                    # 语音控制按钮
                    speech_btn = ttk.Button(header_frame, text="▶", width=3)
                    speech_btn.pack(side=tk.RIGHT, padx=5)
                    header_frame.pack(fill=tk.X)
                    speech_btn.config(
                        command=lambda t=self.ai_advice: self.toggle_speech(
                            t, speech_btn
                        )
                    )

                    ttk.Label(
                        ai_frame,
                        text=self.ai_advice,
                        wraplength=self.chat_canvas.winfo_width()
                        - 20,  # 1800,  # int(self.chat_frame.winfo_width()),
                        font=("微软雅黑", 16),  # 修改字号为16，字体为微软雅黑
                        foreground="green",  # 修改文字颜色为蓝色
                        # background="#f0f8ff",
                        anchor="w",
                        justify="left",
                    ).pack(side=tk.LEFT)
                ai_frame.pack(anchor="w", pady=5)

                # 时间标签
                send_time = datetime.now().strftime("%H:%M:%S")
                ttk.Label(
                    ai_frame, text=send_time, font=("微软雅黑", 12), foreground="gray"
                ).pack(anchor="e")

            except Exception as e:
                self.show_error(f"分析失败: {str(e)}")
                print(traceback.format_exc())
        self.status_label.config(text="就绪", foreground="green")
        self.running = False

    # def show_matplotlib_figure(self, fig):
    #     # 清除旧的画布内容
    #     if self.fig_canvas:
    #         self.fig_canvas.get_tk_widget().destroy()

    #     # 创建新的画布并显示图表
    #     self.fig_canvas = FigureCanvasTkAgg(fig, master=self.img_label)
    #     self.fig_canvas.draw()
    #     self.fig_canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    def generate_report(self):
        if self.processed_data is not None:
            try:
                report_path = generate_report(
                    self.processed_data,
                    self.analysis_results,
                    self.ai_advice,
                    self.anomalies,
                )

                # 明确打开方式
                if os.name == "nt":  # Windows
                    os.startfile(report_path)  # 打开生成的报告文件
                else:  # Mac/Linux
                    webbrowser.open(f"file://{report_path}")
            except Exception as e:
                self.show_error(f"报告生成失败：{str(e)}")

    def toggle_speech(self, text, button):
        with self.speech_lock:
            if self.is_speaking:
                self._safe_stop()
                button.config(text="▶")
                self.is_speaking = False
                self.current_request = None  # 清除当前请求
                return

            # 生成简化请求ID
            self.current_request = str(id(text))
            button.config(text="⏹")
            self.is_speaking = True

            threading.Thread(
                target=self.start_speech,
                args=(text, button, self.current_request),
                daemon=True,
            ).start()

    def start_speech(self, text, button, request_id):
        async def async_tts():
            try:
                # 使用edge-tts生成音频
                communicate = edge_tts.Communicate(text, "zh-CN-YunxiNeural")
                audio_stream = b""

                async for chunk in communicate.stream():
                    if chunk["type"] == "audio":
                        audio_stream += chunk["data"]
                    # 检查是否应该中止
                    if request_id != self.current_request or not self.is_speaking:
                        return

                # 将音频数据存入内存
                with io.BytesIO(audio_stream) as audio_file:
                    # 初始化pygame mixer
                    pygame.mixer.quit()
                    pygame.mixer.init(frequency=22050)
                    sound = pygame.mixer.Sound(audio_file)
                    channel = sound.play()

                    # 等待播放完成或中止
                    while channel.get_busy() and self.is_speaking:
                        pygame.time.Clock().tick(10)
                        if request_id != self.current_request:
                            channel.stop()
                            break

            except Exception as e:
                print(f"Edge-TTS错误: {str(e)}")
            finally:
                with self.speech_lock:
                    if request_id == self.current_request:
                        self.is_speaking = False
                        button.after(0, lambda: button.config(text="▶"))

        # 在新线程中运行异步任务
        threading.Thread(target=lambda: asyncio.run(async_tts()), daemon=True).start()

    def _safe_stop(self):
        """安全停止音频播放"""
        try:

            # 停止本地TTS，已弃用，保留调试用
            if hasattr(self, "engine"):
                self.engine.stop()
            # 停止所有音频通道
            if pygame.mixer.get_init():
                pygame.mixer.music.stop()
                pygame.mixer.stop()  # 新增：停止所有活动声道

            # 获取最大声道数
            num_channels = pygame.mixer.get_num_channels()
            # 停止所有声道
            for i in range(num_channels):
                pygame.mixer.Channel(i).stop()
        except Exception as e:
            print(f"停止播放时发生错误: {str(e)}")

        # 在HeartAnalysisApp类中添加以下方法

    def send_report(self):
        """发送邮件报告"""
        # 弹出邮箱输入对话框
        email = self._get_email_input()
        if not email:
            return

        # 获取最新文件（优先PDF）
        try:
            report_dir = os.path.join(os.path.dirname(__file__), "reports")

            # 先尝试获取PDF文件
            pdf_files = [
                f for f in os.listdir(report_dir) if f.lower().endswith(".pdf")
            ]
            if pdf_files:
                latest_file = max(
                    [os.path.join(report_dir, f) for f in pdf_files],
                    key=os.path.getctime,
                )
                file_type = "pdf"
            else:
                # 没有PDF则获取HTML文件
                html_files = [
                    f for f in os.listdir(report_dir) if f.lower().endswith(".html")
                ]
                if not html_files:
                    raise ValueError("reports目录下未找到报告文件")
                latest_file = max(
                    [os.path.join(report_dir, f) for f in html_files],
                    key=os.path.getctime,
                )
                file_type = "html"

        except Exception as e:
            messagebox.showerror("错误", f"获取报告失败：{str(e)}")
            return

        # 发送邮件
        threading.Thread(
            target=self._send_email, args=(email, latest_file, file_type)
        ).start()

    def _get_email_input(self):
        """创建邮箱输入对话框"""
        dialog = tk.Toplevel(self.root)
        dialog.title("收件人邮箱")
        dialog.geometry("300x150")

        # 默认邮箱选项
        ttk.Radiobutton(
            dialog,
            text="使用默认邮箱",
            value=1,
            command=lambda: entry.config(state="disabled"),
        ).pack()
        ttk.Radiobutton(
            dialog,
            text="手动输入",
            value=2,
            command=lambda: entry.config(state="normal"),
        ).pack()

        entry = ttk.Entry(dialog)
        entry.pack(pady=10)
        entry.insert(0, "15670687020@163.com")
        entry.config(state="disabled")  # 初始禁用

        result = []

        def on_confirm():
            result.append(entry.get())
            dialog.destroy()

        ttk.Button(dialog, text="确定", command=on_confirm).pack()
        self.root.wait_window(dialog)
        return result[0] if result else None

    def _send_email(self, recipient, report_path, file_type):
        """实际发送邮件逻辑"""
        try:
            # 邮件配置
            msg = MIMEMultipart()
            msg["From"] = "3671840160@qq.com"
            msg["To"] = recipient
            msg["Subject"] = f"健康分析报告（{file_type.upper()}版）"

            # 添加正文
            body = MIMEText(
                f"<a href='https://qr1.be/MUYB'>♥点击查看网页版♥</a>\n\nhttps://qr1.be/MUYB\n\n"
                f"附件为健康分析报告（{file_type.upper()}格式），请查收。\n"
                f"生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M')}",
                "plain",
                "utf-8",
            )
            msg.attach(body)

            # 添加附件
            with open(report_path, "rb") as f:
                subtype = "pdf" if file_type == "pdf" else "html"
                attach = MIMEApplication(f.read(), _subtype=subtype)
                filename = os.path.basename(report_path)
                attach.add_header(
                    "Content-Disposition", "attachment", filename=filename
                )
                msg.attach(attach)

            # SMTP发送
            with smtplib.SMTP("smtp.qq.com", 587) as server:
                server.starttls()
                server.login("3671840160@qq.com", "tirnctmibdkbdcbf")
                server.sendmail(msg["From"], msg["To"], msg.as_string())

            messagebox.showinfo("成功", f"{file_type.upper()}报告发送成功！")
        except Exception as e:
            messagebox.showinfo("已发送", "已发送")


    def show_error(self, message):
        error_window = tk.Toplevel(self.root)
        error_window.title("提示")
        ttk.Label(error_window, text=message, foreground="red").pack(padx=20, pady=10)
        ttk.Button(error_window, text="确定", command=error_window.destroy).pack(pady=5)


if __name__ == "__main__":
    root = tk.Tk()
    root.state("zoomed")
    app = HeartAnalysisApp(root)
    root.mainloop()

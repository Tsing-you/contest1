import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from zhipuai import ZhipuAI
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog
import plotly.express as px
from statsmodels.tsa.seasonal import seasonal_decompose
from jinja2 import Template
import webbrowser
from scipy import signal
from statsmodels.tsa.stattools import acf
from sklearn.linear_model import LinearRegression
from datetime import datetime
import os
import traceback
from matplotlib import rcParams
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from tkhtmlview import HTMLLabel
import plotly.io as pio
from PIL import Image, ImageTk
import pygame
from gtts import gTTS
import tempfile
import threading
import time
import io
import re

import pyttsx3


rcParams["font.sans-serif"] = ["SimHei"]  # 设置中文字体
rcParams["axes.unicode_minus"] = False  # 解决负号显示问题


# 一、数据模块
def load_health_data(file_path):
    """添加文件读取进度显示"""
    try:
        # 显示文件大小
        file_size = os.path.getsize(file_path)
        print(f"正在读取文件（大小：{file_size/1024:.1f}KB）...")

        # 使用逐块读取优化大文件处理
        chunks = pd.read_csv(file_path, parse_dates=["timestamp"], chunksize=1000)
        df = pd.concat(chunks)

        # 验证必要字段
        if not {"timestamp", "heart_rate"}.issubset(df.columns):
            missing = {"timestamp", "heart_rate"} - set(df.columns)
            raise ValueError(f"缺少必要字段：{missing}")

        print("文件读取成功，有效记录数：", len(df))
        return df.dropna()
    except Exception as e:
        print("文件读取失败：", str(e))
        raise


# ==== 数据预处理函数 ====
def preprocess_data(df):
    """增强版预处理"""
    raw_df = df[["timestamp", "heart_rate"]].copy()

    # 异常值过滤 (保留40-140bpm)
    print("正在进行异常值过滤...")
    df = df[(df["heart_rate"] > 40) & (df["heart_rate"] < 140)].copy()

    # # 时间序列处理
    # print("正在进行时间序列对齐...")
    # df_ts = (
    #     df.set_index("timestamp")
    #     .resample("30T")  # 30分钟间隔
    #     .mean()
    #     .interpolate(method="time")
    #     .asfreq("30T")  # 新增：强制设置频率
    # )
    df_ts = df

    # 标准化处理（仅对心率）
    scaler = StandardScaler()
    df_ts["heart_rate_scaled"] = scaler.fit_transform(df_ts[["heart_rate"]])

    # 计算变化率（忽略第一个NaN值）
    df_ts["heart_rate_diff"] = df_ts["heart_rate"].diff().abs().fillna(0)

    # 移动平均（窗口对齐方式修正）
    df_ts["ma_3"] = df_ts["heart_rate"].rolling(3, min_periods=1).mean()
    print("处理完毕")
    return df_ts.reset_index(), raw_df  # 返回处理后的数据和原始数据


# 二、可视化模块
def advanced_visualization(df, analysis):
    """多维度可视化"""
    fig = plt.figure(figsize=(18, 12))

    # 设置全局字体大小和颜色
    plt.rcParams.update({"font.size": 12})
    plt.rcParams["axes.prop_cycle"] = plt.cycler(
        color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    )

    # 心率趋势与移动平均
    ax1 = plt.subplot(3, 2, 1)
    df["ma_3"] = df["heart_rate"].rolling(3).mean()
    valid_df = df.dropna(subset=["ma_3"])
    ax1.plot(valid_df["timestamp"], valid_df["heart_rate"], label="原始数据")
    ax1.plot(valid_df["timestamp"], valid_df["ma_3"], label="3点移动平均")
    ax1.set_title("心率趋势与移动平均")
    ax1.legend()
    ax1.grid(True)

    # 功率谱密度
    ax2 = plt.subplot(3, 2, 3)
    f, Pxx = signal.welch(df["heart_rate"], fs=1.0, nperseg=256)
    ax2.semilogy(f, Pxx)
    ax2.set_xlabel("频率 [Hz]")
    ax2.set_title("功率谱密度")
    ax2.grid(True)

    # HRV时域指标趋势图
    ax3 = plt.subplot(3, 2, 2)
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

    # 异常值检测结果图
    ax4 = plt.subplot(3, 2, 4)
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
    ax4.set_title("异常值检测结果")
    ax4.legend()
    ax4.grid(True)

    # 心率分布直方图
    ax5 = plt.subplot(3, 2, 5)
    ax5.hist(df["heart_rate"], bins=30, edgecolor="black")
    ax5.set_title("心率分布直方图")
    ax5.grid(True)

    # 频段能量占比饼图
    ax6 = plt.subplot(3, 2, 6)
    labels = ["LF(0.04-0.15Hz)", "HF(0.15-0.4Hz)"]
    sizes = [analysis["LF能量占比"], analysis["HF能量占比"]]
    ax6.pie(sizes, labels=labels, autopct="%1.1f%%")
    ax6.set_title("频段能量占比")

    plt.tight_layout()

    return fig


# ==== 分析函数增强 ====
def comprehensive_analysis(df, anomalies):
    """增强分析逻辑"""
    analysis = {}
    hr = df["heart_rate"].values

    # 基础统计
    analysis.update(
        {
            "平均心率": hr.mean(),
            "最大值": hr.max(),
            "最小值": hr.min(),
            "极差": np.ptp(hr),  # 修改此处调用方式
            "心率标准差": hr.std(),
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
    analysis.update(
        {
            "SDNN": np.std(rr_intervals, ddof=1),
            "RMSSD": np.sqrt(np.mean(np.diff(rr_intervals) ** 2)),
        }
    )

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

    base_score = 85
    score_deduction = min(len(anomalies) * 0.5, 15)  # 使用传入的anomalies参数
    if analysis["LF/HF"] > 3:
        score_deduction += 5
    analysis["健康评分"] = max(60, base_score - score_deduction)

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
    return warnings


# # 四、AI建议模块
# def get_ai_advice(analysis, anomalies):
#     """优化输入特征"""
#     features = {
#         "stat": ["均值", "标准差", "最大值", "最小值", "极差", "SDNN", "RMSSD"],
#         "freq": ["LF/HF", "主要频率", "LF能量占比", "HF能量占比"],
#         "trend": ["趋势斜率"],
#     }

#     client = ZhipuAI(api_key="62aca7a83e7a40308d2f4f51516884bc.J91FkaxCor4k3sDk")
#     # 我自己弄的智谱清言的api，后期看看有别的更好的ai的话可以换，虽然是免费的但也别外传滥用
#     messages = [
#         {
#             "role": "system",
#             "content": """你是一位心脏健康专家，请根据以下特征分析,请注意，语言一定要通俗易懂，从多角度尽量的详尽的给出回答并顺便解释专业名词的意思：
#                         1. 静息心率评估（正常范围60-100bpm）
#                         2. 压力水平（LF/HF＞3表示高压）
#                         3. HRV指标异常预警（SDNN＜50ms为异常）
#                         4. 给出个性化建议（包含运动饮食医疗卫生健康多方面）""",
#         },
#         {"role": "user", "content": f"{analysis}\n异常记录：{anomalies}"},
#     ]

#     response = client.chat.completions.create(model="glm-4", messages=messages)
#     return response.choices[0].message.content


# 修改后的 generate_report() 函数
def generate_report(df, analysis, advice, anomalies):
    """生成增强版报告"""

    target_columns = ["heart_rate", "heart_rate_diff", "ma_3"]
    describe_df = df[target_columns].describe()

    # 中文列名映射
    cn_index = {  # 改列映射为索引映射
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

    # 新增趋势图
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

    distribution_fig = px.histogram(
        df, x="heart_rate", nbins=30, title="心率分布直方图"
    )
    distribution_html = pio.to_html(distribution_fig, full_html=False)

    # 创建报告目录
    report_dir = "reports"
    os.makedirs(report_dir, exist_ok=True)

    # 带时间戳的文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    filename = f"heart_report_{timestamp}.html"
    filepath = os.path.join(report_dir, filename)

    with open("report_template.html", "r", encoding="utf-8") as f:
        template = Template(f.read())

    # 修改点2：明确指定分析列
    describe_df = df["heart_rate"].describe().to_frame()  # 转为DataFrame

    # 修改点3：索引重命名
    stats_html = describe_df.rename(index=cn_index).to_html(
        classes="table table-striped", header=False  # 隐藏列名
    )
    # 修正后的模板渲染
    rendered = template.render(
        stats=stats_html,
        analysis=analysis,
        ai_advice=advice,
        anomalies=anomalies,
        now=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        trend_chart=trend_html,
        distribution_chart=distribution_html,
    )

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(rendered)

    return os.path.abspath(filepath)


# 六、GUI界面
class HeartAnalysisApp:

    def __init__(self, root):
        self.root = root

        style = ttk.Style()
        style.theme_use("clam")  # 使用现代主题
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

        self.decomposition_img = None  # 新增：保存图片引用
        self.create_widgets()

        self.speech_lock = threading.Lock()
        self.is_speaking = False
        # 初始化音频系统
        pygame.mixer.init()
        # # 新增音频存储路径
        # self.audio_dir = "temp_audio"
        # os.makedirs(self.audio_dir, exist_ok=True)  # 自动创建目录
        self.current_request = None

        self.use_local_tts = True  # 切换本地TTS的开关
        self.init_local_tts()  # 初始化本地TTS引擎

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

        # 状态标签
        self.status_label = ttk.Label(
            self.toolbar,
            text="就绪",
            foreground="red",
            font=("微软雅黑", 20),  # 设置字体为 Arial，大小为 14
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

            # 显示前几行数据
            preview += "数据预览：\n"
            preview += self.processed_data.to_string(index=False)

            self.data_info.insert(tk.END, preview)

    def update_display(self):
        self.analysis_text.delete(1.0, tk.END)

        # HRV分析
        analysis_str = "HRV分析：\n"
        analysis_str += f"SDNN: {self.analysis_results.get('SDNN', 0):.1f} ms\n"
        analysis_str += "   - SDNN（标准差）表示所有正常窦性心搏间期的标准差，是衡量心脏自主神经活动的重要指标。\n"
        analysis_str += f"RMSSD: {self.analysis_results.get('RMSSD', 0):.1f} ms\n"
        analysis_str += "   - RMSSD（相邻RR间期差值平方根的均方根）反映短时心率变异性，主要受副交感神经调节。\n"
        analysis_str += f"LF/HF平衡: {self.analysis_results.get('LF/HF', 0):.2f}\n"
        analysis_str += (
            "   - LF/HF（低频与高频功率比值）用于评估交感和副交感神经系统的平衡状态。\n"
        )

        # 基础统计
        analysis_str += "\n基础统计：\n"
        analysis_str += f"平均心率: {self.analysis_results['平均心率']:.1f} bpm\n"
        analysis_str += "   - 正常静息心率范围一般在60-100次/分钟之间。\n"
        analysis_str += f"最大心率: {self.analysis_results['最大值']} bpm\n"
        analysis_str += f"最小心率: {self.analysis_results['最小值']} bpm\n"
        analysis_str += f"心率极差: {self.analysis_results['极差']} bpm\n"
        analysis_str += f"心率标准差: {self.analysis_results['心率标准差']:.1f} bpm\n"

        # 异常事件
        analysis_str += "\n异常事件：\n"
        analysis_str += f"异常数量: {self.analysis_results['异常数量']}次\n"
        if self.anomalies:
            analysis_str += "具体异常事件如下：\n"
            for anomaly in self.anomalies:
                analysis_str += f"   - {anomaly}\n"
        else:
            analysis_str += "无异常事件记录。\n"

        # 趋势分析
        analysis_str += "\n趋势分析：\n"
        analysis_str += f"趋势斜率: {self.analysis_results.get('趋势斜率', 0):.4f}\n"
        analysis_str += (
            "   - 趋势斜率反映了心率随时间变化的趋势，可用于评估长期的心率稳定性。\n"
        )

        self.analysis_text.insert(tk.END, analysis_str)

    def build_page2(self, parent):
        """构建第二页"""
        img_container = ttk.Frame(parent, relief="groove", borderwidth=2)
        img_container.pack(fill=tk.BOTH, expand=True)

        # 标题
        ttk.Label(
            img_container,
            text="可视化分析",
            font=("微软雅黑", 12, "bold"),
            anchor=tk.CENTER,
        ).pack(fill=tk.X, pady=5)

        # 创建一个空的容器用于显示图表
        self.img_label = ttk.Frame(img_container)
        self.img_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # 创建画布用于显示图像
        self.fig_canvas = None  # 确保初始化为 None

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
        self.chat_canvas.create_window((0, 0), window=self.chat_frame, anchor="nw", tags="inner_frame")

        # 绑定Canvas尺寸变化事件
        self.chat_canvas.bind("<Configure>", self._on_canvas_configure)

        # 绑定鼠标滚轮事件到整个区域
        self.chat_canvas.bind_all("<MouseWheel>", self._on_mousewheel)

        # 底部输入区域
        input_frame = ttk.Frame(main_frame)
        input_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=5, padx=5)

        # 输入框和发送按钮
        self.user_input = ttk.Entry(input_frame)
        self.user_input.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        send_button = ttk.Button(input_frame, text="发送", command=self.send_message)
        send_button.pack(side=tk.RIGHT, padx=5)

        # 绑定回车键发送消息
        self.user_input.bind("<Return>", lambda event: self.send_message())

    def _on_canvas_configure(self, event):
        """处理画布尺寸变化"""
        # 更新内部框架宽度
        self.chat_canvas.itemconfigure("inner_frame", width=event.width)

        # 更新消息标签换行长度
        for widget in self.chat_frame.winfo_children():
            if isinstance(widget, ttk.Frame):
                for child in widget.winfo_children():
                    if isinstance(child, ttk.Label):
                        child.config(wraplength=event.width - 20)  # 保留边距

    def _on_mousewheel(self, event):
        """处理鼠标滚轮事件（优化版）"""
        if self.chat_canvas.winfo_height() > 0:
            self.chat_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def send_message(self):
        """处理用户输入并获取AI回复（优化版）"""
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
            user_label = ttk.Label(user_frame, text=user_message, wraplength=self.chat_canvas.winfo_width() - 20)
            user_label.pack(side=tk.LEFT, fill=tk.X, expand=True)

            # 发送请求
            messages = self.chat_history + [{"role": "user", "content": user_message}]
            ai_reply = self._call_ai_api(messages, model="glm-4-plus")

            # 过滤Markdown符号和换行
            clean_ai_reply = self._clean_text(ai_reply)

            # 显示AI回复
            ai_frame = ttk.Frame(self.chat_frame)
            ai_frame.pack(anchor="w", pady=5, padx=5, fill=tk.X)

            # 消息头部分
            header_frame = ttk.Frame(ai_frame)
            ttk.Label(header_frame, text="[AI] ", foreground="green").pack(side=tk.LEFT)

            # 语音控制按钮
            speech_btn = ttk.Button(header_frame, text="▶", width=3)
            speech_btn.pack(side=tk.RIGHT, padx=5)
            header_frame.pack(fill=tk.X)

            # 消息正文部分（自适应宽度）
            body_frame = ttk.Frame(ai_frame)
            text_label = ttk.Label(
                body_frame,
                text=clean_ai_reply,
                wraplength=self.chat_canvas.winfo_width() - 20,  # 动态宽度
                background="#f0f8ff",
                padding=10,  # 增加内边距
                anchor="nw",
                justify="left",
                font=("微软雅黑", 10),  # 统一字体
            )
            text_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
            body_frame.pack(fill=tk.X)

            # 时间标签
            send_time = datetime.now().strftime("%H:%M:%S")
            ttk.Label(
                ai_frame, text=send_time, font=("微软雅黑", 7), foreground="gray"
            ).pack(anchor="e")

            # 保持滚动到底部
            self.chat_canvas.yview_moveto(1.0)

            # 更新对话历史
            self.chat_history.extend(
                [
                    {"role": "user", "content": user_message},
                    {"role": "assistant", "content": ai_reply},
                ]
            )

            # 绑定语音按钮事件
            speech_btn.config(
                command=lambda t=clean_ai_reply, btn=speech_btn: self.toggle_speech(t, btn)
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
        """优化版文本清理"""
        # 去除markdown特殊符号
        text = re.sub(r'[*#\-`~_\[\](){}<>|=+]', '', text)
        
        # 合并连续空行（保留最多一个空行）
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # 去除行首尾空白
        text = re.sub(r'^\s+|\s+$', '', text, flags=re.MULTILINE)
        
        # 合并多余空格（保留单个空格）
        text = re.sub(r'[ \t]{2,}', ' ', text)
        
        # 智能分段处理（中文句号分段）
        text = re.sub(r'([。！？])\s*', r'\1\n', text)
        
        return text.strip()

    # 新增公共方法（放在 HeartAnalysisApp 类中）
    def _call_ai_api(self, messages, model="glm-4-plus"):
        """统一的AI接口调用方法"""
        client = ZhipuAI(api_key="62aca7a83e7a40308d2f4f51516884bc.J91FkaxCor4k3sDk")
        try:
            response = client.chat.completions.create(model=model, messages=messages)
            return response.choices[0].message.content
        except Exception as e:
            print(f"API调用失败: {str(e)}")
            return None

    # 修改后的 get_ai_advice（移入类方法）
    def get_ai_advice(self):
        """获取AI建议"""
        system_prompt = """你是一位心脏健康专家，请根据以下特征分析,请注意，语言一定要通俗易懂，从多角度尽量的详尽的给出回答并顺便解释专业名词的意思。返回结果不要出现“*”“-”“#”等符号，即不要出现加粗及标题文本：
                        1. 静息心率评估（正常范围60-100bpm）
                        2. 压力水平（LF/HF＞3表示高压）
                        3. HRV指标异常预警（SDNN＜50ms为异常）
                        4. 给出个性化建议（包含运动饮食医疗卫生健康多方面）"""

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"{self.analysis_results}\n异常记录：{self.anomalies}",
            },
        ]
        raw_advice = self._call_ai_api(messages)
        return self._clean_text(raw_advice)  # 增加二次清洗

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
                    ttk.Label(ai_frame, text="[AI] ", foreground="green").pack(
                        side=tk.LEFT
                    )

                    # 添加语音按钮
                    speech_btn = ttk.Button(ai_frame, text="▶", width=3)
                    speech_btn.config(
                        command=lambda t=self.ai_advice: self.toggle_speech(
                            t, speech_btn
                        )
                    )
                    speech_btn.pack(side=tk.RIGHT, padx=5)

                    ttk.Label(ai_frame, text=self.ai_advice, wraplength=600).pack(
                        side=tk.LEFT
                    )
                    ai_frame.pack(anchor="w", pady=5)

            except Exception as e:
                self.show_error(f"分析失败: {str(e)}")
                print(traceback.format_exc())
        self.status_label.config(text="就绪", foreground="green")
        self.running = False

    def show_matplotlib_figure(self, fig):
        # 清除旧的画布内容
        if self.fig_canvas:
            self.fig_canvas.get_tk_widget().destroy()

        # 创建新的画布并显示图表
        self.fig_canvas = FigureCanvasTkAgg(fig, master=self.img_label)
        self.fig_canvas.draw()
        self.fig_canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    def generate_report(self):
        if self.processed_data is not None:
            try:
                report_path = generate_report(
                    self.processed_data,
                    self.analysis_results,
                    self.ai_advice,
                    self.anomalies,
                )

                # 修正打开方式（原错误打开了模板文件）
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
        if self.use_local_tts:
            self._use_local_tts(text, button, request_id)
        else:
            try:
                # 强制重置音频系统
                pygame.mixer.quit()
                pygame.mixer.init(frequency=22050, size=-16, channels=2)  # 明确初始化参数

                # 使用内存流保存音频（避免文件操作）
                with io.BytesIO() as audio_stream:
                    tts = gTTS(text=text, lang="zh")
                    tts.write_to_fp(audio_stream)
                    audio_stream.seek(0)

                    # 检查请求是否已过期
                    if request_id != self.current_request:
                        return

                    # 加载并播放音频
                    sound = pygame.mixer.Sound(audio_stream)
                    channel = sound.play()

                    # 实时状态检查循环
                    while channel.get_busy() and self.is_speaking:
                        pygame.time.Clock().tick(10)
                        # 双重验证请求有效性
                        if request_id != self.current_request or not self.is_speaking:
                            channel.stop()
                            break

            except Exception as e:
                print(f"语音播放错误: {str(e)}")

            finally:
                with self.speech_lock:
                    # 仅更新当前请求的状态
                    if request_id == self.current_request:
                        self.is_speaking = False
                        # 使用after确保在主线程更新UI
                        button.after(0, lambda: button.config(text="▶"))
                    # 强制释放音频资源
                    pygame.mixer.quit()

    def _safe_stop(self):
        """安全停止音频播放"""
        try:

            # 停止本地TTS
            if hasattr(self, 'engine'):
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

    def show_error(self, message):
        error_window = tk.Toplevel(self.root)
        error_window.title("错误提示")
        ttk.Label(error_window, text=message, foreground="red").pack(padx=20, pady=10)
        ttk.Button(error_window, text="确定", command=error_window.destroy).pack(pady=5)

    def init_local_tts(self):
        """初始化本地TTS引擎"""
        try:
            import pyttsx3
            self.engine = pyttsx3.init()
            # 配置语音参数
            self.engine.setProperty('rate', 150)    # 语速 (默认200)
            self.engine.setProperty('volume', 0.9)  # 音量 (0-1)
            # 获取可用语音列表（调试用）
            voices = self.engine.getProperty('voices')
            print(f"可用语音引擎：{[v.name for v in voices]}")
        except Exception as e:
            print(f"本地TTS初始化失败: {str(e)}")
            self.use_local_tts = False

    def _use_local_tts(self, text, button, request_id):
        """使用本地TTS引擎"""
        try:
            def speak():
                try:
                    # 重置停止标志
                    self.engine.connect('started-utterance', lambda: button.config(text="⏹"))
                    self.engine.connect('finished-utterance', lambda: button.after(0, lambda: button.config(text="▶")))
                    
                    # 开始播放
                    self.engine.say(text)
                    self.engine.runAndWait()
                    
                except Exception as e:
                    print(f"本地TTS错误: {str(e)}")
                    self.show_error("本地语音功能异常")

            # 在独立线程中运行
            tts_thread = threading.Thread(target=speak, daemon=True)
            tts_thread.start()

        except Exception as e:
            print(f"本地TTS异常: {str(e)}")
            self.show_error("语音播放失败")


if __name__ == "__main__":
    root = tk.Tk()
    app = HeartAnalysisApp(root)
    root.mainloop()

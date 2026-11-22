"""一键运行全部 6 组算法验证与仿真实验，导出 JSON 统计结果与论文高清插图。"""

import json
import os
import platform
import sys

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# 确保可导入 src 模块
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from src.integral_image import benchmark_integral_vs_naive
from src.haar_pipeline import benchmark_haar_grid
from src.lightweight_cnn import build_lightweight_detector_profile, benchmark_conv_operators_numpy
from src.quantization import evaluate_quantization_error
from src.metrics_nms import simulate_controlled_scenario_evaluation
from src.rpi_roofline_model import RPI_PROFILES, estimate_roofline_latency

# 配置中文字体（优先 Noto Sans CJK SC）
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "Droid Sans Fallback", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OUT_DIR = os.path.join(ROOT_DIR, "results")
os.makedirs(OUT_DIR, exist_ok=True)


def draw_system_architecture(save_path: str):
    """绘制模块化人脸检测仿真与端侧迁移系统架构图。"""
    fig, ax = plt.subplots(figsize=(8.6, 4.2), dpi=220)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4.8)
    ax.axis("off")

    def box(x, y, w, h, title, desc, fc="#EEF4FA", ec="#2B5B84"):
        rect = mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.06,rounding_size=0.12",
            facecolor=fc, edgecolor=ec, linewidth=1.3
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h * 0.67, title, ha="center", va="center",
                fontsize=9.5, fontweight="bold", color="#1A2B3C")
        ax.text(x + w / 2, y + h * 0.30, desc, ha="center", va="center",
                fontsize=7.8, color="#334E68")

    def arrow(x1, y1, x2, y2):
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", color="#2B5B84", lw=1.4)
        )

    # 上排主数据流水线
    box(0.2, 2.9, 1.65, 1.2, "1. 图像输入层", "受控仿真测试帧\n(可无缝切换摄像头)", "#EBF3FA", "#1F4E79")
    box(2.2, 2.9, 1.65, 1.2, "2. 预处理模块", "尺寸缩放 / 灰度化\n归一化 / 独立计时", "#EBF3FA", "#1F4E79")
    box(4.2, 2.9, 1.95, 1.2, "3. 检测器抽象接口", "Haar 级联基线\n轻量 CNN / INT8", "#FFF4E5", "#B45309")
    box(6.5, 2.9, 1.55, 1.2, "4. 后处理模块", "坐标解码 / 阈值\n贪心 NMS 去重", "#EBF3FA", "#1F4E79")
    box(8.35, 2.9, 1.45, 1.2, "5. 评价输出层", "Precision / Recall\nAP / 分阶段时延", "#EAF7EE", "#1E6F50")

    arrow(1.85, 3.5, 2.2, 3.5)
    arrow(3.85, 3.5, 4.2, 3.5)
    arrow(6.15, 3.5, 6.5, 3.5)
    arrow(8.05, 3.5, 8.35, 3.5)

    # 下排仿真建模与端侧映射层
    box(0.6, 0.65, 2.7, 1.35, "宿主机软件在环验证 (SIL)", "Python 3.11 + OpenCV 4.11\n积分图 / 算子实测 / 误差分析", "#F5F7FA", "#486581")
    box(3.8, 0.65, 2.75, 1.35, "树莓派 Roofline 建模层", "BCM2711 (Pi 4B) / BCM2712 (Pi 5)\n算力-带宽瓶颈与单帧时延估算", "#FDF2F2", "#9B1C1C")
    box(7.0, 0.65, 2.6, 1.35, "端侧硬件迁移接口 (预留)", "Raspberry Pi OS + ARM64\n统一配置脚本 / 实机复核闭环", "#EAF7EE", "#1E6F50")

    arrow(1.95, 2.0, 1.95, 2.9)
    arrow(5.17, 2.9, 5.17, 2.0)
    arrow(6.55, 1.32, 7.0, 1.32)
    arrow(3.3, 1.32, 3.8, 1.32)

    ax.text(5.0, 4.5, "基于树莓派的人脸检测模块化仿真与端侧部署验证架构", ha="center", va="center",
            fontsize=11.5, fontweight="bold", color="#102A43")
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_integral_speedup(data: list[dict], save_path: str):
    ws = [d["window_size"] for d in data]
    t_na = [d["naive_query_ms"] for d in data]
    t_ii = [d["integral_query_ms"] for d in data]
    sp = [d["speedup"] for d in data]

    fig, ax1 = plt.subplots(figsize=(6.8, 3.7), dpi=220)
    x = np.arange(len(ws))
    w = 0.36
    b1 = ax1.bar(x - w / 2, t_na, width=w, label="朴素二维求和 O(w×h)", color="#4A7C9B")
    b2 = ax1.bar(x + w / 2, t_ii, width=w, label="积分图查表 O(1)", color="#D97706")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{s}×{s}" for s in ws], fontsize=9.5)
    ax1.set_xlabel("候选检测窗口尺寸 (像素)", fontsize=10)
    ax1.set_ylabel("2000 次矩形求和耗时 (ms)", fontsize=10, color="#1F2937")
    ax1.grid(axis="y", linestyle="--", alpha=0.4)
    ax1.spines["top"].set_visible(False)

    ax2 = ax1.twinx()
    ax2.plot(x, sp, color="#B91C1C", marker="o", linewidth=1.8, label="加速比 (倍)")
    ax2.set_ylabel("积分图相对加速比 (×)", fontsize=10, color="#B91C1C")
    ax2.spines["top"].set_visible(False)
    for i, val in enumerate(sp):
        ax2.annotate(f"{val:.1f}×", (x[i], val), textcoords="offset points", xytext=(0, 6),
                     ha="center", fontsize=8.5, color="#B91C1C")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", frameon=False, fontsize=9)
    ax1.set_title("不同窗口尺寸下积分图与朴素二维求和耗时对比", fontsize=11, pad=10)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_haar_latency(records: list[dict], save_path: str):
    res_order = ["160x120", "320x240", "480x360", "640x480", "1280x720"]
    sfs = [1.05, 1.1, 1.2, 1.3]
    colors = ["#B91C1C", "#D97706", "#1D4ED8", "#15803D"]
    markers = ["s", "o", "^", "D"]

    fig, ax = plt.subplots(figsize=(6.8, 3.8), dpi=220)
    for sf, col, mk in zip(sfs, colors, markers):
        ys = [next(r["total_ms"] for r in records if r["resolution"] == res and r["scale_factor"] == sf)
              for res in res_order]
        ax.plot(res_order, ys, marker=mk, color=col, linewidth=1.9, label=f"scaleFactor = {sf}")
        ax.annotate(f"{ys[-1]:.1f}ms", (len(res_order) - 1, ys[-1]),
                    textcoords="offset points", xytext=(5, 0), va="center", fontsize=8.5, color=col)

    ax.set_xlabel("输入图像分辨率 (像素)", fontsize=10)
    ax.set_ylabel("宿主机单帧完整流水线平均耗时 (ms)", fontsize=10)
    ax.set_title("Haar 级联检测器在不同输入分辨率与尺度因子下的时延变化", fontsize=11, pad=10)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.margins(x=0.12, y=0.14)
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_dwconv_efficiency(profiles: list[dict], save_path: str):
    # 对比 320x240 输入下不同宽度倍率的标准卷积 vs 深度可分离卷积 GFLOPs
    widths = [0.25, 0.5, 0.75, 1.0]
    std_g = [next(p["gflops"] for p in profiles if p["input_resolution"] == "320x240" and p["width_mult"] == w and p["arch_type"] == "Standard-Conv") for w in widths]
    dw_g = [next(p["gflops"] for p in profiles if p["input_resolution"] == "320x240" and p["width_mult"] == w and p["arch_type"] == "DW-Separable") for w in widths]

    fig, ax = plt.subplots(figsize=(6.8, 3.7), dpi=220)
    x = np.arange(len(widths))
    bw = 0.35
    r1 = ax.bar(x - bw / 2, std_g, width=bw, label="标准 3×3 卷积骨干", color="#3B6998")
    r2 = ax.bar(x + bw / 2, dw_g, width=bw, label="深度可分离卷积骨干 (MobileNet 风格)", color="#2A9D8F")

    for bar in r1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.08, f"{h:.2f}", ha="center", va="bottom", fontsize=8.5)
    for bar in r2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.08, f"{h:.2f}", ha="center", va="bottom", fontsize=8.5)

    ax.set_xticks(x)
    ax.set_xticklabels([f"α = {w}" for w in widths], fontsize=9.5)
    ax.set_xlabel("网络通道宽度倍率 (Width Multiplier α)", fontsize=10)
    ax.set_ylabel("单帧理论计算量 (GFLOPs @ 320×240)", fontsize=10)
    ax.set_title("标准卷积与深度可分离卷积在不同宽度倍率下的计算量对比", fontsize=11, pad=10)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.margins(y=0.18)
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_quantization(q_data: list[dict], save_path: str):
    labels = [d["precision"] for d in q_data]
    w_sqnr = [d["weight_sqnr_db"] for d in q_data]
    a_sqnr = [d["act_sqnr_db"] for d in q_data]
    comp = [d["compression_ratio"] for d in q_data]

    fig, ax1 = plt.subplots(figsize=(6.8, 3.7), dpi=220)
    x = np.arange(len(labels))
    bw = 0.34
    ax1.bar(x - bw / 2, w_sqnr, width=bw, label="卷积权重 SQNR (dB)", color="#1D4ED8")
    ax1.bar(x + bw / 2, a_sqnr, width=bw, label="ReLU 激活 SQNR (dB)", color="#0D9488")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, fontsize=9.5)
    ax1.set_xlabel("数据表示精度", fontsize=10)
    ax1.set_ylabel("量化信噪比 SQNR (dB)", fontsize=10)
    ax1.grid(axis="y", linestyle="--", alpha=0.4)
    ax1.spines["top"].set_visible(False)
    ax1.margins(y=0.18)

    ax2 = ax1.twinx()
    ax2.plot(x, comp, color="#C2410C", marker="D", linewidth=1.9, label="相对 FP32 体积压缩比 (×)")
    ax2.set_ylabel("权重存储压缩比 (×)", fontsize=10, color="#C2410C")
    ax2.spines["top"].set_visible(False)
    for i, c in enumerate(comp):
        ax2.annotate(f"{c:.0f}×", (x[i], c), textcoords="offset points", xytext=(0, 6),
                     ha="center", fontsize=8.5, color="#C2410C")

    l1, lb1 = ax1.get_legend_handles_labels()
    l2, lb2 = ax2.get_legend_handles_labels()
    ax1.legend(l1 + l2, lb1 + lb2, loc="upper right", frameon=False, fontsize=8.8)
    ax1.set_title("不同量化位宽下的量化信噪比与模型存储压缩比对比", fontsize=11, pad=10)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_roofline_projection(roof_data: list[dict], save_path: str):
    # 对比 alpha=0.5 轻量检测器在 3 种分辨率下 Pi 4B vs Pi 5 (FP32 / INT8) 估算 FPS
    res_list = ["160x120", "320x240", "640x480"]
    boards = list(RPI_PROFILES.keys())
    series = [
        ("Pi 4B (FP32 估算)", boards[0], "FP32", "#475569"),
        ("Pi 4B (INT8 估算)", boards[0], "INT8", "#2563EB"),
        ("Pi 5 (FP32 估算)", boards[1], "FP32", "#D97706"),
        ("Pi 5 (INT8 估算)", boards[1], "INT8", "#059669"),
    ]

    fig, ax = plt.subplots(figsize=(7.0, 3.9), dpi=220)
    x = np.arange(len(res_list))
    bw = 0.19
    for idx, (lbl, bname, prec, col) in enumerate(series):
        vals = [next(d["est_fps"] for d in roof_data
                     if d["board"] == bname and d["precision"] == prec
                     and d["resolution"] == r and d["width_mult"] == 0.5)
                for r in res_list]
        offset = (idx - 1.5) * bw
        bars = ax.bar(x + offset, vals, width=bw, label=lbl, color=col)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 1.5, f"{h:.0f}",
                    ha="center", va="bottom", fontsize=7.8)

    ax.axhline(25.0, color="#B91C1C", linestyle="--", linewidth=1.2, label="实时视频参考阈值 (25 FPS)")
    ax.set_xticks(x)
    ax.set_xticklabels(res_list, fontsize=9.5)
    ax.set_xlabel("轻量检测器 (α=0.5) 输入分辨率", fontsize=10)
    ax.set_ylabel("Roofline 理论估算帧率 (FPS)", fontsize=10)
    ax.set_title("基于 Roofline 模型的树莓派 4B / 5 端侧推理帧率理论估算", fontsize=11, pad=10)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.margins(y=0.20)
    ax.legend(loc="upper right", frameon=False, fontsize=8.3, ncol=2)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def plot_pr_scenarios(sc_data: dict, save_path: str):
    scenarios = list(sc_data.keys())
    haar_ap = [sc_data[s]["Haar级联(模拟)"]["ap"] for s in scenarios]
    cnn_ap = [sc_data[s]["轻量深度检测器(模拟)"]["ap"] for s in scenarios]

    fig, ax = plt.subplots(figsize=(6.8, 3.7), dpi=220)
    x = np.arange(len(scenarios))
    bw = 0.34
    b1 = ax.bar(x - bw / 2, haar_ap, width=bw, label="Haar 级联检测器 (仿真基准)", color="#64748B")
    b2 = ax.bar(x + bw / 2, cnn_ap, width=bw, label="轻量单阶段深度检测器 (仿真基准)", color="#1D4ED8")

    for bar in b1:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.015, f"{h:.3f}", ha="center", va="bottom", fontsize=8.5)
    for bar in b2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.015, f"{h:.3f}", ha="center", va="bottom", fontsize=8.5)

    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, fontsize=9.5)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("合成受控基准 AP@0.5 (评价流水线输出)", fontsize=10)
    ax.set_title("受控干扰场景下评价流水线模拟 AP 对比（非实机数据集实测）", fontsize=10.8, pad=10)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)


def main():
    print("Running Experiment 1: Integral Image vs Naive Summation...")
    exp1 = benchmark_integral_vs_naive()

    print("Running Experiment 2: Haar Cascade Resolution & ScaleFactor Grid...")
    exp2 = benchmark_haar_grid()

    print("Running Experiment 3: Lightweight CNN Complexity & Operator Benchmark...")
    profiles = []
    for (w, h) in [(160, 120), (320, 240), (640, 480)]:
        for alpha in [0.25, 0.5, 0.75, 1.0]:
            profiles.append(build_lightweight_detector_profile(h, w, alpha, use_dw_sep=False))
            profiles.append(build_lightweight_detector_profile(h, w, alpha, use_dw_sep=True))
    op_bench = benchmark_conv_operators_numpy()

    print("Running Experiment 4: Affine Quantization Error & SQNR Simulation...")
    exp4 = evaluate_quantization_error()

    print("Running Experiment 5: Raspberry Pi 4B/5 Roofline Projection...")
    roof_results = []
    for p in profiles:
        if p["arch_type"] == "DW-Separable" and p["width_mult"] in (0.25, 0.5, 1.0):
            for board in RPI_PROFILES:
                for prec in ("FP32", "INT8"):
                    roof_results.append(estimate_roofline_latency(p, board, prec))

    print("Running Experiment 6: Controlled Scenario NMS & AP Pipeline Verification...")
    exp6 = simulate_controlled_scenario_evaluation()

    all_results = {
        "environment": {
            "os": platform.platform(),
            "arch": platform.machine(),
            "cpu_count": os.cpu_count(),
            "python": platform.python_version(),
            "opencv": cv2.__version__,
            "numpy": np.__version__,
        },
        "exp1_integral_image": exp1,
        "exp2_haar_grid": exp2,
        "exp3_cnn_profiles": profiles,
        "exp3_operator_bench": op_bench,
        "exp4_quantization": exp4,
        "exp5_rpi_roofline": roof_results,
        "exp6_scenario_pipeline": {
            sc: {m: {k: v for k, v in d.items() if k not in ("recalls", "precisions")}
                 for m, d in models.items()}
            for sc, models in exp6.items()
        },
    }

    json_path = os.path.join(OUT_DIR, "experiment_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"Saved results to {json_path}")

    # 生成全部论文插图
    draw_system_architecture(os.path.join(OUT_DIR, "fig1_system_architecture.png"))
    plot_integral_speedup(exp1, os.path.join(OUT_DIR, "fig2_integral_speedup.png"))
    plot_haar_latency(exp2, os.path.join(OUT_DIR, "fig3_haar_resolution_scale.png"))
    plot_dwconv_efficiency(profiles, os.path.join(OUT_DIR, "fig4_dwconv_efficiency.png"))
    plot_quantization(exp4, os.path.join(OUT_DIR, "fig5_quantization_sqnr.png"))
    plot_roofline_projection(roof_results, os.path.join(OUT_DIR, "fig6_rpi_roofline_projection.png"))
    plot_pr_scenarios(exp6, os.path.join(OUT_DIR, "fig7_pr_curves_scenarios.png"))
    print("Generated all 7 figures successfully.")


if __name__ == "__main__":
    main()

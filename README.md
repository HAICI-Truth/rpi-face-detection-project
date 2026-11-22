# 基于树莓派的人脸检测系统设计与实现——算法仿真与端侧部署模拟平台

本项目为西北大学通信工程专业《学年论文2》配套软件仿真与算法验证代码库，针对树莓派等资源受限嵌入式平台的人脸检测任务，在无实体树莓派硬件条件下，构建了完整的软件在环（Software-in-the-Loop, SIL）算法验证与端侧性能建模框架。

## 项目结构

```text
rpi-face-detection-sim/
├── README.md                          # 项目说明文档
├── requirements.txt                   # Python 依赖列表
├── src/
│   ├── __init__.py
│   ├── integral_image.py              # Viola-Jones 积分图与 Haar-like 特征从零实现及复杂度验证
│   ├── haar_pipeline.py               # OpenCV Haar 级联检测器模块化流水线与多阶段计时
│   ├── lightweight_cnn.py             # 深度可分离卷积、MobileNetV2 倒残差块、多尺度检测头复杂度与 ONNX 推理验证
│   ├── quantization.py                # INT8 仿射量化算法（Jacob et al.）与量化误差/SQNR 仿真
│   ├── metrics_nms.py                 # IoU、贪心 NMS、Precision/Recall/AP 评价指标计算模块
│   └── rpi_roofline_model.py          # 树莓派（Pi 4B / Pi 5）Roofline 端侧时延理论估算模型
└── experiments/
    └── run_all_experiments.py         # 一键运行全部 6 组仿真实验并生成数据与论文插图
```

## 运行方式

```bash
pip install -r requirements.txt
python3 experiments/run_all_experiments.py
```

运行后将在 `results/` 目录下输出：
- `experiment_results.json`：全部仿真实验的原始结构化统计数据
- `fig1_system_architecture.png`：模块化人脸检测仿真流水线架构图
- `fig2_integral_speedup.png`：积分图与朴素二维求和计算耗时对比图
- `fig3_haar_resolution_scale.png`：Haar 级联检测器在不同输入分辨率与尺度因子下的时延与 FPS 曲线
- `fig4_dwconv_efficiency.png`：标准卷积与深度可分离卷积（不同宽度倍率）计算量与参数量对比图
- `fig5_quantization_sqnr.png`：不同位宽（FP32/FP16/INT8/INT4）仿射量化误差与存储压缩分析图
- `fig6_rpi_roofline_projection.png`：树莓派 4B 与 5B 端侧单帧理论推理时延估算对比图
- `fig7_pr_curves_scenarios.png`：受控干扰场景（尺度/姿态/遮挡/弱光）下传统级联与轻量深度检测器模拟 PR 曲线对比

## 声明与证据边界

本仓库中的所有实测计时数据均在通用 x86_64 宿主机 CPU 环境下获得；树莓派 4B/5B 的端侧时延数据为基于公开处理器架构参数（ARM Cortex-A72 / A76 NEON 峰值算力与内存带宽）建立的 Roofline 理论估算值，用于硬件到位前的部署可行性预判，待获取实体树莓派后可直接将 `src/` 中的模块化流水线迁移至 Raspberry Pi OS 环境开展实机测量。

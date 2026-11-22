"""基于 Roofline 模型的树莓派（Raspberry Pi 4B / 5）端侧推理时延理论估算模块。

说明：在无实体树莓派硬件条件下，依据公开芯片架构参数建立计算受限（Compute-bound）与
访存受限（Memory-bound）的 Roofline 理论下界模型，并引入有效算力利用系数 eta 估算单帧推理耗时。
该输出为理论估算值（Analytical Projection），不可替代后续实体树莓派实测数据。
"""


RPI_PROFILES = {
    "Raspberry Pi 4B (BCM2711, 4x Cortex-A72 @ 1.5GHz)": {
        # 4核 * 1.5GHz * 8 FP32 FLOPs/cycle (NEON FMA) = 48 GFLOP/s 理论峰值
        "peak_fp32_gflops": 48.0,
        # ARMv8.0-A 无原生 dotprod INT8 指令，INT8 加速主要来自 8-bit 访存减半与宽向量，有效等效算力约 1.8x
        "peak_int8_gops": 86.4,
        # LPDDR4-3200 实测可持续内存带宽约 4.5 GB/s
        "mem_bandwidth_gbs": 4.5,
        # 端侧 CPU 推理框架（NCNN/ONNX Runtime）典型有效利用率（含线程同步与缓存开销）
        "compute_eff_fp32": 0.28,
        "compute_eff_int8": 0.24,
        "mem_eff": 0.60,
        "pre_post_overhead_ms": 6.5,
    },
    "Raspberry Pi 5 (BCM2712, 4x Cortex-A76 @ 2.4GHz)": {
        # 4核 * 2.4GHz * 8 FP32 FLOPs/cycle = 76.8 GFLOP/s 理论峰值
        "peak_fp32_gflops": 76.8,
        # ARMv8.2-A 支持 SDOT/UDOT INT8 指令，每周期 INT8 吞吐显著提升（4核约 307 GOPS 理论峰值）
        "peak_int8_gops": 307.2,
        # LPDDR4X-4267 实测可持续内存带宽约 12.0 GB/s
        "mem_bandwidth_gbs": 12.0,
        "compute_eff_fp32": 0.32,
        "compute_eff_int8": 0.26,
        "mem_eff": 0.65,
        "pre_post_overhead_ms": 3.8,
    },
}


def estimate_roofline_latency(model_profile: dict, board_name: str, precision: str = "FP32") -> dict:
    """根据 Roofline 模型估算单帧网络推理时间与端到端流水线时间。

    Roofline 性能上界:
        P_eff = min(P_peak * eta_c, I_arith * BW * eta_m)
    推理时间估算:
        T_inf = FLOPs / P_eff
    """
    hw = RPI_PROFILES[board_name]
    flops = model_profile["gflops"]  # GFLOPs (10^9 ops)
    if precision == "FP32":
        mem_gb = model_profile["mem_traffic_mb_fp32"] / 1024.0
        eff_compute = hw["peak_fp32_gflops"] * hw["compute_eff_fp32"]
    else:
        mem_gb = model_profile["mem_traffic_mb_int8"] / 1024.0
        eff_compute = hw["peak_int8_gops"] * hw["compute_eff_int8"]

    eff_bw = hw["mem_bandwidth_gbs"] * hw["mem_eff"]
    t_compute_ms = (flops / eff_compute) * 1000.0
    t_memory_ms = (mem_gb / eff_bw) * 1000.0
    t_inf_ms = max(t_compute_ms, t_memory_ms)
    bottleneck = "Compute-bound" if t_compute_ms >= t_memory_ms else "Memory-bound"
    t_total_ms = t_inf_ms + hw["pre_post_overhead_ms"]

    return {
        "board": board_name,
        "resolution": model_profile["input_resolution"],
        "width_mult": model_profile["width_mult"],
        "precision": precision,
        "gflops": model_profile["gflops"],
        "t_compute_ms": round(t_compute_ms, 2),
        "t_memory_ms": round(t_memory_ms, 2),
        "est_inference_ms": round(t_inf_ms, 2),
        "est_pipeline_ms": round(t_total_ms, 2),
        "est_fps": round(1000.0 / max(t_total_ms, 1e-6), 2),
        "bottleneck": bottleneck,
    }

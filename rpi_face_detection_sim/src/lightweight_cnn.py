"""轻量化单阶段人脸检测网络（深度可分离卷积 / MobileNet-SCRFD 风格）复杂度分析与真实算子基准模块。"""

import time
import numpy as np


def conv2d_complexity(h_out: int, w_out: int, c_in: int, c_out: int, k: int = 3) -> dict:
    """计算标准二维卷积与深度可分离卷积的理论参数量与乘加运算量（MACs）。

    标准卷积:
        Params_std = K * K * C_in * C_out
        MACs_std   = H_out * W_out * K * K * C_in * C_out
    深度可分离卷积 (MobileNet):
        Params_dw  = K * K * C_in + C_in * C_out
        MACs_dw    = H_out * W_out * (K * K * C_in + C_in * C_out)
    理论压缩比:
        Ratio = 1 / C_out + 1 / (K * K)
    """
    params_std = k * k * c_in * c_out
    macs_std = h_out * w_out * params_std

    params_dw = k * k * c_in + c_in * c_out
    macs_dw = h_out * w_out * params_dw

    ratio = macs_dw / macs_std
    return {
        "params_std": params_std,
        "macs_std": macs_std,
        "params_dw": params_dw,
        "macs_dw": macs_dw,
        "theoretical_ratio": ratio,
        "reduction_factor": 1.0 / ratio,
    }


def build_lightweight_detector_profile(
    input_h: int,
    input_w: int,
    width_mult: float = 0.5,
    use_dw_sep: bool = True,
) -> dict:
    """构建面向边缘端的人脸检测骨干（MobileNet 风格）+ 轻量 FPN 检测头的逐层复杂度剖析。"""
    def ch(c: int) -> int:
        return max(8, int(round(c * width_mult)))

    # 骨干网络结构: (名称, 输出步长, 输入通道, 输出通道, 卷积核, 是否使用深度可分离)
    stages = [
        ("stem_conv", 2, 3, ch(32), 3, False),
        ("stage1_conv1", 2, ch(32), ch(64), 3, use_dw_sep),
        ("stage2_conv1", 4, ch(64), ch(128), 3, use_dw_sep),
        ("stage2_conv2", 4, ch(128), ch(128), 3, use_dw_sep),
        ("stage3_conv1", 8, ch(128), ch(256), 3, use_dw_sep),
        ("stage3_conv2", 8, ch(256), ch(256), 3, use_dw_sep),
        ("stage4_conv1", 16, ch(256), ch(256), 3, use_dw_sep),
        ("stage4_conv2", 16, ch(256), ch(256), 3, use_dw_sep),
        ("stage5_conv1", 32, ch(256), ch(256), 3, use_dw_sep),
        # 轻量 3 层 FPN + 检测头 (stride 8, 16, 32, 每尺度 2 anchor: cls 2 + box 4 + kps 10 = 16通道)
        ("fpn_p3_lat", 8, ch(256), ch(64), 1, False),
        ("fpn_p4_lat", 16, ch(256), ch(64), 1, False),
        ("fpn_p5_lat", 32, ch(256), ch(64), 1, False),
        ("head_p3", 8, ch(64), 32, 3, use_dw_sep),
        ("head_p4", 16, ch(64), 32, 3, use_dw_sep),
        ("head_p5", 32, ch(64), 32, 3, use_dw_sep),
    ]

    total_params = 0
    total_macs = 0
    peak_activation_bytes_fp32 = 0
    total_memory_traffic_bytes_fp32 = 0

    for name, stride, cin, cout, k, is_dw in stages:
        hout = max(1, input_h // stride)
        wout = max(1, input_w // stride)
        if is_dw and k > 1:
            params = k * k * cin + cin * cout
            macs = hout * wout * params
        else:
            params = k * k * cin * cout
            macs = hout * wout * params

        in_act_bytes = (hout * wout * cin) * 4
        out_act_bytes = (hout * wout * cout) * 4
        w_bytes = params * 4
        peak_activation_bytes_fp32 = max(peak_activation_bytes_fp32, in_act_bytes + out_act_bytes)
        total_memory_traffic_bytes_fp32 += in_act_bytes + out_act_bytes + w_bytes
        total_params += params
        total_macs += macs

    # 1 MAC = 2 FLOPs (1 乘 + 1 加)
    total_flops = total_macs * 2
    return {
        "input_resolution": f"{input_w}x{input_h}",
        "width_mult": width_mult,
        "arch_type": "DW-Separable" if use_dw_sep else "Standard-Conv",
        "params": total_params,
        "params_m": round(total_params / 1e6, 4),
        "macs": total_macs,
        "gmacs": round(total_macs / 1e9, 4),
        "gflops": round(total_flops / 1e9, 4),
        "fp32_size_mb": round(total_params * 4 / (1024 * 1024), 3),
        "int8_size_mb": round(total_params * 1 / (1024 * 1024), 3),
        "peak_act_mb_fp32": round(peak_activation_bytes_fp32 / (1024 * 1024), 3),
        "mem_traffic_mb_fp32": round(total_memory_traffic_bytes_fp32 / (1024 * 1024), 3),
        "mem_traffic_mb_int8": round(total_memory_traffic_bytes_fp32 * 0.25 / (1024 * 1024), 3),
        "arithmetic_intensity_fp32": round(total_flops / max(total_memory_traffic_bytes_fp32, 1), 2),
        "arithmetic_intensity_int8": round(total_flops / max(total_memory_traffic_bytes_fp32 * 0.25, 1), 2),
    }


def benchmark_conv_operators_numpy(
    h: int = 40,
    w: int = 30,
    c_in: int = 64,
    c_out: int = 128,
    k: int = 3,
    repeats: int = 10,
    seed: int = 42,
) -> dict:
    """在宿主机 CPU 上真实运行标准 3x3 卷积与深度可分离卷积（im2col + GEMM），对比实测耗时与理论加速比。"""
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((c_in, h + 2, w + 2)).astype(np.float32)

    # 提取 3x3 局部 patch 矩阵: (C_in * 9, H * W)
    patches = np.lib.stride_tricks.sliding_window_view(x, (k, k), axis=(1, 2))
    # patches shape: (C_in, H, W, 3, 3)

    w_std = rng.standard_normal((c_out, c_in * k * k)).astype(np.float32)
    w_dw = rng.standard_normal((c_in, k * k)).astype(np.float32)
    w_pw = rng.standard_normal((c_out, c_in)).astype(np.float32)

    # 标准卷积实测
    t0 = time.perf_counter()
    for _ in range(repeats):
        col = patches.transpose(0, 3, 4, 1, 2).reshape(c_in * k * k, h * w)
        _ = w_std @ col
    t_std_ms = (time.perf_counter() - t0) * 1000.0 / repeats

    # 深度可分离卷积实测 (Depthwise + Pointwise 1x1)
    t1 = time.perf_counter()
    for _ in range(repeats):
        dw_col = patches.reshape(c_in, h * w, k * k)
        dw_out = np.einsum("cnk,ck->cn", dw_col, w_dw)
        _ = w_pw @ dw_out
    t_dw_ms = (time.perf_counter() - t1) * 1000.0 / repeats

    theory = conv2d_complexity(h, w, c_in, c_out, k)
    return {
        "shape": f"{c_in}->{c_out} @ {w}x{h}",
        "std_ms": round(t_std_ms, 3),
        "dw_ms": round(t_dw_ms, 3),
        "measured_speedup": round(t_std_ms / max(t_dw_ms, 1e-9), 2),
        "theoretical_speedup": round(theory["reduction_factor"], 2),
        "macs_std_m": round(theory["macs_std"] / 1e6, 3),
        "macs_dw_m": round(theory["macs_dw"] / 1e6, 3),
    }

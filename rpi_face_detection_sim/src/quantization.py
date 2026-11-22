"""面向边缘端整数推理的仿射量化仿真模块（依据 Jacob et al., CVPR 2018）。"""

import numpy as np


def affine_quantize_dequantize(
    tensor: np.ndarray,
    num_bits: int = 8,
    symmetric: bool = True,
) -> tuple[np.ndarray, float, int]:
    """执行均匀仿射量化与反量化：
        q = clip(round(r / S) + Z, q_min, q_max)
        r_hat = S * (q - Z)
    """
    if symmetric:
        q_max = (1 << (num_bits - 1)) - 1
        q_min = -q_max
        max_abs = float(np.max(np.abs(tensor)))
        scale = max_abs / max(q_max, 1) if max_abs > 0 else 1.0
        zero_point = 0
    else:
        q_min = 0
        q_max = (1 << num_bits) - 1
        r_min = float(np.min(tensor))
        r_max = float(np.max(tensor))
        scale = (r_max - r_min) / max(q_max - q_min, 1) if r_max > r_min else 1.0
        zero_point = int(np.clip(round(-r_min / scale), q_min, q_max))

    q = np.clip(np.round(tensor / scale) + zero_point, q_min, q_max)
    r_hat = scale * (q - zero_point)
    return r_hat.astype(np.float32), scale, zero_point


def evaluate_quantization_error(
    num_elements: int = 262144,
    seed: int = 42,
) -> list[dict]:
    """仿真卷积权重（近似零均值高斯分布）与 ReLU 激活（非负截断分布）在不同量化位宽下的误差指标。"""
    rng = np.random.default_rng(seed)
    weights = rng.normal(loc=0.0, scale=0.08, size=num_elements).astype(np.float32)
    activations = np.maximum(0.0, rng.normal(loc=0.25, scale=0.65, size=num_elements)).astype(np.float32)

    results = []
    # FP16 基线
    w_fp16 = weights.astype(np.float16).astype(np.float32)
    a_fp16 = activations.astype(np.float16).astype(np.float32)

    def calc_metrics(orig, recon):
        err = orig - recon
        mse = float(np.mean(err ** 2))
        sig_power = float(np.mean(orig ** 2))
        sqnr_db = 10.0 * np.log10(sig_power / max(mse, 1e-15))
        cos_sim = float(np.dot(orig, recon) / (np.linalg.norm(orig) * np.linalg.norm(recon) + 1e-12))
        return mse, sqnr_db, cos_sim

    w_mse, w_sqnr, w_cos = calc_metrics(weights, w_fp16)
    a_mse, a_sqnr, a_cos = calc_metrics(activations, a_fp16)
    results.append({
        "precision": "FP16",
        "bits": 16,
        "compression_ratio": 2.0,
        "weight_sqnr_db": round(w_sqnr, 2),
        "weight_cos_sim": round(w_cos, 6),
        "act_sqnr_db": round(a_sqnr, 2),
        "act_cos_sim": round(a_cos, 6),
    })

    for bits in (8, 6, 4):
        w_hat, _, _ = affine_quantize_dequantize(weights, num_bits=bits, symmetric=True)
        a_hat, _, _ = affine_quantize_dequantize(activations, num_bits=bits, symmetric=False)
        w_mse, w_sqnr, w_cos = calc_metrics(weights, w_hat)
        a_mse, a_sqnr, a_cos = calc_metrics(activations, a_hat)
        results.append({
            "precision": f"INT{bits}",
            "bits": bits,
            "compression_ratio": round(32.0 / bits, 2),
            "weight_sqnr_db": round(w_sqnr, 2),
            "weight_cos_sim": round(w_cos, 6),
            "act_sqnr_db": round(a_sqnr, 2),
            "act_cos_sim": round(a_cos, 6),
        })
    return results

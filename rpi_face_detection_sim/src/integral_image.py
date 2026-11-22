"""Viola-Jones 积分图与 Haar-like 矩形特征计算模块。"""

import time
import numpy as np


def compute_integral_image(gray: np.ndarray) -> np.ndarray:
    """计算二维灰度图像的积分图（带首行首列零填充，便于 O(1) 边界无分支查询）。

    定义:
        ii(x, y) = sum_{x' < x, y' < y} I(x', y')
    使用递推关系:
        s(x, y) = s(x, y-1) + I(x, y)
        ii(x, y) = ii(x-1, y) + s(x, y)
    """
    h, w = gray.shape
    ii = np.zeros((h + 1, w + 1), dtype=np.int64)
    ii[1:, 1:] = np.cumsum(np.cumsum(gray.astype(np.int64), axis=0), axis=1)
    return ii


def rect_sum_integral(ii: np.ndarray, x: int, y: int, w: int, h: int) -> int:
    """利用积分图在 O(1) 时间内（仅需 4 次查表与 3 次加减）计算矩形区域像素和。"""
    return int(ii[y + h, x + w] - ii[y, x + w] - ii[y + h, x] + ii[y, x])


def rect_sum_naive(gray: np.ndarray, x: int, y: int, w: int, h: int) -> int:
    """朴素二维遍历求和，时间复杂度 O(w * h)。"""
    return int(np.sum(gray[y:y + h, x:x + w], dtype=np.int64))


def haar_two_rect_vertical(ii: np.ndarray, x: int, y: int, w: int, h: int) -> int:
    """计算经典两矩形垂直边缘 Haar-like 特征（上暗下亮，常用于眼部-面颊区域）。"""
    half_h = h // 2
    top_sum = rect_sum_integral(ii, x, y, w, half_h)
    bot_sum = rect_sum_integral(ii, x, y + half_h, w, h - half_h)
    return bot_sum - top_sum


def benchmark_integral_vs_naive(
    img_shape=(480, 640),
    window_sizes=(16, 24, 32, 48, 64, 96, 128),
    num_queries=2000,
    seed=42,
):
    """对比朴素求和与积分图查询在不同窗口尺寸下的计算耗时与数值一致性。"""
    rng = np.random.default_rng(seed)
    gray = rng.integers(0, 256, size=img_shape, dtype=np.uint8)
    h_img, w_img = img_shape

    t0 = time.perf_counter()
    ii = compute_integral_image(gray)
    ii_build_ms = (time.perf_counter() - t0) * 1000.0

    results = []
    for ws in window_sizes:
        xs = rng.integers(0, w_img - ws, size=num_queries)
        ys = rng.integers(0, h_img - ws, size=num_queries)

        # 验证数值完全一致
        max_abs_err = 0
        for i in range(min(50, num_queries)):
            v_ii = rect_sum_integral(ii, int(xs[i]), int(ys[i]), ws, ws)
            v_na = rect_sum_naive(gray, int(xs[i]), int(ys[i]), ws, ws)
            max_abs_err = max(max_abs_err, abs(v_ii - v_na))

        # 积分图计时
        t1 = time.perf_counter()
        acc_ii = 0
        for i in range(num_queries):
            acc_ii += rect_sum_integral(ii, int(xs[i]), int(ys[i]), ws, ws)
        t_ii_ms = (time.perf_counter() - t1) * 1000.0

        # 朴素求和计时
        t2 = time.perf_counter()
        acc_na = 0
        for i in range(num_queries):
            acc_na += rect_sum_naive(gray, int(xs[i]), int(ys[i]), ws, ws)
        t_na_ms = (time.perf_counter() - t2) * 1000.0

        assert acc_ii == acc_na, "Integral image result mismatch!"
        results.append({
            "window_size": ws,
            "num_queries": num_queries,
            "ii_build_ms": round(ii_build_ms, 4),
            "integral_query_ms": round(t_ii_ms, 4),
            "naive_query_ms": round(t_na_ms, 4),
            "speedup": round(t_na_ms / max(t_ii_ms, 1e-9), 2),
            "max_abs_err": int(max_abs_err),
        })
    return results

"""OpenCV Haar 级联人脸检测模块化流水线与控制变量计时基准测试。"""

import os
import time
import cv2
import numpy as np


def generate_structured_scene(width: int, height: int, seed: int = 42) -> np.ndarray:
    """生成具有多尺度纹理、亮度渐变和类人脸局部对比度结构的受控测试帧。"""
    rng = np.random.default_rng(seed)
    y_grad = np.linspace(40, 210, height, dtype=np.float32)[:, None]
    x_grad = np.linspace(30, 180, width, dtype=np.float32)[None, :]
    base = 0.5 * y_grad + 0.5 * x_grad
    noise = rng.normal(0, 12.0, size=(height, width)).astype(np.float32)
    frame_gray = np.clip(base + noise, 0, 255).astype(np.uint8)
    frame_bgr = cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR)
    # 添加若干椭圆与局部边缘结构模拟前景目标纹理
    cx, cy = width // 2, height // 2
    rx, ry = max(20, width // 8), max(26, height // 6)
    cv2.ellipse(frame_bgr, (cx, cy), (rx, ry), 0, 0, 360, (195, 175, 160), -1)
    cv2.circle(frame_bgr, (cx - rx // 3, cy - ry // 5), max(3, rx // 8), (45, 40, 38), -1)
    cv2.circle(frame_bgr, (cx + rx // 3, cy - ry // 5), max(3, rx // 8), (45, 40, 38), -1)
    cv2.ellipse(frame_bgr, (cx, cy + ry // 3), (rx // 3, ry // 8), 0, 0, 180, (70, 60, 60), 2)
    return frame_bgr


class HaarFaceDetectorPipeline:
    """模块化 Haar 级联人脸检测流水线（采集/预处理/推理/后处理分离计时）。"""

    def __init__(self, xml_name: str = "haarcascade_frontalface_default.xml"):
        xml_path = os.path.join(cv2.data.haarcascades, xml_name)
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"Haar cascade file not found: {xml_path}")
        self.xml_path = xml_path
        self.model_size_kb = os.path.getsize(xml_path) / 1024.0
        self.classifier = cv2.CascadeClassifier(xml_path)

    def run_frame(
        self,
        frame_bgr: np.ndarray,
        target_size: tuple[int, int] | None = None,
        scale_factor: float = 1.1,
        min_neighbors: int = 5,
        min_size: tuple[int, int] = (30, 30),
        use_equalize: bool = False,
    ) -> dict:
        t_start = time.perf_counter()

        # 1. 预处理阶段：尺寸缩放 + 灰度化 (+ 可选直方图均衡化)
        t_pre0 = time.perf_counter()
        if target_size is not None and (frame_bgr.shape[1], frame_bgr.shape[0]) != target_size:
            resized = cv2.resize(frame_bgr, target_size, interpolation=cv2.INTER_LINEAR)
        else:
            resized = frame_bgr
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        if use_equalize:
            gray = cv2.equalizeHist(gray)
        t_pre_ms = (time.perf_counter() - t_pre0) * 1000.0

        # 2. 级联检测器推理阶段
        t_inf0 = time.perf_counter()
        boxes = self.classifier.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=min_size,
            flags=cv2.CASCADE_SCALE_IMAGE,
        )
        t_inf_ms = (time.perf_counter() - t_inf0) * 1000.0

        # 3. 后处理阶段：格式化边界框坐标
        t_post0 = time.perf_counter()
        detections = []
        if len(boxes) > 0:
            for (x, y, w, h) in boxes:
                detections.append((int(x), int(y), int(w), int(h)))
        t_post_ms = (time.perf_counter() - t_post0) * 1000.0

        t_total_ms = (time.perf_counter() - t_start) * 1000.0
        return {
            "detections": detections,
            "preprocess_ms": t_pre_ms,
            "inference_ms": t_inf_ms,
            "postprocess_ms": t_post_ms,
            "total_ms": t_total_ms,
        }


def benchmark_haar_grid(
    resolutions=((160, 120), (320, 240), (480, 360), (640, 480), (1280, 720)),
    scale_factors=(1.05, 1.1, 1.2, 1.3),
    warmup_runs=3,
    timed_runs=15,
):
    """在宿主机 CPU 上运行 Haar 级联检测器多分辨率与多尺度因子控制变量测试。"""
    pipeline = HaarFaceDetectorPipeline("haarcascade_frontalface_default.xml")
    base_frame = generate_structured_scene(1280, 720, seed=2026)

    records = []
    for (w, h) in resolutions:
        for sf in scale_factors:
            for _ in range(warmup_runs):
                pipeline.run_frame(base_frame, target_size=(w, h), scale_factor=sf)

            pre_list, inf_list, post_list, tot_list = [], [], [], []
            for _ in range(timed_runs):
                out = pipeline.run_frame(base_frame, target_size=(w, h), scale_factor=sf)
                pre_list.append(out["preprocess_ms"])
                inf_list.append(out["inference_ms"])
                post_list.append(out["postprocess_ms"])
                tot_list.append(out["total_ms"])

            avg_tot = float(np.mean(tot_list))
            avg_inf = float(np.mean(inf_list))
            records.append({
                "resolution": f"{w}x{h}",
                "width": w,
                "height": h,
                "scale_factor": sf,
                "model_size_kb": round(pipeline.model_size_kb, 1),
                "preprocess_ms": round(float(np.mean(pre_list)), 3),
                "inference_ms": round(avg_inf, 3),
                "postprocess_ms": round(float(np.mean(post_list)), 3),
                "total_ms": round(avg_tot, 3),
                "std_total_ms": round(float(np.std(tot_list)), 3),
                "fps_pipeline": round(1000.0 / max(avg_tot, 1e-6), 2),
                "fps_inference_only": round(1000.0 / max(avg_inf, 1e-6), 2),
            })
    return records

"""人脸检测后处理（IoU / Greedy NMS）与算法评价指标（Precision / Recall / AP）模块。"""

import numpy as np


def compute_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """计算两个边界框 [x1, y1, x2, y2] 的交并比 IoU。"""
    ix1 = max(box_a[0], box_b[0])
    iy1 = max(box_a[1], box_b[1])
    ix2 = min(box_a[2], box_b[2])
    iy2 = min(box_a[3], box_b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def greedy_nms(boxes: np.ndarray, scores: np.ndarray, iou_thresh: float = 0.4) -> list[int]:
    """贪心非极大值抑制（Greedy NMS），返回保留候选框的索引列表。"""
    if len(boxes) == 0:
        return []
    order = np.argsort(scores)[::-1]
    keep = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rem = order[1:]
        ious = np.array([compute_iou(boxes[i], boxes[j]) for j in rem], dtype=np.float32)
        order = rem[ious <= iou_thresh]
    return keep


def compute_pr_ap(
    gt_boxes_by_img: dict[int, list[np.ndarray]],
    preds: list[tuple[int, float, np.ndarray]],
    iou_thresh: float = 0.5,
) -> dict:
    """按置信度降序匹配真值框，计算全插值 Precision-Recall 曲线与 AP@IoU。"""
    preds_sorted = sorted(preds, key=lambda x: x[1], reverse=True)
    matched = {img_id: [False] * len(b_list) for img_id, b_list in gt_boxes_by_img.items()}
    total_gt = sum(len(b_list) for b_list in gt_boxes_by_img.values())

    tp = np.zeros(len(preds_sorted), dtype=np.float64)
    fp = np.zeros(len(preds_sorted), dtype=np.float64)

    for idx, (img_id, score, pbox) in enumerate(preds_sorted):
        gt_list = gt_boxes_by_img.get(img_id, [])
        best_iou = 0.0
        best_j = -1
        for j, gbox in enumerate(gt_list):
            iou = compute_iou(pbox, gbox)
            if iou > best_iou:
                best_iou = iou
                best_j = j
        if best_iou >= iou_thresh and best_j >= 0 and not matched[img_id][best_j]:
            tp[idx] = 1.0
            matched[img_id][best_j] = True
        else:
            fp[idx] = 1.0

    cum_tp = np.cumsum(tp)
    cum_fp = np.cumsum(fp)
    recalls = cum_tp / max(total_gt, 1)
    precisions = cum_tp / np.maximum(cum_tp + cum_fp, 1e-12)

    # 包络线平滑（单调递减插值）计算 AP
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    change_idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = float(np.sum((mrec[change_idx + 1] - mrec[change_idx]) * mpre[change_idx + 1]))

    return {
        "total_gt": int(total_gt),
        "total_tp": int(cum_tp[-1]) if len(cum_tp) else 0,
        "total_fp": int(cum_fp[-1]) if len(cum_fp) else 0,
        "final_precision": round(float(precisions[-1]), 4) if len(precisions) else 0.0,
        "final_recall": round(float(recalls[-1]), 4) if len(recalls) else 0.0,
        "ap": round(ap, 4),
        "recalls": recalls.tolist(),
        "precisions": precisions.tolist(),
    }


def simulate_controlled_scenario_evaluation(seed: int = 2026) -> dict:
    """在合成受控检测基准（模拟 WIDER FACE 尺度/姿态/遮挡/光照因素）上验证评价流水线与 NMS 行为。

    说明：该仿真实验用于验证检测评价协议（IoU 匹配、NMS 抑制与 PR/AP 积分）的正确性，
    并通过设定受控信噪比衰减模型直观展示不同干扰因素下检测灵敏度变化趋势，不代表真实数据集实测 AP。
    """
    rng = np.random.default_rng(seed)
    n_images = 60
    gt_boxes_by_img = {}
    for img_id in range(n_images):
        # 每帧 2 个真值人脸框
        b1 = np.array([100.0, 80.0, 180.0, 180.0])
        b2 = np.array([340.0, 110.0, 390.0, 175.0])
        gt_boxes_by_img[img_id] = [b1, b2]

    # 设定四类受控场景的仿真检出概率与误检率参数（仅用于软件在环评价流水线验证）
    scenarios = {
        "正面常规光照": {"haar_det": 0.90, "haar_fp": 0.12, "cnn_det": 0.96, "cnn_fp": 0.05},
        "小尺度远距离": {"haar_det": 0.48, "haar_fp": 0.25, "cnn_det": 0.85, "cnn_fp": 0.09},
        "大角度侧脸": {"haar_det": 0.35, "haar_fp": 0.22, "cnn_det": 0.82, "cnn_fp": 0.10},
        "局部遮挡弱光": {"haar_det": 0.42, "haar_fp": 0.28, "cnn_det": 0.80, "cnn_fp": 0.11},
    }

    summary = {}
    for sc_name, cfg in scenarios.items():
        summary[sc_name] = {}
        for model_key, det_p, fp_p, jitter in [
            ("Haar级联(模拟)", cfg["haar_det"], cfg["haar_fp"], 7.0),
            ("轻量深度检测器(模拟)", cfg["cnn_det"], cfg["cnn_fp"], 3.5),
        ]:
            preds = []
            for img_id, gboxes in gt_boxes_by_img.items():
                raw_boxes, raw_scores = [], []
                for gb in gboxes:
                    if rng.random() < det_p:
                        # 主预测框 + 1个重叠冗余框（检验 NMS 去重能力）
                        for dup in range(2):
                            noise = rng.normal(0, jitter * (1.0 + 0.5 * dup), size=4)
                            pb = gb + noise
                            sc = float(np.clip(rng.normal(0.84 - 0.18 * dup, 0.08), 0.15, 0.99))
                            raw_boxes.append(pb)
                            raw_scores.append(sc)
                if rng.random() < fp_p:
                    fx, fy = rng.uniform(20, 500), rng.uniform(20, 350)
                    raw_boxes.append(np.array([fx, fy, fx + 55.0, fy + 65.0]))
                    raw_scores.append(float(np.clip(rng.normal(0.45, 0.15), 0.10, 0.85)))

                if raw_boxes:
                    rb_arr = np.vstack(raw_boxes)
                    rs_arr = np.array(raw_scores)
                    keep = greedy_nms(rb_arr, rs_arr, iou_thresh=0.4)
                    for k in keep:
                        preds.append((img_id, float(rs_arr[k]), rb_arr[k]))

            res = compute_pr_ap(gt_boxes_by_img, preds, iou_thresh=0.5)
            summary[sc_name][model_key] = res
    return summary

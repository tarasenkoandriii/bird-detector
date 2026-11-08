"""Об’єднання детекцій звичайного й горизонтально віддзеркаленого кадру."""

import numpy as np
import cv2


def unflip_boxes(boxes, width):
    rows = np.asarray(boxes, dtype=float).reshape(-1, 5).copy()
    if len(rows):
        rows[:, [0, 2]] = width - rows[:, [2, 0]]
    return rows


def fuse_two_views(first, second, match_iou=0.55, nms_iou=0.5, unmatched_weight=0.5):
    """Зважуємо пару bbox за confidence; непідтверджені рамки мають меншу вагу."""
    a = np.asarray(first, dtype=float).reshape(-1, 5)
    b = np.asarray(second, dtype=float).reshape(-1, 5)
    rows = np.concatenate([a, b])
    views = np.r_[np.zeros(len(a), int), np.ones(len(b), int)]
    if not len(rows):
        return []
    remaining = list(np.argsort(-rows[:, 4], kind="stable"))
    fused = []
    while remaining:
        anchor = remaining.pop(0)
        pair = None
        best = match_iou
        for index in remaining:
            if views[index] == views[anchor]:
                continue
            lo = np.maximum(rows[anchor, :2], rows[index, :2])
            hi = np.minimum(rows[anchor, 2:4], rows[index, 2:4])
            inter = np.maximum(hi - lo, 0).prod()
            area = (
                np.maximum(rows[anchor, 2:4] - rows[anchor, :2], 0).prod()
                + np.maximum(rows[index, 2:4] - rows[index, :2], 0).prod()
            )
            iou = inter / max(area - inter, 1e-12)
            if iou >= best:
                pair = index
                best = iou
        if pair is None:
            fused.append([*rows[anchor, :4], rows[anchor, 4] * unmatched_weight])
        else:
            remaining.remove(pair)
            scores = rows[[anchor, pair], 4]
            coords = (rows[[anchor, pair], :4] * scores[:, None]).sum(0) / max(scores.sum(), 1e-12)
            fused.append([*coords, float(scores.mean())])
    xywh = [[x0, y0, x1 - x0, y1 - y0] for x0, y0, x1, y1, _ in fused]
    keep = cv2.dnn.NMSBoxes(xywh, [float(r[4]) for r in fused], 0.0, nms_iou)
    return [list(map(float, fused[int(i)])) for i in np.asarray(keep).reshape(-1)]

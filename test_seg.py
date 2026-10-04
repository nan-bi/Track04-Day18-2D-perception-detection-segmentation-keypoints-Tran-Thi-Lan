import numpy as np
import cv2
import torch

# Dummy implementation of _run to capture the error
def _run(name, steps, quiet=False):
    for desc, test in steps:
        try:
            hint = test()
        except Exception as e:
            print(f"❌ {name} — {desc}\n   ↳ code báo lỗi → {type(e).__name__}: {e}")
            return
        if hint:
            print(f"❌ {name} — {desc}\n   ↳ {hint}")
            return
    print(f"✅ {name} đạt")

PROGRESS = {}
G = globals()

def _fill(poly, h, w):
    m = np.zeros((h, w), np.uint8)
    cv2.fillPoly(m, [np.round(poly).astype(np.int32)], 1)
    return m.astype(bool)

def _np_iou(a, b):
    return float((a & b).sum() / max((a | b).sum(), 1))

def _parse_seg_line(line, h, w):
    parts = line.split()
    return int(parts[0]), np.array(parts[1:], dtype=float).reshape(-1, 2) * [w, h]

def check_mask_to_yolo_seg(quiet=False):
    f = lambda *a, **k: G["mask_to_yolo_seg"](*a, **k)
    H, W = 200, 300
    circle = np.zeros((H, W), np.uint8); cv2.circle(circle, (150, 100), 60, 1, -1); circle = circle.astype(bool)

    def fmt():
        line = f(circle, 3)
        if not isinstance(line, str) or not line.strip():
            return "phải trả về một chuỗi 'class x1 y1 x2 y2 ...' (đang rỗng hoặc sai kiểu)."
        parts = line.split()
        if parts[0] != "3":
            return f"token đầu phải là class id '3', bạn ra '{parts[0]}'."
        if (len(parts) - 1) % 2 or len(parts) < 7:
            return f"sau class id phải là các cặp (x, y), ít nhất 3 đỉnh; bạn có {len(parts) - 1} số."

    def normalized():
        vals = np.array(f(circle, 3).split()[1:], dtype=float)
        if vals.max() > 1.0 + 1e-6 or vals.min() < 0:
            return f"toạ độ phải nằm trong [0, 1], bạn có giá trị tới {vals.max():.2f}. Bước 3: x chia cho W, y chia cho H."

    def roundtrip():
        _, poly = _parse_seg_line(f(circle, 3), H, W)
        v = _np_iou(circle, _fill(poly, H, W))
        if v < 0.95:
            return f"vẽ lại polygon thì chỉ trùng {v:.2f} với mask gốc (cần > 0.95). Hay gặp: chia x cho H và y cho W (ngược)."

    def largest():
        for corner in ((5, 5), (190, 290)):
            m = circle.copy(); m[corner[0]:corner[0] + 5, corner[1]:corner[1] + 5] = True
            _, poly = _parse_seg_line(f(m, 0), H, W)
            if _np_iou(circle, _fill(poly, H, W)) < 0.9:
                return "mask có thêm một mảnh vụn 5×5 thì bạn lại lấy mảnh vụn. Bước 1: chọn contour có diện tích lớn nhất, không phải contours[0]."

    _run("mask_to_yolo_seg", [("định dạng dòng nhãn", fmt), ("toạ độ chuẩn hoá", normalized),
                              ("mask → polygon → mask", roundtrip), ("chọn mảnh lớn nhất", largest)], quiet)

def mask_to_yolo_seg(mask: np.ndarray, cls_id: int, simplify: float = 0.002) -> str:
    h, w = mask.shape
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return ""
    # Bước 1
    cnt = max(contours, key=cv2.contourArea)
    # Bước 2
    cnt = cv2.approxPolyDP(cnt, simplify * cv2.arcLength(cnt, True), True).reshape(-1, 2)
    if len(cnt) < 3:
        return ""
    # Bước 3
    norm = cnt / np.array([w, h])
    return f"{cls_id} " + " ".join(f"{v:.6f}" for v in norm.flatten())

check_mask_to_yolo_seg()

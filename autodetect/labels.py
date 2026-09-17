"""Hình học box + dựng nhãn YOLO cuối cùng từ votes/ của stage 2 — CHỈ stdlib.

Đây là NƠI DUY NHẤT quyết định "box này thuộc class nào, có được giữ không":
stage 2 chỉ chấm điểm rồi ghi votes/, không tự cắt ngưỡng. Nhờ vậy đổi ngưỡng
không phải chạy lại model (xem rebuild_labels.py).
"""

import json
from pathlib import Path


def iou_ios(a, b) -> tuple:
    """(IoU, giao / diện tích box nhỏ hơn) của 2 box xyxy."""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0, 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter), inter / min(area_a, area_b)


def suppress(items, iou_thr: float, ios_thr: float) -> list:
    """Bỏ box trùng / box nằm lọt trong box khác, giữ box điểm cao hơn.

    items: [(score, box)]. IoU bắt 2 box vẽ lệch nhau trên cùng vật thể, IoS
    (giao / diện tích box NHỎ) bắt box bé nằm gọn trong box to — stage 1 ở
    ngưỡng thấp hay vẽ cả cụm box lồng nhau trên một vật.
    """
    keep = []
    for item in sorted(items, key=lambda it: -it[0]):
        if any(iou > iou_thr or ios > ios_thr
               for iou, ios in (iou_ios(item[1], k[1]) for k in keep)):
            continue
        keep.append(item)
    return keep


def class_ids(classes: dict, data_yaml: Path = None) -> dict:
    """{tên class: id} — mặc định theo THỨ TỰ KHAI TRONG classes.json.

    Chỉ cần data.yaml khi muốn ép id theo một bộ nhãn có sẵn (vd model đã train
    với thứ tự class khác). Khi đó tên class phải khớp khít giữa hai file.
    """
    if data_yaml and Path(data_yaml).exists():
        import yaml

        names = yaml.safe_load(Path(data_yaml).read_text(encoding="utf-8"))["names"]
        ids = ({v: int(k) for k, v in names.items()} if isinstance(names, dict)
               else {name: i for i, name in enumerate(names)})
        missing = [c for c in classes if c not in ids]
        if missing:
            raise SystemExit(f"❌ class thiếu trong {Path(data_yaml).name}: {missing}")
        return ids
    return {name: i for i, name in enumerate(classes)}


def write_data_yaml(path: Path, ids: dict):
    """Ghi data.yaml cạnh nhãn để bộ nhãn xuất ra tự mô tả được chính nó."""
    names = [c for c, _ in sorted(ids.items(), key=lambda kv: kv[1])]
    Path(path).write_text(
        "nc: {}\nnames:\n{}\n".format(len(names), "\n".join(f"  - {n}" for n in names)),
        encoding="utf-8")


def build_labels(vote_dirs: list, ids: dict, out_dir: Path, min_conf: float,
                 nms_iou: float, nms_ios: float) -> tuple:
    """votes/ của MỘT HAY NHIỀU backend -> nhãn YOLO cuối cùng. Trả (số ảnh, số nhãn).

    Mỗi box về tay class có confidence cao nhất và chỉ được giữ nếu >= min_conf.
    conf = 0 nghĩa là stage 2 KHÔNG hề nhắc tới box đó -> luôn bỏ, kể cả khi
    min_conf = 0 (min_conf = 0 nghĩa là "khớp bất kỳ prompt nào cũng lấy",
    không phải "lấy tất").

    Nhiều backend: mỗi backend đã được stage 2 chấm RIÊNG (tự top-K công bằng
    nội bộ) nên confidence của chúng CÙNG MỘT THANG 0..1 do chính VLX-Seek quyết
    định — gộp ở đây bằng NMS trên confidence đó là công bằng thật sự: ai chấm
    cao hơn cho cùng một vật thì thắng. (KHÔNG được gộp proposal của nhiều
    backend TRƯỚC stage 2 rồi mới top-K: score đề xuất thô không cùng thang đo
    giữa các backend nên box thật của backend điểm thấp bị đá văng oan.)

    Dedup chỉ TRONG TỪNG CLASS: box lồng nhau khác class là chuyện bình thường
    (biển số nằm trong xe, mũ nằm trên đầu người).
    """
    vote_dirs = [Path(d) for d in vote_dirs if Path(d).exists()]
    stems = sorted({p.stem for d in vote_dirs for p in d.glob("*.json")})
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    n_img = n_box = 0
    for stem in stems:
        w = h = None
        by_class = {}
        for vdir in vote_dirs:
            vfile = vdir / f"{stem}.json"
            if not vfile.exists():
                continue
            data = json.loads(vfile.read_text(encoding="utf-8"))
            w, h = data["w"], data["h"]
            for rec in data["boxes"]:
                if not rec["conf"]:
                    continue
                cls_name, conf = max(rec["conf"].items(), key=lambda kv: kv[1])
                if conf > 0 and conf >= min_conf:
                    by_class.setdefault(cls_name, []).append((conf, rec["box"]))
        if w is None:
            continue
        lines = []
        for cls_name, items in by_class.items():
            for _, (x1, y1, x2, y2) in suppress(items, nms_iou, nms_ios):
                lines.append(f"{ids[cls_name]} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} "
                             f"{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
        Path(out_dir, f"{stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        n_img += 1
        n_box += len(lines)
    return n_img, n_box

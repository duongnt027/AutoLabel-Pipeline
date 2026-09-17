#!/usr/bin/env python3
"""Stage 1 — backend "llm": LLMDet (Grounding DINO) sinh box proposal cho mọi hint.

CHẠY BẰNG PYTHON CỦA ENV `duongnt` (torch 2.2.1 + mmcv 2.2.0 + transformers
4.37.2) và cwd PHẢI LÀ REPO LLMDet: repo tự vendor mmdet, còn config trỏ model
bằng đường dẫn tương đối '../huggingface/...'.

    cd $LLMDET_REPO
    $LLMDET_PYTHON /path/to/auto_detect/stage1/llmdet.py \
        --images <dir ảnh> --out <ws>/proposals_llm.jsonl \
        --classes /path/to/auto_detect/classes/classes.json \
        --config configs/grounding_dino_swin_l.py \
        --weights pretrained_models/large.pth

Ghi ra jsonl theo HỢP ĐỒNG CHUNG của stage 1 (xem autodetect/common.py).
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path

# mmdet là bản nằm TRONG repo LLMDet (không cài vào site-packages). Chạy script
# bằng đường dẫn tuyệt đối thì python chỉ thêm thư mục chứa script vào sys.path,
# không thêm cwd -> phải tự thêm repo (cwd) vào.
sys.path.insert(0, os.getcwd())
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image  # noqa: E402

from autodetect.common import JsonlWriter, clip_boxes, jobs_from_args, load_hints, plan_jobs  # noqa: E402


def tiles(W: int, H: int, size: int, overlap: float) -> list:
    """Danh sách ô cắt (x1, y1, x2, y2) phủ kín ảnh, các ô chồng mép lên nhau.

    LLMDet resize ảnh về ~1333px cạnh dài trước khi infer nên vật nhỏ trong ảnh
    2880x1620 mất chi tiết. Cắt ô rồi infer từng ô giữ được độ phân giải gốc:
    đo trên 20 ảnh thực tế, recall proposal 0.756 -> 0.914, đổi lại chậm ~14 lần.
    """
    step = max(1, int(size * (1 - overlap)))
    xs = range(0, max(1, W - size + step), step)
    ys = range(0, max(1, H - size + step), step)
    return sorted({(min(x, max(0, W - size)), min(y, max(0, H - size)),
                    min(x + size, W), min(y + size, H)) for x in xs for y in ys})


def suppress_per_hint(boxes, scores, hints, iou_thr=0.6, ios_thr=0.9) -> list:
    """Index các box giữ lại sau khi bỏ trùng TRONG TỪNG HINT.

    Chỉ dùng cho chế độ SAHI: 2 ô cạnh nhau cùng nhìn thấy 1 vật ở phần chồng mép.
    """
    from autodetect.labels import iou_ios

    keep = []
    for i in sorted(range(len(boxes)), key=lambda k: -scores[k]):
        dup = False
        for j in keep:
            if hints[j] != hints[i]:
                continue
            iou, ios = iou_ios(boxes[i], boxes[j])
            if iou > iou_thr or ios > ios_thr:
                dup = True
                break
        if not dup:
            keep.append(i)
    return keep


def main():
    ap = argparse.ArgumentParser(description="LLMDet sinh box proposal theo hint")
    ap.add_argument("--manifest", type=Path,
                    help="JSON [{name, images, preds}] để chạy nhiều thư mục trong 1 lần nạp model")
    ap.add_argument("--images", type=Path, help="thư mục ảnh (khi chạy 1 thư mục)")
    ap.add_argument("--out", type=Path, help="file jsonl kết quả (khi chạy 1 thư mục)")
    ap.add_argument("--classes", type=Path, required=True)
    ap.add_argument("--config", default="configs/grounding_dino_swin_l.py")
    ap.add_argument("--weights", default="pretrained_models/large.pth")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--score-thr", type=float, default=0.15)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--sahi", action="store_true",
                    help="cắt ảnh thành ô chồng mép rồi infer từng ô (bắt vật nhỏ, chậm hơn nhiều)")
    ap.add_argument("--sahi-slice", type=int, default=1024, help="cạnh ô cắt (px)")
    ap.add_argument("--sahi-overlap", type=float, default=0.2, help="tỉ lệ chồng mép giữa 2 ô")
    ap.add_argument("--limit", type=int, default=0, help="chỉ chạy N ảnh đầu mỗi thư mục")
    args = ap.parse_args()

    hints = load_hints(args.classes)
    # Định dạng custom entities của Grounding DINO: "a . b . c ." -> MỌI hint
    # được hỏi trong CÙNG 1 lượt forward.
    texts = " . ".join(hints) + " ."
    print(f"   {len(hints)} hint: {texts}")

    todo = plan_jobs(jobs_from_args(args), args.limit)
    if not todo:
        return

    from mmdet.apis import DetInferencer

    inferencer = DetInferencer(model=args.config, weights=args.weights,
                               device=args.device, palette="none")
    inferencer.model.test_cfg.chunked_size = -1

    def infer(paths, offsets):
        """1 lô ảnh/ô -> [(hint, score, box)] theo TỪNG input, toạ độ đã dời về ảnh gốc."""
        res = inferencer([str(p) for p in paths], batch_size=len(paths), texts=texts,
                         custom_entities=True, no_save_vis=True, no_save_pred=True,
                         out_dir="", print_result=False)
        out = []
        for (ox, oy), pred in zip(offsets, res["predictions"]):
            out.append([(hints[lab], float(score),
                         [box[0] + ox, box[1] + oy, box[2] + ox, box[3] + oy])
                        for lab, score, box in zip(pred["labels"], pred["scores"], pred["bboxes"])
                        if score >= args.score_thr and lab < len(hints)])
        return out

    def infer_sahi(path, W, H):
        """1 lượt ảnh đầy + từng ô cắt (ô ghi ra file tạm cho chắc thứ tự)."""
        im = Image.open(path).convert("RGB")
        with tempfile.TemporaryDirectory() as td:
            queue = [(path, (0, 0))]
            for k, (x1, y1, x2, y2) in enumerate(tiles(W, H, args.sahi_slice, args.sahi_overlap)):
                tile_path = Path(td) / f"{k}.jpg"
                im.crop((x1, y1, x2, y2)).save(tile_path, quality=90)
                queue.append((tile_path, (x1, y1)))
            found = []
            for i in range(0, len(queue), args.batch_size):
                part = queue[i:i + args.batch_size]
                for items in infer([q for q, _ in part], [o for _, o in part]):
                    found += items
        return found

    writer, n_box, start = JsonlWriter(), 0, 0
    try:
        while start < len(todo):
            # 1 lô chỉ gồm ảnh của CÙNG 1 thư mục để ghi thẳng vào jsonl của nó
            out = todo[start][0]
            chunk = [p for o, p in todo[start:start + args.batch_size] if o == out]
            sizes = [Image.open(p).size for p in chunk]
            if args.sahi:            # mỗi ảnh tự chia lô cho các ô của nó
                found_per_image = [infer_sahi(p, w, h) for p, (w, h) in zip(chunk, sizes)]
            else:
                found_per_image = infer(chunk, [(0, 0)] * len(chunk))

            for path, (w, h), found in zip(chunk, sizes, found_per_image):
                boxes, scores, labels = clip_boxes(found, w, h)
                if args.sahi:        # ô chồng mép -> cùng 1 vật ra nhiều box
                    keep = suppress_per_hint(boxes, scores, labels)
                    boxes = [boxes[i] for i in keep]
                    scores = [scores[i] for i in keep]
                    labels = [labels[i] for i in keep]
                writer.write(out, path.stem, w, h, boxes, scores, labels)
                n_box += len(boxes)
            start += len(chunk)
            print(f"   [{start}/{len(todo)}] {n_box} box (score >= {args.score_thr})")
    finally:
        files = writer.files
        writer.close()

    print(f"   ✅ {len(todo)} ảnh, {n_box} box → {files}")


if __name__ == "__main__":
    main()

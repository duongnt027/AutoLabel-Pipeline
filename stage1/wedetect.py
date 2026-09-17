#!/usr/bin/env python3
"""Stage 1 — backend "we": WeDetect (github.com/WeChatCV/WeDetect) sinh box proposal.

WeDetect là detector kiểu YOLO-World (dual-tower, không fusion layer): mọi hint
được nhúng thành "trọng số phân loại" MỘT LẦN qua model.reparameterize(texts),
sau đó MỖI ẢNH chỉ cần 1 lượt forward là chấm điểm đồng thời với TẤT CẢ hint —
nhanh hơn hẳn LLMDet (đo thực tế: 63 hint, reparameterize 0.12s, mỗi ảnh sau đó
0.04-0.32s, trong khi LLMDet ~0.7-1.3s/ảnh).

2 LƯU Ý rút ra khi wire thật (đã chạy kiểm chứng, không phải đoán):

1. README chính thức nói phải dùng tên class TIẾNG TRUNG (model train bằng
   prompt tiếng Trung). Đo thực tế: hint TIẾNG ANH VẪN RA KẾT QUẢ HỢP LÝ vì
   text encoder là XLM-RoBERTa đa ngôn ngữ. Muốn chất lượng cao hơn thì dịch
   hints sang tiếng Trung, lưu thành 1 file classes.json riêng rồi trỏ
   WEDETECT_CLASSES trong .env vào đó.
2. mmdet PHẢI cài THẬT vào env (không vendor như LLMDet), và bản pip mmdet 3.3.0
   tự chặn mmcv >= 2.2.0 trong khi LLMDet cần đúng mmcv == 2.2.0 -> phải comment
   assertion trong mmdet/__init__.py (KHÔNG hạ version mmcv thật). Cũng cần cài
   thêm `webdataset` vì mmdet/datasets/__init__.py import nó vô điều kiện. Xem
   setup/install_env_llmdet.sh — cả hai đã nằm trong env `duongnt`, dùng chung.

CHẠY BẰNG PYTHON CỦA ENV `duongnt` và cwd PHẢI LÀ REPO WeDetect: config dùng
custom_imports=["wedetect"] + đường dẫn tương đối './xlm-roberta-base/' để nạp
text encoder.

    cd $WEDETECT_REPO
    $WEDETECT_PYTHON /path/to/auto_detect/stage1/wedetect.py \
        --images <dir ảnh> --out <ws>/proposals_we.jsonl \
        --classes /path/to/auto_detect/classes/classes.json \
        --config config/wedetect_base.py --weights /path/to/wedetect_base.pth

Ghi ra jsonl theo HỢP ĐỒNG CHUNG của stage 1 (xem autodetect/common.py).
"""

import argparse
import os
import sys
from pathlib import Path

# repo tự vendor package "wedetect" (custom_imports trong config) -> phải có
# trong sys.path; chạy bằng đường dẫn tuyệt đối thì python chỉ thêm thư mục
# CHỨA SCRIPT, không thêm cwd, nên tự thêm cwd (= repo WeDetect).
sys.path.insert(0, os.getcwd())
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image  # noqa: E402

from autodetect.common import JsonlWriter, clip_boxes, jobs_from_args, load_hints, plan_jobs  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="WeDetect sinh box proposal theo hint")
    ap.add_argument("--manifest", type=Path,
                    help="JSON [{name, images, preds}] để chạy nhiều thư mục trong 1 lần nạp model")
    ap.add_argument("--images", type=Path, help="thư mục ảnh (khi chạy 1 thư mục)")
    ap.add_argument("--out", type=Path, help="file jsonl kết quả (khi chạy 1 thư mục)")
    ap.add_argument("--classes", type=Path, required=True)
    ap.add_argument("--config", default="config/wedetect_base.py",
                    help="đường dẫn TƯƠNG ĐỐI so với cwd = repo WeDetect")
    ap.add_argument("--weights", required=True, help="checkpoint .pth (base/large/tiny)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--score-thr", type=float, default=0.01)
    ap.add_argument("--limit", type=int, default=0, help="chỉ chạy N ảnh đầu mỗi thư mục")
    args = ap.parse_args()

    hints = load_hints(args.classes)
    texts = [[h] for h in hints] + [[" "]]     # [" "] = category rỗng, giống demo gốc
    print(f"   {len(hints)} hint (gộp 1 lần reparameterize -> mỗi ảnh chỉ 1 lượt forward)")

    todo = plan_jobs(jobs_from_args(args), args.limit)
    if not todo:
        return

    import torch
    from mmdet.utils import register_all_modules

    register_all_modules()
    from mmdet.apis import init_detector
    from mmengine.config import Config
    from mmengine.dataset import Compose

    cfg = Config.fromfile(args.config)
    print(f"   load {args.config} + {args.weights} lên {args.device}")
    model = init_detector(cfg, checkpoint=args.weights, device=args.device, palette=["red"])
    test_pipeline = Compose(cfg.test_pipeline)
    model.reparameterize(texts)

    writer, n_box = JsonlWriter(), 0
    try:
        for i, (out, path) in enumerate(todo, 1):
            w, h = Image.open(path).size
            data_info = test_pipeline(dict(img_id=0, img_path=str(path), texts=texts))
            data_batch = dict(inputs=data_info["inputs"].unsqueeze(0),
                              data_samples=[data_info["data_samples"]])
            with torch.no_grad():
                output = model.test_step(data_batch)[0]
            pred = output.pred_instances
            pred = pred[pred.scores.float() >= args.score_thr].cpu().numpy()

            found = [(hints[lab], score, bbox)
                     for bbox, score, lab in zip(pred["bboxes"], pred["scores"], pred["labels"])
                     if lab < len(hints)]   # lab == len(hints) là category rỗng [" "]
            boxes, scores, labels = clip_boxes(found, w, h)
            writer.write(out, path.stem, w, h, boxes, scores, labels)
            n_box += len(boxes)
            print(f"   [{i}/{len(todo)}] {n_box} box (score >= {args.score_thr})")
    finally:
        files = writer.files
        writer.close()

    print(f"   ✅ {len(todo)} ảnh, {n_box} box → {files}")


if __name__ == "__main__":
    main()

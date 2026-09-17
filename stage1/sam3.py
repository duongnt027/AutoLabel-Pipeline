#!/usr/bin/env python3
"""Stage 1 — backend "s3": SAM3 sinh box proposal cho mọi hint bằng text prompt.

CHẠY BẰNG PYTHON CỦA ENV `vlx` (transformers >= 5.x mới có Sam3Model /
Sam3Processor) — dùng chung env với stage 2, không cần env riêng.

facebook/sam3 là GATED REPO trên HuggingFace: phải accept license tại
https://huggingface.co/facebook/sam3 rồi đặt HF_TOKEN trong .env.

Khác LLMDet (1 lượt gọi custom-entities cho MỌI hint cùng lúc), SAM3 phải hỏi
TỪNG hint riêng — mỗi hint là 1 "concept" text độc lập, không gộp câu được.

    $VLX_PYTHON stage1/sam3.py --images <dir ảnh> --out <ws>/proposals_s3.jsonl \
        --classes classes/classes.json --score-thr 0.05

Ghi ra jsonl theo HỢP ĐỒNG CHUNG của stage 1 (xem autodetect/common.py).
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image  # noqa: E402

from autodetect.common import JsonlWriter, clip_boxes, jobs_from_args, load_hints, plan_jobs  # noqa: E402


def hf_login():
    """facebook/sam3 gated -> phải đăng nhập HuggingFace trước khi tải."""
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        return
    from huggingface_hub import login

    login(token=token, add_to_git_credential=False)


def main():
    ap = argparse.ArgumentParser(description="SAM3 sinh box proposal theo hint (text prompt)")
    ap.add_argument("--manifest", type=Path,
                    help="JSON [{name, images, preds}] để chạy nhiều thư mục trong 1 lần nạp model")
    ap.add_argument("--images", type=Path, help="thư mục ảnh (khi chạy 1 thư mục)")
    ap.add_argument("--out", type=Path, help="file jsonl kết quả (khi chạy 1 thư mục)")
    ap.add_argument("--classes", type=Path, required=True)
    ap.add_argument("--model", default=os.getenv("SAM3_MODEL", "facebook/sam3"))
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--score-thr", type=float, default=0.05)
    ap.add_argument("--batch-size", type=int, default=1,
                    help="số ẢNH trong 1 lượt forward (mỗi ảnh còn nhân bản theo số hint)")
    ap.add_argument("--hint-chunk", type=int, default=64,
                    help="số hint tối đa gộp vào 1 lượt forward (bộ hint lớn -> OOM nếu để hết)")
    ap.add_argument("--limit", type=int, default=0, help="chỉ chạy N ảnh đầu mỗi thư mục")
    args = ap.parse_args()

    hints = load_hints(args.classes)
    print(f"   {len(hints)} hint (mỗi hint 1 lượt hỏi riêng, không gộp câu như LLMDet)")

    todo = plan_jobs(jobs_from_args(args), args.limit)
    if not todo:
        return

    import torch
    from transformers import Sam3Model, Sam3Processor

    hf_login()
    print(f"   load {args.model} lên {args.device}")
    processor = Sam3Processor.from_pretrained(args.model)
    model = Sam3Model.from_pretrained(args.model, dtype=torch.bfloat16).to(args.device).eval()

    def infer_hint_chunk(imgs, hint_chunk):
        """1 lô ẢNH x 1 lô HINT -> list (theo ảnh) [(hint, score, box)]."""
        flat_images, flat_texts = [], []
        for im in imgs:
            flat_images += [im] * len(hint_chunk)
            flat_texts += hint_chunk

        encoding = processor(images=flat_images, text=flat_texts, return_tensors="pt")
        inputs = {}
        for key in ("pixel_values", "input_ids", "attention_mask"):
            if key not in encoding:
                continue
            value = encoding[key].to(args.device)
            inputs[key] = value.to(model.dtype) if value.is_floating_point() else value

        with torch.inference_mode():
            outputs = model(**inputs)
        results = processor.post_process_object_detection(
            outputs, threshold=args.score_thr,
            target_sizes=[(im.height, im.width) for im in imgs for _ in hint_chunk])

        out = []
        for k in range(len(imgs)):
            part = results[k * len(hint_chunk):(k + 1) * len(hint_chunk)]
            out.append([(hint, float(score), b)
                        for hint, res in zip(hint_chunk, part)
                        for b, score in zip(res["boxes"].float().tolist(), res["scores"].tolist())])
        return out

    def infer(paths):
        """1 lô ẢNH -> list (theo ảnh) [(hint, score, box theo toạ độ ảnh gốc)].

        Chia hint thành từng lô --hint-chunk để giới hạn kích thước batch phẳng
        (ảnh × hint) của 1 lượt forward: 274 hint đã đo thực tế là OOM ở ~34GB,
        trong khi 62 hint chỉ cần ~38GB tổng dù ảnh to hơn — hint càng nhiều
        batch càng phình, bất kể kích thước ảnh.
        """
        imgs = [Image.open(p).convert("RGB") for p in paths]
        per_image = [[] for _ in imgs]
        for i in range(0, len(hints), args.hint_chunk):
            for acc, items in zip(per_image, infer_hint_chunk(imgs, hints[i:i + args.hint_chunk])):
                acc.extend(items)
        return per_image

    writer, n_box, start = JsonlWriter(), 0, 0
    try:
        while start < len(todo):
            out = todo[start][0]
            chunk = [p for o, p in todo[start:start + args.batch_size] if o == out]
            sizes = [Image.open(p).size for p in chunk]
            for path, (w, h), found in zip(chunk, sizes, infer(chunk)):
                boxes, scores, labels = clip_boxes(found, w, h)
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

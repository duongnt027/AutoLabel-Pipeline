#!/usr/bin/env python3
"""CẢ 2 STAGE: thư mục ảnh -> nhãn YOLO.

    stage 1  open-vocabulary detector (LLMDet / SAM3 / WeDetect) vẽ box cho mọi
             HINT của mọi class — ngưỡng thấp, ưu tiên không bỏ sót.
    stage 2  VLX-Seek đọc ảnh + box, chấm từng box theo PROMPT của từng class —
             khâu lọc, đồng thời phân biệt các class dùng chung hint.

    python run_pipeline.py /duong/dan/anh
    python run_pipeline.py /duong/dan/anh --models llm,s3 --min-conf 0.3
    python run_pipeline.py /duong/dan/anh --limit 20     # thử 20 ảnh đầu

Kết quả trong runs/<tên thư mục ảnh>/:
    labels/*.txt            nhãn YOLO cuối cùng (cùng tên với ảnh)
    data.yaml               tên class theo đúng id đã dùng
    proposals_<be>.jsonl    box thô của stage 1
    votes_<be>/*.json       điểm stage 2 cho TỪNG box (đổi ngưỡng không cần chạy lại)

Chạy lại là chạy TIẾP chỗ dở ở cả 2 stage (--fresh để làm lại từ đầu).
Nhiều backend: mỗi backend được stage 2 chấm RIÊNG rồi mới gộp quyết định cuối
theo confidence của VLX — xem autodetect/labels.py::build_labels.
Chạy bằng python3 nào cũng được — mỗi stage tự được gọi bằng python của env
riêng theo .env.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from autodetect import pipeline as P
from autodetect.common import load_classes
from autodetect.labels import build_labels, class_ids, write_data_yaml


def main():
    ap = P.add_common_args(argparse.ArgumentParser(
        description="Auto-label thư mục ảnh: detector (stage 1) + VLX-Seek (stage 2)"))
    ap.add_argument("--min-conf", type=float, default=P.MIN_CONF,
                    help=f"giữ box khi VLX chấm >= mức này (mặc định {P.MIN_CONF})")
    ap.add_argument("--data-yaml", type=Path, default=P.DATA_YAML,
                    help="tuỳ chọn: ép id class theo file này thay vì thứ tự trong classes.json")
    args = ap.parse_args()
    models = P.apply_common_args(args)

    print(f"\n\033[1m=== AUTO LABEL {len(args.images)} thư mục "
          f"(stage 1: {'+'.join(models)} → stage 2: VLX-Seek) ===\033[0m")
    total = 4

    P.step(1, "Kiểm tra env + workspace", total)
    P.check_envs(models, need_stage2=True)
    jobs = P.make_jobs(args.images, args.out, models, args.fresh)

    P.step(2, f"Stage 1 — sinh box proposal ({'+'.join(models)})", total)
    for key in models:
        print(f"   ── backend {key} ──")
        P.run_stage1(key, jobs, args.limit, args.sahi)
    P.summary_stage1(jobs, models)

    P.step(3, "Stage 2 — VLX-Seek chấm điểm từng box theo prompt của class", total)
    for key in models:
        print(f"   ── backend {key} ──")
        P.run_stage2(key, jobs, args.limit)

    P.step(4, f"Dựng nhãn YOLO (min_conf={args.min_conf})", total)
    classes = load_classes(P.CLASSES_JSON)
    ids = class_ids(classes, args.data_yaml)
    for job in jobs:
        n_img, n_box = build_labels([job["votes"][k] for k in models], ids, job["labels"],
                                    args.min_conf, P.NMS_IOU, P.NMS_IOS)
        write_data_yaml(job["root"] / "data.yaml", ids)
        print(f"✅ {job['name']}: {n_box} nhãn trên {n_img} ảnh → {job['labels']}")

    print(f"\n   Đổi ngưỡng mà KHÔNG phải chạy lại model:")
    print(f"   python rebuild_labels.py {jobs[0]['root']} --min-conf 0.5")


if __name__ == "__main__":
    main()

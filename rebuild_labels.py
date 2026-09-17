#!/usr/bin/env python3
"""Dựng lại nhãn từ điểm đã lưu (votes_*/) với ngưỡng khác — KHÔNG chạy lại model.

Stage 2 ghi điểm của MỌI box ra votes_<backend>/, nên đổi min-conf chỉ mất vài
giây thay vì cả lượt chạy GPU.

    python rebuild_labels.py runs/anh_test --min-conf 0.5
    python rebuild_labels.py runs/anh_test --min-conf 0.5 --out runs/anh_test/labels_050
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from autodetect import pipeline as P
from autodetect.common import load_classes
from autodetect.labels import build_labels, class_ids, write_data_yaml


def main():
    ap = argparse.ArgumentParser(description="Dựng lại nhãn từ votes/ với ngưỡng mới")
    ap.add_argument("run_dir", type=Path, help="thư mục workspace, vd runs/anh_test")
    ap.add_argument("--min-conf", type=float, default=P.MIN_CONF)
    ap.add_argument("--out", type=Path, default=None,
                    help="thư mục nhãn (mặc định <run_dir>/labels)")
    ap.add_argument("--classes", type=Path, default=P.CLASSES_JSON)
    ap.add_argument("--data-yaml", type=Path, default=P.DATA_YAML)
    args = ap.parse_args()

    vote_dirs = sorted(args.run_dir.glob("votes_*"))
    if not vote_dirs:
        raise SystemExit(f"❌ không thấy votes_*/ trong {args.run_dir} — chạy run_pipeline.py trước")

    ids = class_ids(load_classes(args.classes), args.data_yaml)
    out = args.out or args.run_dir / "labels"
    n_img, n_box = build_labels(vote_dirs, ids, out, args.min_conf, P.NMS_IOU, P.NMS_IOS)
    write_data_yaml(args.run_dir / "data.yaml", ids)
    print(f"✅ {n_box} nhãn trên {n_img} ảnh (min_conf={args.min_conf}, "
          f"{', '.join(d.name for d in vote_dirs)}) → {out}")


if __name__ == "__main__":
    main()

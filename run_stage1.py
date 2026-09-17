#!/usr/bin/env python3
"""CHỈ STAGE 1: sinh box proposal cho một (hay nhiều) thư mục ảnh.

Dùng khi muốn xem/đánh giá riêng chất lượng proposal, hoặc chạy stage 1 trên máy
này rồi mang jsonl sang máy khác chấm stage 2.

    python run_stage1.py /duong/dan/anh
    python run_stage1.py /duong/dan/anh --models llm,s3 --limit 20
    python run_stage1.py /duong/dan/anh --sahi          # bắt vật nhỏ, chậm ~14 lần

Kết quả: runs/<tên thư mục ảnh>/proposals_<backend>.jsonl
(1 dòng/ảnh — xem hợp đồng jsonl ở autodetect/common.py)

Chạy lại là chạy TIẾP chỗ dở: ảnh đã có trong jsonl thì bỏ qua (--fresh để làm lại).
Chạy bằng python3 nào cũng được — mỗi backend tự được gọi bằng python của env
riêng theo .env.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from autodetect import pipeline as P


def main():
    ap = P.add_common_args(argparse.ArgumentParser(
        description="Stage 1: sinh box proposal cho thư mục ảnh"))
    args = ap.parse_args()
    models = P.apply_common_args(args)

    print(f"\n\033[1m=== STAGE 1: {'+'.join(models)} trên {len(args.images)} thư mục ===\033[0m")
    total = 2

    P.step(1, "Kiểm tra env + workspace", total)
    P.check_envs(models, need_stage2=False)
    jobs = P.make_jobs(args.images, args.out, models, args.fresh)

    P.step(2, f"Sinh box proposal ({'+'.join(models)})", total)
    for key in models:
        print(f"   ── backend {key} ──")
        P.run_stage1(key, jobs, args.limit, args.sahi)

    print()
    P.summary_stage1(jobs, models)
    print(f"\n   Chấm điểm + ra nhãn: python run_pipeline.py {args.images[0]} "
          f"--models {','.join(models)}   (stage 1 đã xong sẽ được dùng lại)")


if __name__ == "__main__":
    main()

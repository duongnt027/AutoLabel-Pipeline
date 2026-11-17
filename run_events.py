#!/usr/bin/env python3
"""CHẾ ĐỘ EVENT: thư mục ảnh -> nhãn YOLO theo HÀNH VI (bán hàng rong, rửa xe...).

    stage 1  detector vẽ box cho mọi hint của các đối tượng chủ thể   (như thường)
    stage 2  VLX chấm từng box theo prompt của class                (như thường)
    stage 3  gộp box cùng class chồng nhau -> gộp đối tượng 1 hành vi thành box
             nhóm -> VLX mô tả box nhóm + chấm theo caption của từng event
    nhãn     so caption sinh ra với caption gốc + requires (chỉ thứ nhìn thấy)
             -> event hoặc loại (khớp caption reject)

    python run_events.py /duong/dan/anh
    python run_events.py /duong/dan/anh --limit 20
    python run_events.py /duong/dan/anh --rebuild --min-conf 0.5   # không chạy model

Bộ class + event: classes/street_events.json (hoặc EVENT_CLASSES / --classes).

Kết quả trong runs/<tên thư mục ảnh>/:
    labels_events/*.txt     nhãn YOLO theo event (id theo thứ tự trong "events")
    data_events.yaml        tên event theo id
    events_summary.jsonl    mỗi box nhóm 1 dòng: thành phần, caption, event, lý do loại
    events_votes/*.json     điểm thô stage 3 (đổi ngưỡng không cần chạy lại model)
    labels_actors/*.txt     nhãn đối tượng chủ thể (như run_pipeline.py)
    proposals_<be>.jsonl, votes_<be>/   như run_pipeline.py
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from autodetect import events as E
from autodetect import pipeline as P
from autodetect.common import env_path, load_events
from autodetect.labels import build_labels, class_ids, write_data_yaml


def main():
    ap = P.add_common_args(argparse.ArgumentParser(
        description="Phát hiện hành vi theo mô tả: detection -> gộp nhóm -> VLX caption -> so khớp"))
    ap.add_argument("--min-conf", type=float, default=None,
                    help="giữ box nhóm khi điểm event >= mức này (mặc định EVENT_MIN_CONF)")
    ap.add_argument("--actor-min-conf", type=float, default=None,
                    help="ngưỡng giữ đối tượng ở nhãn labels_actors/ (mặc định AUTOLABEL_MIN_CONF)")
    ap.add_argument("--rebuild", action="store_true",
                    help="chỉ dựng lại nhãn từ events_votes/ đã có, không chạy model")
    args = ap.parse_args()
    if args.classes is None:
        args.classes = env_path("EVENT_CLASSES", "classes/street_events.json")
    models = P.apply_common_args(args)
    classes, events, reject = load_events(P.CLASSES_JSON)
    p = E.params()
    if args.min_conf is not None:
        p["min_conf"] = args.min_conf

    total = 2 if args.rebuild else 5
    print(f"\n\033[1m=== EVENT MODE {len(args.images)} thư mục, {len(classes)} đối tượng, "
          f"{len(events)} event ({P.CLASSES_JSON.name}) ===\033[0m")
    P.step(1, "Workspace", total)
    if not args.rebuild:
        P.check_envs(models, need_stage2=True)
    jobs = P.make_jobs(args.images, args.out, models, args.fresh and not args.rebuild)
    for job in jobs:
        job["events_votes"] = job["root"] / "events_votes"
        job["event_labels"] = job["root"] / "labels_events"
        job["actor_labels"] = job["root"] / "labels_actors"

    if not args.rebuild:
        P.step(2, f"Stage 1 — box proposal đối tượng ({'+'.join(models)})", total)
        for key in models:
            print(f"   ── backend {key} ──")
            P.run_stage1(key, jobs, args.limit, args.sahi)
        P.summary_stage1(jobs, models)

        P.step(3, "Stage 2 — VLX chấm từng box theo class (detection thường)", total)
        for key in models:
            print(f"   ── backend {key} ──")
            P.run_stage2(key, jobs, args.limit, describe=False)

        P.step(4, "Stage 3 — gộp nhóm + VLX mô tả/chấm box nhóm theo event", total)
        P.run_stage3(jobs, models, args.limit)

    P.step(total, f"Dựng nhãn (event min_conf={p['min_conf']}, "
                  f"w_vlx={p['w_vlx']}, w_text={p['w_text']})", total)
    ids = class_ids(classes)
    actor_conf = args.actor_min_conf if args.actor_min_conf is not None else P.MIN_CONF
    for job in jobs:
        n_img, n_box = build_labels([job["votes"][k] for k in models], ids, job["actor_labels"],
                                    actor_conf, P.NMS_IOU, P.NMS_IOS)
        write_data_yaml(job["root"] / "data_actors.yaml", ids)
        print(f"   {job['name']}: {n_box} nhãn đối tượng trên {n_img} ảnh → {job['actor_labels']}")
        if not job["events_votes"].is_dir():
            print(f"   ⚠️  {job['name']}: chưa có {job['events_votes']} — chạy không --rebuild trước")
            continue
        n_img, n_lbl = E.build_event_labels(job["events_votes"], events, reject,
                                            job["event_labels"], p,
                                            job["root"] / "events_summary.jsonl")
        write_data_yaml(job["root"] / "data_events.yaml", E.event_ids(events))
        print(f"✅ {job['name']}: {n_lbl} nhãn event trên {n_img} ảnh → {job['event_labels']}")
        print(f"   chi tiết từng box nhóm (caption, lý do loại): {job['root'] / 'events_summary.jsonl'}")

    if not args.rebuild:
        print(f"\n   Đổi ngưỡng / trọng số / keywords mà KHÔNG chạy lại model:")
        print(f"   python run_events.py {' '.join(str(i) for i in args.images)} --rebuild --min-conf 0.5")


if __name__ == "__main__":
    main()

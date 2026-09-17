#!/usr/bin/env python3
"""Stage 2: VLX-Seek chấm điểm từng box proposal của stage 1 theo prompt của class.

Với mỗi ảnh và mỗi NHÓM CLASS (gom theo bộ hint), gửi VLX ĐÚNG MỘT lượt gọi
trong đó mọi prompt của các class trong nhóm là các category. Model gán mỗi box
nhiều nhất một category; độ tự tin lấy từ logits lúc sinh token <objN>: VLX
không trả ra confidence, nhưng <obj0>..<obj99> là token đơn trong vocab nên
softmax trên nhóm token đó chính là xác suất model gán cho từng box.

Vì sao 1 lượt gọi thay vì bầu từng prompt: đo trên 1 bộ 250 box GT, gộp 10
prompt vào 1 lượt cho ra ĐÚNG cùng tập box với phép hợp của 10 lượt gọi riêng
mà nhanh 2,2-2,5 lần. Chấm đúng 235/250 box (94%), riêng nhóm conf >= 0.5 đúng
197/199 (99%).

File này CHỈ GHI ĐIỂM ra votes/<stem>.json, KHÔNG cắt ngưỡng và không ghi nhãn:
việc đó là của autodetect/labels.py (run_pipeline.py gọi sau, hoặc chạy lại
rebuild_labels.py để đổi ngưỡng trong vài giây mà không phải chạy lại model).

CHẠY BẰNG PYTHON CỦA ENV `vlx` (torch 2.8 + transformers 5.13).

(Tên file KHÔNG được là vlx_seek.py: python đặt thư mục chứa script lên đầu
sys.path, file cùng tên sẽ che mất package `vlx_seek` của repo VLX-Seek.)

    $VLX_PYTHON stage2/score_vlx.py --manifest jobs.json --shard 0/2
"""

import argparse
import json
import os
import sys
import time
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autodetect.common import (IMG_EXTS, env_float, env_int, env_path, env_str,  # noqa: E402
                               iter_jsonl, load_classes, load_env)

load_env()

# GPU VẬT LÝ chạy VLX. PHẢI ghim TRƯỚC mọi import đụng torch: nạp thẳng "cuda:1"
# thì VLX lỗi lẫn device (buffer của vision tower phụ vẫn nằm ở cuda:0).
# VLX_GPU do run_pipeline.py đặt cho từng shard; chạy tay thì lấy GPU đầu tiên
# trong VLX_GPUS của .env.
os.environ["CUDA_VISIBLE_DEVICES"] = (
    os.getenv("VLX_GPU") or (env_str("VLX_GPUS", "0")).split(",")[0].strip())

VLX_REPO = env_path("VLX_REPO", "third_party/VLX-Seek")
VLX_CHECKPOINT = env_str("VLX_CHECKPOINT", "omlab/VLX-Seek-1.5-10B")
VLX_MAX_NEW_TOKENS = env_int("VLX_MAX_NEW_TOKENS", 256)
VLX_TASK = "detection"
# Số box tối đa mỗi nhóm class đưa VLX xem 1 lượt: 256 token đầu ra chỉ đủ in ra
# khoảng 40-50 <objN>, quá số này thì câu trả lời bị cắt cụt, box cuối mất điểm oan.
TOPK = env_int("AUTOLABEL_TOPK", 40)
NMS_IOU = env_float("AUTOLABEL_NMS_IOU", 0.6)
NMS_IOS = env_float("AUTOLABEL_NMS_IOS", 0.9)

from PIL import Image  # noqa: E402

from autodetect.labels import suppress  # noqa: E402


def load_worker():
    """Nạp VLX-Seek. Luôn là cuda:0 vì đã ghim CUDA_VISIBLE_DEVICES ở trên."""
    sys.path.insert(0, str(VLX_REPO))
    from vlx_seek_worker import VLXSeekWorker

    print(f"   load {VLX_CHECKPOINT} lên GPU vật lý {os.environ['CUDA_VISIBLE_DEVICES']}")
    worker = VLXSeekWorker(VLX_CHECKPOINT, device="cuda:0")

    # VLX không trả ra độ tự tin -> vá generate để giữ lại logits từng bước.
    worker.last = {}
    _generate = worker.model.generate

    def generate_keep_scores(**kwargs):
        kwargs.update(output_scores=True, return_dict_in_generate=True)
        out = _generate(**kwargs)
        # Chỉ giữ logits của nhóm token <objN> (~100 số/bước) thay vì cả vocab
        # (~250k số/bước × 256 bước = 256MB VRAM bị giữ giữa các lượt gọi).
        # Softmax trên nhóm này bằng đúng softmax cả vocab rồi chuẩn hoá lại.
        worker.last = {
            "scores": [step[0, worker.obj_ids].float().cpu() for step in out.scores],
            "seq": out.sequences,
            "n_in": kwargs["inputs"].shape[1],
        }
        return out.sequences

    # Chỉ lấy những <objN> THẬT SỰ có trong vocab: checkpoint này có <obj0>..<obj99>,
    # hỏi <obj100> trở đi thì convert_tokens_to_ids trả None và làm hỏng phép lập
    # chỉ mục trong generate_keep_scores.
    vocab = worker.tokenizer.get_vocab()
    worker.obj_ids = []
    while f"<obj{len(worker.obj_ids)}>" in vocab:
        worker.obj_ids.append(vocab[f"<obj{len(worker.obj_ids)}>"])
    print(f"   model tham chiếu được tối đa {len(worker.obj_ids)} box mỗi lượt gọi")
    worker.model.generate = generate_keep_scores
    return worker


def score_boxes(worker, image, boxes, subset) -> dict:
    """1 lượt gọi cho cả nhóm class -> {class: [confidence từng box]}.

    Category = TOÀN BỘ prompt của mọi class trong nhóm, gửi trong cùng một lượt:
    model coi chúng loại trừ nhau nên tự chọn mô tả khớp nhất cho từng box, tức
    tự phân biệt các class dùng chung hint (helmet / no helmet, car / bus / truck).
    Box khớp bất kỳ prompt nào của một class thì thuộc class đó.
    """
    import torch

    cats = [(text, cls_name)                     # [(text category, tên class)]
            for cls_name, spec in subset.items()
            for text in spec["prompts"]]
    cat_class = {text.strip().lower(): cls_name for text, cls_name in cats}

    worker.run_task(image, VLX_TASK, [text for text, _ in cats], lang="en",
                    bbox_list=boxes, max_new_tokens=VLX_MAX_NEW_TOKENS,
                    temperature=0.0, top_p=1.0)

    # model sắp xếp box lại bên trong (mm_bbox_order_mode=raster) -> map ngược
    _, sorted_to_caller = worker._order_boxes(worker._validate_boxes(boxes, image))
    ids = worker.obj_ids[:len(boxes)]
    conf = {c: [0.0] * len(boxes) for c in subset}
    tok = worker.tokenizer
    cur, buf = None, ""
    for token, step in zip(worker.last["seq"][0, worker.last["n_in"]:].tolist(),
                           worker.last["scores"]):
        buf += tok.decode([token])
        if "</ground>" in buf:                   # model đang liệt kê box cho category nào
            cur = cat_class.get(buf.split("<ground>")[-1].split("</ground>")[0].strip().lower())
            buf = ""
        if cur and token in ids:
            prob = torch.softmax(step[:len(ids)], -1).tolist()
            row = conf[cur]
            for si, p in enumerate(prob):
                ci = sorted_to_caller[si]
                if p > row[ci]:
                    row[ci] = p
    return conf


def score_image(worker, rec, image, classes) -> tuple:
    """Chấm mọi proposal của 1 ảnh -> ([{"box","score","conf"}], số lượt gọi).

    Gom class theo BỘ HINT: class cùng bộ hint (helmet / no helmet) dùng chung
    danh sách proposal và chung 1 lượt gọi. Box của một class được gộp từ MỌI
    hint của nó rồi mới bỏ trùng — một chiếc xe bị 9 hint ("car", "sedan",
    "SUV"...) cùng vẽ thì chỉ còn 1 box, nếu không danh sách proposal phình lên
    và câu trả lời bị cắt cụt ở 256 token.
    """
    groups = {}
    for hint, score, box in zip(rec["hints"], rec["scores"], rec["boxes"]):
        groups.setdefault(hint, []).append((score, box))

    by_hints = {}
    for cls_name, spec in classes.items():
        by_hints.setdefault(tuple(spec["hints"]), []).append(cls_name)

    records = {}
    # Không đưa nhiều box hơn số token <objN> model có: box thừa model không có
    # cách nào chỉ tới.
    cap = min(TOPK, len(worker.obj_ids))
    n_calls = 0
    for hints, names in by_hints.items():
        items = suppress([it for h in hints for it in groups.get(h, ())], NMS_IOU, NMS_IOS)[:cap]
        if not items:
            continue
        conf = score_boxes(worker, image, [b for _, b in items], {c: classes[c] for c in names})
        n_calls += 1
        for cls_name in names:
            for (score, box), p in zip(items, conf[cls_name]):
                key = tuple(round(v, 2) for v in box)
                rec_box = records.setdefault(key, {"box": box, "score": score, "conf": {}})
                rec_box["conf"][cls_name] = p
    return list(records.values()), n_calls


def load_jobs(args) -> list:
    """[{"name","preds","images","votes"}] từ --manifest hoặc các cờ lẻ."""
    if args.manifest:
        jobs = json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    elif args.preds and args.images and args.votes:
        jobs = [{"name": Path(args.images).name, "preds": str(args.preds),
                 "images": str(args.images), "votes": str(args.votes)}]
    else:
        raise SystemExit("❌ cần --manifest, hoặc đủ cả --preds --images --votes")
    for job in jobs:
        job.setdefault("name", Path(job["images"]).name)
    return jobs


def main():
    ap = argparse.ArgumentParser(description="VLX-Seek chấm điểm box proposal của stage 1")
    ap.add_argument("--manifest", type=Path,
                    help="JSON [{name, preds, images, votes}] để chạy nhiều thư mục 1 lần nạp model")
    ap.add_argument("--preds", type=Path, help="file jsonl proposal (khi chạy 1 thư mục)")
    ap.add_argument("--images", type=Path)
    ap.add_argument("--votes", type=Path, help="thư mục ghi điểm")
    ap.add_argument("--classes", type=Path,
                    default=env_path("AUTOLABEL_CLASSES", "classes/classes.json"))
    ap.add_argument("--shard", default="", help='"0/2" = phần 0 trong 2 tiến trình song song')
    ap.add_argument("--limit", type=int, default=0, help="chỉ chạy N ảnh đầu mỗi thư mục")
    args = ap.parse_args()

    classes = load_classes(args.classes)
    jobs = load_jobs(args)

    # LẬP KẾ HOẠCH TRƯỚC, CHỈ GIỮ TÊN ẢNH (không giữ box): bản ghi proposal của
    # ảnh dày vật thể nặng ~0.9MB, giữ cả bộ là hàng GB mỗi tiến trình. Box được
    # đọc lại theo từng dòng ở vòng chạy bên dưới.
    # Trải phẳng (job, stem) của MỌI thư mục rồi mới chia shard: các GPU luôn đầy
    # việc, kể cả khi các thư mục to nhỏ lệch nhau.
    plan = []
    for k, job in enumerate(jobs):
        Path(job["votes"]).mkdir(parents=True, exist_ok=True)
        for i, rec in enumerate(iter_jsonl(job["preds"])):
            if args.limit and i >= args.limit:
                break
            plan.append((k, rec["stem"]))
    if args.shard:
        i, n = (int(v) for v in args.shard.split("/"))
        plan = plan[i::n]
        print(f"   shard {i}/{n}: {len(plan)} ảnh")

    todo = {}                       # {chỉ số job: set stem cần chạy}
    for k, stem in plan:
        if not Path(jobs[k]["votes"], f"{stem}.json").exists():
            todo.setdefault(k, set()).add(stem)
    n_todo = sum(len(v) for v in todo.values())
    print(f"   {len(plan)} ảnh của {len(jobs)} thư mục, đã có {len(plan) - n_todo}, "
          f"cần chạy {n_todo}")
    if not n_todo:
        return

    worker = load_worker()
    t0 = t_prev = time.time()
    recent = deque(maxlen=20)   # thời gian 20 ảnh gần nhất -> ETA theo tốc độ HIỆN
                                # TẠI, không bị vài ảnh đầu (nhiễu nhất) kéo lệch.
    n_in = n_calls = i = 0
    for k, want in todo.items():
        job = jobs[k]
        # Đọc lại jsonl theo dòng: mỗi lúc chỉ 1 bản ghi nằm trong RAM.
        for rec in iter_jsonl(job["preds"]):
            if rec["stem"] not in want:
                continue
            i += 1
            path = next((p for p in Path(job["images"]).glob(f"{rec['stem']}.*")
                         if p.suffix.lower() in IMG_EXTS), None)
            if path is None:
                print(f"   [{i}/{n_todo}] {rec['stem']}: ❌ không tìm thấy ảnh, bỏ qua")
                continue
            image = Image.open(path).convert("RGB")
            records, calls = score_image(worker, rec, image, classes)
            Path(job["votes"], f"{rec['stem']}.json").write_text(json.dumps(
                {"stem": rec["stem"], "w": rec["w"], "h": rec["h"], "boxes": records}),
                encoding="utf-8")
            n_in += len(rec["boxes"])
            n_calls += calls
            now = time.time()
            recent.append(now - t_prev)
            t_prev = now
            eta = (sum(recent) / len(recent)) * (n_todo - i) / 60
            print(f"   [{i}/{n_todo}] {job['name']}/{rec['stem']}: {len(rec['boxes'])} box "
                  f"→ {len(records)} box được chấm ({calls} lượt gọi, còn ~{eta:.0f} phút)")

    print(f"\n   ✅ {n_todo} ảnh, {n_calls} lượt gọi VLX, {n_in} box proposal "
          f"({(time.time() - t0) / 60:.1f} phút)")
    for job in jobs:
        print(f"   📁 {job['name']}: điểm ở {job['votes']}")


if __name__ == "__main__":
    main()

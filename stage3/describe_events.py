#!/usr/bin/env python3
"""Stage 3 (chế độ EVENT): gộp đối tượng thành box nhóm, VLX mô tả + chấm theo event.

Mỗi ảnh:
  1. đọc votes/ của stage 2 (mọi backend) -> đối tượng (box về tay class tự tin nhất)
  2. gộp box CÙNG class chồng nhau           -> autodetect/events.py::merge_same_class
  3. gộp đối tượng của 1 hành vi thành nhóm  -> autodetect/events.py::group_objects
  4. MÔ TẢ: brief_region_caption cho box nhóm (đã nới DESC_PAD lấy ngữ cảnh)
     -> caption tự do, KHÔNG gợi ý event (mẫu task của VLX không nhận câu hỏi tự do,
     và câu hỏi có hướng thì model chiều theo, box nào cũng "khớp")
  5. CHẤM: 1 lượt detection trên MỌI box nhóm, category = caption của các event
     (+ caption reject) -> VLX(event) cho từng nhóm; event cạnh tranh nhau như stage 2
  6. SO KHỚP: caption sinh ra ở bước 4 so với caption gốc + keywords của từng event
     -> text(event). Mặc định so theo từ (stdlib); đặt EVENT_TEXT_MODEL (vd
     sentence-transformers/all-MiniLM-L6-v2) để so bằng embedding.
Ghi events_votes/<stem>.json (điểm thô). Quyết định cuối + nhãn: autodetect/events.py.

CHẠY BẰNG PYTHON CỦA ENV `vlx` (cùng env với stage 2).

    $VLX_PYTHON stage3/describe_events.py --manifest jobs.json --classes classes/street_events.json
"""

import argparse
import hashlib
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "stage2"))

# score_vlx ghim CUDA_VISIBLE_DEVICES lúc import -> phải import TRƯỚC mọi thứ đụng torch
import score_vlx as sv  # noqa: E402

from autodetect import events as E  # noqa: E402
from autodetect.common import IMG_EXTS, env_int, env_str, load_events  # noqa: E402

# Caption cho box nhóm cần dài hơn chế độ --describe (ở đó caption bị chép lại
# làm category nên phải ngắn); ở đây caption chỉ đem đi so khớp.
CAPTION_MAX_WORDS = env_int("EVENT_CAPTION_MAX_WORDS", 30)
TEXT_MODEL = env_str("EVENT_TEXT_MODEL", "")


def caption(worker, image, boxes) -> list:
    """brief_region_caption cho từng box (chia lô DESC_BATCH) -> [caption đã chuẩn hoá | ""].

    Viết riêng ở đây, KHÔNG dùng score_vlx.caption: mẫu task cố định là
    "Give a brief description of {}." và {} PHẢI là các tham chiếu
    <obj0><obj1>... (như VLXSeekWorker.describe_region) — nhét câu hỏi tự do vào
    thì prompt thành vô nghĩa. Box được xếp SẴN theo thứ tự raster model dùng
    bên trong, nên <objN> ở câu hỏi lẫn câu trả lời đều là chỉ số trong `ordered`.
    """
    out = [""] * len(boxes)
    for s in range(0, len(boxes), sv.DESC_BATCH):
        chunk = boxes[s:s + sv.DESC_BATCH]
        try:
            ordered, sorted_to_caller = worker._order_boxes(worker._validate_boxes(chunk, image))
            ordered = [list(b) for b in ordered]
        except Exception:
            ordered, sorted_to_caller = chunk, list(range(len(chunk)))
        refs = "".join(f"<obj{i}>" for i in range(len(ordered)))
        res = worker.run_task(image, "brief_region_caption", refs, lang="en",
                              bbox_list=ordered, max_new_tokens=sv.DESC_MAX_NEW_TOKENS,
                              temperature=0.0, top_p=1.0)
        answer = res["answer"] if isinstance(res, dict) else str(res)
        marks = list(re.finditer(r"<obj(\d+)>", answer))
        if not marks and len(chunk) == 1:        # 1 box: model có thể trả thẳng câu mô tả
            out[s] = sv.norm_text(answer)
        for i, m in enumerate(marks):
            n = int(m.group(1))
            if n >= len(sorted_to_caller):
                continue
            end = marks[i + 1].start() if i + 1 < len(marks) else len(answer)
            ci = s + sorted_to_caller[n]
            if not out[ci]:
                out[ci] = sv.norm_text(answer[m.end():end])
    return [" ".join(c.split()[:CAPTION_MAX_WORDS]) for c in out]


def make_sim_fn(events: dict, reject: dict):
    """(tên model text, hàm sim(caption, refs)) — embedding nếu có EVENT_TEXT_MODEL."""
    if not TEXT_MODEL:
        return "lexical", None
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(TEXT_MODEL, device="cuda:0")
    refs = [c for ev in events.values() for c in ev["captions"]] + reject["captions"]
    cache = dict(zip(refs, model.encode(refs, normalize_embeddings=True)))
    print(f"   so khớp caption bằng embedding: {TEXT_MODEL}")

    def sim(caption, texts):
        q = model.encode([caption], normalize_embeddings=True)[0]
        return max((max(0.0, float(q @ cache[t])) for t in texts), default=0.0)

    return TEXT_MODEL, sim


def signature(events: dict, reject: dict, p: dict, roles: dict) -> str:
    """Đổi caption event / tham số gộp nhóm thì ảnh phải chấm lại -> lưu kèm chữ ký.

    KHÔNG gồm keywords, trọng số, ngưỡng cuối: những thứ đó --rebuild áp lại được.
    """
    key = {"events": {e: ev["captions"] for e, ev in events.items()},
           "reject": reject["captions"], "roles": roles, "text_model": TEXT_MODEL,
           "caption_words": CAPTION_MAX_WORDS, "desc_pad": sv.DESC_PAD,
           **{k: p[k] for k in ("actor_min_conf", "merge_pad", "group_pad", "group_max_area")}}
    return hashlib.sha1(json.dumps(key, sort_keys=True).encode()).hexdigest()[:12]


def done_sig(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("sig")
    except (OSError, json.JSONDecodeError):
        return None


def score_groups(worker, image, groups, events, reject, sim_fn) -> tuple:
    """Bước 4-6 cho mọi nhóm của 1 ảnh. Trả số lượt gọi VLX."""
    w, h = image.size
    caps = caption(worker, image, [sv.pad_box(g["box"], w, h, sv.DESC_PAD) for g in groups])
    calls = -(-len(groups) // sv.DESC_BATCH)

    # Chỉ đưa vào category những event có ít nhất 1 nhóm thoả `requires`: ít
    # category hơn thì câu trả lời ngắn hơn, không bị cắt cụt ở max_new_tokens.
    present = [{m["class"] for m in g["members"]} for g in groups]
    cand = [e for e, ev in events.items() if any(E.requires_ok(ev, pr) for pr in present)]
    cats, seen = [], set()
    for e in cand:
        for c in events[e]["captions"]:
            t = sv.norm_text(c)
            if t and t not in seen:          # 2 event trùng 1 câu: giữ ở event khai trước
                seen.add(t)
                cats.append((t, e))
    for c in reject["captions"]:
        t = sv.norm_text(c)
        if t and t not in seen:
            seen.add(t)
            cats.append((t, E.REJECT))

    conf, conf_text = {}, {}
    if cand:
        conf, conf_text = sv.ground(worker, image, [g["box"] for g in groups], cats,
                                    sv.DESC_MAX_NEW_TOKENS)
        calls += 1
    for i, (g, cap) in enumerate(zip(groups, caps)):
        g["caption"] = cap
        g["vlx"] = {e: round(v[i], 4) for e, v in conf.items() if v[i] > 0}
        best = max(conf_text.items(), key=lambda kv: kv[1][i], default=(None, [0.0] * len(groups)))
        if best[0] and best[1][i] > 0:
            g["vlx_caption"] = best[0]           # caption event VLX đã gán cho nhóm
        g["text"] = {e: round(v, 4) for e, v in
                     E.text_scores(cap, events, reject, sim_fn).items() if v > 0}
    return calls


def main():
    ap = argparse.ArgumentParser(description="Chế độ event: gộp nhóm + VLX mô tả/chấm box nhóm")
    ap.add_argument("--manifest", type=Path, required=True,
                    help='JSON [{name, images, votes: [thư mục votes...], events_votes}]')
    ap.add_argument("--classes", type=Path, required=True, help="file event (street_events.json)")
    ap.add_argument("--shard", default="")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    classes, events, reject = load_events(args.classes)
    roles = {c: spec.get("role", "anchor") for c, spec in classes.items()}
    p = E.params()
    sig = signature(events, reject, p, roles)
    jobs = json.loads(args.manifest.read_text(encoding="utf-8"))

    plan = []
    for k, job in enumerate(jobs):
        Path(job["events_votes"]).mkdir(parents=True, exist_ok=True)
        stems = sorted({f.stem for d in job["votes"] if Path(d).is_dir()
                        for f in Path(d).glob("*.json")})
        plan += [(k, s) for s in (stems[:args.limit] if args.limit else stems)]
    if args.shard:
        i, n = (int(v) for v in args.shard.split("/"))
        plan = plan[i::n]
    todo = [(k, s) for k, s in plan
            if done_sig(Path(jobs[k]["events_votes"], f"{s}.json")) != sig]
    print(f"   {len(plan)} ảnh, đã có {len(plan) - len(todo)}, cần chạy {len(todo)} (sig {sig})")
    if not todo:
        return

    worker = sv.load_worker()
    text_model, sim_fn = make_sim_fn(events, reject)
    cap = min(sv.TOPK, len(worker.obj_ids))
    t0, n_calls = time.time(), 0
    for n, (k, stem) in enumerate(todo, 1):
        job = jobs[k]
        img_path = next((q for q in Path(job["images"]).glob(f"{stem}.*")
                         if q.suffix.lower() in IMG_EXTS), None)
        vote_files = [Path(d, f"{stem}.json") for d in job["votes"]
                      if Path(d, f"{stem}.json").exists()]
        if img_path is None or not vote_files:
            print(f"   [{n}/{len(todo)}] {stem}: ❌ thiếu ảnh hoặc votes, bỏ qua")
            continue
        image = sv.Image.open(img_path).convert("RGB")
        w, h = image.size

        dets = E.detections(vote_files, classes, p["actor_min_conf"])
        objs = E.merge_same_class(dets, p["merge_pad"])
        groups = E.group_objects(objs, roles, w, h, p["group_pad"], p["group_max_area"])
        # nhóm có anchor tự tin nhất trước; quá số <objN> model có thì cắt bớt
        groups.sort(key=lambda g: -max(objs[m]["conf"] for m in g["members"]))
        dropped = max(0, len(groups) - cap)
        groups = [{"box": [round(v, 1) for v in g["box"]],
                   "members": [{"class": objs[m]["class"], "conf": round(objs[m]["conf"], 4),
                                "n": objs[m]["n"], "box": [round(v, 1) for v in objs[m]["box"]]}
                               for m in g["members"]]}
                  for g in groups[:cap]]
        calls = score_groups(worker, image, groups, events, reject, sim_fn) if groups else 0
        n_calls += calls

        out = {"stem": stem, "w": w, "h": h, "sig": sig, "text_model": text_model,
               "n_detections": len(dets), "n_objects": len(objs), "dropped_groups": dropped,
               "groups": groups}
        Path(job["events_votes"], f"{stem}.json").write_text(
            json.dumps(out, ensure_ascii=False), encoding="utf-8")
        print(f"   [{n}/{len(todo)}] {job['name']}/{stem}: {len(dets)} box → {len(objs)} đối tượng "
              f"→ {len(groups)} nhóm ({calls} lượt gọi)")
    print(f"\n   ✅ {len(todo)} ảnh, {n_calls} lượt gọi VLX ({(time.time() - t0) / 60:.1f} phút)")


if __name__ == "__main__":
    main()

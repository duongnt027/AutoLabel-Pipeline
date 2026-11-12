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
    $VLX_PYTHON stage2/score_vlx.py --manifest jobs.json --describe

--describe (hoặc AUTOLABEL_DESCRIBE=1): prompt KHÔNG lấy cố định từ "prompts" mà
VLX tự sinh theo TỪNG ẢNH — xem describe_group(). Mỗi nhóm class tốn 2-3 lượt
gọi thay vì 1. File votes ghi kèm "mode" nên đổi chế độ thì ảnh được chấm lại,
không lẫn điểm của 2 chế độ.
"""

import argparse
import json
import os
import re
import sys
import time
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autodetect.common import (IMG_EXTS, env_bool, env_float, env_int, env_path,  # noqa: E402
                               env_str, iter_jsonl, load_classes, load_env)

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

# ── Chế độ --describe: VLX tự sinh prompt theo từng ảnh ─────────────────────
DESCRIBE = env_bool("AUTOLABEL_DESCRIBE", False)
# Task + câu hỏi mô tả vùng. {hint} = hint đầu tiên của nhóm class ("head",
# "car"...): câu hỏi trung lập, KHÔNG được nhắc tên class.
DESC_TASK = env_str("VLX_DESC_TASK", "brief_region_caption")
DESC_TEXT = env_str("VLX_DESC_TEXT", "Briefly describe this {hint}.")
DESC_PAD = env_float("VLX_DESC_PAD", 0.25)            # nới box mỗi phía 25% khi mô tả
DESC_BATCH = env_int("VLX_DESC_BATCH", 8)             # số box mỗi lượt mô tả
DESC_PER_CLASS = env_int("VLX_DESC_PER_CLASS", 6)     # mô tả tối đa N box/class/ảnh (0 = mọi box)
DESC_MIN_ALIAS = env_float("VLX_DESC_MIN_ALIAS", 0.3) # box phải khớp alias >= mức này mới được mô tả
DESC_MAX_WORDS = env_int("VLX_DESC_MAX_WORDS", 12)    # cắt mô tả còn N từ
DESC_MAX_PROMPTS = env_int("VLX_DESC_MAX_PROMPTS", 6) # giữ tối đa N mô tả/class sau lọc trùng
DESC_DEDUP = env_float("VLX_DESC_DEDUP", 0.8)         # Jaccard theo từ >= mức này = trùng
# Lượt verify có nhiều category hơn, mỗi category bị chép lại vào đầu ra -> cần
# nhiều token hơn VLX_MAX_NEW_TOKENS.
DESC_MAX_NEW_TOKENS = env_int("VLX_DESC_MAX_NEW_TOKENS", 512)

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


def norm_text(text: str) -> str:
    """Chuẩn hoá 1 câu prompt/mô tả để so trùng VÀ để làm khoá tra ngược.

    Câu gửi cho VLX luôn là bản đã chuẩn hoá, nên chữ model lặp lại trong
    <ground>...</ground> đem chuẩn hoá lại vẫn khớp đúng khoá.
    """
    text = re.sub(r"<[^>]+>", " ", text.lower())
    text = re.sub(r"[^a-z0-9'\- ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip(" -'")


def ground(worker, image, boxes, cats, max_new_tokens) -> tuple:
    """1 lượt gọi detection: mỗi box được gán cho NHIỀU NHẤT 1 category.

    cats = [(text, tên class)]. Trả (conf, conf_text):
        conf      {class: [confidence từng box]}
        conf_text {text:  [confidence từng box]}  -> biết prompt nào đã khớp
    Mọi category gửi cùng 1 lượt nên model coi chúng LOẠI TRỪ NHAU và tự chọn
    câu khớp nhất cho từng box — đó là thứ phân biệt helmet / no helmet.
    """
    import torch

    cat_class = {norm_text(text): cls_name for text, cls_name in cats}
    conf = {c: [0.0] * len(boxes) for c in dict.fromkeys(c for _, c in cats)}
    conf_text = {t: [0.0] * len(boxes) for t in cat_class}

    worker.run_task(image, VLX_TASK, list(cat_class), lang="en", bbox_list=boxes,
                    max_new_tokens=max_new_tokens, temperature=0.0, top_p=1.0)

    # model sắp xếp box lại bên trong (mm_bbox_order_mode=raster) -> map ngược
    _, sorted_to_caller = worker._order_boxes(worker._validate_boxes(boxes, image))
    ids = worker.obj_ids[:len(boxes)]
    tok = worker.tokenizer
    cur, buf = None, ""
    for token, step in zip(worker.last["seq"][0, worker.last["n_in"]:].tolist(),
                           worker.last["scores"]):
        buf += tok.decode([token])
        if "</ground>" in buf:                   # model đang liệt kê box cho category nào
            cur = norm_text(buf.split("<ground>")[-1].split("</ground>")[0])
            cur = cur if cur in cat_class else None
            buf = ""
        if cur and token in ids:
            prob = torch.softmax(step[:len(ids)], -1).tolist()
            row, trow = conf[cat_class[cur]], conf_text[cur]
            for si, p in enumerate(prob):
                ci = sorted_to_caller[si]
                row[ci] = max(row[ci], p)
                trow[ci] = max(trow[ci], p)
    return conf, conf_text


def caption(worker, image, boxes, text) -> list:
    """Mô tả từng box (chia lô DESC_BATCH box/lượt) -> [mô tả đã chuẩn hoá hoặc ""]."""
    out = [""] * len(boxes)
    for s in range(0, len(boxes), DESC_BATCH):
        chunk = boxes[s:s + DESC_BATCH]
        res = worker.run_task(image, DESC_TASK, text, lang="en", bbox_list=chunk,
                              max_new_tokens=DESC_MAX_NEW_TOKENS, temperature=0.0, top_p=1.0)
        answer = res["answer"] if isinstance(res, dict) else str(res)
        try:
            _, sorted_to_caller = worker._order_boxes(worker._validate_boxes(chunk, image))
        except Exception:
            sorted_to_caller = list(range(len(chunk)))
        marks = list(re.finditer(r"<obj(\d+)>", answer))
        if not marks and len(chunk) == 1:        # 1 box: model có thể trả thẳng câu mô tả
            out[s] = norm_text(answer)
        for i, m in enumerate(marks):
            n = int(m.group(1))
            if n >= len(sorted_to_caller):
                continue
            end = marks[i + 1].start() if i + 1 < len(marks) else len(answer)
            ci = s + sorted_to_caller[n]
            if not out[ci]:
                out[ci] = norm_text(answer[m.end():end])
    # Cắt bớt chữ: mỗi category bị model chép lại nguyên văn vào đầu ra của bước
    # verify, mô tả dài là ăn hết ngân sách token trước khi kịp liệt kê box.
    return [" ".join(c.split()[:DESC_MAX_WORDS]) for c in out]


def pad_box(box, w, h, ratio):
    """Nới box ra mỗi phía `ratio` lần cạnh tương ứng, kẹp trong ảnh -> có ngữ cảnh."""
    x1, y1, x2, y2 = box
    dx, dy = (x2 - x1) * ratio, (y2 - y1) * ratio
    return [max(0.0, x1 - dx), max(0.0, y1 - dy), min(float(w), x2 + dx), min(float(h), y2 + dy)]


def _similar(a: set, b: set) -> bool:
    return bool(a | b) and len(a & b) / len(a | b) >= DESC_DEDUP


def dedup_prompts(by_class: dict, anchors: dict) -> dict:
    """{class: [mô tả, xếp từ đáng tin nhất]} -> {class: [mô tả đã lọc]}. LUÔN chạy.

    1. trong 1 class: bỏ câu trùng / gần trùng (Jaccard theo từ >= DESC_DEDUP), bỏ
       câu trùng alias (alias đã có sẵn trong lượt verify), giữ tối đa
       DESC_MAX_PROMPTS câu -> ít category hơn, VLX trả lời nhanh hơn.
    2. giữa các class: câu (gần) trùng nhau ở 2 class thì bỏ ở CẢ HAI — câu đó
       không phân biệt được 2 class, để lại chỉ làm VLX gán box lung tung.
    """
    kept = {}
    for cls_name, texts in by_class.items():
        seen = [set(a.split()) for a in anchors.get(cls_name, [])]
        out = []
        for t in texts:
            ws = set(t.split())
            if not t or any(_similar(ws, s) for s in seen):
                continue
            seen.append(ws)
            out.append(t)
            if len(out) >= DESC_MAX_PROMPTS:
                break
        kept[cls_name] = out

    words = {(c, t): set(t.split()) for c, ts in kept.items() for t in ts}
    clash = {key for key in words for other in words
             if key[0] != other[0] and _similar(words[key], words[other])}
    return {c: [t for t in ts if (c, t) not in clash] for c, ts in kept.items()}


def describe_group(worker, image, boxes, subset, hint) -> tuple:
    """Chế độ --describe cho 1 nhóm class -> (conf, thông tin để lưu lại).

    1. ALIAS  (box gốc)   : 1 lượt detection với alias của mọi class trong nhóm
                            -> mỗi box tạm thuộc class nào, tự tin bao nhiêu.
    2. MÔ TẢ (box có pad) : chỉ hỏi box tự tin nhất của từng class. Câu hỏi
                            nhắc HINT ("this head"), KHÔNG nhắc class ("this
                            helmet") — hỏi có hướng thì model chiều theo câu hỏi
                            và box nào cũng tự xác nhận được chính nó.
    3. LỌC TRÙNG          : dedup_prompts.
    4. VERIFY (box gốc)   : 1 lượt detection, category = alias + mô tả của MỌI
                            class trong nhóm, cạnh tranh nhau như chế độ thường.
    Không còn mô tả nào sau khi lọc thì dùng luôn kết quả bước 1, đỡ 1 lượt gọi.
    """
    w, h = image.size
    alias = {}
    for cls_name, spec in subset.items():
        alias[cls_name] = list(dict.fromkeys(norm_text(a) for a in spec["alias"] if norm_text(a)))
    owners = {}
    for cls_name, texts in alias.items():
        for t in texts:
            owners.setdefault(t, set()).add(cls_name)
    alias = {c: [t for t in ts if len(owners[t]) == 1] for c, ts in alias.items()}
    cats = [(t, c) for c, ts in alias.items() for t in ts]
    if not cats:
        return {c: [0.0] * len(boxes) for c in subset}, {}, 0

    conf_a, _ = ground(worker, image, boxes, cats, VLX_MAX_NEW_TOKENS)
    n_calls = 1

    # box -> class alias tự tin nhất; mỗi class lấy tối đa DESC_PER_CLASS box
    pick = {}
    for i in range(len(boxes)):
        cls_name, p = max(((c, conf_a[c][i]) for c in conf_a), key=lambda kv: kv[1])
        if p >= DESC_MIN_ALIAS:
            pick.setdefault(cls_name, []).append((p, i))
    chosen = []
    for cls_name, items in pick.items():
        items.sort(reverse=True)
        chosen += [(cls_name, p, i) for p, i in (items[:DESC_PER_CLASS] if DESC_PER_CLASS else items)]
    if not chosen:
        return conf_a, {"alias": alias, "captions": {}, "prompts": {}}, n_calls

    question = DESC_TEXT.format(hint=hint)
    caps = caption(worker, image, [pad_box(boxes[i], w, h, DESC_PAD) for _, _, i in chosen],
                   question)
    n_calls += -(-len(chosen) // DESC_BATCH)

    by_class = {}
    for (cls_name, p, i), cap in sorted(zip(chosen, caps), key=lambda x: -x[0][1]):
        by_class.setdefault(cls_name, []).append(cap)
    prompts = dedup_prompts(by_class, alias)
    info = {"alias": alias, "prompts": prompts,
            "captions": {i: cap for (_, _, i), cap in zip(chosen, caps) if cap}}
    if not any(prompts.values()):
        return conf_a, info, n_calls

    cats += [(t, c) for c, ts in prompts.items() for t in ts]
    conf_v, _ = ground(worker, image, boxes, cats, DESC_MAX_NEW_TOKENS)
    return conf_v, info, n_calls + 1


def score_image(worker, rec, image, classes, describe=False) -> tuple:
    """Chấm mọi proposal của 1 ảnh -> ([{"box","score","conf"}], số lượt gọi, prompt đã dùng).

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
    used = {}
    # Không đưa nhiều box hơn số token <objN> model có: box thừa model không có
    # cách nào chỉ tới.
    cap = min(TOPK, len(worker.obj_ids))
    n_calls = 0
    for hints, names in by_hints.items():
        items = suppress([it for h in hints for it in groups.get(h, ())], NMS_IOU, NMS_IOS)[:cap]
        if not items:
            continue
        boxes = [b for _, b in items]
        subset = {c: classes[c] for c in names}
        captions = {}
        if describe:
            conf, info, calls = describe_group(worker, image, boxes, subset, hints[0])
            captions = info.get("captions", {})
            used[" | ".join(names)] = {"alias": info.get("alias", {}),
                                       "prompts": info.get("prompts", {})}
        else:
            cats = [(t, c) for c in names for t in classes[c]["prompts"]]
            conf, _ = ground(worker, image, boxes, cats, VLX_MAX_NEW_TOKENS)
            calls = 1
        n_calls += calls
        for i, (score, box) in enumerate(items):
            key = tuple(round(v, 2) for v in box)
            rec_box = records.setdefault(key, {"box": box, "score": score, "conf": {}})
            for cls_name in names:
                rec_box["conf"][cls_name] = conf.get(cls_name, [0.0] * len(boxes))[i]
            if i in captions:
                rec_box.setdefault("caption", {})[" | ".join(names)] = captions[i]
    return list(records.values()), n_calls, used


def vote_mode(path: Path):
    """Chế độ đã chấm file votes này (None = chưa có). File cũ không ghi mode = prompts."""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("mode", "prompts")
    except json.JSONDecodeError:
        return None                  # file ghi dở (Ctrl+C) -> chấm lại


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
    ap.add_argument("--describe", action="store_true", default=DESCRIBE,
                    help="VLX tự sinh prompt theo từng ảnh (alias -> mô tả -> verify)")
    args = ap.parse_args()
    mode = "describe" if args.describe else "prompts"
    print(f"   chế độ chấm: {mode}")

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
        if vote_mode(Path(jobs[k]["votes"], f"{stem}.json")) != mode:
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
            records, calls, used = score_image(worker, rec, image, classes, args.describe)
            out = {"stem": rec["stem"], "w": rec["w"], "h": rec["h"], "mode": mode,
                   "boxes": records}
            if used:
                out["prompts"] = used        # prompt VLX đã tự sinh cho ảnh này
            Path(job["votes"], f"{rec['stem']}.json").write_text(json.dumps(out),
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

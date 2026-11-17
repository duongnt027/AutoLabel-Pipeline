"""Chế độ EVENT: phát hiện HÀNH VI theo mô tả từ các đối tượng đã detect — CHỈ stdlib.

    votes/ (stage 2, mỗi box 1 class)
      -> detections()        box về tay class tự tin nhất, >= actor_min_conf
      -> merge_same_class()  box CÙNG class chồng lên nhau gộp thành 1 đối tượng
      -> group_objects()     đối tượng (đã gộp) của 1 hành vi gộp thành 1 BOX NHÓM
      -> [stage3] VLX mô tả box nhóm + chấm box nhóm theo caption của từng event
      -> decide()            so caption sinh ra với caption gốc + requires -> event / loại

Phần hình học + so khớp + quyết định nằm ở đây (không model) để:
  * đổi ngưỡng / trọng số / keyword rồi dựng lại nhãn trong vài giây (--rebuild);
  * thử được logic gộp nhóm mà không cần GPU.
"""

import json
import math
import re
from pathlib import Path

from autodetect.common import env_float, load_env

REJECT = "__reject__"


def params() -> dict:
    """Tham số chế độ event từ .env (xem .env.example, mục 'Chế độ event')."""
    load_env()
    return {
        # box stage 2 phải được VLX chấm >= mức này mới được coi là 1 đối tượng
        "actor_min_conf": env_float("EVENT_ACTOR_MIN_CONF", 0.3),
        # gộp cùng class: nới mỗi box ra `merge_pad` x cạnh dài rồi xét chạm nhau
        "merge_pad": env_float("EVENT_MERGE_PAD", 0.0),
        # gộp nhóm hành vi: nới box ra `group_pad` x cạnh dài rồi xét chạm nhau
        "group_pad": env_float("EVENT_GROUP_PAD", 0.15),
        # box nhóm không được to quá tỉ lệ này của ảnh (chặn cả phố thành 1 nhóm)
        "group_max_area": env_float("EVENT_GROUP_MAX_AREA", 0.35),
        # điểm cuối = w_vlx * điểm VLX + w_text * điểm so caption
        "w_vlx": env_float("EVENT_W_VLX", 0.6),
        "w_text": env_float("EVENT_W_TEXT", 0.4),
        "min_conf": env_float("EVENT_MIN_CONF", 0.35),
        # event phải hơn reject ít nhất chừng này mới được giữ
        "reject_margin": env_float("EVENT_REJECT_MARGIN", 0.0),
    }


# ── Hình học ─────────────────────────────────────────────────────────────────
def area(b) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def union_box(a, b) -> list:
    return [min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])]


def grow(b, ratio: float) -> list:
    """Nới box ra mọi phía `ratio` x CẠNH DÀI (đều 4 phía, khác pad_box của stage 2).

    Theo cạnh dài chứ không theo từng cạnh: người đứng cạnh sạp hàng là box cao
    hẹp, nới theo chiều ngang của chính nó thì gần như không chạm được tới sạp.
    """
    d = max(b[2] - b[0], b[3] - b[1]) * ratio
    return [b[0] - d, b[1] - d, b[2] + d, b[3] + d]


def inter_area(a, b) -> float:
    return (max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
            * max(0.0, min(a[3], b[3]) - max(a[1], b[1])))


class _DSU:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, i):
        while self.p[i] != i:
            self.p[i] = self.p[self.p[i]]
            i = self.p[i]
        return i

    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)


def detections(vote_files: list, classes: dict, min_conf: float) -> list:
    """votes/<stem>.json của 1 hay nhiều backend -> [{"class","conf","box"}].

    Mỗi box về tay class tự tin nhất (như build_labels), chỉ lấy class có trong
    bộ class của chế độ event. Box trùng giữa các backend để merge_same_class gộp.
    """
    out = []
    for vfile in vote_files:
        for rec in json.loads(Path(vfile).read_text(encoding="utf-8"))["boxes"]:
            conf = {c: v for c, v in rec.get("conf", {}).items() if c in classes}
            if not conf:
                continue
            cls_name, p = max(conf.items(), key=lambda kv: kv[1])
            if p > 0 and p >= min_conf:
                out.append({"class": cls_name, "conf": p, "box": list(rec["box"])})
    return out


def merge_same_class(dets: list, pad: float = 0.0) -> list:
    """Box CÙNG class chồng nhau (sau khi nới `pad`) -> 1 đối tượng = box bao của cả cụm.

    Lặp tới khi ổn định: 2 cụm ban đầu không chạm nhau nhưng box bao sau khi gộp
    lại chạm nhau thì cũng phải gộp (vd dãy hàng hoá bày nối nhau trên vỉa hè).
    Trả [{"class","box","conf","n"}], conf = max của cụm, n = số box gốc.
    """
    objs = []
    for cls_name in dict.fromkeys(d["class"] for d in dets):
        cur = [{"class": cls_name, "box": d["box"], "conf": d["conf"], "n": 1}
               for d in dets if d["class"] == cls_name]
        while True:
            dsu = _DSU(len(cur))
            grown = [grow(o["box"], pad) for o in cur]
            for i in range(len(cur)):
                for j in range(i + 1, len(cur)):
                    if inter_area(grown[i], grown[j]) > 0:
                        dsu.union(i, j)
            merged = {}
            for i, o in enumerate(cur):
                m = merged.setdefault(dsu.find(i), {"class": cls_name, "box": o["box"],
                                                    "conf": 0.0, "n": 0})
                m["box"] = union_box(m["box"], o["box"])
                m["conf"] = max(m["conf"], o["conf"])
                m["n"] += o["n"]
            if len(merged) == len(cur):
                break
            cur = list(merged.values())
        objs += cur
    return objs


def group_objects(objs: list, roles: dict, w: int, h: int, pad: float, max_area: float) -> list:
    """Đối tượng đã gộp -> nhóm hành vi [{"box", "members": [chỉ số trong objs]}].

    1. ANCHOR (hàng hoá, xe đẩy, biển, vật liệu...) chạm nhau sau khi nới `pad`
       thì vào chung nhóm — cặp chồng nhiều nhất được ghép trước, và không ghép
       nếu box nhóm vượt `max_area` x diện tích ảnh.
    2. CONTEXT (người, xe, cây, cột) gắn vào ĐÚNG 1 nhóm anchor nó chạm nhiều
       nhất. Context không tự lập nhóm (người đi bộ một mình không phải vi phạm)
       và không làm cầu nối 2 nhóm (người đứng giữa 2 sạp không gộp 2 sạp làm một).
    Nhóm nào cũng có >= 1 anchor.
    """
    limit = max_area * w * h
    anchors = [i for i, o in enumerate(objs) if roles.get(o["class"], "anchor") == "anchor"]
    contexts = [i for i, o in enumerate(objs) if roles.get(o["class"], "anchor") != "anchor"]

    clusters = {i: {"box": list(objs[i]["box"]), "members": [i]} for i in anchors}
    owner = {i: i for i in anchors}
    pairs = []
    for x, i in enumerate(anchors):
        gi = grow(objs[i]["box"], pad)
        for j in anchors[x + 1:]:
            gj = grow(objs[j]["box"], pad)
            inter = inter_area(gi, gj)
            if inter > 0:
                pairs.append((inter / min(area(gi), area(gj)), i, j))
    for _, i, j in sorted(pairs, reverse=True):
        a, b = owner[i], owner[j]
        if a == b:
            continue
        box = union_box(clusters[a]["box"], clusters[b]["box"])
        if area(box) > limit:
            continue
        clusters[a]["box"] = box
        clusters[a]["members"] += clusters[b]["members"]
        for m in clusters.pop(b)["members"]:
            owner[m] = a

    for i in sorted(contexts, key=lambda k: -objs[k]["conf"]):
        gi = grow(objs[i]["box"], pad)
        best, best_ov = None, 0.0
        for key, cl in clusters.items():
            ov = inter_area(gi, grow(cl["box"], pad)) / max(area(gi), 1e-6)
            if ov > best_ov and area(union_box(cl["box"], objs[i]["box"])) <= limit:
                best, best_ov = key, ov
        if best is not None:
            clusters[best]["box"] = union_box(clusters[best]["box"], objs[i]["box"])
            clusters[best]["members"].append(i)
    return list(clusters.values())


# ── So khớp caption ──────────────────────────────────────────────────────────
_STOP = set("""a an the of on in at to and or with by for from is are was were be been being
this that these those there it its into onto near next some several two one many
side front back while being which who has have having""".split())


def _stem(word: str) -> str:
    """Rút gọn thô: selling/sells -> sell, tables -> table, parked -> park.

    Không cắt "es" (tables -> "tabl" sẽ lệch với table) và giữ >= 3 chữ (bus, gas).
    """
    for suf in ("ing", "ed", "s"):
        if word.endswith(suf) and len(word) - len(suf) >= 3:
            return word[: -len(suf)]
    return word


def tokens(text: str) -> set:
    return {_stem(w) for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOP}


def _cos(a: set, b: set) -> float:
    return len(a & b) / math.sqrt(len(a) * len(b)) if a and b else 0.0


def lexical_sim(caption: str, refs: list) -> float:
    """Độ giống theo từ (cosine trên tập từ đã rút gọn), max qua các câu gốc."""
    ct = tokens(caption)
    return max((_cos(ct, tokens(r)) for r in refs), default=0.0)


def keyword_score(caption: str, keywords: list) -> float:
    """Tỉ lệ keyword xuất hiện trong caption, bão hoà ở 2 keyword (= 1.0)."""
    ct = tokens(caption)
    hits = sum(1 for k in keywords if tokens(k) and tokens(k) <= ct)
    return min(1.0, hits / 2)


def text_scores(caption: str, events: dict, reject: dict, sim_fn=None) -> dict:
    """{event | REJECT: điểm so caption 0..1} = 0.5 * độ giống câu + 0.5 * keyword.

    sim_fn(caption, refs) -> 0..1: mặc định lexical_sim; stage 3 truyền hàm
    embedding vào nếu EVENT_TEXT_MODEL được đặt.
    """
    sim_fn = sim_fn or lexical_sim
    if not caption:
        return {e: 0.0 for e in [*events, REJECT]}
    out = {e: 0.5 * sim_fn(caption, ev["captions"]) + 0.5 * keyword_score(caption, ev["keywords"])
           for e, ev in events.items()}
    out[REJECT] = (0.5 * sim_fn(caption, reject["captions"])
                   + 0.5 * keyword_score(caption, reject["keywords"])) if reject["captions"] else 0.0
    return out


# ── Luật + quyết định ────────────────────────────────────────────────────────
def requires_ok(event: dict, present) -> bool:
    """requires = AND của các OR: [["signboard"], ["tree", "utility pole"]]."""
    present = set(present)
    return all(any(c in present for c in alt) for alt in event.get("requires", []))


def decide(group: dict, events: dict, p: dict, stem: str) -> dict:
    """1 box nhóm đã chấm -> {"event": tên | None, "score", "reason", "scores"}.

    score(e) = w_vlx * VLX(e) + w_text * text(e), = 0 nếu nhóm thiếu class mà
    event yêu cầu (requires). Giữ event điểm cao nhất nếu >= min_conf VÀ hơn
    điểm reject.

    CHỈ xét thứ NHÌN THẤY trong ảnh. Điều kiện pháp lý ảnh không thể hiện
    (giờ quy định, tuyến phố cấm, có giấy phép hay không) lọc ở hệ thống bên
    ngoài dựa trên stem/camera/thời điểm trong events_summary.jsonl.
    """
    vlx, text = group.get("vlx", {}), group.get("text", {})
    present = {m["class"] for m in group["members"]}

    def score(e):
        return p["w_vlx"] * vlx.get(e, 0.0) + p["w_text"] * text.get(e, 0.0)

    scores = {e: round(score(e), 4) if requires_ok(ev, present) else 0.0
              for e, ev in events.items()}
    rej = round(score(REJECT), 4)
    best = max(scores, key=scores.get) if any(scores.values()) else None
    res = {"event": None, "best": best, "score": scores.get(best, 0.0), "reject": rej,
           "scores": scores, "reason": ""}
    if rej > 0 and rej + p["reject_margin"] >= res["score"]:
        res["reason"] = "reject_caption"          # box rác: khớp caption reject hơn mọi event
        return res
    if best is None or res["score"] < p["min_conf"]:
        res["reason"] = "low_score"
        return res
    res["event"] = best
    return res


def event_ids(events: dict) -> dict:
    return {name: i for i, name in enumerate(events)}


def build_event_labels(vote_dir: Path, events: dict, reject: dict, out_dir: Path,
                       p: dict, summary: Path = None) -> tuple:
    """events_votes/ -> nhãn YOLO theo EVENT + events_summary.jsonl. Trả (số ảnh, số nhãn).

    Không chạy model: đổi EVENT_* (ngưỡng, trọng số) hay keywords/luật trong
    file event rồi gọi lại hàm này là xong. Điểm text của bản lexical được tính
    LẠI từ caption đã lưu (nên sửa keywords có tác dụng ngay); bản embedding thì
    dùng điểm đã lưu. Đổi `captions` của event thì phải chạy lại stage 3, vì
    đó là category VLX đã chấm.
    """
    ids = event_ids(events)
    vote_dir, out_dir = Path(vote_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = Path(summary or out_dir.parent / "events_summary.jsonl")
    n_img = n_lbl = 0
    with open(summary, "w", encoding="utf-8") as fs:
        for vfile in sorted(vote_dir.glob("*.json")):
            data = json.loads(vfile.read_text(encoding="utf-8"))
            w, h, stem = data["w"], data["h"], data["stem"]
            lines = []
            for gi, g in enumerate(data["groups"]):
                if data.get("text_model", "lexical") == "lexical":
                    g["text"] = text_scores(g.get("caption", ""), events, reject)
                d = decide(g, events, p, stem)
                fs.write(json.dumps({"stem": stem, "group": gi, "box": g["box"],
                                     "classes": sorted({m["class"] for m in g["members"]}),
                                     "caption": g.get("caption", ""),
                                     "event": d["event"], "best": d["best"],
                                     "name_vi": events[d["best"]].get("name_vi", "") if d["best"] else "",
                                     "score": d["score"], "reject": d["reject"],
                                     "reason": d["reason"]},
                                    ensure_ascii=False) + "\n")
                if d["event"] is None:
                    continue
                x1, y1, x2, y2 = g["box"]
                lines.append(f"{ids[d['event']]} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} "
                             f"{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
            Path(out_dir, f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""),
                                                    encoding="utf-8")
            n_img += 1
            n_lbl += len(lines)
    return n_img, n_lbl

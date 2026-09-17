"""Tiện ích dùng chung cho mọi stage — CHỈ stdlib.

Các backend stage 1 chạy bằng python của những conda env KHÁC NHAU (xem
docs/SETUP.md), nên module này không được import torch/transformers/dotenv:
env nào cũng phải import được nó.
"""

import json
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp", ".gif")


# ── .env ─────────────────────────────────────────────────────────────────────
def load_env(path: Path = None):
    """Nạp KEY=VALUE từ .env vào os.environ. Biến đặt sẵn ở shell luôn thắng.

    Tự viết thay vì python-dotenv để runner chạy được bằng python3 bất kỳ,
    không phải cài thêm gì.
    """
    path = Path(path or REPO_ROOT / ".env")
    if not path.exists():
        return
    # Khai TRÙNG KEY trong .env thì DÒNG CUỐI thắng — giống shell và python-dotenv.
    # (Dùng setdefault thì dòng ĐẦU thắng, và dòng ghi đè thêm ở cuối file sẽ im
    # lặng không có tác dụng — rất khó nhận ra.)
    from_file = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key in os.environ and key not in from_file:
            continue                      # biến đặt sẵn ở shell vẫn thắng cả file
        os.environ[key] = value.strip().strip('"').strip("'")
        from_file.add(key)


def env_str(key: str, default: str = "") -> str:
    return (os.getenv(key) or "").strip() or default


def env_float(key: str, default: float) -> float:
    return float(env_str(key) or default)


def env_int(key: str, default: int) -> int:
    return int(env_str(key) or default)


def env_bool(key: str, default: bool) -> bool:
    raw = env_str(key).lower()
    return raw in ("1", "true", "yes", "y", "on") if raw else default


def env_list(key: str) -> list:
    return [v.strip() for v in env_str(key).replace(",", " ").split() if v.strip()]


# ── Class / hint ─────────────────────────────────────────────────────────────
def load_classes(path: Path) -> dict:
    """{class: {"hints": [...], "prompts": [...]}} — nguồn duy nhất của bộ nhãn.

    hints  = từ khoá cho stage 1 vẽ box (thô, rộng, nhiều từ đồng nghĩa)
    prompts = câu mô tả cho stage 2 chấm điểm (hẹp, phân biệt các class chung hint)
    """
    classes = json.loads(Path(path).read_text(encoding="utf-8"))
    for name, spec in classes.items():
        if not spec.get("hints") or not spec.get("prompts"):
            raise SystemExit(f"❌ class '{name}' thiếu 'hints' hoặc 'prompts' trong {path}")
    return classes


def load_hints(path: Path) -> list:
    """Danh sách hint DUY NHẤT của mọi class, giữ nguyên thứ tự khai báo.

    Nhiều class dùng chung 1 hint là chuyện bình thường (helmet / no helmet đều
    dùng "head"): stage 1 chỉ cần vẽ box cái đầu, phân biệt là việc của stage 2.
    """
    hints = []
    for spec in load_classes(path).values():
        for hint in spec["hints"]:
            if hint not in hints:
                hints.append(hint)
    return hints


# ── Proposal jsonl (HỢP ĐỒNG CHUNG của mọi backend stage 1) ──────────────────
# Mỗi dòng = 1 ảnh:
#   {"stem", "w", "h", "boxes": [[x1,y1,x2,y2],...], "scores": [...], "hints": [...]}
# Toạ độ là PIXEL của ảnh gốc. Backend nào ghi đúng hợp đồng này thì cắm vào
# pipeline được ngay, không phải sửa stage 2.
def iter_jsonl(path: Path):
    """Đọc jsonl THEO TỪNG DÒNG, bỏ qua dòng hỏng ở cuối (lần chạy trước bị Ctrl+C
    giữa lúc ghi).

    Generator chứ không trả list: mỗi bản ghi giữ hàng nghìn box, ảnh dày vật thể
    (SAM3 ~2800 box/ảnh) tốn ~0.9MB RAM/ảnh nếu giữ hết — 10k ảnh là ~9GB MỖI
    tiến trình, mà stage 2 chạy một tiến trình cho mỗi GPU. Đọc từng dòng thì RAM
    phẳng bất kể bộ ảnh to cỡ nào.
    """
    if not Path(path).exists():
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                print(f"   ⚠️  bỏ qua 1 dòng hỏng trong {Path(path).name}")


def read_jsonl(path: Path) -> list:
    """Cả file vào RAM. CHỈ dùng khi biết chắc file nhỏ — không thì dùng iter_jsonl."""
    return list(iter_jsonl(path))


def done_stems(path: Path) -> set:
    """Ảnh đã có kết quả trong file jsonl — chạy lại thì bỏ qua, làm tiếp chỗ dở."""
    return {rec["stem"] for rec in iter_jsonl(path) if "stem" in rec}


def count_jsonl(path: Path) -> tuple:
    """(số ảnh, số box) của 1 file proposal — đọc theo dòng, không giữ gì lại."""
    n_img = n_box = 0
    for rec in iter_jsonl(path):
        n_img += 1
        n_box += len(rec["boxes"])
    return n_img, n_box


def list_images(folder: Path, limit: int = 0) -> list:
    images = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in IMG_EXTS)
    return images[:limit] if limit else images


def plan_jobs(jobs: list, limit: int) -> list:
    """[(file jsonl kết quả, đường dẫn ảnh)] của MỌI job, đã bỏ ảnh làm rồi.

    Gom việc của mọi job TRƯỚC khi nạp model: chạy N thư mục chỉ tốn 1 lần nạp.
    """
    todo = []
    for job in jobs:
        out = Path(job["preds"])
        images = list_images(job["images"], limit)
        skip = done_stems(out)
        job_todo = [p for p in images if p.stem not in skip]
        todo += [(out, p) for p in job_todo]
        print(f"   {job.get('name', out.parent.name)}: {len(images)} ảnh, "
              f"đã có {len(skip)}, cần chạy {len(job_todo)}")
    return todo


def jobs_from_args(args) -> list:
    """[{"name","images","preds"}] từ --manifest (nhiều thư mục) hoặc --images/--out."""
    if getattr(args, "manifest", None):
        return json.loads(Path(args.manifest).read_text(encoding="utf-8"))
    if args.images and args.out:
        return [{"name": Path(args.images).name, "images": str(args.images),
                 "preds": str(args.out)}]
    raise SystemExit("❌ cần --manifest, hoặc đủ cả --images --out")


def clip_boxes(found, w: int, h: int) -> tuple:
    """[(hint, score, box)] -> (boxes, scores, hints) đã cắt về trong ảnh.

    Bỏ box mỏng hơn 1px: model ở ngưỡng thấp thỉnh thoảng trả ra box suy biến,
    để lọt xuống stage 2 thì tốn 1 slot <objN> mà chẳng chấm được gì.
    """
    boxes, scores, hints = [], [], []
    for hint, score, box in found:
        x1, x2 = sorted((max(0.0, float(box[0])), min(float(w), float(box[2]))))
        y1, y2 = sorted((max(0.0, float(box[1])), min(float(h), float(box[3]))))
        if x2 - x1 <= 1 or y2 - y1 <= 1:
            continue
        boxes.append([x1, y1, x2, y2])
        scores.append(float(score))
        hints.append(hint)
    return boxes, scores, hints


class JsonlWriter:
    """Ghi thẳng từng ảnh xuống đĩa (mở file theo nhu cầu, flush ngay).

    Không buffer trong RAM: chạy 1000 ảnh mà đứt giữa chừng thì phần đã làm vẫn
    còn, lần sau done_stems() nhặt tiếp.
    """

    def __init__(self):
        self.handles = {}

    def write(self, out: Path, stem: str, w: int, h: int, boxes, scores, hints):
        out = Path(out)
        if out not in self.handles:
            out.parent.mkdir(parents=True, exist_ok=True)
            self.handles[out] = open(out, "a", encoding="utf-8")
        self.handles[out].write(json.dumps(
            {"stem": stem, "w": w, "h": h, "boxes": boxes,
             "scores": scores, "hints": hints}) + "\n")
        self.handles[out].flush()

    def close(self):
        for f in self.handles.values():
            f.close()

    @property
    def files(self) -> str:
        return ", ".join(str(p) for p in self.handles)


def env_path(key: str, default) -> Path:
    """Đường dẫn lấy từ .env. TƯƠNG ĐỐI thì tính từ GỐC REPO, không phải cwd —
    các backend stage 1 chạy với cwd = repo bên thứ ba (LLMDet/WeDetect)."""
    p = Path(env_str(key) or default)
    return p if p.is_absolute() else (REPO_ROOT / p)

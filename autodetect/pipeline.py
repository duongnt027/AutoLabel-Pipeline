"""Cấu hình + cách gọi từng stage. Dùng chung cho run_stage1.py và run_pipeline.py.

Mỗi backend/stage sống trong MỘT CONDA ENV KHÁC NHAU (LLMDet/WeDetect: torch
2.2.1 + transformers 4.37.2 | SAM3/VLX-Seek: torch 2.8 + transformers 5.13) nên
mỗi bước là một SUBPROCESS gọi thẳng python của env tương ứng, trao đổi dữ liệu
qua file jsonl — không stage nào import trực tiếp stage khác.
"""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from autodetect.common import (REPO_ROOT, env_float, env_int, env_list, env_path, env_str,
                               list_images, load_env)

load_env()

# ── Backend stage 1 ──────────────────────────────────────────────────────────
#   llm = LLMDet (Grounding DINO)  |  s3 = SAM3 (text prompt)
#   we  = WeDetect
# Mỗi backend 1 script riêng, CÙNG HỢP ĐỒNG jsonl (xem autodetect/common.py) nên
# dùng riêng hay kết hợp đều được: --models llm,s3
MODEL_CHOICES = ("llm", "s3", "we")

CLASSES_JSON = env_path("AUTOLABEL_CLASSES", "classes/classes.json")
DATA_YAML = env_path("AUTOLABEL_DATA_YAML", "") if env_str("AUTOLABEL_DATA_YAML") else None
RUNS_ROOT = env_path("RUNS_ROOT", "runs")
THIRD_PARTY = env_path("THIRD_PARTY_DIR", "third_party")

DEFAULT_MODELS = env_list("AUTOLABEL_MODELS") or ["llm"]

LLMDET_PYTHON = env_str("LLMDET_PYTHON", "python")
LLMDET_REPO = env_path("LLMDET_REPO", THIRD_PARTY / "LLMDet")
# config/weights ghi TƯƠNG ĐỐI so với LLMDET_REPO (config trỏ model bằng '../huggingface/...')
LLMDET_CONFIG = env_str("LLMDET_CONFIG", "configs/grounding_dino_swin_l.py")
LLMDET_WEIGHTS = env_str("LLMDET_WEIGHTS", "pretrained_models/large.pth")
LLMDET_GPU = env_str("LLMDET_GPU", "0")
LLMDET_BATCH = env_int("LLMDET_BATCH", 4)
# Ngưỡng proposal thấp: stage 1 chỉ cần vẽ đủ box, stage 2 mới là khâu lọc.
LLMDET_SCORE_THR = env_float("LLMDET_SCORE_THR", 0.15)
LLMDET_SAHI_SLICE = env_int("LLMDET_SAHI_SLICE", 1024)
LLMDET_SAHI_OVERLAP = env_float("LLMDET_SAHI_OVERLAP", 0.2)

VLX_PYTHON = env_str("VLX_PYTHON", "python")
VLX_GPUS = env_list("VLX_GPUS") or ["0"]
SAM3_SCORE_THR = env_float("SAM3_SCORE_THR", 0.05)

WEDETECT_PYTHON = env_str("WEDETECT_PYTHON") or LLMDET_PYTHON
WEDETECT_REPO = env_path("WEDETECT_REPO", THIRD_PARTY / "WeDetect")
WEDETECT_CONFIG = env_str("WEDETECT_CONFIG", "config/wedetect_base.py")
# Giải tương đối theo GỐC REPO: backend we chạy với cwd = repo WeDetect nên
# đường dẫn tương đối trong .env sẽ trỏ sai nếu để nguyên chuỗi.
WEDETECT_WEIGHTS = str(env_path("WEDETECT_WEIGHTS", "")) if env_str("WEDETECT_WEIGHTS") else ""
# WeDetect train bằng prompt tiếng Trung -> có thể trỏ sang 1 classes.json đã dịch
WEDETECT_CLASSES = env_path("WEDETECT_CLASSES", CLASSES_JSON)

# Số thread CPU cho MỖI tiến trình con. torch mặc định lấy nửa số core (56 trên
# máy 112 core) cho MỖI tiến trình — chạy 2-3 GPU song song là 100+ thread giành
# nhau CPU, thrash mà không nhanh hơn: khâu nặng nằm ở GPU, CPU chỉ decode ảnh.
CPU_THREADS = env_str("CPU_THREADS", "8")

# Ngưỡng quyết định cuối (dùng ở autodetect/labels.py)
MIN_CONF = env_float("AUTOLABEL_MIN_CONF", 0.2)
NMS_IOU = env_float("AUTOLABEL_NMS_IOU", 0.6)
NMS_IOS = env_float("AUTOLABEL_NMS_IOS", 0.8)

_START = time.time()


def step(n: int, title: str, total: int):
    print(f"\n\033[1;36m[{n}/{total}]\033[0m {title} "
          f"\033[90m(+{time.time() - _START:.0f}s)\033[0m")


def resolve_models(raw: list) -> list:
    """"all" -> mọi backend; nhận list mã cách nhau dấu phẩy/khoảng trắng."""
    codes = [c for item in raw for c in str(item).replace(",", " ").split()]
    if "all" in codes:
        return list(MODEL_CHOICES)
    bad = [c for c in codes if c not in MODEL_CHOICES]
    if bad:
        raise SystemExit(f"❌ backend lạ: {bad} (chỉ nhận {MODEL_CHOICES} hoặc 'all')")
    return codes or DEFAULT_MODELS


def make_jobs(image_dirs: list, out_root: Path, models: list, fresh: bool) -> list:
    """1 thư mục ảnh -> 1 workspace runs/<tên thư mục>/. Ảnh ĐỌC THẲNG tại chỗ,
    workspace chỉ chứa kết quả (proposal, điểm, nhãn)."""
    import shutil

    jobs = []
    for src in image_dirs:
        src = Path(src).resolve()
        if not src.is_dir():
            raise SystemExit(f"❌ không phải thư mục: {src}")
        n = len(list_images(src))
        if not n:
            raise SystemExit(f"❌ không có ảnh nào trong {src}")
        root = Path(out_root) / src.name
        if fresh and root.exists():
            print(f"   --fresh: xoá {root}")
            shutil.rmtree(root)
        root.mkdir(parents=True, exist_ok=True)
        jobs.append({
            "name": src.name,
            "images": src,
            "root": root,
            "labels": root / "labels",
            "proposals": {k: root / f"proposals_{k}.jsonl" for k in models},
            "votes": {k: root / f"votes_{k}" for k in models},
        })
        print(f"   {src}  ({n} ảnh)  →  {root}")
    return jobs


def write_manifest(path: Path, rows: list) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def check_envs(models: list, need_stage2: bool):
    """Thử import trước, sai env thì báo NGAY.

    Không có bước này thì sai env chỉ lộ ra sau khi backend trước đã chạy xong
    (có khi hàng chục phút). Chỉ kiểm tra backend THỰC SỰ được chọn.
    """
    checks = []
    if "llm" in models:
        checks.append((LLMDET_PYTHON, "mmdet", LLMDET_REPO, f"env LLMDet, cwd={LLMDET_REPO}"))
    if "we" in models:
        checks.append((WEDETECT_PYTHON, "wedetect", WEDETECT_REPO,
                       f"env WeDetect (mmdet đã vá + webdataset), cwd={WEDETECT_REPO}"))
    if "s3" in models or need_stage2:
        checks.append((VLX_PYTHON, "transformers.models.sam3" if "s3" in models
                       else "transformers", REPO_ROOT, "env vlx (transformers >= 5.x)"))
    for python, module, cwd, hint in checks:
        ok = subprocess.run(
            [python, "-c", f"import importlib.util,sys;"
                           f"sys.exit(0 if importlib.util.find_spec('{module}') else 1)"],
            cwd=str(cwd), capture_output=True).returncode == 0
        if not ok:
            raise SystemExit(f"❌ {python} không import được '{module}' ({hint}).\n"
                             f"   Sửa các biến *_PYTHON trong {REPO_ROOT / '.env'} "
                             f"— xem docs/SETUP.md")
        print(f"   {module:34s} ✓  {python}")


def check_vram(gpus, need_gb: float, what: str):
    """Chặn TRƯỚC khi nạp model nếu GPU không còn đủ VRAM trống.

    GPU đang có tiến trình khác (notebook quên tắt) thì model nạp được nửa
    chừng rồi mới OOM giữa lúc infer — mất công chờ.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        print("   ⚠️  không gọi được nvidia-smi, bỏ qua bước kiểm tra VRAM")
        return
    free = {line.split(",")[0].strip(): float(line.split(",")[1]) / 1024
            for line in out.strip().splitlines() if "," in line}
    low = [f"GPU {g}: còn {free.get(str(g), 0):.1f}GB" for g in gpus
           if free.get(str(g), 0) < need_gb]
    if low:
        raise SystemExit(f"❌ {what} cần ~{need_gb:.0f}GB VRAM mỗi GPU nhưng "
                         f"{', '.join(low)}. Đợi tiến trình khác xong hoặc đổi GPU "
                         f"(LLMDET_GPU / VLX_GPUS trong .env).")


def _device(gpu: str) -> str:
    return f"cuda:{gpu}" if gpu.isdigit() else gpu


def _env(**extra) -> dict:
    """Env cho tiến trình con: ghìm số thread CPU (xem CPU_THREADS) + biến thêm.

    setdefault chứ không ghi đè: đặt sẵn OMP_NUM_THREADS ở shell thì vẫn thắng.
    """
    env = os.environ.copy()
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env.setdefault(key, CPU_THREADS)
    env.update(extra)
    return env


def run_stage1(key: str, jobs: list, limit: int, sahi: bool):
    """Chạy 1 backend sinh proposal cho MỌI thư mục trong 1 lần nạp model.

    Mỗi backend ghi vào jsonl RIÊNG (proposals_<key>.jsonl): trùng file giữa các
    backend thì done_stems() của backend B sẽ tưởng backend A đã làm xong rồi bỏ qua.
    """
    manifest = write_manifest(
        RUNS_ROOT / f".manifest_{key}.json",
        [{"name": j["name"], "images": str(j["images"]), "preds": str(j["proposals"][key])}
         for j in jobs])
    common = ["--manifest", str(manifest), "--limit", str(limit), "--device", _device(LLMDET_GPU)]

    if key == "llm":
        check_vram([LLMDET_GPU], 26, "stage 1 (LLMDet)")
        cmd = [LLMDET_PYTHON, str(REPO_ROOT / "stage1" / "llmdet.py"), *common,
               "--classes", str(CLASSES_JSON), "--config", LLMDET_CONFIG,
               "--weights", LLMDET_WEIGHTS, "--score-thr", str(LLMDET_SCORE_THR),
               "--batch-size", str(LLMDET_BATCH)]
        if sahi:
            cmd += ["--sahi", "--sahi-slice", str(LLMDET_SAHI_SLICE),
                    "--sahi-overlap", str(LLMDET_SAHI_OVERLAP)]
        cwd = LLMDET_REPO          # config trỏ model bằng đường dẫn tương đối
    elif key == "s3":
        check_vram([LLMDET_GPU], 12, "stage 1 (SAM3)")
        cmd = [VLX_PYTHON, str(REPO_ROOT / "stage1" / "sam3.py"), *common,
               "--classes", str(CLASSES_JSON), "--score-thr", str(SAM3_SCORE_THR)]
        cwd = REPO_ROOT
    elif key == "we":
        if not WEDETECT_WEIGHTS:
            raise SystemExit("❌ backend 'we' cần WEDETECT_WEIGHTS trong .env "
                             "(xem docs/WEIGHTS.md)")
        check_vram([LLMDET_GPU], 6, "stage 1 (WeDetect)")
        cmd = [WEDETECT_PYTHON, str(REPO_ROOT / "stage1" / "wedetect.py"), *common,
               "--classes", str(WEDETECT_CLASSES), "--config", WEDETECT_CONFIG,
               "--weights", WEDETECT_WEIGHTS,
               "--score-thr", str(env_float("WEDETECT_SCORE_THR", 0.01))]
        # cwd PHẢI là repo WeDetect: config dùng custom_imports=["wedetect"] +
        # đường dẫn tương đối tới xlm-roberta-*/ để nạp text encoder.
        cwd = WEDETECT_REPO
    else:
        raise ValueError(key)

    subprocess.run(cmd, check=True, cwd=str(cwd), env=_env())


def run_stage2(key: str, jobs: list, limit: int):
    """Stage 2 cho proposal của 1 backend: mỗi GPU trong VLX_GPUS một tiến trình,
    chia ảnh của CẢ MẺ xen kẽ.

    Phải là tiến trình riêng (không import) vì VLX-Seek cần ghim
    CUDA_VISIBLE_DEVICES trước khi torch được nạp — tiện thể chạy song song
    nhiều GPU, đây là khâu tốn thời gian nhất của pipeline.
    """
    check_vram(VLX_GPUS, 26, "stage 2 (VLX-Seek)")
    manifest = write_manifest(
        RUNS_ROOT / f".manifest_vlx_{key}.json",
        [{"name": j["name"], "images": str(j["images"]),
          "preds": str(j["proposals"][key]), "votes": str(j["votes"][key])} for j in jobs])

    procs = []
    for i, gpu in enumerate(VLX_GPUS):
        print(f"   GPU {gpu}: shard {i}/{len(VLX_GPUS)}")
        procs.append((gpu, subprocess.Popen(
            [VLX_PYTHON, str(REPO_ROOT / "stage2" / "score_vlx.py"),
             "--manifest", str(manifest), "--classes", str(CLASSES_JSON),
             "--limit", str(limit), "--shard", f"{i}/{len(VLX_GPUS)}"],
            cwd=str(REPO_ROOT), env=_env(VLX_GPU=gpu))))
    failed = [gpu for gpu, p in procs if p.wait() != 0]
    if failed:
        raise SystemExit(f"❌ stage 2 lỗi ở GPU {failed} (xem log phía trên)")


def summary_stage1(jobs: list, models: list):
    from autodetect.common import count_jsonl

    for job in jobs:
        for key in models:
            n_img, n_box = count_jsonl(job["proposals"][key])
            print(f"✅ {job['name']} / {key}: {n_box} box trên {n_img} ảnh "
                  f"→ {job['proposals'][key]}")


def add_common_args(ap):
    ap.add_argument("images", nargs="+", type=Path, help="1 hay nhiều thư mục chứa ảnh")
    ap.add_argument("-o", "--out", type=Path, default=RUNS_ROOT,
                    help=f"thư mục chứa kết quả (mặc định {RUNS_ROOT})")
    ap.add_argument("--models", nargs="+", default=[],
                    help=f"backend stage 1 trong {MODEL_CHOICES} hoặc 'all' "
                         f"(mặc định AUTOLABEL_MODELS trong .env = {'+'.join(DEFAULT_MODELS)})")
    ap.add_argument("--classes", type=Path, default=None,
                    help=f"file class (mặc định {CLASSES_JSON})")
    ap.add_argument("--fresh", action="store_true", help="xoá workspace cũ, làm lại từ đầu")
    ap.add_argument("--sahi", action="store_true",
                    help="LLMDet cắt ảnh thành ô để bắt vật nhỏ (recall +16 điểm, chậm ~14 lần)")
    ap.add_argument("--limit", type=int, default=0, help="chỉ chạy N ảnh đầu mỗi thư mục")
    return ap


def apply_common_args(args):
    """--classes ghi đè cấu hình toàn module (mọi stage đọc cùng 1 file class)."""
    global CLASSES_JSON, WEDETECT_CLASSES, RUNS_ROOT
    if args.classes:
        if WEDETECT_CLASSES == CLASSES_JSON:
            WEDETECT_CLASSES = args.classes
        CLASSES_JSON = args.classes
    RUNS_ROOT = Path(args.out)
    if not CLASSES_JSON.exists():
        raise SystemExit(f"❌ không thấy file class: {CLASSES_JSON}")
    sys.stdout.reconfigure(line_buffering=True)
    return resolve_models(args.models)

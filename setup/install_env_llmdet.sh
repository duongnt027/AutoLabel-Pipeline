#!/usr/bin/env bash
# Env cho STAGE 1 backend llm (LLMDet) và we (WeDetect).
#   torch 2.2.1 + cu121 | mmcv 2.2.0 | mmdet 3.3.0 | transformers 4.37.2 | numpy 1.23.5
# Bộ version này khoá chặt vào nhau, đổi 1 cái là hỏng cả env (xem chú thích).
#
#   bash setup/install_env_llmdet.sh [tên_env]     # mặc định: llmdet
#
# Trên máy đang chạy pipeline, env này tên là `duongnt`.
set -e

# CHỐT AN TOÀN: nếu có gói nào lỡ phải biên dịch (mmcv không tìm thấy wheel
# chẳng hạn), không cho nó đẻ 1 job/core. Máy nhiều core mà build kernel CUDA
# song song là ăn hết RAM rồi bị OOM killer bắn.
export MAX_JOBS="${MAX_JOBS:-4}"
export NVCC_THREADS="${NVCC_THREADS:-2}"

ENV_NAME="${1:-llmdet}"
# Tìm conda + nạp `conda activate` (xem setup/_conda.sh)
. "$(dirname "$0")/_conda.sh"

echo "===== [1/8] Tạo conda env $ENV_NAME (python 3.11) ====="
# Có sẵn thì DÙNG LẠI: lượt cài dài, đứt ở bước 5 mà bắt làm lại từ đầu thì phí.
# Mọi bước pip bên dưới đều idempotent (đã đúng version thì pip bỏ qua).
if conda_env_exists "$ENV_NAME"; then
    echo "   env đã có -> cài tiếp các bước còn thiếu"
else
    conda create -y -n "$ENV_NAME" python=3.11
fi

# KHÔNG dùng `conda activate` rồi gọi `pip` trần. Đã bị một lần: activate báo
# thành công nhưng `pip` vẫn phân giải sang pip của env base (tuỳ trạng thái
# PATH/conda của shell gọi script), khiến gói chui vào base thay vì env mới —
# im lặng, chỉ lộ ra khi torch không có wheel cho python của base.
# Gọi thẳng python của env theo đường dẫn tuyệt đối thì không có cách nào sai.
PY="$CONDA_BASE/envs/$ENV_NAME/bin/python"
PIP="$PY -m pip"
[ -x "$PY" ] || { echo "❌ không thấy $PY sau khi tạo env"; exit 1; }
$PY -c "import sys; assert sys.version_info[:2] == (3, 11), sys.version; \
        print('   python của env:', sys.executable)"

# Cài wheel TRƯỚC: thiếu nó thì một số package rơi về 'setup.py install' kiểu cũ
$PIP install -U pip setuptools wheel

echo "===== [2/8] PyTorch 2.2.1 + CUDA 12.1 ====="
$PIP install torch==2.2.1+cu121 torchvision==0.17.1+cu121 torchaudio==2.2.1+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

echo "===== [3/8] mmengine + mmcv qua openmim (wheel dựng sẵn, KHÔNG build từ source) ====="
# mmcv có extension CUDA: build từ source ăn hết RAM và mất hàng giờ. openmim tự
# tìm wheel dựng sẵn ở download.openmmlab.com theo (cuda, torch, python) — bản
# cần ở đây CÓ sẵn: cu121/torch2.2/mmcv-2.2.0-cp311-manylinux1_x86_64.whl.
# Nhưng nếu mim đọc nhầm phiên bản CUDA/torch, nó ÂM THẦM quay sang build source
# -> kiểm tra ngay sau khi cài, thấy không có .so thì dừng.
$PIP install -U openmim
$PY -m mim install mmengine==0.10.5
$PY -m mim install mmcv==2.2.0
$PY - <<'CHECK'
import sys
import mmcv
try:
    from mmcv import _ext                       # chỉ có khi cài từ wheel/đã build xong
except ImportError:
    sys.exit("❌ mmcv cài xong nhưng thiếu extension biên dịch. Nhiều khả năng mim "
             "không tìm thấy wheel dựng sẵn. Tải tay wheel đúng bản rồi pip install:\n"
             "   https://download.openmmlab.com/mmcv/dist/cu121/torch2.2/index.html")
print("   mmcv", mmcv.__version__, "— có sẵn extension, không phải build từ source")
CHECK

echo "===== [4/8] mmdet 3.3.0 (cho backend WeDetect) + webdataset ====="
# LLMDet tự vendor mmdet trong repo của nó, nhưng WeDetect thì cần mmdet cài thật.
$PIP install mmdet==3.3.0
# mmdet/datasets/__init__.py import webdataset vô điều kiện dù infer không dùng tới
$PIP install webdataset

echo "===== [5/8] Gỡ chốt version mmcv trong mmdet ====="
# mmdet 3.3.0 tự chặn mmcv >= 2.2.0 (assert trong mmdet/__init__.py) trong khi
# LLMDet cần ĐÚNG mmcv == 2.2.0. Comment assertion, KHÔNG hạ version mmcv thật.
$PY - <<'PY'
import importlib.util, pathlib, re, sys

# find_spec CHỈ tra vị trí file, KHÔNG chạy nội dung module — bắt buộc phải vậy:
# `import mmdet` sẽ kích hoạt đúng cái assertion ta đang định gỡ và chết ngay tại
# đây (đã bị: AssertionError "MMCV==2.2.0 is used but incompatible").
spec = importlib.util.find_spec("mmdet")
if spec is None or not spec.origin:
    sys.exit("❌ không tìm thấy mmdet trong env")
p = pathlib.Path(spec.origin)
s = p.read_text()
if "mmcv_maximum_version = '2.3.0'" not in s:
    s = re.sub(r"^(mmcv_maximum_version = ).*$", r"\1'2.3.0'", s, flags=re.M)
    p.write_text(s)
    print(f"   đã nới mmcv_maximum_version -> 2.3.0 trong {p}")
else:
    print("   đã nới từ trước, bỏ qua")
PY

echo "===== [6/8] Các package còn lại ====="
# fairscale là gói THUẦN PYTHON (pip phải dựng từ sdist vì không có wheel, nhưng
# KHÔNG biên dịch gì) — vẫn bắt buộc: mmdet/models/detectors/grounding_dino.py
# import `fairscale.nn.checkpoint` ngay ở đầu file, thiếu là không nạp nổi model.
# KHÔNG cài deepspeed: nó chỉ cần cho huấn luyện, mà build op của nó rất nặng.
$PIP install timm==1.0.9 pycocotools lvis jsonlines fairscale nltk peft terminaltables

echo "===== [7/8] transformers 4.37.2 + ép cứng version dependency đi kèm ====="
# --no-deps: KHÔNG để pip tự kéo tokenizers/huggingface-hub/safetensors bản mới
# nhất — đây là nguyên nhân chính của lỗi ImportError version mismatch.
$PIP install "transformers==4.37.2" --no-deps
$PIP install "tokenizers==0.15.2" "huggingface-hub==0.23.4" "safetensors==0.4.2"

echo "===== [8/8] numpy 1.23.5 + các gói ĐI KÈM numpy — BƯỚC CUỐI ====="
# numpy phải là 1.23.5 (LLMDet/mmdet cần), nên MỌI gói phụ thuộc numpy cũng phải
# lùi về bản còn chấp nhận numpy < 1.25. Bản mới nhất của chúng chặn cứng:
#   matplotlib >= 3.10 -> "Matplotlib requires numpy>=1.25; you have 1.23.5"
#   scipy      >= 1.11 -> chặn tương tự
# mmdet import matplotlib ngay trong đường chạy infer nên thiếu pin này là
# stage 1 chết ngay khi nạp model. --no-deps để chúng không kéo numpy mới về.
$PIP install --force-reinstall --no-deps \
    "numpy==1.23.5" "matplotlib==3.9.2" "scipy==1.10.1"

echo "===== Kiểm tra ====="
$PY -c "
import torch, transformers, mmcv, mmengine, mmdet, numpy, fairscale
from mmcv import _ext                          # extension đã biên dịch sẵn
# torch phải còn nguyên 2.2.1: gói cài sau lỡ kéo bản khác thì mmcv (dựng cho
# torch 2.2 + cu121) sẽ lỗi ABI lúc chạy chứ không lỗi lúc cài.
assert torch.__version__.startswith('2.2.1'), f'torch bị đổi thành {torch.__version__} — cài lại từ bước [2]'
assert numpy.__version__.startswith('1.23'), f'numpy bị đổi thành {numpy.__version__} — chạy lại bước [8]'
import matplotlib, scipy                        # 2 gói hay kéo numpy mới về
from mmdet.apis import DetInferencer            # đúng thứ stage 1 dùng, import thật cho chắc
print('torch        ', torch.__version__, '| CUDA:', torch.cuda.is_available())
print('transformers ', transformers.__version__)
print('mmcv         ', mmcv.__version__)
print('mmengine     ', mmengine.__version__)
print('mmdet        ', mmdet.__version__)
print('numpy        ', numpy.__version__)
print('matplotlib   ', matplotlib.__version__, '| scipy', scipy.__version__)
"
echo
echo "✅ xong. Đặt vào .env:  LLMDET_PYTHON=$PY"

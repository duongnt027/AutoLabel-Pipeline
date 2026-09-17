#!/usr/bin/env bash
# Env cho STAGE 2 (VLX-Seek) và STAGE 1 backend s3 (SAM3).
#   torch 2.8.0 + cu128 | transformers 5.13 (mới có Sam3Model + qwen3_5)
#
#   bash setup/install_env_vlx.sh [tên_env]     # mặc định: vlx
set -e

# CHỐT AN TOÀN: nếu có gói nào lỡ phải biên dịch, không cho nó đẻ 1 job/core.
# Máy nhiều core (112 ở đây) mà build kernel CUDA song song là ăn hết RAM rồi bị
# OOM killer bắn — đúng cái đã xảy ra khi build flash-attn từ source.
export MAX_JOBS="${MAX_JOBS:-4}"
export NVCC_THREADS="${NVCC_THREADS:-2}"

ENV_NAME="${1:-vlx}"
# Tìm conda + nạp `conda activate` (xem setup/_conda.sh)
. "$(dirname "$0")/_conda.sh"

echo "===== [1/6] Tạo conda env $ENV_NAME (python 3.11 + toolkit CUDA) ====="
if conda_env_exists "$ENV_NAME"; then
    echo "   env đã có -> cài tiếp các bước còn thiếu"
else
    conda create -y -n "$ENV_NAME" python=3.11
fi
# CUDA toolkit (TUỲ CHỌN). Phải ghim KÊNH ĐÚNG BẢN 12.8 cho khớp torch cu128:
# dùng `-c nvidia` trần thì conda kéo cuda-version 12.9 và chết ở post-link
# ("This cross-compiler package contains no program .../x86_64-conda-linux-gnu-g++").
# KHÔNG bắt buộc cho việc chạy pipeline: triton 3.4 đã tự mang theo ptxas/nvdisasm/
# cuobjdump, còn flash-attn và causal-conv1d là wheel dựng sẵn nên không biên dịch
# gì lúc chạy. Vì vậy hỏng bước này chỉ CẢNH BÁO, không giết cả lượt cài.
# -n "$ENV_NAME" thay vì activate: chỉ đúng env này, không phụ thuộc PATH.
if conda install -y -n "$ENV_NAME" -c nvidia/label/cuda-12.8.1 \
        cuda-nvcc cuda-cudart-dev cuda-cccl; then
    echo "   đã cài CUDA toolkit 12.8"
else
    echo "   ⚠️  không cài được CUDA toolkit qua conda — BỎ QUA, pipeline vẫn chạy được"
    echo "      (triton tự mang ptxas; flash-attn/causal-conv1d là wheel dựng sẵn)"
fi

# KHÔNG `conda activate` rồi gọi `pip` trần — xem chú thích trong
# install_env_llmdet.sh: đã bị gói chui vào env base vì chuyện đó.
PY="$CONDA_BASE/envs/$ENV_NAME/bin/python"
PIP="$PY -m pip"
[ -x "$PY" ] || { echo "❌ không thấy $PY sau khi tạo env"; exit 1; }
$PY -c "import sys; assert sys.version_info[:2] == (3, 11), sys.version; \
        print('   python của env:', sys.executable)"

$PIP install -U pip setuptools wheel

echo "===== [2/6] PyTorch 2.8.0 + CUDA 12.8 ====="
# GHIM ĐÚNG 2.8.0: các wheel dựng sẵn của flash-attn / causal-conv1d bên dưới
# build theo đúng bản torch này. Chọn torch khác là phải build 2 gói đó từ
# source — việc đó ăn hết RAM máy và mất hàng giờ, ĐỪNG làm.
$PIP install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128

echo "===== [3/6] flash-attn + causal-conv1d (WHEEL DỰNG SẴN, KHÔNG build source) ====="
# ⛔ TUYỆT ĐỐI KHÔNG chạy `pip install flash-attn` / `pip install causal-conv1d`:
# hai gói này build kernel CUDA từ source, ăn sạch RAM máy (đã bị một lần) và
# mất hàng giờ. Luôn cài bằng ĐÚNG file .whl dựng sẵn dưới đây.
#
# Tên wheel phải khớp CẢ BỐN: cu12 · torch2.8 · cxx11abiTRUE · cp311.
#   - torch2.8   : đúng bản torch cài ở bước [2]
#   - cxx11abiTRUE: khớp torch._C._GLIBCXX_USE_CXX11_ABI (torch 2.8 cu128 = True)
#   - cp311      : python 3.11 của env này
# Sai bất kỳ mảnh nào thì pip báo "not a supported wheel on this platform" và
# script DỪNG (set -e) — nó KHÔNG âm thầm quay sang build source.
#
# ⚠️ Số hiệu TAG không phải lúc nào cũng bằng version trong tên file: wheel
# causal_conv1d-1.6.1 nằm dưới tag v1.6.2. Link đổi thì tra ở trang releases
# chứ đừng cài theo tên gói.
FLASH_WHL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl"
CONV_WHL="https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.6.2/causal_conv1d-1.6.1+cu12torch2.8cxx11abiTRUE-cp311-cp311-linux_x86_64.whl"
# --only-binary=:all: là chốt cuối: có trục trặc gì thì pip BÁO LỖI chứ không build.
$PIP install --no-build-isolation --only-binary=:all: "$FLASH_WHL"
$PIP install --no-build-isolation --only-binary=:all: "$CONV_WHL"

echo "===== [4/6] transformers 5.13 + fla-core (linear attention của Qwen3.5) ====="
$PIP install "transformers==5.13.0" accelerate==1.4.0 timm==1.0.9 einops==0.6.1
$PIP install "fla-core==0.5.2"

echo "===== [5/6] Tiện ích ====="
$PIP install pillow pyyaml huggingface_hub

echo "===== [6/6] Kiểm tra ====="
# Quan trọng nhất: torch PHẢI vẫn là 2.8.0. Một gói cài sau lỡ kéo torch bản khác
# thì flash-attn/causal-conv1d dựng cho 2.8 sẽ lỗi ABI lúc chạy chứ không lỗi lúc cài.
$PY -c "
import torch, transformers
from transformers import Sam3Model            # stage 1 backend s3
import transformers.models.qwen3_5            # stage 2 VLX-Seek
import flash_attn, flash_attn_2_cuda          # import cả .so -> chắc chắn khớp ABI
import causal_conv1d
assert torch.__version__.startswith('2.8.0'), f'torch bị đổi thành {torch.__version__} — cài lại bước [2] rồi bước [3]'
print('torch        ', torch.__version__, '| CUDA:', torch.cuda.is_available())
print('transformers ', transformers.__version__)
print('flash_attn   ', flash_attn.__version__, '(wheel dựng sẵn, .so nạp được)')
print('causal_conv1d', causal_conv1d.__version__)
print('Sam3Model    ', Sam3Model.__name__)
"
echo
echo "✅ xong. Đặt vào .env:  VLX_PYTHON=$PY"

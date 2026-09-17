#!/usr/bin/env bash
# Clone các repo bên thứ ba + tải weight cho stage 1 & 2.
# Không repo/weight nào được commit vào repo này (vài GB mỗi cái) — xem docs/WEIGHTS.md
# nếu muốn tải tay hoặc chỉ tải phần cho backend mình dùng.
#
#   bash setup/get_third_party.sh            # LLMDet + VLX-Seek (mặc định, đủ chạy)
#   bash setup/get_third_party.sh all        # thêm SAM3 + WeDetect
set -e

WHAT="${1:-default}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# HF_TOKEN nằm trong .env chứ không phải biến shell -> tự lấy ra. Thiếu token thì
# chỉ SAM3 bị bỏ qua (gated repo), phần còn lại vẫn tải bình thường.
if [ -z "$HF_TOKEN" ] && [ -f "$ROOT/.env" ]; then
    HF_TOKEN="$(grep -E '^HF_TOKEN=' "$ROOT/.env" | tail -1 | cut -d= -f2- | tr -d '"'"'"'"' )"
    export HF_TOKEN
fi
TP="$ROOT/third_party"
mkdir -p "$TP"
cd "$TP"

clone() {  # clone <url> <dir> <commit đã kiểm chứng>
    [ -d "$2" ] && { echo "   $2 đã có, bỏ qua"; return; }
    git clone --depth 50 "$1" "$2"
    git -C "$2" checkout "$3"
}

echo "===== LLMDet (stage 1, backend llm) ====="
clone https://github.com/iSEE-Laboratory/LLMDet.git LLMDet 5336624
# Checkpoint chính (~2.8GB) + các model phụ mà config trỏ tới bằng '../huggingface/...'
mkdir -p "$TP/huggingface/mm_grounding_dino" "$TP/LLMDet/pretrained_models"
export HF_HUB_DISABLE_XET=1     # backend tải mới của HF đang 404 ở endpoint refresh token
python - <<PY
from huggingface_hub import snapshot_download, hf_hub_download
import shutil
# 1. LLMDet swin-L đã fine-tune (file large.pth)
p = hf_hub_download(repo_id='fushh7/LLMDet', filename='large.pth')
shutil.copy(p, "$TP/LLMDet/pretrained_models/large.pth")
# 2. BERT (text encoder) + SigLIP (vision) + LLaVA-OneVision đã fine-tune bởi tác giả
# chỉ lấy bản pytorch: repo còn có tf/flax/onnx/coreml, tải hết là 3.3GB thay vì ~450MB
snapshot_download(repo_id='bert-base-uncased', local_dir="$TP/huggingface/bert-base-uncased",
                  allow_patterns=['config.json', 'pytorch_model.bin', 'tokenizer*.json', 'vocab.txt'])
snapshot_download(repo_id='google/siglip-so400m-patch14-384',
                  local_dir="$TP/huggingface/siglip-so400m-patch14-384")
snapshot_download(repo_id='fushh7/LLMDet',
                  allow_patterns=['my_llava-onevision-qwen2-0.5b-ov-2/*'],
                  local_dir="$TP/huggingface")
PY
# 3. Backbone mm_grounding_dino mà config nạp qua load_from
wget -c -P "$TP/huggingface/mm_grounding_dino" \
  https://download.openmmlab.com/mmdetection/v3.0/mm_grounding_dino/grounding_dino_swin-l_pretrain_obj365_goldg/grounding_dino_swin-l_pretrain_obj365_goldg-34dcdc53.pth

echo "===== VLX-Seek (stage 2) ====="
clone https://github.com/om-ai-lab/VLX-Seek.git VLX-Seek 7f0e869
# Checkpoint ~20GB, tự tải về HF cache ở lần chạy đầu. Tải trước cho chắc:
python -c "
from huggingface_hub import snapshot_download
snapshot_download(repo_id='omlab/VLX-Seek-1.5-10B')
"

if [ "$WHAT" = "all" ]; then
    echo "===== SAM3 (stage 1, backend s3) ====="
    # GATED: phải accept license ở https://huggingface.co/facebook/sam3 và đặt
    # HF_TOKEN trong .env. Không có token thì BỎ QUA chứ không làm hỏng cả lượt
    # tải (LLMDet/VLX-Seek phía trên đã xong, mất công tải lại).
    if [ -z "$HF_TOKEN" ]; then
        echo "   ⚠️  chưa có HF_TOKEN trong .env -> bỏ qua SAM3 (backend s3 sẽ không chạy được)"
    else
        python -c "
import os
from huggingface_hub import login, snapshot_download
login(token=os.environ['HF_TOKEN'], add_to_git_credential=False)
snapshot_download(repo_id='facebook/sam3')
"
    fi
    echo "===== WeDetect (stage 1, backend we) ====="
    clone https://github.com/WeChatCV/WeDetect.git WeDetect dd302db
    # Text encoder XLM-RoBERTa: config nạp bằng đường dẫn TƯƠNG ĐỐI './xlm-roberta-base/'
    python - <<PY
from huggingface_hub import snapshot_download
# chỉ cần tokenizer + config: trọng số text encoder nằm sẵn trong checkpoint WeDetect
snapshot_download(repo_id='FacebookAI/xlm-roberta-base', local_dir="$TP/WeDetect/xlm-roberta-base",
                  allow_patterns=['config.json', 'tokenizer.json', 'sentencepiece.bpe.model'])
PY
    python - <<PY
from huggingface_hub import hf_hub_download
import shutil, pathlib
dst = pathlib.Path("$TP/WeDetect/checkpoints"); dst.mkdir(parents=True, exist_ok=True)
shutil.copy(hf_hub_download(repo_id='fushh7/WeDetect', filename='wedetect_base.pth'),
            dst / 'wedetect_base.pth')
print(f"   -> đặt WEDETECT_WEIGHTS={dst / 'wedetect_base.pth'} trong .env")
PY
fi

echo
echo "✅ xong → $TP"

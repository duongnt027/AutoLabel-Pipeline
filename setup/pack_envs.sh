#!/usr/bin/env bash
# Đóng gói 2 conda env thành file .tar.gz để bê nguyên sang máy khác (conda-pack)
# — dùng khi máy đích KHÔNG có mạng hoặc muốn chắc chắn giống hệt máy nguồn.
# Máy có mạng thì cứ chạy 2 script install_env_*.sh, nhanh và nhẹ hơn nhiều.
#
#   bash setup/pack_envs.sh [env_llmdet] [env_vlx] [thư_mục_ra]
#   # mặc định: llmdet  vlx  dist/
#
# File ra KHÔNG commit lên git (mỗi env ~5-8GB nén) — dist/ đã nằm trong .gitignore.
#
# Bung ở máy đích:
#   mkdir -p ~/miniconda3/envs/llmdet && tar -xzf llmdet.tar.gz -C ~/miniconda3/envs/llmdet
#   ~/miniconda3/envs/llmdet/bin/conda-unpack      # sửa lại đường dẫn tuyệt đối
# Máy đích phải cùng kiến trúc (linux-x86_64) và có driver NVIDIA đủ mới.
set -e

ENV_DET="${1:-llmdet}"
ENV_VLX="${2:-vlx}"
OUT_DIR="${3:-dist}"

. "$(dirname "$0")/_conda.sh"
conda activate base
python -c "import conda_pack" 2>/dev/null || conda install -y -c conda-forge conda-pack

mkdir -p "$OUT_DIR"
for env in "$ENV_DET" "$ENV_VLX"; do
    echo "===== đóng gói $env → $OUT_DIR/$env.tar.gz ====="
    # -f: ghi đè file cũ | --ignore-missing-files: bỏ qua file đã bị xoá tay trong env
    conda pack -n "$env" -o "$OUT_DIR/$env.tar.gz" -f --ignore-missing-files
done

ls -lh "$OUT_DIR"
echo "✅ xong."

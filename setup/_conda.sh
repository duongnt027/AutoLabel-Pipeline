#!/usr/bin/env bash
# Tìm conda rồi nạp hàm `conda activate`. Các script khác source file này.
#
# Vì sao cần: shell KHÔNG TƯƠNG TÁC (chạy script qua ssh, cron, CI) không đọc
# ~/.bashrc nên `conda` thường không có trong PATH. Khi đó
# `eval "$(conda shell.bash hook)"` chỉ in "command not found" rồi eval chuỗi
# rỗng — TRẢ VỀ 0, `set -e` không chặn, và script chạy tiếp như không có gì.

_find_conda_base() {
    if [ -n "$CONDA_EXE" ] && [ -x "$CONDA_EXE" ]; then
        dirname "$(dirname "$CONDA_EXE")"; return 0
    fi
    if command -v conda > /dev/null 2>&1; then
        conda info --base; return 0
    fi
    for d in "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/miniforge3" "$HOME/mambaforge" /opt/conda; do
        [ -x "$d/bin/conda" ] && { echo "$d"; return 0; }
    done
    return 1
}

CONDA_BASE="$(_find_conda_base)" || {
    echo "❌ Không tìm thấy conda. Cài miniconda rồi chạy lại, hoặc đặt sẵn biến:" >&2
    echo "   export CONDA_EXE=/duong/dan/den/conda/bin/conda" >&2
    exit 1
}
# shellcheck disable=SC1091
. "$CONDA_BASE/etc/profile.d/conda.sh"

# ⛔ ĐỪNG thêm `export PATH="$CONDA_BASE/bin:$PATH"` ở đây. Đã bị một lần:
# khi shell gọi script đang activate một env KHÁC BASE, `conda activate X` chỉ
# THAY THẾ TẠI CHỖ mục của env cũ trong PATH — bản sao base/bin chèn thêm vẫn
# đứng trước, nên `pip`/`python` trần trúng của BASE dù CONDA_PREFIX đã đúng.
# Hậu quả: gói được cài vào env base mà không ai biết.
# (Không lộ ra khi test với base đang active, vì khi đó conda ghi đè đúng mục đó.)
# Vì vậy các script cài gọi thẳng $CONDA_BASE/envs/<env>/bin/python, không dựa PATH.

# Có env tên này chưa? (khớp CHÍNH XÁC tên, không phải khớp một phần)
conda_env_exists() {
    conda env list | awk '{print $1}' | grep -qx "$1"
}

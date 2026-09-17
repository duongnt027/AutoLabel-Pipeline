#!/usr/bin/env bash
# Xoá sạch những gì setup/try_install.sh vừa tạo.
#
#   bash setup/uninstall.sh                    # xoá env llmdet + vlx, third_party/, runs/
#   bash setup/uninstall.sh env1 env2          # tên env khác
#   bash setup/uninstall.sh --caches           # xoá thêm pip cache + gói conda tải về
#   bash setup/uninstall.sh --yes              # không hỏi lại
#
# In ra DANH SÁCH sẽ xoá kèm dung lượng rồi mới hỏi. Không xoá gì ngoài danh sách đó.
set -e

YES=0
CACHES=0
ARGS=()
for a in "$@"; do
    case "$a" in
        --yes|-y) YES=1 ;;
        --caches) CACHES=1 ;;
        *) ARGS+=("$a") ;;
    esac
done
ENV_DET="${ARGS[0]:-llmdet}"
ENV_VLX="${ARGS[1]:-vlx}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
. "$(dirname "$0")/_conda.sh"      # đặt CONDA_BASE + conda_env_exists()

# Những env này KHÔNG phải do repo tạo ra — chặn cứng để không xoá nhầm env đang chạy
# việc khác trên máy. Muốn xoá thật thì tự chạy `conda env remove -n <tên>`.
PROTECTED="base duongnt vlx_seek"
for env in "$ENV_DET" "$ENV_VLX"; do
    for p in $PROTECTED; do
        [ "$env" = "$p" ] && { echo "❌ '$env' nằm trong danh sách bảo vệ ($PROTECTED) — từ chối xoá."; exit 1; }
    done
done

# ── Liệt kê ──────────────────────────────────────────────────────────────────
TARGETS=()
# `|| true` bắt buộc: thiếu nó thì đường dẫn KHÔNG tồn tại làm hàm trả về 1,
# và `set -e` giết script ngay ở lần add hụt đầu tiên — im lặng, không báo gì.
add() { [ -e "$1" ] && TARGETS+=("$1") || true; }
add "$CONDA_BASE/envs/$ENV_DET"
add "$CONDA_BASE/envs/$ENV_VLX"
add "$ROOT/third_party"
add "$ROOT/runs"
add "$ROOT/dist"
add "$ROOT/.try_install_logs"
# Weight repo này tải vào HF cache (snapshot_download không dùng local_dir)
add "$HOME/.cache/huggingface/hub/models--fushh7--LLMDet"
add "$HOME/.cache/huggingface/hub/models--fushh7--WeDetect"

if [ ${#TARGETS[@]} -eq 0 ]; then
    echo "Không có gì để xoá (chưa cài, hoặc đã xoá rồi)."
    exit 0
fi

echo "=============================================================="
echo " SẼ XOÁ:"
for t in "${TARGETS[@]}"; do
    printf "   %-8s %s\n" "$(du -sh "$t" 2>/dev/null | cut -f1)" "$t"
done
[ $CACHES = 1 ] && echo "   + pip cache ($(du -sh "$HOME/.cache/pip" 2>/dev/null | cut -f1)) và gói conda tải về"
echo
echo " GIỮ NGUYÊN (không đụng tới):"
for keep in "$HOME/.cache/huggingface/hub/models--omlab--VLX-Seek-1.5-10B" \
            "$HOME/.cache/huggingface/hub/models--facebook--sam3"; do
    [ -e "$keep" ] && printf "   %-8s %s\n" "$(du -sh "$keep" 2>/dev/null | cut -f1)" "$keep"
done
echo "   (checkpoint dùng chung với việc khác trên máy — muốn xoá thì rm -rf tay)"
echo "=============================================================="

if [ $YES = 0 ]; then
    read -r -p "Xoá những thứ trên? gõ 'yes' để xác nhận: " answer
    [ "$answer" = "yes" ] || { echo "Huỷ, không xoá gì."; exit 0; }
fi

# ── Xoá ──────────────────────────────────────────────────────────────────────
for env in "$ENV_DET" "$ENV_VLX"; do
    if conda_env_exists "$env"; then
        echo "   gỡ conda env $env"
        conda env remove -n "$env" -y > /dev/null
    fi
done
for t in "${TARGETS[@]}"; do
    [ -e "$t" ] && { echo "   xoá $t"; rm -rf "$t"; }
done
find "$ROOT" -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true

if [ $CACHES = 1 ]; then
    echo "   dọn pip cache + gói conda"
    "$CONDA_BASE/bin/pip" cache purge > /dev/null 2>&1 || true
    "$CONDA_BASE/bin/conda" clean -a -y > /dev/null
fi

# Trả .env cũ về nếu try_install.sh đã sao lưu
if [ -f "$ROOT/.env.bak" ]; then
    mv "$ROOT/.env.bak" "$ROOT/.env"
    echo "   khôi phục .env từ .env.bak"
fi

echo
echo "✅ đã xoá sạch. Đĩa trống: $(df -h "$HOME" | tail -1 | awk '{print $4}')"

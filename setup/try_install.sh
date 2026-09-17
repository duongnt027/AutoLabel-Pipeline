#!/usr/bin/env bash
# Cài THỬ toàn bộ: 2 conda env + (tuỳ chọn) repo/weight, có ĐO RAM ĐỈNH trong
# lúc cài để biết chắc không bị tràn. Xoá sạch lại bằng setup/uninstall.sh.
#
#   bash setup/try_install.sh                      # chỉ 2 env (~20GB đĩa)
#   bash setup/try_install.sh --weights            # thêm LLMDet + VLX-Seek (đủ chạy mặc định)
#   bash setup/try_install.sh --weights-all        # thêm cả SAM3 + WeDetect
#   bash setup/try_install.sh --weights llmdet vlx # đặt tên env khác
#
# ĐÃ CÓ repo/weight ở chỗ khác trên máy thì ĐỪNG tải lại: bỏ cờ --weights và trỏ
# LLMDET_REPO / VLX_REPO / WEDETECT_* trong .env sang đường dẫn sẵn có.
#
# Script DỪNG NGAY nếu tên env đã tồn tại — không bao giờ ghi đè env đang dùng.
set -e
# pipefail BẮT BUỘC: `bash install_env_x.sh | tee log` lấy mã lỗi của TEE (=0),
# thiếu nó thì cài hỏng vẫn chạy tiếp và báo "cài xong".
set -o pipefail

WEIGHTS=0          # 0 = không tải | 1 = default (llm + vlx) | 2 = all (thêm s3 + we)
case "$1" in
    --weights)     WEIGHTS=1; shift ;;
    --weights-all) WEIGHTS=2; shift ;;
esac
ENV_DET="${1:-llmdet}"
ENV_VLX="${2:-vlx}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG_DIR="$ROOT/.try_install_logs"
mkdir -p "$LOG_DIR"

echo "=============================================================="
echo " CÀI THỬ auto_detect"
echo "   env stage 1 : $ENV_DET"
echo "   env stage 2 : $ENV_VLX"
case $WEIGHTS in
    0) W_DESC="không tải (thêm --weights nếu muốn)" ;;
    1) W_DESC="LLMDet + VLX-Seek" ;;
    2) W_DESC="LLMDet + VLX-Seek + SAM3 + WeDetect" ;;
esac
echo "   repo/weight : $W_DESC"
echo "   log         : $LOG_DIR"
echo "=============================================================="

# ── Kiểm tra trước khi động vào gì ───────────────────────────────────────────
. "$(dirname "$0")/_conda.sh"
for env in "$ENV_DET" "$ENV_VLX"; do
    if conda_env_exists "$env"; then
        echo "❌ env '$env' ĐÃ TỒN TẠI. Script này không ghi đè env có sẵn."
        echo "   Đặt tên khác:  bash setup/try_install.sh ${env}_test ..."
        exit 1
    fi
done

# Đo thật trên 2 env tương đương: 7.8GB + 11GB = ~19GB, gần hết là thư viện CUDA
# của nvidia (3.4GB + 4.4GB) và torch. Cộng thêm ~8GB wheel nằm lại trong pip
# cache — phần này lấy lại được bằng `pip cache purge` sau khi cài xong.
# Weight: LLMDet ~15GB (kể cả model phụ), thêm SAM3 + WeDetect ~5GB nữa.
NEED_GB=$((30 + WEIGHTS * 15))
FREE_GB=$(df -BG --output=avail "$HOME" | tail -1 | tr -dc '0-9')
echo "   đĩa trống: ${FREE_GB}GB (cần ~${NEED_GB}GB = env ~19GB + pip cache ~8GB tạm$([ $WEIGHTS -gt 0 ] && echo " + weight"))"
[ "$FREE_GB" -lt "$NEED_GB" ] && { echo "❌ không đủ đĩa"; exit 1; }

# .env đang dùng có thể trỏ tới env khác -> giữ lại trước khi bạn sửa
if [ -f "$ROOT/.env" ] && [ ! -f "$ROOT/.env.bak" ]; then
    cp "$ROOT/.env" "$ROOT/.env.bak"
    echo "   đã sao lưu .env -> .env.bak"
fi

# ── Đo RAM đỉnh của CẢ MÁY trong lúc cài ─────────────────────────────────────
# Đây là câu trả lời cho "có tràn RAM không": nếu có gói nào lỡ build từ source
# thì đỉnh RAM sẽ nhảy vọt và thấy ngay ở đây.
RAM_LOG="$LOG_DIR/ram.log"
: > "$RAM_LOG"
( while true; do
      free -m | awk '/^Mem:/{print $3}' >> "$RAM_LOG"
      sleep 5
  done ) &
SAMPLER=$!
# Dừng bộ đo dù script kết thúc kiểu gì
trap 'kill $SAMPLER 2>/dev/null || true' EXIT

RAM_BEFORE=$(free -m | awk '/^Mem:/{print $3}')
DISK_BEFORE=$(df -BM --output=avail "$HOME" | tail -1 | tr -dc '0-9')
PIP_BEFORE=$(du -sm "$HOME/.cache/pip" 2>/dev/null | cut -f1 || echo 0)
START=$(date +%s)

# ── Cài ──────────────────────────────────────────────────────────────────────
echo
echo "───── [1] env stage 1: $ENV_DET ─────"
bash "$ROOT/setup/install_env_llmdet.sh" "$ENV_DET" 2>&1 | tee "$LOG_DIR/llmdet.log"

echo
echo "───── [2] env stage 2: $ENV_VLX ─────"
bash "$ROOT/setup/install_env_vlx.sh" "$ENV_VLX" 2>&1 | tee "$LOG_DIR/vlx.log"

if [ $WEIGHTS -gt 0 ]; then
    echo
    echo "───── [3] repo bên thứ ba + weight: $W_DESC ─────"
    # get_third_party.sh cần huggingface_hub -> dùng python của env vừa tạo
    PATH="$CONDA_BASE/envs/$ENV_VLX/bin:$PATH" \
        bash "$ROOT/setup/get_third_party.sh" "$([ $WEIGHTS = 2 ] && echo all || echo default)" \
        2>&1 | tee "$LOG_DIR/weights.log"
fi

kill $SAMPLER 2>/dev/null || true
ELAPSED=$(( $(date +%s) - START ))

# ── Báo cáo ──────────────────────────────────────────────────────────────────
RAM_PEAK=$(sort -n "$RAM_LOG" | tail -1)
RAM_TOTAL=$(free -m | awk '/^Mem:/{print $2}')
echo
echo "=============================================================="
echo " ✅ CÀI XONG trong $((ELAPSED / 60)) phút $((ELAPSED % 60)) giây"
echo
printf " RAM cả máy:  trước %d MB  →  đỉnh %d MB  (tổng %d MB)\n" \
       "$RAM_BEFORE" "$RAM_PEAK" "$RAM_TOTAL"
printf " tức là quá trình cài làm tăng tối đa %d MB\n" "$((RAM_PEAK - RAM_BEFORE))"
echo " (số này nhảy lên hàng chục GB = có gói build từ source, KHÔNG được phép)"
echo
# Đĩa: tách phần Ở LẠI (env) và phần TẠM (pip cache) để biết cái nào xoá được
DISK_AFTER=$(df -BM --output=avail "$HOME" | tail -1 | tr -dc '0-9')
PIP_AFTER=$(du -sm "$HOME/.cache/pip" 2>/dev/null | cut -f1 || echo 0)
printf " Đĩa đã dùng: %.1f GB tổng, trong đó ~%.1f GB là pip cache\n" \
       "$(echo "$DISK_BEFORE $DISK_AFTER" | awk '{print ($1-$2)/1024}')" \
       "$(echo "$PIP_BEFORE $PIP_AFTER" | awk '{print ($2-$1)/1024}')"
for env in "$ENV_DET" "$ENV_VLX"; do
    printf "   env %-10s %s\n" "$env" "$(du -sh "$CONDA_BASE/envs/$env" 2>/dev/null | cut -f1)"
done
echo " Lấy lại phần pip cache:  pip cache purge"
echo
echo " Đặt vào .env:"
echo "   LLMDET_PYTHON=$CONDA_BASE/envs/$ENV_DET/bin/python"
echo "   VLX_PYTHON=$CONDA_BASE/envs/$ENV_VLX/bin/python"
echo
echo " Chạy thử:   python3 run_pipeline.py /duong/dan/anh --limit 5"
echo " Xoá sạch:   bash setup/uninstall.sh $ENV_DET $ENV_VLX"
echo "=============================================================="

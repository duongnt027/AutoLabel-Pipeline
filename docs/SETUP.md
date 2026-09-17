# Cài đặt

Pipeline cần **2 conda env tách biệt** vì các model không sống chung được trong
một tiến trình:

| env | dùng cho | version chính |
|---|---|---|
| `llmdet` | stage 1: backend `llm` (LLMDet), `we` (WeDetect) | torch 2.2.1+cu121, mmcv 2.2.0, mmdet 3.3.0, transformers 4.37.2, numpy 1.23.5 |
| `vlx` | stage 1: backend `s3` (SAM3) · stage 2: VLX-Seek | torch 2.8.0+cu128, transformers 5.13.0, flash-attn 2.7.3 |

Runner (`run_pipeline.py`, `run_stage1.py`) chạy bằng **python3 nào cũng được**
(chỉ dùng stdlib): nó gọi từng stage bằng python của env tương ứng theo `.env`.

## 1. Cài — một lệnh

```bash
bash setup/try_install.sh --weights-all      # 2 env + repo + toàn bộ weight
```

Số đo thật của một lượt chạy trọn vẹn trên máy 112 core / 503 GB RAM:

```
✅ CÀI XONG trong 22 phút 38 giây
 RAM cả máy:  244885 MB → đỉnh 245239 MB   ⇒ cả quá trình chỉ tăng 354 MB
 Đĩa đã dùng: 34.6 GB (env llmdet 6.3G + env vlx 9.5G + repo/weight ~18G)
              trong đó 1.6 GB là pip cache, lấy lại bằng `pip cache purge`
```

354 MB là bằng chứng **không gói nào bị build từ source** — số này mà nhảy lên
hàng chục GB thì có chỗ đang biên dịch, xem [mục dưới](#-không-bao-giờ-để-pip-build-từ-source).

`try_install.sh` dừng ngay nếu tên env đã tồn tại (không bao giờ ghi đè env đang
dùng), đo RAM đỉnh 5 giây một lần, và ghi log từng bước vào `.try_install_logs/`.

Các biến thể:

```bash
bash setup/try_install.sh                 # chỉ 2 env (~16 GB)
bash setup/try_install.sh --weights       # + LLMDet + VLX-Seek (đủ chạy mặc định)
bash setup/try_install.sh --weights-all   # + SAM3 + WeDetect
bash setup/try_install.sh --weights e1 e2 # đặt tên env khác
```

### Xoá sạch

```bash
bash setup/uninstall.sh              # liệt kê kèm dung lượng → hỏi 'yes' → xoá
bash setup/uninstall.sh --yes        # không hỏi
bash setup/uninstall.sh --caches     # xoá thêm pip cache + gói conda tải về
```

Chỉ xoá đúng những gì đã liệt kê, từ chối các env trong danh sách bảo vệ, và
**không đụng** checkpoint dùng chung trong HF cache (VLX-Seek 19 GB, SAM3 3.3 GB).

### Hoặc chạy từng script

```bash
bash setup/install_env_llmdet.sh          # env tên llmdet
bash setup/install_env_vlx.sh             # env tên vlx
```

Muốn tên khác thì truyền vào: `bash setup/install_env_llmdet.sh env_cua_toi`.

**Đứt giữa chừng thì cứ chạy lại chính lệnh đó** — env có sẵn sẽ được dùng lại và
chỉ cài tiếp phần còn thiếu (mọi bước pip đều idempotent), không phải tải lại 6 GB
torch. Riêng `try_install.sh` thì từ chối env đã tồn tại, nên trường hợp này gọi
thẳng 2 script trên.

### ⛔ Không bao giờ để pip build từ source

Đây là chỗ duy nhất trong cả repo có thể **ăn sạch RAM máy rồi bị OOM killer bắn**
(đã bị một lần, build hàng giờ rồi chết). Ba gói có extension CUDA:

| gói | cài đúng cách | nếu làm sai |
|---|---|---|
| `flash-attn` | **URL wheel dựng sẵn** trong script | `pip install flash-attn` → build source → hết RAM |
| `causal-conv1d` | **URL wheel dựng sẵn** trong script | `pip install causal-conv1d` → build source → hết RAM |
| `mmcv` | `mim install mmcv==2.2.0` (mim tự lấy wheel) | mim không thấy wheel → âm thầm build source |

Script đã chốt ba lớp:

1. `MAX_JOBS=4` + `NVCC_THREADS=2` export ở đầu — lỡ có gì phải biên dịch thì nó
   cũng không đẻ 112 job song song (mặc định là 1 job/core).
2. `--only-binary=:all:` khi cài flash-attn/causal-conv1d: trục trặc thì pip
   **báo lỗi**, `set -e` dừng script, chứ không quay sang build.
3. Cài mmcv xong kiểm tra `from mmcv import _ext` ngay; thiếu extension là dừng.

Tên wheel phải khớp **cả bốn mảnh**: `cu12` · `torch2.8` · `cxx11abiTRUE` · `cp311`.
Sai một mảnh thì pip báo "not a supported wheel on this platform".

⚠️ Số hiệu tag GitHub **không phải lúc nào cũng bằng version trong tên file**:
wheel `causal_conv1d-1.6.1+cu12torch2.8...` nằm dưới tag **`v1.6.2`**. Link hỏng
thì mở trang releases tra lại tên file, đừng quay sang `pip install <tên gói>`.

Cuối mỗi script có `assert` chốt lại torch (và numpy) vẫn đúng bản: một gói cài
sau lỡ kéo torch khác thì flash-attn/mmcv dựng cho bản cũ sẽ lỗi ABI **lúc chạy**,
không lỗi lúc cài — rất khó truy.

`deepspeed` bị bỏ hẳn (build op rất nặng, chỉ cần khi huấn luyện). `fairscale`
vẫn bắt buộc vì `grounding_dino.py` import nó, nhưng là gói thuần python.

### Các version khoá chặt vào nhau

Vài chỗ dễ vỡ, đã ghim sẵn trong script — **đừng "nâng cấp cho mới"**:

- `transformers==4.37.2` cài với `--no-deps` rồi ghim tay `tokenizers` /
  `huggingface-hub` / `safetensors`; `numpy==1.23.5` cài **sau cùng**. Đây là
  nguyên nhân số một của lỗi `ImportError: version mismatch` trong env này.
- `mmdet 3.3.0` tự chặn `mmcv >= 2.2.0` trong khi LLMDet cần đúng `2.2.0` →
  script nới `mmcv_maximum_version` trong `mmdet/__init__.py`, **không** hạ
  version mmcv thật.
- `matplotlib==3.9.2` và `scipy==1.10.1` phải ghim cùng lúc với numpy: bản mới
  hơn chặn cứng numpy < 1.25 (`ImportError: Matplotlib requires numpy>=1.25;
  you have 1.23.5`). `mmdet` import matplotlib ngay trên đường chạy infer nên
  thiếu pin này là **stage 1 chết lúc nạp model**, không phải lúc cài.
- CUDA toolkit của env `vlx` lấy từ kênh `nvidia/label/cuda-12.8.1` cho khớp
  torch cu128. Dùng `-c nvidia` trần thì conda kéo cuda-version 12.9 và chết ở
  post-link (`This cross-compiler package contains no program ...g++`). Bước này
  không bắt buộc nên hỏng cũng chỉ cảnh báo: triton tự mang `ptxas`, còn
  flash-attn/causal-conv1d là wheel dựng sẵn.

Repo VLX-Seek khai `torch==2.10 / flash_attn==2.8.3` trong `requirements.txt`,
nhưng tổ hợp đã chạy thật ở đây là **torch 2.8.0 + flash-attn 2.7.3** — script
cài đúng tổ hợp này.

### Bê nguyên env sang máy khác (không có mạng)

```bash
bash setup/pack_envs.sh                   # -> dist/llmdet.tar.gz, dist/vlx.tar.gz
```

Bung ở máy đích (cùng linux-x86_64, driver NVIDIA đủ mới):

```bash
mkdir -p ~/miniconda3/envs/llmdet
tar -xzf llmdet.tar.gz -C ~/miniconda3/envs/llmdet
~/miniconda3/envs/llmdet/bin/conda-unpack
```

`setup/requirements-*.lock.txt` là **ảnh chụp toàn bộ** env đang chạy, để tra
cứu/đối chiếu version. Đừng `pip install -r` thẳng vào env trống: torch,
flash-attn, mmcv cần index/wheel riêng — dùng 2 script install ở trên.

## 2. Tải repo bên thứ ba + weight

```bash
bash setup/get_third_party.sh             # LLMDet + VLX-Seek  (đủ để chạy mặc định)
bash setup/get_third_party.sh all         # thêm SAM3 + WeDetect
```

Chi tiết từng file, dung lượng và link tải tay: [WEIGHTS.md](WEIGHTS.md).

## 3. Cấu hình

```bash
cp .env.example .env
```

Tối thiểu phải sửa:

```ini
LLMDET_PYTHON=/duong/dan/den/envs/llmdet/bin/python
VLX_PYTHON=/duong/dan/den/envs/vlx/bin/python
LLMDET_GPU=0        # GPU chạy stage 1
VLX_GPUS=0          # GPU chạy stage 2, nhiều GPU: 0,1
HF_TOKEN=hf_...     # chỉ cần cho backend s3 (SAM3 là gated repo)
```

`WEDETECT_WEIGHTS` thì lấy đúng đường dẫn mà `get_third_party.sh` in ra ở cuối.

Mọi tuỳ chọn khác đã có giá trị mặc định kèm giải thích ngay trong `.env.example`.

⚠️ Khai **trùng key** trong `.env` thì **dòng cuối thắng** (giống shell và
python-dotenv). Biến đặt sẵn ở shell vẫn thắng cả file.

## 4. Kiểm tra

```bash
python3 run_pipeline.py /duong/dan/anh --limit 2
```

Bước đầu tiên của cả 2 runner là thử `import` trong từng env và kiểm tra VRAM
trống, nên sai env/thiếu GPU sẽ báo ngay chứ không chết giữa chừng sau nửa tiếng.

## Sự cố đã gặp khi chạy thật lần đầu

Tất cả đã được vá trong script; bảng này để nhận diện nếu gặp lại biến thể khác.

| triệu chứng | nguyên nhân | chỗ đã vá |
|---|---|---|
| `conda: command not found` rồi script vẫn chạy tiếp | shell không tương tác không đọc `~/.bashrc`; `eval "$(conda shell.bash hook)"` in lỗi nhưng **trả về 0** nên `set -e` không chặn | `setup/_conda.sh` tự dò conda qua `CONDA_EXE` → PATH → các thư mục quen |
| Báo "✅ CÀI XONG" dù vừa lỗi | `bash install_env.sh \| tee log` lấy mã thoát của `tee` (=0) | `set -o pipefail` trong `try_install.sh` |
| Gói chui vào env **base** dù `conda activate` báo thành công | shell gọi script đang activate một env **khác base** → `conda activate X` chỉ thay thế tại chỗ, bản sao `base/bin` chèn thêm vào PATH vẫn đứng trước, nên `pip` trần trúng của base | bỏ hẳn `conda activate`; gọi `$CONDA_BASE/envs/<env>/bin/python -m pip` |
| `AssertionError: MMCV==2.2.0 is used but incompatible` khi đang **vá** chính assertion đó | bước vá dùng `import mmdet` để tìm file, mà import lại kích hoạt assertion | `importlib.util.find_spec()` — tra vị trí file, không chạy module |
| `LinkError: post-link ... cuda-version-12.9` | `-c nvidia` trần giải ra CUDA 12.9 lệch với torch cu128 | ghim `-c nvidia/label/cuda-12.8.1`, và bước này thành không bắt buộc |
| `ImportError: Matplotlib requires numpy>=1.25` lúc **chạy** stage 1 | matplotlib/scipy bản mới chặn numpy 1.23.5 mà LLMDet cần | ghim `matplotlib==3.9.2` + `scipy==1.10.1` cùng lúc với numpy |

Bài học chung: các khối kiểm tra cuối mỗi script cố tình `import` **đúng thứ
pipeline sẽ dùng** (`from mmdet.apis import DetInferencer`, `import
flash_attn_2_cuda`, `from transformers import Sam3Model`) để lỗi lộ ra lúc cài
chứ không phải nửa tiếng sau lúc chạy.

## RAM / CPU

RAM thật ~2.5 GB mỗi tiến trình (`VmRSS` ở `top` hiện ~18 GB nhưng gần hết là
file checkpoint được mmap — dùng chung, hệ điều hành tự thu hồi). Số thread CPU
mỗi tiến trình ghìm bằng `CPU_THREADS` trong `.env` (mặc định 8). Chi tiết số đo
xem mục "RAM / CPU" trong [README](../README.md).

## VRAM cần có

| bước | VRAM |
|---|---|
| stage 1 — LLMDet | ~26 GB |
| stage 1 — SAM3 | ~12 GB |
| stage 1 — WeDetect | ~6 GB |
| stage 2 — VLX-Seek | ~23 GB / tiến trình (mỗi GPU trong `VLX_GPUS` một tiến trình) |

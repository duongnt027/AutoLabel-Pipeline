# Weight & repo bên thứ ba

Không file nào trong danh sách này được commit vào repo. Tải hết bằng:

```bash
bash setup/get_third_party.sh        # LLMDet + VLX-Seek
bash setup/get_third_party.sh all    # thêm SAM3 + WeDetect
```

Tất cả nằm trong `third_party/` (đổi chỗ được bằng `THIRD_PARTY_DIR` trong `.env`).
Dưới đây là chi tiết nếu muốn tải tay hoặc chỉ tải phần cho backend mình dùng.

## Stage 2 — VLX-Seek (BẮT BUỘC)

| thứ | nguồn | dung lượng |
|---|---|---|
| repo | `github.com/om-ai-lab/VLX-Seek` @ `7f0e869` | ~50 MB |
| checkpoint | HF `omlab/VLX-Seek-1.5-10B` | ~19 GB |

```bash
git clone https://github.com/om-ai-lab/VLX-Seek.git third_party/VLX-Seek
huggingface-cli download omlab/VLX-Seek-1.5-10B
```

Checkpoint tải về HF cache (`~/.cache/huggingface`), không cần copy vào repo —
`VLX_CHECKPOINT` trong `.env` là tên repo HF, không phải đường dẫn file.

## Stage 1 — backend `llm`: LLMDet (mặc định)

| thứ | nguồn | dung lượng | đặt ở đâu |
|---|---|---|---|
| repo | `github.com/iSEE-Laboratory/LLMDet` @ `5336624` | ~25 MB | `third_party/LLMDet` |
| checkpoint chính | HF `fushh7/LLMDet` → `large.pth` | ~2.7 GB | `third_party/LLMDet/pretrained_models/large.pth` |
| backbone | `download.openmmlab.com/.../grounding_dino_swin-l_pretrain_obj365_goldg-34dcdc53.pth` | ~1.5 GB | `third_party/huggingface/mm_grounding_dino/` |
| text encoder | HF `bert-base-uncased` (chỉ bản pytorch) | ~450 MB | `third_party/huggingface/bert-base-uncased/` |
| vision encoder | HF `google/siglip-so400m-patch14-384` | ~3.3 GB | `third_party/huggingface/siglip-so400m-patch14-384/` |
| LMM | HF `fushh7/LLMDet` → `my_llava-onevision-qwen2-0.5b-ov-2/` | ~1.7 GB | `third_party/huggingface/my_llava-onevision-qwen2-0.5b-ov-2/` |

⚠️ **Cấu trúc thư mục quan trọng**: config của LLMDet trỏ các model phụ bằng
đường dẫn TƯƠNG ĐỐI `../huggingface/...`, nên `huggingface/` phải nằm **cùng cấp**
với repo `LLMDet/`:

```
third_party/
├── LLMDet/
│   └── pretrained_models/large.pth
└── huggingface/
    ├── bert-base-uncased/
    ├── siglip-so400m-patch14-384/
    ├── my_llava-onevision-qwen2-0.5b-ov-2/
    └── mm_grounding_dino/*.pth
```

Muốn dùng bản nhẹ hơn: `fushh7/LLMDet` còn có `base.pth` / `tiny.pth`, đổi
`LLMDET_WEIGHTS` + `LLMDET_CONFIG` trong `.env` cho khớp (`configs/grounding_dino_swin_b.py`…).

## Stage 1 — backend `s3`: SAM3

| thứ | nguồn | dung lượng |
|---|---|---|
| checkpoint | HF `facebook/sam3` | ~3 GB |

**Gated repo**: phải bấm accept license tại <https://huggingface.co/facebook/sam3>,
lấy token ở <https://huggingface.co/settings/tokens> rồi đặt `HF_TOKEN` trong `.env`.
Không cần clone repo code nào — dùng thẳng `Sam3Model` của `transformers >= 5`.

## Stage 1 — backend `we`: WeDetect

| thứ | nguồn | dung lượng | đặt ở đâu |
|---|---|---|---|
| repo | `github.com/WeChatCV/WeDetect` @ `dd302db` | ~100 MB | `third_party/WeDetect` |
| checkpoint | HF `fushh7/WeDetect` → `wedetect_base.pth` | ~1.1 GB | bất kỳ, trỏ `WEDETECT_WEIGHTS` vào |
| tokenizer | HF `FacebookAI/xlm-roberta-base` (chỉ config + tokenizer) | ~14 MB | `third_party/WeDetect/xlm-roberta-base/` |

Trọng số text encoder nằm sẵn trong checkpoint WeDetect, nên chỉ cần tokenizer +
config; config nạp nó bằng đường dẫn tương đối `./xlm-roberta-base/` nên thư mục
này **phải nằm trong repo WeDetect**.

Sau khi tải xong nhớ điền vào `.env`:

```ini
WEDETECT_WEIGHTS=/duong/dan/den/wedetect_base.pth
```

## Tổng dung lượng

| dùng gì | cần bao nhiêu đĩa |
|---|---|
| chỉ `llm` + stage 2 (mặc định) | ~29 GB |
| thêm `s3` | ~32 GB |
| thêm `we` | ~33 GB |

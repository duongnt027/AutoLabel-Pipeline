# auto_detect

Gán nhãn tự động (YOLO format) cho **một thư mục ảnh**, bằng 2 stage:

```
thư mục ảnh
   │
   ├─ STAGE 1  open-vocabulary detector vẽ box cho mọi HINT của mọi class
   │           (LLMDet / SAM3 / WeDetect — chạy riêng hoặc kết hợp)
   │           ngưỡng THẤP, ưu tiên không bỏ sót → proposals_<backend>.jsonl
   │
   └─ STAGE 2  VLX-Seek (VLM) nhìn ảnh + từng box, chấm điểm theo PROMPT của
               từng class → votes_<backend>/*.json → labels/*.txt
```

Ý tưởng: detector giỏi **tìm chỗ có vật** nhưng dở phân biệt ngữ nghĩa; VLM giỏi
**đọc ngữ cảnh** nhưng không tự vẽ box chuẩn. Ghép lại thì mỗi class khai 2 thứ
trong `classes/classes.json`:

```json
{
  "helmet": {
    "hints":   ["head", "helmet"],
    "prompts": ["a person wearing a safety helmet, clearly visible"]
  },
  "no helmet": {
    "hints":   ["head", "helmet"],
    "prompts": ["a bare human head with no helmet, clearly visible"]
  }
}
```

`hints` (thô, rộng, nhiều từ đồng nghĩa) cho stage 1 vẽ box; `prompts` (câu mô tả
hẹp) cho stage 2 quyết định. Hai class dùng chung hint là bình thường — stage 1
chỉ cần khoanh cái đầu, phân biệt đội mũ hay không là việc của stage 2.

### Chế độ `--describe`: VLX tự sinh prompt theo từng ảnh

Thêm trường `alias` — cụm từ chỉ thẳng vật thể, kiểu VLX hiểu tốt nhất:

```json
"helmet": { "hints": ["head"], "alias": ["a helmet", "a person head wearing a helmet"], ... }
```

`python3 run_pipeline.py /anh --describe` thì với mỗi nhóm class (cùng bộ hint):

1. chấm box gốc theo `alias` → mỗi box tạm thuộc class nào;
2. box tự tin nhất của mỗi class (tối đa `VLX_DESC_PER_CLASS`) được **nới rộng**
   rồi cho VLX mô tả. Câu hỏi chỉ nhắc hint ("this head"), không nhắc class —
   hỏi có hướng thì box nào cũng tự xác nhận được chính nó;
3. **luôn lọc trùng** mô tả: trùng/gần trùng trong 1 class thì bỏ, trùng giữa 2
   class thì bỏ cả hai (câu đó không phân biệt được gì);
4. chấm lại box **gốc** bằng alias + mô tả của mọi class trong nhóm, cạnh tranh
   nhau trong 1 lượt như chế độ thường.

Prompt đã sinh cho từng ảnh lưu trong `votes_*/<ảnh>.json` (`"prompts"`, và
`"caption"` ở từng box) để xem lại. Thiếu `alias` thì dùng `prompts` và ngược lại.
Đổi chế độ thì ảnh được chấm lại (votes ghi kèm `"mode"`).

## Chế độ EVENT: phát hiện hành vi theo mô tả

`python3 run_events.py /anh` — nhãn theo **hành vi** (bán hàng rong, rửa xe, đặt
biển trên vỉa hè...) thay vì theo vật thể. Bộ nhãn: `classes/street_events.json`
gồm `classes` (đối tượng chủ thể, định dạng y như classes.json), `events` (caption
gốc tiếng Anh + `name_vi` + `keywords` + luật) và `reject` (caption "rác").

```
stage 1 + 2   detection thường cho các đối tượng chủ thể (người, hàng hoá, xe đẩy, biển...)
gộp cùng class box cùng class chồng nhau -> 1 đối tượng (box bao)
gộp nhóm      anchor (hàng hoá, biển, vật liệu...) chạm nhau -> 1 nhóm; context
              (người, xe, cây, cột) gắn vào nhóm gần nhất, không tự lập nhóm
stage 3       VLX brief_region_caption mô tả box nhóm (caption tự do)
              + 1 lượt detection: box nhóm x caption của các event + reject
quyết định    điểm = w_vlx * VLX(event) + w_text * so(caption sinh ra, caption gốc + keywords)
              = 0 nếu nhóm thiếu class event yêu cầu (requires); khớp reject hơn -> loại
              (chỉ xét thứ nhìn thấy; giờ / tuyến phố cấm / giấy phép lọc bên ngoài)
```

Kết quả: `labels_events/` (YOLO, id theo thứ tự `events`), `data_events.yaml`,
`events_summary.jsonl` (mỗi box nhóm: thành phần, caption, event, lý do bị loại),
`labels_actors/`. Đổi ngưỡng/trọng số/keywords: `run_events.py /anh --rebuild`.
Đổi `captions` hay tham số gộp nhóm thì stage 3 tự chấm lại những ảnh liên quan.

## Chạy

```bash
# cả 2 stage: thư mục ảnh -> nhãn YOLO
python3 run_pipeline.py /duong/dan/anh

# chỉ stage 1 (xem/đánh giá riêng chất lượng proposal)
python3 run_stage1.py /duong/dan/anh
```

Kết quả nằm ở `runs/<tên thư mục ảnh>/`:

```
runs/anh_test/
├── labels/*.txt           # nhãn YOLO cuối cùng, trùng tên với ảnh
├── data.yaml              # tên class theo đúng id đã dùng
├── proposals_llm.jsonl    # box thô của stage 1 (1 dòng / ảnh)
└── votes_llm/*.json       # điểm stage 2 cho TỪNG box
```

Ảnh gốc **không bị copy hay sửa** — pipeline chỉ đọc.

### Các cờ hay dùng

```bash
python3 run_pipeline.py /anh --limit 20            # thử 20 ảnh đầu
python3 run_pipeline.py /anh --models llm,s3       # chạy 2 backend rồi gộp kết quả
python3 run_pipeline.py /anh --min-conf 0.4        # siết ngưỡng giữ box
python3 run_pipeline.py /anh --sahi                # cắt ô để bắt vật nhỏ (chậm ~14 lần)
python3 run_pipeline.py /anh --classes bo_nhan_khac.json
python3 run_pipeline.py /anh --fresh               # xoá workspace cũ, làm lại
python3 run_pipeline.py /anh1 /anh2 /anh3          # nhiều thư mục, model chỉ nạp 1 lần
```

Chạy lại là **chạy tiếp chỗ dở** ở cả 2 stage: ảnh đã có proposal / đã có điểm
thì bỏ qua. Ngắt giữa chừng rồi chạy lại cũng không mất việc đã làm.

### Đổi ngưỡng không cần chạy lại model

Stage 2 lưu điểm của **mọi** box, nên đổi ngưỡng chỉ mất vài giây:

```bash
python3 rebuild_labels.py runs/anh_test --min-conf 0.5
```

## Cài đặt

Xem [docs/SETUP.md](docs/SETUP.md) — tóm tắt:

```bash
bash setup/try_install.sh --weights-all   # 2 env + repo + weight, một lệnh
cp .env.example .env                      # rồi sửa *_PYTHON, GPU, HF_TOKEN
```

Đo thật một lượt trọn vẹn: **22 phút 38 giây**, đĩa **34.6 GB**, và RAM cả máy chỉ
tăng **354 MB** — không gói nào bị build từ source. Xoá sạch lại:
`bash setup/uninstall.sh`.

Weight tải ở đâu, đặt vào đâu: [docs/WEIGHTS.md](docs/WEIGHTS.md).

## Backend stage 1

| mã | model | env | VRAM | ghi chú |
|---|---|---|---|---|
| `llm` | [LLMDet](https://github.com/iSEE-Laboratory/LLMDet) (Grounding DINO) | `llmdet` | ~26 GB | mặc định; mọi hint hỏi trong 1 lượt forward |
| `s3` | [SAM3](https://huggingface.co/facebook/sam3) text prompt | `vlx` | ~12 GB | mỗi hint 1 lượt hỏi riêng → chậm khi nhiều hint; repo gated |
| `we` | [WeDetect](https://github.com/WeChatCV/WeDetect) | `llmdet` | ~6 GB | nhanh nhất; train bằng prompt tiếng Trung (hint tiếng Anh vẫn chạy được) |

Mọi backend ghi **cùng một hợp đồng jsonl** (xem `autodetect/common.py`), nên
thêm backend mới chỉ là viết thêm 1 file trong `stage1/` — không phải sửa stage 2.

**Nhiều backend thì mỗi backend được stage 2 chấm RIÊNG**, xong mới gộp quyết
định cuối theo confidence của VLX. Không gộp proposal trước rồi mới cắt top-K:
score đề xuất thô không cùng thang đo giữa các backend, gộp sớm sẽ đá văng box
thật của backend có thang điểm thấp hơn trước khi VLX kịp nhìn thấy.

## Cấu trúc

```
run_pipeline.py       chạy cả 2 stage
run_events.py         chế độ event: detection -> gộp nhóm -> caption -> so khớp
run_stage1.py         chỉ stage 1
rebuild_labels.py     dựng lại nhãn với ngưỡng khác, không chạy model
autodetect/
├── common.py         .env, class/hint, hợp đồng jsonl (chỉ stdlib — mọi env import được)
├── events.py         chế độ event: gộp nhóm, so caption, quyết định (stdlib)
├── labels.py         NMS + quyết định class + ghi nhãn YOLO (nơi DUY NHẤT cắt ngưỡng)
└── pipeline.py       cấu hình + cách gọi từng stage bằng env riêng
stage1/{llmdet,sam3,wedetect}.py    backend sinh proposal
stage2/score_vlx.py                 chấm điểm box -> votes/
stage3/describe_events.py           chế độ event: VLX mô tả + chấm box nhóm -> events_votes/
classes/street_events.json          bộ đối tượng + event (chế độ event)
classes/classes.json                bộ nhãn (hints + prompts)
setup/                              script tạo env, đóng gói env, tải weight
```

Mỗi stage là **một subprocess** gọi thẳng python của env tương ứng, trao đổi dữ
liệu qua file — không stage nào import stage khác, nên xung đột version giữa các
model không bao giờ chạm vào nhau.

## RAM / CPU

Đã đo thật (stage 2, ảnh dày 2776 box/ảnh, checkpoint 19 GB):

| | mỗi tiến trình |
|---|---|
| RAM thật (anonymous) | **~2.5 GB** |
| `VmRSS` hiển thị ở `top` | ~18 GB — gần hết là **file checkpoint được mmap**: dùng chung giữa các tiến trình, hệ điều hành tự thu hồi khi thiếu RAM, không phải RAM bị chiếm |
| thread | ghìm bằng `CPU_THREADS` (mặc định 8) |

Nên với `VLX_GPUS=0,1` thì cần ~5 GB RAM thật + page cache cho checkpoint.

Không chỗ nào giữ cả bộ ảnh trong RAM:

- File proposal đọc **theo từng dòng** (`iter_jsonl`) ở mọi khâu — lúc đầu code
  parse cả file, đo được **0.9 MB RAM/ảnh** (10k ảnh ≈ 8.7 GB **mỗi tiến trình**);
  sau khi sửa còn **+3 MB, phẳng bất kể số ảnh**.
- Stage 2 lập kế hoạch bằng danh sách **tên ảnh**, box đọc lại theo dòng lúc chạy.
- Nhãn dựng theo từng ảnh, mỗi lúc chỉ 1 file votes nằm trong RAM.

`CPU_THREADS` có mặt vì torch mặc định lấy **nửa số core cho mỗi tiến trình**
(56 trên máy 112 core): chạy 2-3 GPU song song là trăm thread giành nhau CPU mà
không nhanh hơn, vì khâu nặng nằm ở GPU còn CPU chỉ decode ảnh.

## Giới hạn đã biết

- Stage 2 xem tối đa `AUTOLABEL_TOPK` (mặc định 40) box mỗi nhóm class mỗi ảnh:
  model chỉ có `<obj0>..<obj99>` để chỉ vào box, và 256 token đầu ra chỉ đủ liệt
  kê ~40-50 box. Ảnh cực đông vật thể sẽ bị cắt bớt.
- Stage 2 là khâu chậm nhất (~vài giây/ảnh). Nhiều GPU thì khai `VLX_GPUS=0,1`,
  ảnh được chia đều, nhanh gần gấp số GPU.
- `--sahi` chỉ áp dụng cho backend `llm`.

# Các chỗ demo lệch khỏi spec

Spec gốc: `/Users/researchmac/DATA IMAGES/PROMPT_DenseUAV_Forest_Localization.md`.
Kế hoạch demo: bản rút gọn, chỉ đo DenseUAV định vị ảnh UAV trên **orthomosaic** trước và sau khi train.

| # | Ngày | Spec yêu cầu | Demo làm | Lý do |
|---|---|---|---|---|
| D1 | 2026-11-27 | Ref map là ảnh vệ tinh | Ref map là orthomosaic UAV `Rung_Sac_demo.png` (~2 cm/px, WGS84) | Chưa có ảnh vệ tinh. Ortho có thể được dựng từ chính các ảnh query nên kết quả sẽ lạc quan hơn thực tế. |
| D2 | 2026-11-27 | Có DINOv2, GeM/SALAD, InfoNCE geo-aware (M5–M8) | Chưa làm | Demo chỉ làm phần DenseUAV. |
| D3 | 2026-11-27 | ~50 ảnh test, `buffer_m = 2 × footprint_max`, tối thiểu 200 m | Khoảng 14 ảnh test ở cực Nam tuyến bay, `buffer_m = 1 × footprint_max` (~100 m) | Chỉ có 48 ảnh trên một tuyến dài ~740 m. |
| D4 | 2026-11-27 | Pseudo-query là dữ liệu bổ sung (`pseudo_ratio` 0.3), pseudo-val ~2000 | Pseudo-query từ ortho là nguồn train chính (cách làm hiện tại xem D16) | Có quá ít ảnh UAV thật. |
| D5 | 2026-11-27 | Có Z1 = ViT-S với checkpoint DenseUAV (Urban), zero-shot | **Bỏ Z1** | Link tải checkpoint baseline (README.md dòng 132) đã chết. |
| D6 | 2026-11-27 | (không có) | **Thêm Z0** = ViT-S ImageNet, zero-shot (xem chi tiết bên dưới) | Thay cho Z1 để vẫn có mốc so sánh khi chưa train. |
| D7 | 2026-11-27 | E0a fine-tune từ checkpoint DenseUAV | **E0a train từ ImageNet**: `finetune_trees.sh`, `name=forest_E0a`, bỏ `--load_from`, lr 0.01, 60 epoch, milestones 40,55, `save_interval` 10 | Không có checkpoint baseline. lr giữ như baseline gốc vì head train từ đầu. Số epoch giảm từ 120 xuống 60 vì dữ liệu nhỏ. |
| D8 | 2026-11-27 | Augment lúc train: xoay ±15° (vì đã north-up), màu theo preset C1, giống nhau cho hai nhánh | E0a dùng flag gốc của repo: `cj="uav,satellite"` và `rr="uav,satellite"` cho cả hai nhánh. `rr` = `RotateAndCrop(0.5)` (`datasets/queryDataset.py:140`), xoay **0–360°** với xác suất 0.5. `cj` = `ColorJitter(brightness=0.5, contrast=0.1, saturation=0.1, hue=0)`, không phải C1. Ngoài ra `ra=satellite` (`RandomAffine(180)`) và lật ngang luôn bật | Giữ nguyên code augment của DenseUAV cho baseline E0a; repo không có flag xoay ±15°. (Biến thể ±15° và C1 chỉ áp lúc **sinh** pseudo-query trong `data_prep/`.) |

| D9 | 2026-11-27 | Footprint = `RelativeAltitude × sensor_w / focal` | Nhân thêm `north_up.footprint_scale = 0.836` (`configs/data.yaml`) khi cắt ô positive | Dùng SIFT để fit phép đồng dạng giữa ảnh UAV north-up và ô ortho trên các ảnh **không thuộc test** (0282–0315): tỉ lệ trung vị 0.836 (0.81–0.86). Nhiều khả năng do tán rừng cao hơn điểm cất cánh. Sau khi hiệu chỉnh, tỉ lệ trung vị = 0.996. Footprint dùng để tính buffer vẫn là giá trị danh nghĩa (98.5 m), tức là thận trọng hơn. |
| D10 | 2026-11-27 | (không có) | Không hiệu chỉnh góc la bàn còn lại sau north-up: **−4.2°**, gần như hằng số trên cả 48 ảnh (đo bằng SIFT) | Ảnh test lúc chạy thật cũng có sai số này. Nó nằm trong giả định ±15° của spec. |
| D11 | 2026-11-27 | GPS là nhãn chính xác | Không hiệu chỉnh. Độ lệch giữa tâm GPS và ortho **thay đổi tuyến tính dọc tuyến**, từ khoảng +10 m ở đầu Bắc đến khoảng −7 m ở đầu Nam (đo bằng SIFT) | Sửa thì phải fit trên ảnh test. Hệ quả: sai số định vị dưới khoảng 10 m **không có ý nghĩa** khi đánh giá. |
| D12 | 2026-11-27 | C1: blur σ ∈ [0, 1.0] | Pseudo drone dùng σ ∈ [0, 1.5], áp ở độ phân giải 512 px sau khi cắt | Theo kế hoạch demo. |
| D13 | 2026-11-27 | Val chia theo lưới khối trên ref map | Val là **một khối theo toạ độ dọc tuyến** s ∈ [186, 246] m, có buffer 98.5 m về hai phía | Ref map chỉ là một dải hẹp khoảng 100 m. |
| D14 | 2026-11-27 | Bỏ ô có > 10% no-data | Ảnh test 0329 (37% no-data) và 0328 (8%) vẫn được giữ trong test; chỉ ô positive của chúng có no-data, query là ảnh UAV nguyên vẹn | Bỏ thì còn 13 ảnh test. Ô positive của test chỉ dùng cho oracle và debug. |
| D15 | 2026-11-27 | Pseudo-query phủ ref map lớn | **Đã xử lý bằng D16.** Bản đầu: 1500 tâm pseudo-train dồn vào vùng train nhỏ, trung vị khoảng cách tới tâm gần nhất chỉ **0.85 m** (pseudo-val 1.33 m). Nhiều "lớp" CE gần như trùng ảnh nhau, và triplet sẽ coi chúng là negative | Vùng ortho hợp lệ cho train chỉ khoảng 145 m dọc tuyến. Cần người dùng quyết định trước khi train E0a (xem phần kết M2). |
| D16 | 2026-11-27 | Pseudo-query: tâm ngẫu nhiên, footprint ngẫu nhiên, trộn với tỉ lệ `pseudo_ratio` 0.3; pseudo-val khoảng 2000 | **Mỗi lớp là một tâm của lưới 14 m** (`pseudo.grid_spacing_m`, ≈ 0.25 × F), xoay theo trục tuyến bay, giữ nguyên vùng pseudo-train/val và vòng cấm quanh test. `satellite/<id>/` gồm 1 ô north-up cạnh S = 55.6 m (scale 1.0). `drone/<id>/` gồm **12 biến thể**: lệch tâm ngẫu nhiên ≤ 3 m, cạnh U[0.8, 1.25]×S, xoay ±15°, C1, blur σ ∈ [0, 1.5]. Tâm có ô > 10% no-data bị bỏ. Ảnh UAV thật train/val cách tâm lưới gần nhất < 7 m thì **gộp** vào lớp đó (chỉ thêm vào `drone/`); các ảnh còn lại giữ lớp riêng. Kết quả (sau khi đổi `merge_real_within_m` = 10 m): train 29 lớp (27 lớp lưới + 2 ảnh thật riêng 0308, 0309, nằm trong vòng cấm test nên không có tâm lưới gần), val 12 lớp; NN min giữa mọi tâm lớp = 14.0 m. | Bản ngẫu nhiên có trung vị NN 0.85 m trong khi ô rộng khoảng 55 m, nên nhiều lớp gần như trùng nhau và làm hỏng cả CE lẫn triplet. Đổi lại thì **số lớp rất ít**, vì vùng ortho hợp lệ cho train chỉ dài khoảng 145 m. Ngưỡng gộp lúc đầu là 7 m nên 0307 (cách 7.6 m) bị tách thành lớp riêng; sau khi nâng lên 10 m thì 0307 đã được gộp. |
| D17 | 2026-11-27 | `footprint_set` 5–6 mức theo cấp số nhân, phủ dải độ cao; kế hoạch demo ban đầu là {60, 70, 80, 95, 110} m | `gallery.footprint_set_m = [45, 50, 56, 63, 70]` m (≈ 0.8–1.25 × S, S = 55.6 m), stride 25% cạnh | Cạnh UAV north-up thực tế sau hiệu chỉnh D9 là 55.6 m, nhỏ hơn mức thấp nhất của bộ cũ. Bộ mới phủ đúng dải scale dùng khi sinh pseudo drone. |
| D18 | 2026-11-27 | DenseUAV gốc dùng `sample_num=1` | E0a dùng `sample_num=8` (`finetune_trees.sh`) | Đã đọc `Sampler_University.__iter__` (`datasets/Dataloader_University.py:96`): hoán vị chỉ số lớp rồi `np.repeat(list, sample_num)`, tức mỗi lớp lặp **liên tiếp** 8 lần trong cùng một epoch. Mỗi lần `__getitem__` lại bốc ngẫu nhiên một ảnh drone và một ảnh satellite của lớp, kèm augment riêng. Hiện có 29 lớp: 29×8 = 232 mẫu, chia batch 16 được **14 iteration/epoch** (`train.py:106` bỏ batch thiếu). Đã chạy thử: mỗi batch 16 gồm **đúng 2 lớp × 8 mẫu**, nên triplet có nhiều positive nhưng chỉ có negative từ một lớp khác. Muốn nhiều lớp hơn mỗi batch thì tăng batch hoặc giảm `sample_num`. `sample_num` không ảnh hưởng loss ngoài phép tính `split_num` (`losses/loss.py:57`, không dùng tới). |
| D19 | 2026-11-27 | FAISS `IndexFlatIP` | `torch` matmul trên descriptor đã chuẩn hoá L2 (cùng kết quả với tìm kiếm chính xác theo inner product) | `faiss` chưa được cài trong `.venv`. Gallery nhỏ (vài nghìn ô) nên matmul đủ nhanh, không cần cài thêm. |
| D20 | 2026-11-27 | Nghiệm thu M3: chạy với "oracle" (lấy ô positive làm query), sai số phải ≈ 0 | Có ba kiểm tra: (1) `python -m eval.metrics --selftest`: dữ liệu tổng hợp, GT trùng đúng tâm ô → err = 0, rank = 1, SDM@1 = 1 (đạt). (2) `--model oracle`: gallery có descriptor ngẫu nhiên, query lấy **đúng descriptor của ô có tâm gần GT nhất** → R@1 = 100%, sai số = sàn lượng tử hoá của gallery (median 3.4 m trên test, 3.2 m trên pseudo-val). (3) `--model random` → mức ngẫu nhiên. | Dùng ảnh ô positive làm query thì kết quả phụ thuộc mô hình, nên không chứng minh được metric đúng. Với gallery stride 25%, sai số top-1 không thể bằng 0 vì tâm GT không trùng tâm ô. SDM@1 của oracle chỉ khoảng 0.87 vì công thức repo `exp(-5000·d)` với d tính bằng độ: 3 m ≈ 2.7e-5° cho exp(-0.135) ≈ 0.87. |
| D21 | 2026-11-27 | (kế hoạch) E0a dùng lr 0.01 như baseline gốc, 120 epoch | Lần chạy lr 0.01 (SGD, backbone 0.3×lr, `sample_num=4`) **phân kỳ** và đã bị dừng ở epoch 21. Thư mục lưu ở `checkpoints/forest_E0a_lr0.01_diverged/` (net_009, net_019) và không dùng để đánh giá. Probe `forest_E0a_probe` (lr 0.003, 15 epoch, milestones 10,13) **đạt**, nên E0a chính dùng lr 0.003, 120 epoch, milestones 80,110, `save_interval` 5, chạy 3 seed (0, 1, 2; `train.py --seed`) | Log cho thấy tổng CE (sat + drone) tăng từ 6.6 lên khoảng 8.2, cao hơn mức ngẫu nhiên khoảng 6.7 (2 × ln 29). KL tăng từ 0.10 lên khoảng 2.1, triplet từ 1.0 lên khoảng 1.9; accuracy train chỉ 2–5% (mức ngẫu nhiên 3.4%). Với 29 lớp, lr của baseline gốc là quá lớn. |
| D22 | 2026-11-28 | Chọn checkpoint theo pseudo-val + val địa lý; báo 3 seed | Mỗi seed chọn checkpoint chỉ dựa trên pseudo-val (148 query), trong 24 checkpoint cách nhau 5 epoch: seed 0 → `net_104`, seed 1 → `net_114`, seed 2 → `net_094`. Test chạy một lần cho mỗi checkpoint đã chọn. Seed truyền qua `train.py --seed` (thêm mới; mặc định 666 như repo gốc) | **Hạn chế khi đọc kết quả:** (1) Pseudo-val **gần bão hoà**: median 3.9–4.5 m, sát sàn lượng tử hoá của oracle (3.2 m). Từ epoch 64 trở đi các checkpoint chênh nhau rất ít, nên việc chọn checkpoint không phân biệt được nhiều. (2) Ortho nhiều khả năng được dựng từ **chính các ảnh của chuyến bay này**, kể cả ảnh test (D1). Query test và ô gallery vì vậy cùng ngày và cùng ánh sáng, nên kết quả **lạc quan hơn nhiều** so với khi đối chiếu với ảnh vệ tinh thật. (3) E0a trên test có median khoảng 5–6 m, đã ở mức nhiễu nhãn GPS (D11, tới khoảng 10 m) và gần sàn oracle 3.4 m: tầng truy hồi gần như luôn tìm đúng ô, không còn chỗ để đo thêm cải thiện. |

### Probe lr 0.003 (D21), log `runs/E0a_probe/seed0/train.log`

| Epoch | CE (sat + drone) | KL | Triplet | Acc sat / drone | lr backbone / head |
|---|---|---|---|---|---|
| 0 | 6.52 | 0.01 | 1.08 | 0.00 / 0.01 | 9e-4 / 3e-3 |
| 5 | 4.60 | 0.56 | 0.80 | 0.16 / 0.14 | 9e-4 / 3e-3 |
| 8 | 4.40 | 0.82 | 0.79 | 0.25 / 0.31 | 9e-4 / 3e-3 |
| 10 | 3.72 | 0.91 | 0.78 | 0.28 / 0.35 | 9e-5 / 3e-4 |
| 13 | 2.80 | 0.71 | 0.70 | 0.47 / 0.55 | 9e-6 / 3e-5 |
| 14 | 3.31 | 0.74 | 0.71 | 0.40 / 0.41 | 9e-6 / 3e-5 |

CE giảm xuống dưới mức ngẫu nhiên (khoảng 6.7) và accuracy tăng đều, khác hẳn lần chạy lr 0.01. Sau khi bật `persistent_workers` trong `datasets/make_dataloader.py` (người dùng tự sửa), mỗi epoch chỉ mất khoảng 0.9 s, so với khoảng 43 s trước đó.

## Ghi chú cho M3

- Đã làm: `localize/features.py` (`Extractor`) chép cách làm của `test.py:122` (ảnh gốc + ảnh lật ngang, `outputs[1]`, chuẩn hoá L2), và tự thêm một bản sao khi batch chỉ có 1 ảnh.
- `gallery.footprint_set_m` đã đổi theo D17.

## Định nghĩa Z0

- Model: `timm.create_model("vit_small_patch16_224", pretrained=True, img_size=(224, 224))`, đúng như nhánh `ViTS-224` trong `models/Backbone/backbone.py:30`. Trọng số mặc định là `augreg_in21k_ft_in1k`, đã có trong cache HF.
- **Không dùng head** (`SingleBranch` / `ClassBlock`), vì head chưa được train thì chỉ là phép chiếu ngẫu nhiên.
- Descriptor = `forward_features(x)` → mean-pool các patch token (`[:, 1:]`, bỏ CLS), khớp với `head_pool=avg` của E0a. Phương án phụ là dùng CLS token (`[:, 0]`); nếu chạy phương án này thì ghi riêng thành `Z0-cls`.
- Chuẩn hoá L2. Tiền xử lý giống `test.py:74–85`: Resize 224 bicubic, ToTensor, Normalize ImageNet.
- Dùng cùng gallery, cùng split và cùng bộ metric với E0a.

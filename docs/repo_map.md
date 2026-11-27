# Repo map — DenseUAV (fork, nhánh main, sau commit 871a3f0)

Viết sau khi đọc code thật ngày 2026-11-27. Số dòng tính theo trạng thái hiện tại của file.

## 1. Luồng chính

```
train.py ── make_dataset(opt)        datasets/make_dataloader.py
         ── make_model(opt)          models/taskflow.py
         ── make_optimizer(model,opt) optimizers/make_optimizer.py
         ── Loss(opt)                losses/loss.py
         ── copyfiles2checkpoints(opt) tool/utils.py:56  -> checkpoints/<name>/opts.yaml (+ bản copy code)
test.py  (chạy TRONG checkpoints/<name>/, đọc ./opts.yaml) -> pytorch_result_<mode>.mat
evaluate_gpu.py      (đọc pytorch_result_1.mat) -> Recall@1/5/10, AP -> results.txt
evaluateDistance.py  (đọc pytorch_result_<mode>.mat + <root_dir>/Dense_GPS_ALL.txt) -> SDM@K, MA@K json
```

## 2. Model — `models/`

| Thành phần | File : dòng | Ghi chú |
|---|---|---|
| `Model(nn.Module)` | `models/taskflow.py:8` | `backbone = make_backbone(opt)`, `opt.in_planes = backbone.output_channel`, `head = make_head(opt)`. `forward(drone_image, satellite_image)` trả `(drone_res, satellite_res)`; **dùng chung trọng số** cho 2 nhánh. Truyền `None` để bỏ một nhánh. |
| `Model.load_params(load_from)` | `models/taskflow.py:30` | Chỉ nạp key trùng tên **và** trùng shape → classifier có số lớp khác bị bỏ qua (đúng ý khi fine-tune). `torch.load(load_from)` **không có `map_location`**, xem mục 9. |
| `make_model(opt)` | `models/taskflow.py:38` | Gọi `load_params` nếu `os.path.exists(opt.load_from)`. |
| Backbone registry | `models/Backbone/backbone.py:20` `Backbone.init_backbone` | chuỗi `if/elif` theo `opt.backbone`: `resnet50, RKNet, senet, ViTS-224 (vit_small_patch16_224, 384 ch), ViTS-384, DeitS-224, DeitB-224, Pvtv2b2, ViTB-224, SwinB-224, Swinv2S-256, Swinv2T-256, Convnext-T, EfficientNet-B2/B3/B5/B6, vgg16, cvt13`. `forward` = `timm.forward_features` → ViT trả `(B, 1+N, C)`. |
| Head registry | `models/Head/head.py:18` `Head.init_head` | `SingleBranch, SingleBranchCNN, SingleBranchSwin, NetVLAD, FSRA, FSRA_CNN, LPN, LPN_CNN, GeM`. (`NeXtVLAD.py` có file nhưng không đăng ký.) |
| `SingleBranch` | `models/Head/SingleBranch.py:6` | `head_pool ∈ {global, avg, max, avg+max}`; `avg` = mean patch token (bỏ CLS). Gọi `.squeeze()` → **batch size 1 sẽ mất chiều batch**. Trả `[cls_logits, feature]`. |
| `GeM` head | `models/Head/GeM.py:6` | Dùng `Pooling(h//16*w//16, "gem")` → `Gem_heat` (`models/Head/utils.py:66`): thực chất là **trung bình có trọng số softmax trên token**, không phải GeM chuẩn có số mũ p. |
| `ClassBlock` | `models/Head/utils.py:35` | `Linear(in, num_bottleneck) → BN → Dropout(droprate)` → `Linear(num_bottleneck, nclasses)`. `forward` trả `(cls_, feature_)`; descriptor = `feature_` (512 chiều với baseline). |

| `LPN` | `models/Head/LPN.py:9` | `opt.block` = số vòng (ring) LPN. Tạo `global_classifier` trên CLS token + `block` `ClassBlock` cho từng vòng (`get_part_pool`, dòng 41). Lúc train trả `[list cls (1+block), list feature]`; lúc eval `torch.stack` → `(B, num_bottleneck, 1+block)`. |
| `LPN_CNN` | `models/Head/LPN.py:105` | Như trên cho CNN, không có global classifier. |
| `FSRA`, `FSRA_CNN` | `models/Head/FSRA.py:6, 81` | `opt.block` = số vùng heatmap; `block==1` thì chỉ trả global. |
| `NetVLAD` | `models/Head/NetVLAD.py:7` | `num_clusters = opt.block`, `ClassBlock(in_planes*block, ...)`. |

Tham số head dùng chung (đều lấy từ `opt`, khai báo trong `train.py:get_parse`):
- `--block` (mặc định 2, `train.py:51`): số phần với LPN/FSRA/NetVLAD; `SingleBranch`/`GeM` không dùng. Lưu ý: `test.py:152` chuẩn hoá feature 3 chiều bằng `sqrt(opt.block)`, trong khi LPN trả `1+block` phần → chuẩn L2 cuối không đúng bằng 1 (thứ hạng cosine vẫn đúng tương đối).
- `--head_pool` (mặc định `max`, `train.py:62`): chỉ `SingleBranch` dùng.
- `--num_bottleneck` (mặc định 512, `train.py:58`): chiều descriptor ra khỏi `ClassBlock`.
- `--droprate` (0.5), `opt.nclasses` (= số lớp train, gán ở `train.py:197`), `opt.in_planes` (= `output_channel` của backbone, gán ở `taskflow.py:12`).

Cách gọi timm: `timm.create_model(<tên>, pretrained=True[, img_size=(h,w)])` (chỉ `ViTS-224` truyền `img_size`), rồi `forward_features`.

Thêm backbone/head mới: thêm một nhánh `elif` vào hai hàm registry trên.

### Kiểm tra DINOv2 trong `.venv` (timm 1.0.31, torch 2.15.0)
- `timm.list_models('*dinov2*')` → `vit_small/base/large/giant_patch14_dinov2` và các bản `_reg4_`.
- `vit_base_patch14_dinov2.lvd142m` **có** trong danh sách pretrained.
- Đã thử `timm.create_model('vit_base_patch14_dinov2.lvd142m', pretrained=False, img_size=224)`: lưới patch `(16, 16)` = 256 token, `num_features = 768`; cấu hình gốc là 518×518 nên với 224 thì timm tự nội suy pos-embed. (Chưa tải trọng số pretrained.)

## 3. Loss — `losses/`

| Thành phần | File : dòng | Ghi chú |
|---|---|---|
| `Loss(nn.Module)` | `losses/loss.py:9` | Chọn theo `opt.cls_loss ∈ {CELoss, FocalLoss}`, `opt.feature_loss ∈ {TripletLoss, HardMiningTripletLoss, SameDomainTripletLoss, WeightedSoftTripletLoss, ContrastiveLoss}`, `opt.kl_loss ∈ {KLLoss}`; giá trị khác (vd `no`) = tắt. |
| `Loss.forward` | `losses/loss.py:42` | `loss = CE(drone)+CE(sat) + triplet(cat(f1,f2), cat(labels,labels)) + KL(log_softmax(cls1) ‖ softmax(cls2))`. Các thành phần **cộng với trọng số 1**, không có flag weight. |
| `WeightedSoftTripletLoss` | `losses/TripletLoss.py:281` | |
| `TripletLoss`, `HardMiningTripletLoss`, `SameDomainTripletLoss` | `losses/TripletLoss.py:195, 163, 227` | |
| `FocalLoss` | `losses/FocalLoss.py:5` | |
| `cal_loss.py` | `losses/cal_loss.py:47` | Bản cũ, `train.py` không dùng. |

Nơi `train.py` gọi loss: import `from losses.loss import Loss` (`train.py:17`), khởi tạo `nnloss = Loss(opt)` (`train.py:89`), gọi `loss, cls_loss, f_triplet_loss, kl_loss = nnloss(outputs, outputs2, labels, labels3)` (`train.py:122`), backward ở `train.py:126–132`.
Cấu hình baseline/finetune: `--cls_loss CELoss --feature_loss WeightedSoftTripletLoss --kl_loss KLLoss` (mặc định trong `train.py` là `feature_loss=no`, `kl_loss=no`).
| `ContrastiveLoss` | `pytorch_metric_learning.losses` | Import ở đầu `loss.py` → cần gói `pytorch-metric-learning`. |

## 4. Transform và ý nghĩa thật của các flag — `datasets/make_dataloader.py:8` `make_dataset`

Giá trị flag là chuỗi; điều kiện là `"uav" in opt.xx` / `"satellite" in opt.xx` (kiểm tra chuỗi con).
Nhánh drone dùng `data_transforms['train']`, nhánh vệ tinh dùng `data_transforms['satellite']`.

Thứ tự áp (mỗi nhánh):
1. `--rr` → `RotateAndCrop(0.5)` (`datasets/queryDataset.py:140`): xác suất 0.5, góc ngẫu nhiên nguyên **0–360°**, `cv2.warpPerspective` lấy hình vuông nội tiếp xoay → output **512×512**. Không có tuỳ chọn ±15°.
2. `Resize((h,w), bicubic)`, `Pad(opt.pad, edge)`, `RandomHorizontalFlip()` — **luôn bật** cho cả hai nhánh (flip ngang làm mất tính north-up chặt chẽ nhưng vẫn hợp lệ cho cả hai nhánh như nhau).
3. `--ra` → `transforms.RandomAffine(180)`: **chỉ xoay** ±180°, không tịnh tiến/scale; góc vùng bị lấp đen.
4. `--re` → `RandomErasing(probability=opt.erasing_p)` (`queryDataset.py:199`, mặc định `erasing_p=0.3`, diện tích 2–30%).
5. `--cj` → `ColorJitter(brightness=0.5, contrast=0.1, saturation=0.1, hue=0)` (cố định, không có preset).
6. `--DA` (store_true) → `ImageNetPolicy()` (AutoAugment) **chỉ cho nhánh drone**, chèn lên đầu.
7. `ToTensor`, `Normalize(ImageNet mean/std)`.

Transform val/test (`test.py:74–85`): chỉ `Resize((h,w), bicubic)` + `ToTensor` + `Normalize`. Không có north-up hay crop.

## 5. Dữ liệu train — `datasets/Dataloader_University.py`

| Thành phần | Dòng | Ghi chú |
|---|---|---|
| `Dataloader_University(root, transforms, names=['satellite','drone'])` | 10 | Định dạng: `root/satellite/<cls>/*` và `root/drone/<cls>/*`. Lớp = tên thư mục con, lấy theo `os.listdir(root/satellite)` rồi `sort()` (so sánh **chuỗi**). Mỗi `__getitem__(i)` bốc ngẫu nhiên **1 ảnh** trong mỗi lớp của mỗi view → trả `(img_s, img_d, i)`. `len` = số lớp. |
| | 20–27 | `os.listdir` **không lọc file ẩn** → một `.DS_Store` trong `satellite/` hoặc `drone/` sẽ bị coi là một lớp và gây lỗi. Không để file lạ trong các thư mục này. |
| `Sampler_University` | 81 | Hoán vị chỉ số lớp, lặp mỗi lớp `sample_num` lần. |
| `train_collate_fn` | 106 | Trả `[sat_batch, ids], [drone_batch, ids]`. Lưu ý trong `train.py:103` biến `inputs` là **vệ tinh**, `inputs3` là **drone**, và `model(inputs, inputs3)` đưa vệ tinh vào tham số tên `drone_image` (không sao vì dùng chung trọng số). |

`train.py:106` bỏ batch cuối nếu thiếu → mỗi epoch có `floor(nclasses*sample_num / batchsize)` iteration.

## 6. `test.py`

- Toàn bộ code nằm ở **cấp module** (argparse, đọc `opts.yaml` ở thư mục hiện tại, tạo `ImageFolder`, nạp model ở dòng 179). Chỉ phần trích feature + lưu `.mat` nằm trong `if __name__ == "__main__"` (dòng 210). ⇒ `import test` sẽ chạy argparse và nạp model; **không import hàm `extract_feature` được một cách sạch sẽ**.
- `extract_feature(model, dataloaders, view_index)` (dòng 122): với mỗi batch, chạy ảnh gốc + ảnh lật ngang (`fliplr`, dòng 103), cộng `outputs[1]` (descriptor của `ClassBlock`), rồi chuẩn hoá L2 (nếu 3 chiều thì chia thêm `sqrt(opt.block)`). `view_index=1` → nhánh thứ nhất của `model(...)`, `3` → nhánh thứ hai; hai nhánh cùng trọng số nên kết quả như nhau. Dùng biến toàn cục `opt.block`.
- Thư mục test: `<test_dir>/query_drone/<id>/*`, `<test_dir>/gallery_satellite/<id>/*` (`datasets.ImageFolder`). `get_id` (dòng 164) ép `int(folder_name)` → **id phải là số nguyên**.
- `--h/--w` mặc định 256 nhưng bị ghi đè bởi `opts.yaml` (vòng `setattr` dòng 48), nên lấy theo lúc train.
- `load_network(opt)` (`tool/utils.py:155`): `make_model(opt)` rồi `load_state_dict(torch.load(opt.checkpoint, map_location="cpu"))` (strict). `make_model` cũng có thể nạp `opt.load_from` trước đó.

## 7. `evaluate_gpu.py`

- `evaluate(qf, ql, gf, gl)` (dòng 10): `score = gf @ qf`, sắp giảm dần; good = `gl == ql`; junk = `gl == -1`.
- `compute_mAP(index, good_index, junk_index)` (dòng 31): CMC + AP (nội suy hình thang).
- In `Recall@1/5/10`, `Recall@top1` (= CMC tại 1% gallery), `AP`. Code chạy ở cấp module (không có `__main__` guard) → import sẽ đọc `.mat` ngay.

## 8. `evaluateDistance.py` — công thức SDM

Các hàm ở cấp module (import được, phần chạy nằm trong `if '__main__' == __name__`):

- `euclideanDistance(query, gallery)` (dòng 38): khoảng cách Euclid trên **(lon, lat) tính bằng độ**, giữa query và từng ô top-K.
- `evaluateSingle(distance, K)` (dòng 50):
  ```
  w_i = 1 - i/K,   i = 0..K-1
  SDM@K = Σ w_i · exp(-5000 · d_i) / Σ w_i     (d_i tính bằng độ)
  ```
  Hằng số `5e3` viết cứng (`--M` có nhưng không được dùng).
- `evaluate_SDM(indexOfTopK, queryIndex, K)` (dòng 78) phụ thuộc biến toàn cục `image_datasets`, `configDict` → chỉ tái sử dụng được `euclideanDistance` + `evaluateSingle`.
- `latlog2meter(lata, loga, latb, logb)` (dòng 61): haversine, R = 6378.137 km. `evaluate_MA` (dòng 92) = tỉ lệ top-1 trong ngưỡng 1..100 m.
- `Dense_GPS_ALL.txt`: mỗi dòng `<path>/<id>/<file> E<lon> N<lat> ...`, khoá = tên thư mục `<id>` (dòng 119–124).

## 9. Thiết bị, AMP và các chỗ còn gọi `torch.cuda.*`

`tool/utils.py:15 get_device()` chọn `cuda → mps → cpu`, biến toàn cục `device`.

Các chỗ còn gọi `torch.cuda.*` / phụ thuộc CUDA (tất cả đã được bọc điều kiện, chạy được trên MPS):

| File : dòng | Nội dung |
|---|---|
| `tool/utils.py:17` | `torch.cuda.is_available()` trong `get_device` |
| `tool/utils.py:210–211` | `set_seed`: `torch.cuda.manual_seed_all` nếu có CUDA (không seed riêng MPS; `torch.manual_seed` đã đủ cho MPS) |
| `train.py:86–88` | `use_cuda`, `GradScaler("cuda", enabled=use_cuda and opt.autocast)` |
| `train.py:118` | `torch.autocast(device_type="cuda", enabled=use_cuda)` → **trên MPS không có AMP** |
| `train.py:192–193`, `test.py:69–70`, `test_hard.py:71–72` | `torch.cuda.set_device` khi có CUDA |
| `tool/visual/demo_custom.py:150–151`, `tool/applications/inference_global.py:155–156`, `inference_neibor.py:185–186`, `forwardAllSatelliteHub.py:71–72` | `torch.cuda.set_device` khi có CUDA |
| `tool/visual/heatmap.py:95`, `test.py:130–133`, `test_hard.py:123–126`, `forwardAllSatelliteHub.py:111–114` | chỉ trong comment |
| `tool/visual/grad_cam.py:60` | `use_cuda=False` |

Vấn đề khác liên quan thiết bị:
- `train.py:49` `--autocast` là `store_true` với `default=True` → không tắt được từ dòng lệnh; và chỉ có tác dụng khi có CUDA.
- `models/taskflow.py:31` `torch.load(load_from)` không có `map_location` → checkpoint baseline lưu từ GPU sẽ báo lỗi khi nạp trên máy không có CUDA. Cần `map_location="cpu"` trước khi fine-tune trên Mac.
- `cudnn.*` trong `set_seed` vô hại trên MPS.
- Không có chỗ nào log bộ nhớ GPU (`max_memory_allocated`).

## 10. Optimizer — `optimizers/make_optimizer.py:5`

SGD momentum 0.9, nesterov, wd 5e-4; backbone LR = `0.3·lr`, phần còn lại LR = `lr`; `MultiStepLR(milestones, gamma=0.1)`. Không có warmup (`--warm_epoch` không được dùng trong `train.py`).

## 11. Script

Input/output của từng bước:

| Script | Input | Output |
|---|---|---|
| `train.py` | `--data_dir` (`train/{satellite,drone}/<cls>/`), `--load_from` | `checkpoints/<name>/net_XXX.pth` (mỗi `save_interval` epoch + epoch cuối), `train.log`, `opts.yaml`, bản copy code |
| `test.py` | chạy trong `checkpoints/<name>/`: `./opts.yaml`, `--checkpoint`, `--test_dir` (`query_drone/`, `gallery_satellite/`), `--mode 1\|2` | `pytorch_result_<mode>.mat` (`query_f, query_label, query_path, gallery_f, gallery_label, gallery_path`), `inference_time.txt` |
| `evaluate_gpu.py` | `./pytorch_result_1.mat` | in và ghi `results.txt` (Recall@1/5/10, Recall@top1%, AP) |
| `evaluateDistance.py` | `./pytorch_result_<mode>.mat`, `--root_dir` (cần `Dense_GPS_ALL.txt` + `test/`) | `SDM@K(1,100).json`, `MA@K(1,100)` (không có đuôi .json), in SDM@1/3/5/10 |

- `train_test_local.sh`: chạy train baseline (`ViTS-224`, `SingleBranch`, `avg`, `block=1`, CE+WeightedSoftTriplet+KL, `ra=satellite re=satellite cj=no rr=uav`, lr 0.01) → test → evaluate_gpu → evaluateDistance.
- `finetune_trees.sh` (thí nghiệm E0a): cùng kiến trúc baseline, `name=forest_E0a`, `root_dir=$HOME/DenseUAV_Forest`, **không `--load_from`** (train từ ImageNet qua timm), lr 0.01, 60 epoch, milestones 40,55, `save_interval` 10, `cj/rr` cho cả hai nhánh. Phần test/evaluate cũ ở cuối đã bị comment; đánh giá làm bằng `localize/` + `eval/` (M3).

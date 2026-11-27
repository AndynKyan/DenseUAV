# Train DenseUAV (ViT-S, SingleBranch) on the forest dataset, experiment E0a.
# Starts from ImageNet weights: timm loads vit_small_patch16_224 with pretrained=True.
# No DenseUAV baseline checkpoint is used (the download link is dead).
# Data layout must match DenseUAV:
#   $root_dir/train/drone/<id>/*.JPG        $root_dir/train/satellite/<id>/*.tif
#   $root_dir/test/query_drone/<id>/...     $root_dir/test/gallery_satellite/<id>/...
#   $root_dir/Dense_GPS_ALL.txt             (only needed for the SDM metric)
# <id> folders must be integers (test.py parses them with int()).

# Seed: SEED=1 bash finetune_trees.sh -> seed 1, checkpoints/forest_E0a_s1 (seed 0 keeps the plain name).
seed=${SEED:-0}
name="forest_E0a"
if [ "$seed" != "0" ]; then name="forest_E0a_s$seed"; fi
root_dir="$HOME/DenseUAV_Forest"
data_dir=$root_dir/train
test_dir=$root_dir/test
gpu_ids=0
num_worker=8

# lr 0.01 diverged; a 15-epoch probe with 0.003 trained fine (D21). 120 epochs, lr x0.1 at 80 and 110.
lr=0.003
num_epochs=120
milestones="80,110"
save_interval=5
batchsize=16
# Each class is repeated sample_num times in a row (Sampler_University, np.repeat): with 29 classes,
# 29*4/16 -> 7 iterations per epoch, and every batch holds 4 classes x 4 samples (see docs/demo_deviations.md D18).
sample_num=4

# Same architecture as the DenseUAV baseline.
backbone="ViTS-224"
head="SingleBranch"
head_pool="avg"
block=1
num_bottleneck=512
h=224
w=224

cls_loss="CELoss"
feature_loss="WeightedSoftTripletLoss"
kl_loss="KLLoss"

# Vegetation changes colour with season/light and has no fixed orientation,
# so color jitter is enabled on both views and rotation on both views.
ra="satellite"
re="satellite"
cj="uav,satellite"
rr="uav,satellite"

python train.py --name $name --data_dir $data_dir --gpu_ids $gpu_ids --num_worker $num_worker \
                --lr $lr --num_epochs $num_epochs --milestones $milestones --save_interval $save_interval \
                --batchsize $batchsize --sample_num $sample_num --block $block \
                --backbone $backbone --head $head --head_pool $head_pool --num_bottleneck $num_bottleneck \
                --h $h --w $w \
                --ra $ra --re $re --cj $cj --rr $rr \
                --cls_loss $cls_loss --feature_loss $feature_loss --kl_loss $kl_loss --seed $seed

# Đánh giá bằng localize/ + eval/ (M3), không dùng test.py / evaluate_gpu.py / evaluateDistance.py.
# last_ckpt=$(printf "net_%03d.pth" $((num_epochs - 1)))
# cd checkpoints/$name
# python test.py --name $name --test_dir $test_dir --gpu_ids $gpu_ids --num_worker $num_worker --checkpoint $last_ckpt
# python evaluate_gpu.py
# python evaluateDistance.py --root_dir $root_dir
# cd ../../

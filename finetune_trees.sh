# Fine-tune the DenseUAV baseline on a custom (tree / vegetation) dataset.
# Data layout must match DenseUAV:
#   $root_dir/train/drone/<id>/*.JPG        $root_dir/train/satellite/<id>/*.tif
#   $root_dir/test/query_drone/<id>/...     $root_dir/test/gallery_satellite/<id>/...
#   $root_dir/Dense_GPS_ALL.txt             (only needed for the SDM metric)
# <id> folders must be integers (test.py parses them with int()).

name="trees_ft"
root_dir="/path/to/TreeUAV"                     # <-- change
pretrained="checkpoints/baseline/net_119.pth"   # <-- DenseUAV baseline weights (see README)
data_dir=$root_dir/train
test_dir=$root_dir/test
gpu_ids=0
num_worker=8

# Smaller lr and fewer epochs than training from scratch (0.01 / 120 epochs).
lr=0.003
num_epochs=40
milestones="25,35"
save_interval=10
batchsize=16

# Must match the pretrained baseline so its weights load.
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
                --batchsize $batchsize --sample_num 1 --block $block \
                --backbone $backbone --head $head --head_pool $head_pool --num_bottleneck $num_bottleneck \
                --h $h --w $w --load_from $pretrained \
                --ra $ra --re $re --cj $cj --rr $rr \
                --cls_loss $cls_loss --feature_loss $feature_loss --kl_loss $kl_loss

last_ckpt=$(printf "net_%03d.pth" $((num_epochs - 1)))
cd checkpoints/$name
python test.py --name $name --test_dir $test_dir --gpu_ids $gpu_ids --num_worker $num_worker --checkpoint $last_ckpt
python evaluate_gpu.py
python evaluateDistance.py --root_dir $root_dir
cd ../../

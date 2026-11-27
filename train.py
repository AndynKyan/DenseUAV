# -*- coding: utf-8 -*-

from __future__ import print_function, division
import argparse
import torch
import torch.nn as nn

from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import time
from optimizers.make_optimizer import make_optimizer
# from models.model import make_model
from models.taskflow import make_model
from datasets.make_dataloader import make_dataset
from tool.utils import save_network, copyfiles2checkpoints, get_preds, get_logger, calc_flops_params, set_seed, device
import warnings
from losses.loss import Loss


warnings.filterwarnings("ignore")


def get_parse():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--gpu_ids', default='0', type=str,
                        help='gpu_ids: e.g. 0  0,1,2  0,2')
    parser.add_argument('--name', default='test',
                        type=str, help='the experiment name that will be saved in checkpoints dir in the root')
    parser.add_argument('--data_dir', default='/home/dmmm/Dataset/DenseUAV/data_2022/train',
                        type=str, help='training dir path')
    parser.add_argument('--num_worker', default=0, type=int, help='')
    parser.add_argument('--batchsize', default=2, type=int, help='batchsize')
    parser.add_argument('--pad', default=0, type=int, help='padding')
    parser.add_argument('--h', default=224, type=int, help='height')
    parser.add_argument('--w', default=224, type=int, help='width')
    parser.add_argument('--rr', default="", type=str, help='random rotate')
    parser.add_argument('--ra', default="", type=str, help='random affine')
    parser.add_argument('--re', default="", type=str, help='random erasing')
    parser.add_argument('--cj', default="", type=str, help='color jitter')
    parser.add_argument('--erasing_p', default=0.3, type=float,
                        help='random erasing probability, in [0,1]')
    parser.add_argument('--warm_epoch', default=0, type=int,
                        help='the first K epoch that needs warm up')
    parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
    parser.add_argument('--DA', action='store_true',
                        help='use Color Data Augmentation')
    parser.add_argument('--droprate', default=0.5,
                        type=float, help='drop rate')
    parser.add_argument('--autocast', action='store_true',
                        default=True, help='use mix precision')
    parser.add_argument('--block', default=2, type=int, help='')
    parser.add_argument('--cls_loss', default="CELoss", type=str, help='loss type of representation learning')
    parser.add_argument('--feature_loss', default="no", type=str, help='loss type of metric learning')
    parser.add_argument('--kl_loss', default="no", type=str, help='loss type of mutual learning')
    parser.add_argument('--sample_num', default=1, type=int,
                        help='num of repeat sampling')
    parser.add_argument('--num_epochs', default=120, type=int, help='total epoches for training')
    parser.add_argument('--num_bottleneck', default=512, type=int, help='the dimensions for embedding the feature')
    parser.add_argument('--load_from', default="", type=str, help='checkpoints path for pre-loading')
    parser.add_argument('--backbone', default="cvt13", type=str, help='backbone network for applying')
    parser.add_argument('--head', default="FSRA_CNN", type=str, help='head type for applying')
    parser.add_argument('--head_pool', default="max", type=str, help='head pooling type for applying')
    parser.add_argument('--milestones', default="70,110", type=str,
                        help='epochs at which lr is multiplied by 0.1, e.g. "20,35"')
    parser.add_argument('--save_interval', default=10, type=int,
                        help='save a checkpoint every N epochs (the last epoch is always saved)')
    parser.add_argument('--seed', default=666, type=int, help='random seed (666 = original repo value)')
    

    opt = parser.parse_args()
    print(opt)
    return opt

def train_model(model, opt, optimizer, scheduler, dataloaders, dataset_sizes):
    logger = get_logger(
        "checkpoints/{}/train.log".format(opt.name))

    # thop计算MACs
    # macs, params = calc_flops_params(
    #     model, (1, 3, opt.h, opt.w), (1, 3, opt.h, opt.w))
    # logger.info("model MACs={}, Params={}".format(macs, params))

    use_gpu = opt.use_gpu
    num_epochs = opt.num_epochs
    since = time.time()
    # autocast/GradScaler chi bat khi co CUDA
    use_cuda = device.type == "cuda"
    use_scaler = use_cuda and opt.autocast
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    nnloss = Loss(opt)
    for epoch in range(num_epochs):
        logger.info('Epoch {}/{}'.format(epoch, num_epochs - 1))
        logger.info('-' * 50)

        model.train(True)  # Set model to training mode
        running_cls_loss = 0.0
        running_triplet = 0.0
        running_kl_loss = 0.0
        running_loss = 0.0
        running_corrects = 0.0
        running_corrects2 = 0.0
        for data, data3 in dataloaders:
            # 获取输入无人机和卫星数据
            inputs, labels = data
            inputs3, labels3 = data3
            now_batch_size = inputs.shape[0]
            if now_batch_size < opt.batchsize:  # skip the last batch
                continue
            inputs = inputs.to(device).detach()
            inputs3 = inputs3.to(device).detach()
            labels = labels.to(device).detach()
            labels3 = labels3.to(device).detach()

            # 梯度清零
            optimizer.zero_grad()

            # start_time = time.time()
            # 模型前向传播
            with torch.autocast(device_type="cuda", enabled=use_cuda):
                outputs, outputs2 = model(inputs, inputs3)
            # print("model_time:{}".format(time.time()-start_time))
            # 计算损失
            loss, cls_loss, f_triplet_loss, kl_loss = nnloss(
                outputs, outputs2, labels, labels3)
            # start_time = time.time()
            # 反向传播
            if use_scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            # print("backward_time:{}".format(time.time()-start_time))
            if epoch == 0 and not getattr(opt, "_mem_logged", False):
                opt._mem_logged = True
                if device.type == "mps":
                    logger.info("MPS memory after first iteration: current_allocated {:.1f} MB, driver_allocated {:.1f} MB".format(
                        torch.mps.current_allocated_memory() / 2**20, torch.mps.driver_allocated_memory() / 2**20))
                elif device.type == "cuda":
                    logger.info("CUDA max_memory_allocated after first iteration: {:.1f} MB".format(
                        torch.cuda.max_memory_allocated() / 2**20))

            # 统计损失
            running_loss += loss.item() * now_batch_size
            running_cls_loss += cls_loss.item()*now_batch_size
            running_triplet += f_triplet_loss.item() * now_batch_size
            running_kl_loss += kl_loss.item() * now_batch_size

            # 统计精度
            preds, preds2 = get_preds(outputs[0], outputs2[0])
            if isinstance(preds, list) and isinstance(preds2, list):
                running_corrects += sum([float(torch.sum(pred == labels.data))
                                        for pred in preds])/len(preds)
                running_corrects2 += sum([float(torch.sum(pred == labels3.data))
                                         for pred in preds2]) / len(preds2)
            else:
                running_corrects += float(torch.sum(preds == labels.data))
                running_corrects2 += float(torch.sum(preds2 == labels3.data))

        # 统计损失和精度
        epoch_cls_loss = running_cls_loss/dataset_sizes['satellite']
        epoch_kl_loss = running_kl_loss / dataset_sizes['satellite']
        epoch_triplet_loss = running_triplet/dataset_sizes['satellite']
        epoch_loss = running_loss / dataset_sizes['satellite']
        epoch_acc = running_corrects / dataset_sizes['satellite']
        epoch_acc2 = running_corrects2 / dataset_sizes['satellite']

        lr_backbone = optimizer.state_dict()['param_groups'][0]['lr']
        lr_other = optimizer.state_dict()['param_groups'][1]['lr']
        logger.info('Loss: {:.4f} Cls_Loss:{:.4f} KL_Loss:{:.4f} Triplet_Loss {:.4f} Satellite_Acc: {:.4f}  Drone_Acc: {:.4f} lr_backbone:{:.6f} lr_other {:.6f}'
                    .format(epoch_loss, epoch_cls_loss, epoch_kl_loss,
                            epoch_triplet_loss, epoch_acc,
                            epoch_acc2, lr_backbone, lr_other))

        scheduler.step()
        if (epoch + 1) % opt.save_interval == 0 or epoch == num_epochs - 1:
            save_network(model, opt.name, epoch)

        time_elapsed = time.time() - since
        since = time.time()
        logger.info('Training complete in {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))


if __name__ == '__main__':
    opt = get_parse()
    set_seed(opt.seed)
    str_ids = opt.gpu_ids.split(',')
    gpu_ids = []
    for str_id in str_ids:
        gid = int(str_id)
        if gid >= 0:
            gpu_ids.append(gid)

    use_gpu = device.type != "cpu"
    opt.use_gpu = use_gpu
    print("Using device: {}".format(device))
    # set gpu ids (chi khi co CUDA)
    if device.type == "cuda" and len(gpu_ids) > 0:
        torch.cuda.set_device(gpu_ids[0])
        cudnn.benchmark = True

    dataloaders, class_names, dataset_sizes = make_dataset(opt)
    opt.nclasses = len(class_names)

    model = make_model(opt)

    optimizer_ft, exp_lr_scheduler = make_optimizer(model, opt)

    model = model.to(device)
    # 移动文件到指定文件夹
    copyfiles2checkpoints(opt)

    train_model(model, opt, optimizer_ft, exp_lr_scheduler,
                dataloaders, dataset_sizes)

import argparse
import logging
import os
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel.distributed import DistributedDataParallel
import torch.utils.data.distributed
from torch.nn.utils import clip_grad_norm_
from torch.nn import CrossEntropyLoss

from utils import losses
from config.config import config as cfg
from utils.dataset import MXFaceDataset, DataLoaderX, FaceDatasetFolder
from utils.utils_callbacks import CallBackVerification, CallBackLogging, CallBackModelCheckpoint
from utils.utils_logging import AverageMeter, init_logging

from backbones.iresnet import iresnet100, iresnet50, iresnet18, iresnet34
from cleanup import clean_folder

torch.backends.cudnn.benchmark = True



def format_output_folder(experiment, net, loss, auth_ds, synth_ds, auth_ids, synth_ids, cmt=""):
    base = cfg.output
    base += f"{experiment}/"
    base += f"{net}_"
    base += f"{loss}_"
    base += f"{synth_ds}_{synth_ids//1000}K_" if synth_ids > 0 else ""
    base += f"{auth_ds}_{auth_ids//1000}K_" if auth_ids > 0 else ""
    base += f"/{cmt}_" if cmt != "" else ""
    return base


def get_exp_config(experiment, cmt, dataset):
    if experiment in ["baseline", "reference"]:
        return None
    data_path = os.path.join(f"metrics/data2/{dataset}", f"{experiment}.pkl")
    data = np.load(data_path, allow_pickle=True)
    print(f"Using {dataset}: Sorting via {experiment} in {cmt} order")
    return (data, cmt)

def main(args):
    dist.init_process_group(backend='nccl', init_method='env://')
    local_rank = args.local_rank
    torch.cuda.set_device(local_rank)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    auth_ids = int(args.auth_id)
    synth_ids = int(args.synth_id)
    experiment = args.experiment
    cfg.auth_dataset = args.auth_ds
    cfg.synt_dataset = args.synth_ds
    cfg.synthetic_root = cfg.synt_dict[cfg.synt_dataset]
    cfg.rec = cfg.auth_dict[cfg.auth_dataset]
    iteration = args.iter
    cmt = args.cmt



    cfg.output = format_output_folder(
        experiment=experiment,
        net=cfg.network,
        loss=cfg.loss,
        auth_ds=cfg.auth_dataset,
        synth_ds=cfg.synt_dataset,
        auth_ids=auth_ids,
        synth_ids=synth_ids,
        cmt=cmt
    )

    experimental_config = get_exp_config(
        experiment=experiment,
        cmt=cmt,
        dataset=cfg.auth_dataset if auth_ids > 0 else cfg.synt_dataset
    )

    if iteration != -1:
        cfg.output += f"/{iteration}"

    global_step = cfg.global_step
    if not os.path.exists(cfg.output) and rank==0:
        os.makedirs(cfg.output)
    elif os.path.exists(cfg.output):
        content = os.listdir(cfg.output)
        content = [int(c.replace("backbone.pth", "").replace("header.pth", "")) for c in content if c.endswith(".pth")]
        if len(content) > 0:
            global_step = max(content)
            print(f"Training will resume from step {global_step}")
    else:
        time.sleep(2)

    log_root = logging.getLogger()
    init_logging(log_root, rank, cfg.output)



    logging.info(f"Dataset: {cfg.synthetic_root if synth_ids > 0 else cfg.rec}")

    trainset = FaceDatasetFolder(
        root_dir=cfg.synthetic_root,
        local_rank=local_rank,
        root2=cfg.rec,
        synth_ids=synth_ids,
        auth_ids=auth_ids,
        shuffle=iteration!=-1,
        criterion=experimental_config
    )

    num_ids = trainset.num_ids
    cfg.num_classes = num_ids[0] +1
    cfg.num_image = len(trainset.imgidx)
    cfg.eval_step = int(len(trainset) / cfg.batch_size / world_size)

    if local_rank == 0:
        logging.info(f"Classes: {cfg.num_classes - num_ids[1]} synthetic, {num_ids[1]} real - {cfg.num_image} images - eval: {cfg.eval_step}")
        #logging.info(
        #    f"Total Samples: {len(trainset)}")
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        trainset, shuffle=True)

    train_loader = DataLoaderX(
        local_rank=local_rank, dataset=trainset, batch_size=cfg.batch_size,
        sampler=train_sampler, num_workers=0, pin_memory=True, drop_last=True)

    # load model
    if cfg.network == "iresnet100":
        backbone = iresnet100(num_features=cfg.embedding_size, use_se=cfg.SE).to(local_rank)
    elif cfg.network == "iresnet50":
        backbone = iresnet50(dropout=0.4,num_features=cfg.embedding_size, use_se=cfg.SE).to(local_rank)
    elif cfg.network == "iresnet34":
        backbone = iresnet34(dropout=0.4,num_features=cfg.embedding_size, use_se=cfg.SE).to(local_rank)
    else:
        backbone = None
        logging.info("load backbone failed!")
        exit()

    if os.path.exists(cfg.output) and rank == 0 and global_step > 0:
        try:
            backbone_pth = os.path.join(cfg.output, str(global_step) + "backbone.pth")
            backbone.load_state_dict(torch.load(backbone_pth, map_location=torch.device(torch.device("cuda:0"))))

            if rank == 0:
                logging.info("backbone resume loaded successfully!")
        except (FileNotFoundError, KeyError, IndexError, RuntimeError):
            logging.info(f"load backbone resume init, failed! {global_step}")

    for ps in backbone.parameters():
        dist.broadcast(ps, 0)

    backbone = DistributedDataParallel(
        module=backbone, broadcast_buffers=False, device_ids=[local_rank])
    backbone.train()

    # get header
    if cfg.loss == "ElasticArcFace":
        header = losses.ElasticArcFace(in_features=cfg.embedding_size, out_features=cfg.num_classes, s=cfg.s, m=cfg.m,std=cfg.std).to(local_rank)
    elif cfg.loss == "ElasticArcFacePlus":
        header = losses.ElasticArcFace(in_features=cfg.embedding_size, out_features=cfg.num_classes, s=cfg.s, m=cfg.m,
                                       std=cfg.std, plus=True).to(local_rank)
    elif cfg.loss == "ElasticCosFace":
        header = losses.ElasticCosFace(in_features=cfg.embedding_size, out_features=cfg.num_classes, s=cfg.s, m=cfg.m,std=cfg.std).to(local_rank)
    elif cfg.loss == "ElasticCosFacePlus":
        header = losses.ElasticCosFace(in_features=cfg.embedding_size, out_features=cfg.num_classes, s=cfg.s, m=cfg.m,
                                       std=cfg.std, plus=True).to(local_rank)
    elif cfg.loss == "ArcFace":
        header = losses.ArcFace(in_features=cfg.embedding_size, out_features=cfg.num_classes, s=cfg.s, m=cfg.m).to(local_rank)
    elif cfg.loss == "CosFace":
        header = losses.CosFace(in_features=cfg.embedding_size, out_features=cfg.num_classes, s=cfg.s, m=cfg.m).to(
            local_rank)
    elif cfg.loss == "AdaFace":
        header = losses.AdaFace(in_features=cfg.embedding_size, out_features=cfg.num_classes, s=cfg.s, m=cfg.m).to(
            local_rank)
    else:
        print("Header not implemented")
    if os.path.exists(cfg.output) and rank == 0 and global_step > 0:
        try:
            header_pth = os.path.join(cfg.output, str(global_step) + "header.pth")
            header.load_state_dict(torch.load(header_pth, map_location=torch.device(local_rank)))

            if rank == 0:
                logging.info("header resume loaded successfully!")
        except (FileNotFoundError, KeyError, IndexError, RuntimeError):
            logging.info("header resume init, failed!")
    
    header = DistributedDataParallel(
        module=header, broadcast_buffers=False, device_ids=[local_rank])
    header.train()

    opt_backbone = torch.optim.SGD(
        params=[{'params': backbone.parameters()}],
        lr=cfg.lr / 512 * cfg.batch_size * world_size,
        momentum=0.9, weight_decay=cfg.weight_decay)
    opt_header = torch.optim.SGD(
        params=[{'params': header.parameters()}],
        lr=cfg.lr / 512 * cfg.batch_size * world_size,
        momentum=0.9, weight_decay=cfg.weight_decay)

    scheduler_backbone = torch.optim.lr_scheduler.LambdaLR(
        optimizer=opt_backbone, lr_lambda=cfg.lr_func)
    scheduler_header = torch.optim.lr_scheduler.LambdaLR(
        optimizer=opt_header, lr_lambda=cfg.lr_func)        

    criterion = CrossEntropyLoss()

    start_epoch = 0
    total_step = int(len(trainset) / cfg.batch_size / world_size * cfg.num_epoch)
    if rank == 0: logging.info("Total Step is: %d" % total_step)

    if os.path.exists(cfg.output) and rank == 0 and global_step > 0:
        rem_steps = (total_step - global_step)
        cur_epoch = cfg.num_epoch - int(cfg.num_epoch / total_step * rem_steps)
        logging.info("resume from estimated epoch {}".format(cur_epoch))
        logging.info("remaining steps {}".format(rem_steps))
        
        start_epoch = cur_epoch
        scheduler_backbone.last_epoch = cur_epoch
        scheduler_header.last_epoch = cur_epoch

        # --------- this could be solved more elegant ----------------
        opt_backbone.param_groups[0]['lr'] = scheduler_backbone.get_lr()[0]
        opt_header.param_groups[0]['lr'] = scheduler_header.get_lr()[0]

        print("last learning rate: {}".format(scheduler_header.get_lr()))
        # ------------------------------------------------------------

    callback_verification = CallBackVerification(cfg.eval_step, rank, cfg.val_targets, cfg.val_root)
    callback_logging = CallBackLogging(50, rank, total_step, cfg.batch_size, world_size, writer=None)
    callback_checkpoint = CallBackModelCheckpoint(rank, cfg.output)
    loss = AverageMeter()
    global_step = global_step
    for epoch in range(start_epoch, cfg.num_epoch):
        train_sampler.set_epoch(epoch)
        for _, (idx, img, label, fn) in enumerate(train_loader):
            global_step += 1
            img = img.cuda(local_rank, non_blocking=True)
            label = label.cuda(local_rank, non_blocking=True)

            features = F.normalize(backbone(img))

            thetas = header(features, label)
            loss_v = criterion(thetas, label)
            loss_v.backward()

            clip_grad_norm_(backbone.parameters(), max_norm=5, norm_type=2)

            opt_backbone.step()
            opt_header.step()

            opt_backbone.zero_grad()
            opt_header.zero_grad()

            loss.update(loss_v.item(), 1)
            
            callback_logging(global_step, loss, epoch)
            callback_verification(global_step, backbone)

        scheduler_backbone.step()
        scheduler_header.step()

        callback_checkpoint(global_step, backbone, header)

    clean_folder(cfg.output)
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch margin penalty loss training')
    parser.add_argument('--local_rank', type=int, default=0, help='local_rank')
    parser.add_argument('--resume', type=int, default=0, help="resume training")
    parser.add_argument("--auth_id", type=int, default=0, help="authentic identities")
    parser.add_argument("--auth_ds", type=str, default="WF", help="authentic dataset")
    parser.add_argument("--iter", type=int, default=-1, help="sampling iteration")
    parser.add_argument("--synth_id", type=int, default=0, help="synthetic identities")
    parser.add_argument("--synth_ds", type=str, default="GC", help="synthetic dataset")
    parser.add_argument("--experiment", type=str, default="", help="experiment")
    parser.add_argument("--cmt", type=str, default="", help="additional comments")

    args_ = parser.parse_args()
    main(args_)

import logging
import os

#import cv2
import sys
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np 

sys.path.append("../")
from utils import losses
from utils.dataset import DataLoaderX, FaceDatasetFolder
from utils.utils_callbacks import CallBackVerification
from utils.utils_logging import init_logging
from config import config as cfg

from backbones.iresnet import iresnet100, iresnet50, iresnet18

if __name__ == "__main__":
    gpu_id = 0
    log_root = logging.getLogger()

    if cfg.network == "iresnet100":
        print("iresnet100")
        backbone = iresnet100(num_features=512).to(f"cuda:{gpu_id}")
    elif cfg.network == "iresnet50":
        print("iresnet50")
        backbone = iresnet50(num_features=512).to(f"cuda:{gpu_id}")
    elif cfg.network == "iresnet18":
        print("iresnet18")
        backbone = iresnet18(num_features=512).to(f"cuda:{gpu_id}")
    else:
        backbone = None
        exit()

    local_rank=f"cuda:{gpu_id}"

    trainset = FaceDatasetFolder(root_dir=cfg.rec, local_rank=local_rank)
    num_ids = trainset.num_ids
    cfg.num_classes = num_ids +1
    cfg.num_image = len(trainset.imgidx)
    
    if local_rank == 0:
        print(f"Classes: {num_ids+1} real - {cfg.num_image} images - eval: {cfg.eval_step}")

    train_sampler = torch.utils.data.RandomSampler(trainset)

    train_loader = DataLoaderX(
        local_rank=local_rank, dataset=trainset, batch_size=cfg.batch_size,
        sampler=train_sampler, num_workers=0, pin_memory=True, drop_last=True)

    header = losses.CosFace(in_features=cfg.embedding_size, out_features=cfg.num_classes, s=cfg.s, m=cfg.m).to(
            local_rank)

    weights_bb = "output_IDiffFace500/R18_CosFace_500_webface/1560backbone.pth"
    backbone.load_state_dict(torch.load(weights_bb))

    weights_header = "output_IDiffFace500/R18_CosFace_500_webface/1560header.pth"
    header.load_state_dict(torch.load(weights_header))
    
    name_dataset = weights_bb.split('_')[1].split('/')[0]

    output_path=f"criterion_npy"
    header = torch.nn.DataParallel(header, device_ids=[gpu_id])
    backbone = torch.nn.DataParallel(backbone, device_ids=[gpu_id])

    softmax = nn.Softmax(dim=1)
    
    M, N = cfg.num_image, 2
    s=(M,N)

    arr=np.zeros(s)

    backbone.eval()
    header.eval()

    with torch.no_grad(): #per non considerare i gradienti

        for _, (idx, img, label) in enumerate(train_loader):
            img = img.cuda(local_rank, non_blocking=True)
            label = label.cuda(local_rank, non_blocking=True)

            features = F.normalize(backbone(img))

            #distribuzione delle probabilità di tutte le classi
            thetas = header(features, label)
                
            output = softmax(thetas)

            positive_sims = output[torch.arange(output.size(0)).unsqueeze(1), label.view(-1, 1)]
            
            arr[index, 0] = label.cpu().numpy()[i]
            arr[index, 1] = positive_sims.cpu().numpy().flatten()[i]


    os.makedirs(f'criterion_npy/{name_dataset}', exist_ok=True)
    np.save(f'criterion_npy/{name_dataset}/confidence_array.npy', arr)
    

        
        



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

from backbones.iresnet import iresnet100, iresnet50, iresnet34

gpu_id = 0
log_root = logging.getLogger()
output_path = f"{cfg.output}/{cfg.dataset}_classification_confidence.npy"
softmax = nn.Softmax(dim=1)

print(cfg.network)
if cfg.network == "iresnet100":
    backbone = iresnet100(num_features=512).to(f"cuda:{gpu_id}")
elif cfg.network == "iresnet50":
    backbone = iresnet50(num_features=512).to(f"cuda:{gpu_id}")
elif cfg.network == "iresnet34":
    backbone = iresnet34(num_features=512).to(f"cuda:{gpu_id}")
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

backbone.load_state_dict(torch.load(cfg.data_dict[cfg.dataset]["pretrained"]["backbone"]))
backbone = torch.nn.DataParallel(backbone, device_ids=[gpu_id])
header.load_state_dict(torch.load(cfg.data_dict[cfg.dataset]["pretrained"]["header"]))
header = torch.nn.DataParallel(header, device_ids=[gpu_id])




backbone.eval()
header.eval()

class_confidences = {}

with torch.no_grad():
    for _, (idx, img, label) in enumerate(train_loader):
        img = img.cuda(local_rank, non_blocking=True)
        label = label.cuda(local_rank, non_blocking=True)

        features = F.normalize(backbone(img))

        # distribuzione delle probabilità di tutte le classi
        thetas = header(features, label)
        output = softmax(thetas)

        for i in range(output.size(0)):
            class_id = label[i].item()
            confidence = output[i, class_id].item()
            if class_id not in class_confidences:
                class_confidences[class_id] = []
            class_confidences[class_id].append(confidence)

# Calcola la media delle confidenze per ogni classe
mean_confidences = {class_id: np.mean(confidences) for class_id, confidences in class_confidences.items()}

# Ordina le classi in ordine crescente di confidenza
sorted_classes = sorted(mean_confidences.items(), key=lambda x: x[1])

#classi ordinate e le loro confidenze
sorted_confidences = np.array(sorted_classes)

os.makedirs(f'{output_path}/{name_dataset}', exist_ok=True)
np.save(f'{output_path}/{name_dataset}/confidence_array.npy', sorted_confidences)

print(f"Array di confidenza salvato in: {output_path}/{name_dataset}/confidence_array.npy")
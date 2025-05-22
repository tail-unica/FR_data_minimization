
import logging
import os
import pickle
import sys
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np 
from tqdm import tqdm

sys.path.append("../")
from utils import losses
from utils.dataset import DataLoaderX, FaceDatasetFolder
from config import config as cfg

from backbones.iresnet import iresnet100, iresnet50, iresnet34

PAD_CHAR = 95
gpu_id = 0
log_root = logging.getLogger()




for dataset in cfg.data_dict.keys():

    print(f"Inferencing {dataset}: \n data: {cfg.data_dict[dataset]['data']} \n pretrained: {cfg.data_dict[dataset]['pretrained']['backbone']}")

    base_putpath = f"data2/{dataset}/"
    output_path = f"data2/{dataset}/confidence.pkl"
    if os.path.exists(output_path):
        print(f"Skipping {dataset}: already exists")
    else:
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

        trainset = FaceDatasetFolder(root_dir=cfg.data_dict[dataset]['data'], local_rank=local_rank)
        num_ids = len(list(set(trainset.labels)))
        cfg.num_classes = num_ids
        cfg.num_image = len(trainset.imgidx)

        if local_rank == 0:
            print(f"Classes: {num_ids+1} real - {cfg.num_image} images - eval: {cfg.eval_step}")

        train_sampler = torch.utils.data.RandomSampler(trainset)

        train_loader = DataLoaderX(
            local_rank=local_rank, dataset=trainset, batch_size=cfg.batch_size,
            sampler=train_sampler, num_workers=0, pin_memory=True, drop_last=True)

        header = losses.CosFace(in_features=cfg.embedding_size, out_features=cfg.num_classes, s=cfg.s, m=cfg.m).to(
                local_rank)

        backbone.load_state_dict(torch.load(cfg.data_dict[dataset]["pretrained"]["backbone"], weights_only=True))
        backbone = torch.nn.DataParallel(backbone, device_ids=[gpu_id])
        header.load_state_dict(torch.load(cfg.data_dict[dataset]["pretrained"]["header"], weights_only=True))
        header = torch.nn.DataParallel(header, device_ids=[gpu_id])


        backbone.eval()
        header.eval()

        class_confidences = {}
        progress_bar = tqdm(total=len(train_loader))
        with torch.no_grad():
            for _, (idx, img, label, folder_name) in enumerate(train_loader):
                #print(folder_name)
                img = img.cuda(local_rank, non_blocking=True)
                label = label.cuda(local_rank, non_blocking=True)

                features = F.normalize(backbone(img))

                # distribuzione delle probabilità di tutte le classi
                thetas = header(features, label)
                output = thetas[torch.arange(thetas.size(0)).unsqueeze(1), label.view(-1, 1)]
                output /= cfg.s
                output.acos_()
                output += cfg.m
                output.cos_()

                for i in range(output.size(0)):
                    class_id = ''.join([chr(c.item()) for c in folder_name[i] if c != PAD_CHAR])#label[i].item()
                    confidence = output[i].item()
                    if class_id not in class_confidences:
                        class_confidences[class_id] = []
                    class_confidences[class_id].append(confidence)

                progress_bar.update(1)

        # Calcola la media delle confidenze per ogni classe
        mean_confidences = {class_id: np.mean(confidences) for class_id, confidences in class_confidences.items()}

        #cls = np.asarray(list(mean_confidences.keys()))
        #cnf = np.asarray(list(mean_confidences.values()))
        #to_save = np.hstack([cls.reshape(-1, 1), cnf.reshape(-1, 1)])
        to_save = (list(mean_confidences.keys()), list(mean_confidences.values()))

        os.makedirs(base_putpath, exist_ok=True)
        with open(output_path, 'wb') as fp:
            pickle.dump(to_save, fp)
        #np.save(output_path, to_save)

        print(f"Results saved in: {output_path}")
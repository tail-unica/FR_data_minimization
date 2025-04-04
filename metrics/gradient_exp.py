import logging
import os

#import cv2
import sys
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torch.nn import CrossEntropyLoss

sys.path.append("../")
from utils import losses
from utils.dataset import DataLoaderX, FaceDatasetFolder
from utils.utils_callbacks import CallBackVerification
from utils.utils_logging import init_logging
from eval.verification import calculate_roc
from config import config as cfg

from backbones.iresnet import iresnet100, iresnet50, iresnet18

def compute_last_layer_grads(loss_v, backbone, local_rank):
    last_layer_grads = torch.zeros(size=(loss_v.shape[0],)).cuda(local_rank)
    for i in range(loss_v.shape[0]):
        g = abs(torch.autograd.grad(loss_v[i], list(backbone.parameters())[-1], retain_graph=True)[0]).mean()
        last_layer_grads[i] = g
        backbone.zero_grad()
    return last_layer_grads

criterion = CrossEntropyLoss(reduction="none")

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

    weights_bb = "output_BUPT500/R18_CosFace_500_webface/1120backbone.pth"
    backbone.load_state_dict(torch.load(weights_bb))

    weights_header = "output_BUPT500/R18_CosFace_500_webface/1120header.pth"
    header.load_state_dict(torch.load(weights_header))

    name_dataset = weights_bb.split('_')[1].split('/')[0]

    #output_path ="criterion_npy/Casia500"
    
    header = torch.nn.DataParallel(header, device_ids=[gpu_id])
    backbone = torch.nn.DataParallel(backbone, device_ids=[gpu_id])

    softmax = nn.Softmax(dim=1)
    
    M, N = cfg.num_image, 2
    s=(M,N)

    arr=np.zeros(s)

    backbone.eval()
    header.eval()


    for _, (idx, img, label) in enumerate(train_loader):

        embeddings=[] 
        labels=[]

        img = img.cuda(local_rank, non_blocking=True)
        img.requires_grad = True
        label = label.cuda(local_rank, non_blocking=True)
        print(idx)
        print(label)
        features = F.normalize(backbone(img))

        

        '''popolo i vettori creati prima'''
        embeddings.append(features.cpu().detach().numpy())
        labels.append(label.cpu().numpy())


        #distribuzione delle probabilità di tutte le classi
        thetas = header(features, label)

        loss_v = criterion(thetas, label)

        loss_v.mean().backward()

        grads = img.grad.mean(dim=(1,2,3)).abs()
        arr[idx.cpu().numpy(),0]=label.cpu().numpy()
        arr[idx.cpu().numpy(),1]=grads.cpu().numpy().flatten()

        backbone.zero_grad()
        header.zero_grad()
    
    '''salvo i vettori .npy'''
    embeddings = np.concatenate(embeddings, axis=0)
    labels = np.concatenate(labels, axis=0)
    np.save('embeddings.npy', embeddings)
    np.save('labels.npy', labels)

    '''inizializzo questi vettori da mandare a compute_roc'''
    n_samples = embeddings.shape[0]
    embeddings1, embeddings2 = [], []
    actual_issame = []

    '''popolo i vettori'''
    for i in range(n_samples):
        for j in range(i + 1, n_samples):
            embeddings1.append(embeddings[i])
            embeddings2.append(embeddings[j])
            actual_issame.append(labels[i] == labels[j])

    '''li converto in numpy'''
    embeddings1 = np.array(embeddings1)
    embeddings2 = np.array(embeddings2)
    actual_issame = np.array(actual_issame)

    '''da inserire nel calculate_roc'''
    thresholds = np.arange(0, 4, 0.01)

    #calculate_roc(thresholds, embeddings1, embeddings2, actual_issame, nrof_folds=10, pca=0, output_dir=output_dir_name)


    np.save(f'criterion_npy/{name_dataset}/gradients_array_abs.npy', arr)

    classes = np.unique(arr[:, 0])

    arr_avg = np.zeros(len(classes))
    for cls in classes:
        subset = arr[arr[:, 0].astype(np.int32) == int(cls)]
        arr_avg[int(cls-1)] = np.mean(subset[:, 1])
    np.save(f"criterion_npy/{name_dataset}/mean_grad_magnitude_abs_per_cls.npy", arr_avg)
        
        


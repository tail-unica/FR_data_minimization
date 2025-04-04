import logging
import os

#import cv2
import sys
import torch
sys.path.append("../")
from utils.utils_callbacks import CallBackVerification
from utils.utils_logging import init_logging

from config.config_eval import config as cfg

from backbones.iresnet import iresnet100, iresnet50

if __name__ == "__main__":
    gpu_id = 0
    log_root = logging.getLogger()

    if cfg.network == "iresnet100":
        print("iresnet100")
        backbone = iresnet100(num_features=512).to(f"cuda:{gpu_id}")
    elif cfg.network == "iresnet50":
        print("iresnet50")
        backbone = iresnet50(num_features=512).to(f"cuda:{gpu_id}")
    else:
        backbone = None
        exit()

    weights = "/home/aatzori/codebase/cross-domain/output/62496backbone.pth"
    backbone.load_state_dict(torch.load(weights))
    init_logging(log_root, 0, "/home/aatzori/codebase/cross-domain/output/",logfile="testmix.log")
    callback_verification = CallBackVerification(1, 0, cfg.val_targets, cfg.val_root)




    model = torch.nn.DataParallel(backbone, device_ids=[gpu_id])
    callback_verification(1, model)


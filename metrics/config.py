from easydict import EasyDict as edict

config = edict()
config.dataset = "GC"  # training dataset
config.embedding_size = 512  # embedding size of model
config.batch_size = 128  # batch size per GPU
config.output = "output/"  # train model output folder
config.s = 64.0
config.m = 0.35
config.std = 0.05

config.loss = "CosFace"  # Option : ElasticArcFace, ArcFace, ElasticCosFace, CosFace, MLLoss, ElasticArcFacePlus, ElasticCosFacePlus, AdaFace

if config.loss == "ElasticArcFacePlus":
    config.s = 64.0
    config.m = 0.50
    config.std = 0.0175
elif config.loss == "ElasticArcFace":
    config.s = 64.0
    config.m = 0.50
    config.std = 0.05
if config.loss == "ElasticCosFacePlus":
    config.s = 64.0
    config.m = 0.35
    config.std = 0.02
elif config.loss == "ElasticCosFace":
    config.s = 64.0
    config.m = 0.35
    config.std = 0.05
elif config.loss == "AdaFace":
    config.s = 64.0
    config.m = 0.4
elif config.loss == "ArcFace":
    config.s = 64.0
    config.m = 0.5


config.data_dict = {
    "GC": {"data": "/data/Synthetic/GAN_Control_class_images/images",
           "pretrained": {
               "backbone": "../output/reference/iresnet34_CosFace_GC_10K_/187480backbone.pth",
               "header": "../output/reference/iresnet34_CosFace_GC_10K_/187480header.pth"
           }
           },
    "DC": {"data": "/data/Synthetic/dcface_0.5m_oversample_xid/images",
           "pretrained": {
               "backbone": "../output/reference/iresnet34_CosFace_DC_10K_/171840backbone.pth",
               "header": "../output/reference/iresnet34_CosFace_DC_10K_/171840header.pth",
           }
           },
    "IDF": {"data": "/data/Synthetic/Idifface",
            "pretrained": {
               "backbone": "../output/reference/iresnet34_CosFace_IDF_10K_/156240backbone.pth",
               "header": "../output/reference/iresnet34_CosFace_IDF_10K_/156240header.pth",
            }
            },
    "WF": {"data": "/data/Authentic/casia_training",
           "pretrained": {
               "backbone": "../output/reference/iresnet34_CosFace_WF_10K_/148320backbone.pth",
               "header": "../output/reference/iresnet34_CosFace_WF_10K_/148320header.pth",
           }
           },
    "M2-S": {"data": "/data/Authentic/faces_emore_10k/images",
             "pretrained":{
               "backbone": "../output/reference/iresnet34_CosFace_M2-S_10K_/236920backbone.pth",
               "header": "../output/reference/iresnet34_CosFace_M2-S_10K_/236920header.pth",
             }
             },
}

config.rec = config.data_dict[config.dataset]["data"]
config.network = "iresnet34"
config.SE = False  # SEModule


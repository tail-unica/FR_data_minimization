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
               "backbone": "",
               "header": ""
           }
           },
    "DC": {"data": "/data/Synthetic/dcface_0.5m_oversample_xid/images",
           "pretrained": {
               "backbone": "",
               "header": ""
           }
           },
    "IDF": {"data": "/data/Synthetic/Idifface",
            "pretrained": {
               "backbone": "",
               "header": ""
            }
            },
    "WF": {"data": "/data/Authentic/casia_training",
           "pretrained": {
               "backbone": "",
               "header": ""
           }
           },
    "M2-S": {"data": "/data/Authentic/faces_emore_10k/images",
             "pretrained":{
               "backbone": "",
               "header": ""
             }
             },
}

config.rec = config.data_dict[config.dataset]["data"]
config.network = "iresnet34"
config.SE = False  # SEModule


from easydict import EasyDict as edict

config = edict()
config.dataset = "webface"  # training dataset
config.synt_dataset = "GC"  # training dataset
config.embedding_size = 512  # embedding size of model
config.momentum = 0.9
config.weight_decay = 5e-4
config.batch_size = 128  # batch size per GPU
config.lr = 0.1
config.output = "output/"  # train model output folder
config.global_step = 0  # step to resume
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

# type of network to train [iresnet100 | iresnet50]
# "/home/unica/datasets/FaceRecognition/Synthetic/dcface_0.5m_oversample_xid/images"
# "/home/unica/datasets/FaceRecognition/Synthetic/GAN_Control_class_images/images"
# "/home/unica/datasets/FaceRecognition/Synthetic/Idifface"

config.synt_dict = {
    "GC": "/data/Synthetic/GAN_Control_class_images/images",
    "DC": "/data/Synthetic/dcface_0.5m_oversample_xid/images",
    "IDF": "/data/Synthetic/Idifface"
}
config.synthetic_root = config.synt_dict[config.synt_dataset]


config.val_root = "/data/FR_Benchmark"
config.network = "iresnet34"
config.SE = False  # SEModule

if config.dataset == "emoreIresNet":
    config.rec = "/home/unica/datasets/FaceRecognition/MS1MV2/faces_emore/imgs/tmp/tmp.PTU8Enp8LP/"#"/home/unica/datasets/FaceRecognition/MS1MV2/faces_emore/"
    config.num_classes = 85742
    config.num_image = 5822653
    config.num_epoch = 40 #26
    config.warmup_epoch = -1
    config.val_targets = ["lfw", "cfp_fp", "agedb_30", "calfw", "cplfw", "African_test", "Asian_test", "Caucasian_test", "Indian_test"]
    config.eval_step = 5686


    #def lr_step_func(epoch):
    #    return ((epoch + 1) / (4 + 1)) ** 2 if epoch < -1 else 0.1 ** len(
    #        [m for m in [8, 14, 20, 25] if m - 1 <= epoch])  # [m for m in [8, 14,20,25] if m - 1 <= epoch])
    def lr_step_func(epoch):
        return ((epoch + 1) / (4 + 1)) ** 2 if epoch < config.warmup_epoch else 0.1 ** len(
            [m for m in [22, 30, 36] if m - 1 <= epoch])

    config.lr_func = lr_step_func

elif config.dataset == "webface":
    config.rec = "/data/Authentic/casia_training"
    config.num_classes = 10572
    config.num_image = 501195
    config.num_epoch = 40  # [22, 30, 35]
    config.warmup_epoch = -1
    config.val_targets = ["lfw", "cfp_fp", "agedb_30", "calfw", "cplfw"]
    config.eval_step = 958  # 33350


    def lr_step_func(epoch):
        return ((epoch + 1) / (4 + 1)) ** 2 if epoch < config.warmup_epoch else 0.1 ** len(
            [m for m in [22, 30, 36] if m - 1 <= epoch])


    config.lr_func = lr_step_func

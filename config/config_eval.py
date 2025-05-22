from easydict import EasyDict as edict

config = edict()
config.val_root = "/home/aatzori/codebase/cross-domain/eval"
config.val_targets = ["agedb_wacv", "bupt_wacv", "cfp_wacv", "rof_wacv"]
#config.val_root = "/home/unica/datasets/FaceRecognition/RFW_test"
#config.val_targets = ["lfw", "African_test", "Asian_test", "Caucasian_test", "Indian_test"]#,
config.network = "iresnet50"


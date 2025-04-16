import datetime
import math
import os
from detectron2.utils.logger import setup_logger
setup_logger()

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, BestCheckpointer
from detectron2.evaluation import COCOEvaluator
from detectron2.data.datasets import register_coco_instances
from detectron2.data import build_detection_train_loader
from detectron2.data import DatasetMapper
from detectron2.data import transforms as T
import random
import cv2
import numpy as np

register_coco_instances("my_dataset_train", {}, "nycu-hw2-data/train.json", "nycu-hw2-data/train")
register_coco_instances("my_dataset_val", {}, "nycu-hw2-data/valid.json", "nycu-hw2-data/valid")
class RandomRotate(T.Augmentation):
    def __init__(self, angle_range=(-10, 10)):
        super().__init__()
        self.angle_range = angle_range

    def get_transform(self, image):
        angle = random.uniform(*self.angle_range)
        return T.RotationTransform(
            h=image.shape[0],
            w=image.shape[1],
            angle=angle
        )

def build_mapper(cfg, is_train=True):
    augs = [
        T.ResizeShortestEdge(short_edge_length=(800, 1000), max_size=1333, sample_style='choice'),
        RandomRotate(angle_range=(-15, 15)),    # rotate ±15 degrees
        T.RandomBrightness(0.9, 1.1),
        T.RandomContrast(0.9, 1.1),
    ]

    if is_train and cfg.INPUT.CROP.ENABLED:
        augs.insert(0, T.RandomCrop("relative_range", cfg.INPUT.CROP.SIZE))

    return DatasetMapper(cfg, is_train=is_train, augmentations=augs)

class MyTrainer(DefaultTrainer):
    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_train_loader(cfg, mapper=build_mapper(cfg, is_train=True))
    
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        return COCOEvaluator(dataset_name, cfg, True, output_folder)
    
    def build_hooks(self):
        hooks = super().build_hooks()
        hooks.append(BestCheckpointer(cfg.TEST.EVAL_PERIOD, self.checkpointer, 'bbox/AP'))
        return hooks

cfg = get_cfg()
model_name = "faster_rcnn_X_101_32x8d_FPN_3x.yaml"
cfg.merge_from_file(model_zoo.get_config_file(f"COCO-Detection/{model_name}"))

cfg.DATASETS.TRAIN = ("my_dataset_train",)
cfg.DATASETS.TEST = ("my_dataset_val",)
cfg.DATALOADER.NUM_WORKERS = 4
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(f"COCO-Detection/{model_name}")

now = datetime.now()
timestamp = now.strftime("%Y%m%d_%H%M%S") 
cfg.OUTPUT_DIR = "X_anchor8-256/{timestamp}"
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

cfg.SOLVER.MAX_ITER = 50000
cfg.TEST.EVAL_PERIOD = 1000
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.001
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 10
cfg.MODEL.RPN.BBOX_REG_LOSS_TYPE = "smooth_l1"
cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_TYPE = "giou"
cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[16, 32, 64, 128, 256]]

cfg.INPUT.RANDOM_FLIP = "none"

cfg.INPUT.CROP.ENABLED = True
cfg.INPUT.CROP.TYPE = "relative_range"
cfg.INPUT.CROP.SIZE = [0.8, 0.8]

cfg.INPUT.MIN_SIZE_TRAIN = (640, 800)
cfg.INPUT.MAX_SIZE_TRAIN = 1333
cfg.INPUT.MIN_SIZE_TEST = 800
cfg.INPUT.MAX_SIZE_TEST = 1333

cfg.MODEL.DEVICE = "cuda:0"

config_file_path = os.path.join(cfg.OUTPUT_DIR, "config.yaml")
with open(config_file_path, "w") as f:
    f.write(cfg.dump())

trainer = MyTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

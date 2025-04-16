from datetime import datetime
import os

from detectron2.utils.logger import setup_logger
setup_logger()

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, BestCheckpointer
from detectron2.evaluation import COCOEvaluator
from detectron2.data.datasets import register_coco_instances

register_coco_instances("my_dataset_train", {}, "nycu-hw2-data/train.json", "nycu-hw2-data/train")
register_coco_instances("my_dataset_val", {}, "nycu-hw2-data/valid.json", "nycu-hw2-data/valid")

class MyTrainer(DefaultTrainer):  
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        return COCOEvaluator(dataset_name, cfg, True, output_folder)
    
    def build_hooks(self):
        hooks = super().build_hooks()
        hooks.append(BestCheckpointer(cfg.TEST.EVAL_PERIOD, self.checkpointer, 'bbox/AP', mode="max"))
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
cfg.OUTPUT_DIR = f"output/{timestamp}"
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

cfg.SOLVER.MAX_ITER = 50000
cfg.TEST.EVAL_PERIOD = 1000
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.CHECKPOINT_PERIOD = 50000
cfg.SOLVER.BASE_LR = 0.001

cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[8, 16, 32, 64, 128]]
cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.5, 0.75, 1.0, 1.5, 2.0]]

cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN = 2000
cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 1000
cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN = 1000
cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 1000

cfg.MODEL.ROI_HEADS.NUM_CLASSES = 10
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
cfg.MODEL.ROI_HEADS.POSITIVE_FRACTION = 0.5

cfg.MODEL.RPN.BBOX_REG_LOSS_TYPE = "giou"
cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_LOSS_TYPE = "giou"

cfg.INPUT.MIN_SIZE_TRAIN = (800, 1000, 1200)
cfg.INPUT.MAX_SIZE_TRAIN = 1600
cfg.INPUT.MIN_SIZE_TEST = 1000
cfg.INPUT.MAX_SIZE_TEST = 1600

cfg.INPUT.CROP.ENABLED = True
cfg.INPUT.CROP.TYPE = "relative_range"
cfg.INPUT.CROP.SIZE = [0.9, 0.9]

cfg.INPUT.RANDOM_FLIP = 'none'

cfg.TEST.DETECTIONS_PER_IMAGE = 300

# cfg.VIS_PERIOD = 1000
cfg.MODEL.DEVICE = "cuda:0"

config_file_path = os.path.join(cfg.OUTPUT_DIR, "config.yaml")
with open(config_file_path, "w") as f:
    f.write(cfg.dump())

trainer = MyTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

import os
import json
import pandas as pd
from tqdm import tqdm
import cv2

from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo

def setup_cfg():
    cfg = get_cfg()
    cfg.merge_from_file("/mnt/HDD1/lamle/tuong/LamLe/Selected/HW2/detectron2/X_anchor8-256/config.yaml")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.4
    cfg.MODEL.WEIGHTS = "/mnt/HDD1/lamle/tuong/LamLe/Selected/HW2/detectron2/X_anchor8-256/model_best.pth"  # path to your trained Detectron2 model
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 10  # 10 digits (0-9)
    cfg.MODEL.DEVICE = "cuda:6" if torch.cuda.is_available() else "cpu"
    return cfg

def load_image_paths(folder):
    return [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".png")]

def xyxy_to_xywh(box):
    x_min, y_min, x_max, y_max = box
    width = x_max - x_min
    height = y_max - y_min
    return [x_min, y_min, width, height]

def main():
    cfg = setup_cfg()
    predictor = DefaultPredictor(cfg)

    image_folder = "/mnt/HDD1/lamle/tuong/LamLe/Selected/HW2/nycu-hw2-data/test"
    image_paths = load_image_paths(image_folder)

    pred_json = []
    pred_csv = []

    for image_path in tqdm(image_paths):
        image_id = int(os.path.splitext(os.path.basename(image_path))[0])
        image = cv2.imread(image_path)
        outputs = predictor(image)

        boxes = outputs["instances"].pred_boxes.tensor.cpu().numpy()
        scores = outputs["instances"].scores.cpu().numpy()
        labels = outputs["instances"].pred_classes.cpu().numpy()

        digits = []
        for box, label, score in zip(boxes, labels, scores):
            if score > 0.6:
                x_min, y_min, x_max, y_max = box
                box_xywh = xyxy_to_xywh([x_min, y_min, x_max, y_max])
                pred_json.append({
                    "image_id": image_id,
                    "bbox": [round(float(x), 2) for x in box_xywh],
                    "score": round(float(score), 3),
                    "category_id": int(label + 1)  # Detectron2 categories are 0-indexed
                })
                digits.append((x_min, y_min, int(label)))

        if digits:
            digits = sorted(digits, key=lambda x: (x[0]))  # x_min then y_min
            digit_str = ''.join(str(d[2]) for d in digits)
            pred_csv.append({"image_id": image_id, "pred_label": int(digit_str)})
        else:
            pred_csv.append({"image_id": image_id, "pred_label": -1})

    with open("pred.json", "w") as f:
        json.dump(pred_json, f)

    pd.DataFrame(pred_csv).to_csv("pred.csv", index=False)

if __name__ == "__main__":
    import torch
    main()

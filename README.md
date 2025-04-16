# SelectedTopic25
#ID: 312540020

## Introduction
In this task, I focus on recognizing digits from natural images using the Faster R-CNN model. The dataset includes 30,062 training images, 3,340 validation images, and 13,068 test images. Each image may contain one or more digits, which are often very small and appear in complex backgrounds.
My code is base on git detectron2: https://github.com/facebookresearch/detectron2/tree/main

### 🔹 Key Enhancements:  
- **Anchor size**  
- **giou loss**  
- **Data Augmentation** (ineffective)
- **Focal Loss**  (ineffective)

## Training 
First install detectron2
```
git clone https://github.com/facebookresearch/detectron2.git
```
```
pip install -U torch==1.5 torchvision==0.6 -f https://download.pytorch.org/whl/cu101/torch_stable.html
pip install cython pyyaml==5.1
pip install -U 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
pip install detectron2==0.1.3 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu101/torch1.5/index.html
```

The train.py is the best setting I submit on leaderboard.
```
python train.py
```
The train_aug.py is for augmentation.
```
python train_aug.py
```
The train_focal.py is for apply focal loss.
```
python train_focal.py
```
## Val
To get the pred.csv and pred.json.
```
python pred.py
```
## Performance Snapshot
![Performance Snapshot](./Snapshot.png)  

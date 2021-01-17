# ASL-Detection
美式手语实时检测系统 2021 SDU

## Environment
Python==3.8

Pytorch==1.6.0

## Datasets
PASCAL VOC2012  

自制American Sign Language数据集（训练集3200张+验证集630张）  

## Function
Daily Life：基于VOC2012数据集，可检测日常生活的20类（VOC的20类），应用的算法有Faster RCNN、Mask RCNN、Cascade RCNN、SSD、YOLOv5

**ASL Detection：基于ASL数据集的美式手语实时检测，目前可检测基本的26个字母，应用的算法有Cascade RCNN、SSD、YOLOv5**

## Algorithm
Faster RCNN：混合精度训练

Mask RCNN：基于Detectron2，带实例分割

Cascade RCNN：基于Detectron2，不带实例分割

SSD：混合精度训练

YOLOv5：带稀疏训练及剪枝

## Performance
ASL Detection（YOLOv5）：FPS 55帧/秒左右，mAP@0.5:0.95 65%

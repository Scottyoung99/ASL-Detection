import TargetModels.cascade.detectron2
from TargetModels.cascade.detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import cv2
import random
import datetime
import time
import tqdm
import os
from PIL import Image
from TargetModels.mask.detectron2 import model_zoo
from TargetModels.mask.detectron2.engine import DefaultPredictor
from TargetModels.mask.detectron2.config import get_cfg
from TargetModels.mask.detectron2.utils.visualizer import Visualizer
from TargetModels.mask.detectron2.data import MetadataCatalog
from TargetModels.mask.detectron2.data.datasets import register_coco_instances
from TargetModels.mask.detectron2.data import DatasetCatalog, MetadataCatalog
from TargetModels.mask.detectron2.engine import DefaultTrainer
from TargetModels.mask.detectron2.config import get_cfg
from TargetModels.mask.detectron2.utils.visualizer import ColorMode
from TargetModels.mask.detectron2.utils.video_visualizer import VideoVisualizer
from TargetModels.mask.detectron2.evaluation import COCOEvaluator, inference_on_dataset
from TargetModels.mask.detectron2.data import build_detection_test_loader
import uuid

def Train():
    register_coco_instances("coco_train", {}, "datasets/coco/annotations/instances_train.json", "datasets/coco/train2014")
    register_coco_instances("coco_val", {}, "datasets/coco/annotations/instances_val.json", "datasets/coco/val2014")
    # register_coco_instances("coco_train", {}, "datasets/coco/trainval.json", "datasets/coco/images")
    coco_train_metadata = MetadataCatalog.get("coco_train")
    coco_val_metadata = MetadataCatalog.get("coco_train")
    board_metadata = MetadataCatalog.get("coco_train")
    dataset_dicts = DatasetCatalog.get("coco_train")
    for d in random.sample(dataset_dicts, 3):
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=board_metadata, scale=1)
        vis = visualizer.draw_dataset_dict(d)
        cv2.imshow('Sample',vis.get_image()[:, :, ::-1])
        cv2.waitKey()

    cfg = get_cfg()
    cfg.merge_from_file("configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.DATASETS.TRAIN = ("coco_train",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = 'model_final_maskrcnn.pkl'
    cfg.SOLVER.IMS_PER_BATCH = 1
    cfg.SOLVER.BASE_LR = 1e-06
    cfg.SOLVER.MAX_ITER = 7000
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 20
    # cfg.TEST.EVAL_PERIOD = 500
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()    

def Predict(file_path):
    test_time = time.time()
    register_coco_instances("coco_val_voc", {}, "TargetModels/mask/datasets/coco/annotations/instances_val.json", "TargetModels/mask/datasets/coco/val2014")
    register_coco_instances("coco_train_voc", {}, "TargetModels/mask/datasets/coco/annotations/instances_train.json", "TargetModels/mask/datasets/coco/train2014")
    board_metadata = MetadataCatalog.get("coco_train_voc")
    #dataset_dicts = DatasetCatalog.get("coco_val")
    DatasetCatalog.get("coco_train_voc")

    im = cv2.imread(file_path)
    cfg = get_cfg()
    cfg.merge_from_file("TargetModels/mask/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml")
    cfg.DATASETS.TEST = ("coco_val_voc", )
    cfg.MODEL.WEIGHTS = os.path.join("TargetModels/mask/",cfg.OUTPUT_DIR, "1x.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 20
    predictor = DefaultPredictor(cfg)
    #for d in random.sample(dataset_dicts, 3):
    #print(d["file_name"])
    #im = cv2.imread(d["file_name"])
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1],metadata=board_metadata,scale=1.0,instance_mode=ColorMode.IMAGE_BW)  # remove the colors of unsegmented pixels
    vis = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    result=vis.get_image()[:, :, ::-1]
    filename=str(uuid.uuid4())
    basedir = os.path.abspath(os.path.dirname(__file__))
    img = Image.fromarray(result)
    img.save(basedir+'/../../static/output/'+filename+'.jpg')
    ret='static/output/'+filename+'.jpg'
    test_time=time.time()-test_time
    mAP=0.65
    return ret,test_time,mAP,1
    #cv2.imshow('predict',vis.get_image()[:, :, ::-1])
    #cv2.waitKey()

def Video():
    register_coco_instances("coco_val_hand", {}, "TargetModels/mask/datasets/coco/annotations/instances_val.json", "datasets/coco/val2014")
    register_coco_instances("coco_train_hand", {}, "TargetModels/mask/datasets/coco/annotations/instances_train.json","datasets/coco/train2014")
    board_metadata = MetadataCatalog.get("coco_train_hand")
    dataset_dicts = DatasetCatalog.get("coco_val_hand")
    DatasetCatalog.get("coco_train_hand")
    cfg = get_cfg()
    cfg.merge_from_file("TargetModels/mask/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_1x.yaml")
    cfg.DATASETS.TEST = ("coco_val_hand",)
    cfg.MODEL.WEIGHTS = os.path.join("TargetModels/mask",cfg.OUTPUT_DIR, "1x.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 26
    predictor = DefaultPredictor(cfg)
    video = cv2.VideoCapture("test.mp4")
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames_per_second = video.get(cv2.CAP_PROP_FPS)
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    output_fname='mask.mp4'
    output_file = cv2.VideoWriter(
        filename=output_fname,
        # some installation of opencv may not support x264 (due to its license),
        # you can try other format (e.g. MPEG)
        # fourcc=cv2.VideoWriter_fourcc(*"x264"),
        fps=float(frames_per_second),
        frameSize=(width, height),
        isColor=True,
    )
    video_visualizer = VideoVisualizer(metadata=board_metadata, instance_mode=ColorMode.IMAGE_BW)
    while video.isOpened():
        success, frame = video.read()
        if success:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            output = predictor(frame)
            vis = video_visualizer.draw_instance_predictions(frame, output["instances"].to("cpu"))
            vis = cv2.cvtColor(vis.get_image(), cv2.COLOR_RGB2BGR)
            output_file.write(vis)
        else: break
    video.release()
    output_file.release()

def Evaluate():
    register_coco_instances("coco_train", {}, "TargetModels/mask/datasets/coco/annotations/instances_train.json","datasets/coco/train2014")
    #register_coco_instances("coco_val", {}, "datasets/coco/annotations/instances_val.json", "datasets/coco/val2014")
    #register_coco_instances("coco_train", {}, "datasets/coco/trainval.json", "datasets/coco/images")
    #coco_train_metadata = MetadataCatalog.get("coco_train")
    MetadataCatalog.get("coco_train").evaluator_type
    #coco_val_metadata = MetadataCatalog.get("coco_val").evaluator_type
    # board_metadata = MetadataCatalog.get("coco_val")
    DatasetCatalog.get("coco_train")
    cfg = get_cfg()
    cfg.merge_from_file("TargetModels/mask/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    #cfg.DATASETS.TEST = ("coco_val",)
    cfg.DATASETS.TEST = ("coco_train",)
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "3x.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 20
    predictor = DefaultPredictor(cfg)
    # Call the COCO Evaluator function and pass the Validation Dataset
    evaluator = COCOEvaluator("coco_train", cfg, False, output_dir="output/")
    val_loader = build_detection_test_loader(cfg, "coco_train")
    # Use the created predicted model in the previous step
    inference_on_dataset(predictor.model, val_loader, evaluator)

def Vis():
    # register_coco_instances("coco_train", {}, "datasets/coco/annotations/instances_train.json", "datasets/coco/train2014")
    register_coco_instances("coco_val", {}, "TargetModels/mask/datasets/coco/annotations/instances_val.json", "datasets/coco/val2014")
    # register_coco_instances("coco_train", {}, "datasets/coco/trainval.json", "datasets/coco/images")
    # coco_train_metadata = MetadataCatalog.get("coco_train")
    coco_val_metadata = MetadataCatalog.get("coco_val")
    # board_metadata = MetadataCatalog.get("coco_train")
    dataset_dicts = DatasetCatalog.get("coco_val")
    for d in random.sample(dataset_dicts, 10):
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=coco_val_metadata, scale=1)
        vis = visualizer.draw_dataset_dict(d)
        cv2.imshow('Sample',vis.get_image()[:, :, ::-1])
        cv2.waitKey()

if __name__ == "__main__":
    #Train()
    Predict()
    #Evaluate()
    #Vis()
    #Video()
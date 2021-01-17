import tqdm
from TargetModels.cascade.demo.predictor import VisualizationDemo
import TargetModels.cascade.detectron2
from TargetModels.cascade.detectron2.utils.logger import setup_logger
from TargetModels.cascade.detectron2.utils.video_visualizer import VideoVisualizer
import uuid
setup_logger()

# import some common libraries
import numpy as np
import cv2
import random
import datetime
import time
import os
from PIL import Image
from TargetModels.cascade.detectron2 import model_zoo
from TargetModels.cascade.detectron2.engine import DefaultPredictor
from TargetModels.cascade.detectron2.config import get_cfg
from TargetModels.cascade.detectron2.utils.visualizer import Visualizer
from TargetModels.cascade.detectron2.data import MetadataCatalog
from TargetModels.cascade.detectron2.data.datasets import register_coco_instances
from TargetModels.cascade.detectron2.data import DatasetCatalog, MetadataCatalog
from TargetModels.cascade.detectron2.engine import DefaultTrainer
from TargetModels.cascade.detectron2.config import get_cfg
from TargetModels.cascade.detectron2.utils.visualizer import ColorMode
from TargetModels.cascade.detectron2.evaluation import COCOEvaluator, inference_on_dataset
from TargetModels.cascade.detectron2.data import build_detection_test_loader
import uuid




register_coco_instances("coco_train_voc", {}, "TargetModels/cascade/datasets/coco/annotations/instances_train.json", "TargetModels/cascade/datasets/coco/train2014")
coco_val_metadata_voc = MetadataCatalog.get("coco_train_voc")
DatasetCatalog.get("coco_train_voc")
register_coco_instances("coco_train", {}, "TargetModels/cascade/datasets/coco/annotations/train.json", "TargetModels/cascade/datasets/coco/train_imgs")
coco_val_metadata = MetadataCatalog.get("coco_train")
DatasetCatalog.get("coco_train")

def Train():

    register_coco_instances("coco_train", {}, "datasets/coco/annotations/train.json", "datasets/coco/train_imgs")
    register_coco_instances("coco_val", {}, "datasets/coco/annotations/val.json", "datasets/coco/val_imgs")
    custom_metadata = MetadataCatalog.get("coco_train")
    dataset_dicts = DatasetCatalog.get("coco_train")
    for d in random.sample(dataset_dicts, 3):
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=custom_metadata, scale=1)
        vis = visualizer.draw_dataset_dict(d)
        cv2.imshow('Sample',vis.get_image()[:, :, ::-1])
        cv2.waitKey()


    cfg = get_cfg()
    cfg.merge_from_file(
        "configs/Misc/cascade_mask_rcnn_R_50_FPN_3x.yaml"
    )
    cfg.DATASETS.TRAIN = ("coco_train",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.MODEL.WEIGHTS = 'model_final_cascadercnn3x_hand.pkl'
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.0001
    cfg.SOLVER.MAX_ITER = 15000
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 26
    cfg.MODEL.MASK_ON = False
    cfg.MODEL.KEYPOINT_ON = False

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()    


def Predict(file_path):

    basedir = os.path.abspath(os.path.dirname(__file__))

    im = cv2.imread(file_path)
    print(im.__class__)
    cfg = get_cfg()
    cfg.merge_from_file("TargetModels/cascade/configs/Misc/cascade_mask_rcnn_R_50_FPN_3x.yaml")
    #cfg.DATASETS.TEST = ("coco_val", )
    cfg.MODEL.WEIGHTS = os.path.join("TargetModels/cascade",cfg.OUTPUT_DIR, "model_final_cascade3x_12000_0.00025.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 20
    cfg.MODEL.MASK_ON = False
    cfg.MODEL.KEYPOINT_ON = False
    predictor = DefaultPredictor(cfg)
    # for d in random.sample(dataset_dicts, 3):
    # im = cv2.imread(d["file_name"])
    test_time = time.time()
    outputs = predictor(im)

    v = Visualizer(im[:, :, ::-1],metadata=coco_val_metadata_voc,scale=1.0,instance_mode=ColorMode.IMAGE)  # remove the colors of unsegmented pixels
    vis = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    img = vis.get_image()[:, :, ::-1]
    
    #cv2.imshow('predict',vis.get_image()[:, :, ::-1])
    #cv2.waitKey()
    filename = str(uuid.uuid4())
    img = Image.fromarray(img)
    img.save(basedir+'/../../static/output/'+filename+'.jpg')
    ret = 'static/output/'+filename+'.jpg'
    test_time = time.time()-test_time
    mAP = 0.541
    return ret, test_time, mAP, 1


def Video(flag=0):


    cfg = get_cfg()
    cfg.merge_from_file("TargetModels/cascade/configs/Misc/cascade_mask_rcnn_R_50_FPN_3x.yaml")
    cfg.DATASETS.TEST = ("coco_val",)
    cfg.MODEL.WEIGHTS = os.path.join("TargetModels/cascade",cfg.OUTPUT_DIR, "model_final_cascade3x_hand.pth")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 26
    cfg.MODEL.MASK_ON = False
    cfg.MODEL.KEYPOINT_ON = False
    predictor = DefaultPredictor(cfg)

    cam = cv2.VideoCapture(0)

    while(cam.isOpened()):
        start = time.time()
        ret, frame = cam.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        outputs = predictor(frame)

        v = VideoVisualizer(metadata=coco_val_metadata, instance_mode=ColorMode.IMAGE)
        vis = v.draw_instance_predictions(frame, outputs["instances"].to("cpu"))
        vis_frame = cv2.cvtColor(vis.get_image(), cv2.COLOR_RGB2BGR)
        seconds = time.time()-start
        fps = 1/seconds
        fps = round(fps,2)
        cv2.putText(vis_frame, "FPS:{0}".format(fps), (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),
                    3)
        
        if flag==1: raise StopIteration
        #cv2.imshow("hand video", vis_frame)
        ret, im1 = cv2.imencode('.jpg', vis_frame)
        frame = im1.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n'+frame+b'\r\n\r\n')

        if cv2.waitKey(1) == 27:
            break  # esc to quit
    cam.release()
    cv2.destroyAllWindows()

    # cv2.waitKey()


def Evaluate():
    register_coco_instances("coco_train", {}, "datasets/coco/annotations/train.json", "datasets/coco/train_imgs")
    register_coco_instances("coco_val", {}, "datasets/coco/annotations/val.json", "datasets/coco/val_imgs")
    #register_coco_instances("coco_train", {}, "datasets/coco/trainval.json", "datasets/coco/images")
    #coco_train_metadata = MetadataCatalog.get("coco_train")
    coco_val_metadata = MetadataCatalog.get("coco_val").evaluator_type
    board_metadata = MetadataCatalog.get("coco_train")
    DatasetCatalog.get("coco_train")
    cfg = get_cfg()
    cfg.merge_from_file("configs/Misc/cascade_mask_rcnn_R_50_FPN_3x.yaml")
    cfg.DATASETS.TEST = ("coco_val",)
    cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final_cascade3x_15000_1e-4_hand.pth")
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 26
    cfg.MODEL.MASK_ON = False
    cfg.MODEL.KEYPOINT_ON = False
    predictor = DefaultPredictor(cfg)
    # Call the COCO Evaluator function and pass the Validation Dataset
    evaluator = COCOEvaluator("coco_val", cfg, False, output_dir="output/")
    val_loader = build_detection_test_loader(cfg, "coco_val")
    # Use the created predicted model in the previous step
    inference_on_dataset(predictor.model, val_loader, evaluator)



if __name__ == "__main__":
    #Train()
    Evaluate()
    #Predict()
    #Video()
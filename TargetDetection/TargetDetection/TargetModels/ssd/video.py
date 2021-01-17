#-------------------------------------#
#       调用摄像头检测
#-------------------------------------#
import torch
import torchvision
from ssd.draw_box_utils import draw_box
from PIL import Image
import json
import matplotlib.pyplot as plt
from ssd.src.ssd_model import SSD300, Backbone
import transform
import time
import os
import cv2
import uuid
import requests
import re
import numpy as np
from translate import Translator
from torchvision import transforms

def create_model(num_classes):
    backbone = Backbone()
    model = SSD300(backbone=backbone, num_classes=num_classes)

    return model

def ssd_video(flag=0):
    # get devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    basedir = os.path.abspath(os.path.dirname(__file__))
    print(basedir)
    # create model
    model = create_model(num_classes=21)

    # load train weights
    train_weights = basedir + "/save_weights/ssd300-hand.pth"
    train_weights_dict = torch.load(train_weights, map_location=device)['model']

    model.load_state_dict(train_weights_dict, strict=False)
    model.to(device)

    # read class_indict
    category_index = {}
    try:
        json_file = open(basedir+'./pascal_voc_classes.json', 'r')
        class_dict = json.load(json_file)
        category_index = {v: k for k, v in class_dict.items()}
    except Exception as e:
        print(e)
        exit(-1)

    # 调用摄像头

    capture=cv2.VideoCapture(0)


    fps = 0.0
    while(True):
        t1 = time.time()
        # 读取某一帧
        ref,frame=capture.read()
        # 格式转变，BGRtoRGB
        frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        # 转变成Image
        frame = Image.fromarray(np.uint8(frame))

        data_transform = transform.Compose([transform.Resize(),
                                            transform.ToTensor(),
                                            transform.Normalization()])
        img, _ = data_transform(frame)
        # expand batch dimension
        img = torch.unsqueeze(img, dim=0)

        model.eval()
        with torch.no_grad():
            predictions = model(img.to(device))[0]  # bboxes_out, labels_out, scores_out
            predict_boxes = predictions[0].to("cpu").numpy()
            predict_boxes[:, [0, 2]] = predict_boxes[:, [0, 2]] * frame.size[0]
            predict_boxes[:, [1, 3]] = predict_boxes[:, [1, 3]] * frame.size[1]
            predict_classes = predictions[1].to("cpu").numpy()
            predict_scores = predictions[2].to("cpu").numpy()

            if len(predict_boxes) == 0:
                print("没有检测到任何目标!")

            draw_box(frame,
                    predict_boxes,
                    predict_classes,
                    predict_scores,
                    category_index,
                    thresh=0.5,
                    line_thickness=3)


            # RGBtoBGR满足opencv显示格式
            frame = cv2.cvtColor(np.array(frame), cv2.COLOR_RGBA2BGRA)
            fps = (fps + (1. / (time.time() - t1))) / 2
            print("fps= %.2f" % (fps))
            frame = cv2.putText(frame, "fps= %.2f" % (fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # cv2.imshow("video", frame)
            if flag==1: raise StopIteration
            ret, im1 = cv2.imencode('.jpg', frame)
            frame = im1.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n'+frame+b'\r\n\r\n')



        c= cv2.waitKey(1) & 0xff
        if c==27:
            capture.release()
            break


ssd_video()
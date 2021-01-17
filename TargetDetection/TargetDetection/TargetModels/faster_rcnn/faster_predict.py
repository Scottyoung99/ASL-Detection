import torch
import torchvision
from torchvision import transforms
from TargetModels.faster_rcnn.network_files.faster_rcnn_framework import FasterRCNN, FastRCNNPredictor
from TargetModels.faster_rcnn.backbone.resnet50_fpn_model import resnet50_fpn_backbone
from TargetModels.faster_rcnn.network_files.rpn_function import AnchorsGenerator
from TargetModels.faster_rcnn.draw_box_utils import draw_box
from PIL import Image
import json
import matplotlib.pyplot as plt
import cv2
import time
import os
import uuid

import json
import requests
import re
import numpy as np
from translate import Translator
'''
def translator(str):
    """
    input : str 需要翻译的字符串
    output：translation 翻译后的字符串
    """
    # API
    url = 'http://fanyi.youdao.com/translate?smartresult=dict&smartresult=rule&smartresult=ugc&sessionFrom=null'
    # 传输的参数， i为要翻译的内容
    key = {
        'type': "AUTO",
        'i': str,
        "doctype": "json",
        "version": "2.1",
        "keyfrom": "fanyi.web",
        "ue": "UTF-8",
        "action": "FY_BY_CLICKBUTTON",
        "typoResult": "true"
    }
    # key 这个字典为发送给有道词典服务器的内容
    response = requests.post(url, data=key)
    # 判断服务器是否相应成功
    if response.status_code == 200:
        # 通过 json.loads 把返回的结果加载成 json 格式
        result = json.loads(response.text)
#         print ("输入的词为：%s" % result['translateResult'][0][0]['src'])
#         print ("翻译结果为：%s" % result['translateResult'][0][0]['tgt'])
        translation = result['translateResult'][0][0]['tgt']
        return translation
    else:
        print("有道词典调用失败")
        # 相应失败就返回空
        return None


'''



def create_model(num_classes):

    # resNet50+fpn+faster_RCNN
    backbone = resnet50_fpn_backbone()
    model = FasterRCNN(backbone=backbone, num_classes=num_classes)

    return model


def predict(file_path, target='None'):
    # get devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    basedir = os.path.abspath(os.path.dirname(__file__))
    # create model
    model = create_model(num_classes=21)

    # load train weights
    train_weights = basedir + "/save_weights/resNetFpn-model-14.pth"
    model.load_state_dict(torch.load(train_weights)["model"])
    model.to(device)

    # read class_indict
    category_index = {}
    try:
        json_file = open(basedir+'/pascal_voc_classes.json', 'r')
        class_dict = json.load(json_file)
        category_index = {v: k for k, v in class_dict.items()}
    except Exception as e:
        print(e)
        exit(-1)

    # load image
    original_img = Image.open(file_path)

    # from pil image to tensor, do not normalize image
    data_transform = transforms.Compose([transforms.ToTensor()])
    img = data_transform(original_img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)



    model.eval()
    with torch.no_grad():
        test_time = time.time()
        predictions = model(img.to(device))[0]
        predict_boxes = predictions["boxes"].to("cpu").numpy()
        predict_classes = predictions["labels"].to("cpu").numpy()
        predict_scores = predictions["scores"].to("cpu").numpy()

        if len(predict_boxes) == 0:
            print("没有检测到任何目标!")
        fg = 0

        class_dict["people"] = 15
        if target != 'None':
            translator = Translator(from_lang="chinese", to_lang="english")
            eng_str = translator.translate(target).lower()
            if eng_str in class_dict.keys():
                predict_box = []
                predict_class = []
                predict_score = []
                label = class_dict[eng_str]
                for i in range(len(predict_classes)):
                    if label == predict_classes[i]:
                        predict_box.append(predict_boxes[i])
                        predict_class.append(predict_classes[i])
                        predict_score.append(predict_scores[i])
                        fg = 1
                predict_boxes = np.array(predict_box)
                predict_classes = np.array(predict_class)
                predict_scores = np.array(predict_score)
        else:
            fg = 1

        draw_box(original_img,
                predict_boxes,
                predict_classes,
                predict_scores,
                category_index,
                thresh=0.5,
                line_thickness=3)
        plt.imshow(original_img)
        plt.show()

        filename = str(uuid.uuid4())
        original_img.save(basedir+'/../../static/output/' + filename +'.jpg')
        ret = 'static/output/' + filename +'.jpg'
        test_time = time.time() - test_time
        mAP = 0.735
        return ret, test_time, mAP, fg



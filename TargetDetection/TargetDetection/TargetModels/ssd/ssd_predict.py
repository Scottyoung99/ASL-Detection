import torch
from TargetModels.ssd.draw_box_utils import draw_box
from PIL import Image
import json
import matplotlib.pyplot as plt
from TargetModels.ssd.src.ssd_model import SSD300, Backbone
import TargetModels.ssd.transform as transform
import time
import os
import uuid

def create_model(num_classes):
    backbone = Backbone()
    model = SSD300(backbone=backbone, num_classes=num_classes)

    return model

def ssd_predict(file_path, target='None'):
    # get devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    basedir = os.path.abspath(os.path.dirname(__file__))
    # create model
    model = create_model(num_classes=21)

    # load train weights
    train_weights = basedir+"/save_weights/ssd300-normal.pth"
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

    # load image
    original_img = Image.open(file_path)

    # from pil image to tensor, do not normalize image
    data_transform = transform.Compose([transform.Resize(),
                                        transform.ToTensor(),
                                        transform.Normalization()])
    img, _ = data_transform(original_img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    model.eval()
    with torch.no_grad():
        test_time = time.time()
        predictions = model(img.to(device))[0]  # bboxes_out, labels_out, scores_out
        predict_boxes = predictions[0].to("cpu").numpy()
        predict_boxes[:, [0, 2]] = predict_boxes[:, [0, 2]] * original_img.size[0]
        predict_boxes[:, [1, 3]] = predict_boxes[:, [1, 3]] * original_img.size[1]
        predict_classes = predictions[1].to("cpu").numpy()
        predict_scores = predictions[2].to("cpu").numpy()

        if len(predict_boxes) == 0:
            print("没有检测到任何目标!")

        draw_box(original_img,
                predict_boxes,
                predict_classes,
                predict_scores,
                category_index,
                thresh=0.5,
                line_thickness=5)
        plt.imshow(original_img)
        plt.show()

        filename = str(uuid.uuid4())
        original_img.save(basedir+'/../../static/output/' + filename +'.jpg')
        ret = 'static/output/' + filename +'.jpg'
        test_time = time.time() - test_time
        mAP = 0.685
        return ret, test_time, mAP, 1


# ssd_predict('test.jpg')
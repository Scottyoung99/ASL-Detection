#!usr/bin/env python  
#-*- coding:utf-8 -*-  

import os,sys
root_path = os.getcwd()
# root_path = "D:\\mycode\\py-workspace\\TargetDetection\\TargetDetection\\"
sys.path.insert(0,root_path+"./TargetModels/yolov5Master") 

from werkzeug.utils import secure_filename
from flask import Flask, render_template, jsonify, request, make_response, send_from_directory, abort,Response
import time
import base64
import uuid
import json
import gl
from PIL import Image
from TargetModels.faster_rcnn import faster_predict
from TargetModels.cascade import cas_predict
from TargetModels.mask import mask_predict
from TargetModels.yolov5Master import yolo_detect
from TargetModels.ssd import ssd_predict

#这个后面的"/yolov5" 文件夹应该是指向的 yolov5根目录.


app = Flask(__name__)
UPLOAD_FOLDER = 'static/upload'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
basedir = os.path.abspath(os.path.dirname(__file__))  # 这个是获取当前路径
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'JPG', 'PNG', 'gif', 'GIF'])
video_flag=0

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

#########################################################
# func代表预测函数，返回值为生成图片的相对路径，最好将图片输出到static/output中
def func(filename,tgt,mdl):
    if mdl=='Faster_Rcnn':  output, test_time, mAP, flag = faster_predict.predict(filename,tgt)
    elif mdl=='Mask': output, test_time, mAP, flag = mask_predict.Predict(filename)
    elif mdl=='yolov5': output, test_time, mAP, flag = yolo_detect.yolo_pic_detect(filename)
    elif mdl=='Cascade': output, test_time, mAP, flag = cas_predict.Predict(filename)
    elif mdl=='SSD': output, test_time, mAP, flag = ssd_predict.ssd_predict(filename)
    return output,test_time,mAP,flag

##########################################################

@app.route('/video_request', methods=['POST','GET']) 
def video_request():  
    rqst = request.values.get('object')
    video_flag = 0
    if rqst=='yolov5': 
        return '1'
    elif rqst=='Cascade':
        return '2'
    elif rqst=='SSD':
        return '3'

###############################################################

@app.route('/stop_video', methods=['POST','GET']) 
def stop_video():  
    rqst = request.values.get('stp')
    global video_flag
    video_flag = 1
    return '1'

@app.route('/yolo_video_feed')  # 这个地址返回视频流响应
def yolo_video_feed():  
    return Response(yolo_detect.yolo_video_detect(video_flag),
                     mimetype='multipart/x-mixed-replace; boundary=frame')  

@app.route('/ssd_video_feed')  # 这个地址返回视频流响应
def ssd_video_feed():  
    return Response(yolo_detect.yolo_video_detect(video_flag),
                     mimetype='multipart/x-mixed-replace; boundary=frame')  

@app.route('/cas_video_feed')  # 这个地址返回视频流响应
def cas_video_feed():  
    return Response(cas_predict.Video(video_flag),
                     mimetype='multipart/x-mixed-replace; boundary=frame') 

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/product/list")
def product_list():
    return render_template("video.html")

@app.route("/user/list")
def user_list():
    return render_template("userList.html")

@app.route("/record/list")
def record_list():
    return render_template("recordList.html")

@app.route('/pred', methods=['POST','GET'])
def pred():
    file_dir = os.path.join(basedir, app.config['UPLOAD_FOLDER'])
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    f = request.files.get('file')
    mdl = request.values.get('mdl')
    target = request.values.get('object')
    if target=='': target = 'None'
    if f and allowed_file(f.filename):  #如果文件合法
        fname = secure_filename(f.filename)
        ext = fname.rsplit('.', 1)[1]
        new_filename =  str(uuid.uuid4()) + "." + ext
        f.save(os.path.join(file_dir, new_filename))
############################################################################
        gl.pred_file,gl.Time,gl.Map,gl.flag = func(file_dir + '/' + new_filename,target,mdl) #func(源图片相对路径)目标检测函数
        gl.pred_file.replace("\\",'/')     
############################################################################
    d = {'name': gl.pred_file, 'Time': gl.Time, 'Map': gl.Map,'flag':gl.flag}
    return jsonify(d)

if __name__ == "__main__":
    app.run()
    
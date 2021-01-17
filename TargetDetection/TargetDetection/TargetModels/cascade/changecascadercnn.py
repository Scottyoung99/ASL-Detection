import torch
import numpy as np
import pickle
num_class = 26
with open('checkpoints/model_final_480dd8.pkl', 'rb') as f:
    obj = f.read()
weights = pickle.loads(obj, encoding='latin1')

weights['model']['roi_heads.box_predictor.0.cls_score.weight']=np.zeros([num_class+1,1024], dtype='float32')
weights['model']['roi_heads.box_predictor.0.cls_score.bias']=np.zeros([num_class+1], dtype='float32')
weights['model']['roi_heads.box_predictor.1.cls_score.weight']=np.zeros([num_class+1,1024], dtype='float32')
weights['model']['roi_heads.box_predictor.1.cls_score.bias']=np.zeros([num_class+1], dtype='float32')
weights['model']['roi_heads.box_predictor.2.cls_score.weight']=np.zeros([num_class+1,1024], dtype='float32')
weights['model']['roi_heads.box_predictor.2.cls_score.bias']=np.zeros([num_class+1], dtype='float32')


weights['model']['roi_heads.box_predictor.0.bbox_pred.weight']=np.zeros([4,1024], dtype='float32')
weights['model']['roi_heads.box_predictor.0.bbox_pred.bias']=np.zeros([4], dtype='float32')
weights['model']['roi_heads.box_predictor.1.bbox_pred.weight']=np.zeros([4,1024], dtype='float32')
weights['model']['roi_heads.box_predictor.1.bbox_pred.bias']=np.zeros([4], dtype='float32')
weights['model']['roi_heads.box_predictor.2.bbox_pred.weight']=np.zeros([4,1024], dtype='float32')
weights['model']['roi_heads.box_predictor.2.bbox_pred.bias']=np.zeros([4], dtype='float32')

weights['model']['roi_heads.mask_head.predictor.weight']=np.zeros([num_class,256,1,1], dtype='float32')
weights['model']['roi_heads.mask_head.predictor.bias']=np.zeros([num_class], dtype='float32')


f = open('model_final_cascadercnn3x_hand.pkl', 'wb')
pickle.dump(weights, f)
f.close()
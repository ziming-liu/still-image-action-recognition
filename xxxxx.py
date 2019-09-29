

import numpy as xp
import torch

import mmcv
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
from mmdet.apis import inference_detector, show_result
import mmcv
from mmcv.runner import load_checkpoint
from mmdet.models import build_detector
from mmdet.apis import inference_detector, show_result
import os
"""
os.environ["DISPLAY"] = ":22"
cfg = mmcv.Config.fromfile('/home/liuziming/show/configs/faster_rcnn_r50_fpn_1x.py')
cfg.model.pretrained = None

# construct the model and load checkpoint
model = build_detector(cfg.model, test_cfg=cfg.test_cfg)
_ = load_checkpoint(model, './MODELZOO/faster_rcnn_r50_fpn_1x_20181010-3d1b3351.pth')

# test a single image
img = mmcv.imread('./test.png')#img 是ndarray或者 file
result = inference_detector(model, img, cfg)
print(len(result))
show_result(img, result,show=False,out_file='./tt.png')


# test a list of images
imgs = ['test1.jpg', 'test2.jpg']
for i, result in enumerate(inference_detector(model, imgs, cfg, device='cuda:0')):
    print(i, imgs[i])
  show_result(imgs[i], result)
"""
CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
           'train', 'truck', 'boat', 'traffic_light', 'fire_hydrant',
           'stop_sign', 'parking_meter', 'bench', 'bird', 'cat', 'dog',
           'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
           'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
           'skis', 'snowboard', 'sports_ball', 'kite', 'baseball_bat',
           'baseball_glove', 'skateboard', 'surfboard', 'tennis_racket',
           'bottle', 'wine_glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
           'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
           'hot_dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
           'potted_plant', 'bed', 'dining_table', 'toilet', 'tv', 'laptop',
           'mouse', 'remote', 'keyboard', 'cell_phone', 'microwave',
           'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
           'vase', 'scissors', 'teddy_bear', 'hair_drier', 'toothbrush')
cat2label = {
            cat_id: i + 1
            for i, cat_id in enumerate(CLASSES)
        }
label2cat = {i+1:cat_id for i,cat_id in enumerate(CLASSES)}

result = mmcv.load('stanford_train_bbox_new.pkl')
print(result)
print(len(result[230]))
print(len(result[230][0]))
#print(result[230][0][3])
for ii,item in enumerate(result[230]):
    if item.any():
        print('=============================')
        print('class is {}'.format(label2cat[ii+1]))
        print(item)


from mmdet.apis import DetInferencer
import os.path as osp
from pprint import pprint

current_path = osp.dirname(osp.abspath(__file__))
inferencer = DetInferencer(
    model=f'{current_path}/configs/deepin/faster-rcnn_r101_fpn_2x_coco.py',
    weights=f'{current_path}/work_dirs/faster-rcnn_r101_fpn_2x_coco/epoch_24.pth',
    device="cuda:0"
)

res = inferencer(f'{current_path}/data/coco/train2017/20241104095629.jpg', out_dir=f"{current_path}/outputs")
pprint(res)

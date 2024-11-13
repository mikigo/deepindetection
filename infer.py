from mmdet.apis import DetInferencer
import os.path as opt

current_path = opt.dirname(opt.abspath(__file__))
print(current_path)
# 初始化模型
inferencer = DetInferencer(
    model=f'{current_path}/configs/deepin/faster-rcnn_r101_fpn_2x_coco.py',
    weights=f'{current_path}/work_dirs/faster-rcnn_r101_fpn_2x_coco/epoch_24.pth',
    device="cuda:0"
)

# 推理示例图片
inferencer(f'{current_path}/data/coco/train2017/20241104095629.jpg', out_dir=f"{current_path}/outputs")
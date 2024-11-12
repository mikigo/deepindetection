from mmdet.apis import DetInferencer
import os.path as opt

current_path = opt.dirname(opt.abspath(__file__))
print(current_path)
# 初始化模型
inferencer = DetInferencer(
    model=f'{current_path}/rtmdet_tiny_8xb32-300e_coco.py',
    weights=f'{current_path}/rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth',
    device="cpu"
)

# 推理示例图片
inferencer(f'{current_path}/demo/demo.jpg', out_dir=f"{current_path}/output")
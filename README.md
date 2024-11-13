# DeepinDetection2

## install

```bash
pip install torch torchvision
pip install -U openmim
pip install platformdirs
mim install mmengine
pip install -U setuptools
mim install "mmcv==2.1.0"
pip install -e .
```

## check 

```bash
mim download mmdet --config rtmdet_tiny_8xb32-300e_coco --dest .
python demo/image_demo.py demo/demo.jpg rtmdet_tiny_8xb32-300e_coco.py --weights rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth --device cpu
```

## cvrt

python tools/dataset_converters/pascal_voc.py data/VOCdevkit/ -o data/coco/ --out-format coco

## train

python tools/train.py configs/deepin/faster-rcnn_r101_fpn_2x_coco.py


## test

python tools/test.py configs/deepin/faster-rcnn_r101_fpn_2x_coco.py work_dirs/faster-rcnn_r101_fpn_2x_coco/epoch_24.pth


## driver

https://www.nvidia.cn/drivers
https://developer.nvidia.com/cuda-toolkit-archive


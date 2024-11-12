_base_ = '../faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py'

# 我们还需要更改 head 中的 num_classes 以匹配数据集中的类别数
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=2, type="Shared2FCBBoxHead"),
        mask_head=dict(num_classes=2, type="FCNMaskHead")
    )
)

# 修改数据集相关配置
data_root = 'data/coco/'
metainfo = {
    'classes': ('dde_file_manager_icon', 'dde_launcher_icon'),
    'palette': [
        (220, 20, 60),  # 红色，对应 'dde_file_manager_icon'
        (0, 255, 0)     # 绿色，对应 'dde_launcher_icon'
    ]
}
train_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/train.json',
        data_prefix=dict(img='train/')))
val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/val.json',
        data_prefix=dict(img='val/')))
test_dataloader = val_dataloader

# 修改评价指标相关配置
val_evaluator = dict(ann_file=data_root + 'annotations/val.json')
test_evaluator = val_evaluator

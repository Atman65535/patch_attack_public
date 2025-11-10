_base_ = [
    '../bisenetv2/bisenetv2_fcn_4xb4-amp-160k_rellis-1024x1024.py'
]


val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])

epoch = 1
lr = 0.001
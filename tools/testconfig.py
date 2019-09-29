# model settings
model = 'resnet50'
# dataset settings
#data_root = '/mnt/SSD/dataset/cifar10'
mean = [0.4914, 0.4822, 0.4465]
std = [0.2023, 0.1994, 0.2010]
batch_size = 12

# optimizer and learning rate
optimizer = dict(type='Adagrad', lr=1e-4, )#momentum=0.9, weight_decay=5e-4
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='fixed',)

# runtime settings
work_dir = './demo'
gpus = range(2)
dist_params = dict(backend='nccl',)
data_workers = 2  # data workers per gpu
checkpoint_config = dict(interval=10)  # save checkpoint at every epoch
workflow = [('train', 3), ('val', 1)]
total_epochs = 300
resume_from = None
load_from = None

# logging settings
log_level = 'INFO'
log_config = dict(
    interval=10,  # log at every 50 iterations
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook'),
    ])

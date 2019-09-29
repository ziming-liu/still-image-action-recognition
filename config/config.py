# coding:utf-8

from yacs.config import CfgNode as CN

_C = CN()

_C.SYSTEM = CN()
_C.SYSTEM.NAME = 'action-rcg'
_C.SYSTEM.VIS_PORT = 8097
_C.SYSTEM.DEVICE = "0"
_C.SYSTEM.ROOT = "/home/share/LabServer/GLnet"


_C.MODEL = CN()
_C.MODEL.NAME = ''
_C.MODEL.DIST = False
_C.MODEL.NUM_CLASS = 40
_C.MODEL.NONLOCAL = ['concatenation', 'embedded_gaussian', 'gaussian', 'dot_product', ]
_C.MODEL.SAVE_MODE = 'best'  # 'all'  #'best'
_C.MODEL.SAVE_IN = '/home/share/LabServer/GLnet/MODELZOO/'
_C.MODEL.LOAD_IN = '/home/share/LabServer/GLnet/MODELZOO/'
_C.MODEL.PRETRAINED = '/home/share/LabServer/GLnet/ckp/keypoint_resversion/keypoint_resversion_best.pth'
    #'/home/share/LabServer/GLnet/ckp/resnet50_baseline/resnet50_baseline.pth'  #'./resnet18-pre.pth' #'./resnet18_places365.pth.tar'
_C.MODEL.BASELINE = '../ckp/resnet18_baseline/resnet18_baseline_best.pth'
_C.MODEL.TRAIN = True
_C.MODEL.VAL = True
_C.MODEL.TEST = False
_C.MODEL.RESUME= 'MODELZOO/un_att_pascal_0001.pth'
_C.MODEL.RESULT='MODELZOO/'
_C.MODEL.CKP_DURING=2


_C.DATASET = CN()
#tow still image datasets
_C.DATASET.STANFORD40 = '/home/share/DATASET/stanford40'
_C.DATASET.PASCALVOC = '/home/share/DATASET/VOCdevkit/VOC2012'
#TODO:video datasets
#_C.DATASET.train_img_folder = '../data/VOCdevkit/train'
#_C.DATASET.val_img_folder = '../data/VOCdevkit/val'
#_C.DATASET.test_img_folder = '../data/VOCdevkit/test'
#_C.DATASET.SCENELABELS = '/home/share/LabServer/GLnet/data/sceneLabels'
_C.DATASET.IMAGE_SIZE = (224,224)
_C.DATASET.CROP_BBOX = False
_C.DATASET.HIGH_RESOLUTION=224
_C.DATASET.LOW_RESOLUTION= 56
_C.DATASET.INTERNAL= 8
_C.DATASET.ALPHA = 1.0

_C.OPTIM = CN()
_C.OPTIM.WEIGHT_DECAY = 1E-5
_C.OPTIM.LR = 1e-4
_C.OPTIM.LR_DECAY = 0.1  # 0.5-0.1


_C.TRAIN = CN()
_C.TRAIN.BATCH_SIZE = 10
_C.TRAIN.BEGIN_EPOCH = 1


_C.TEST = CN()
_C.TEST.BATCH_SIZE = 32

# Exporting as cfg is a nice convention
cfg = _C





"""
class Config(object):
    env = 'actoin'
    vis_port = 8097

    #train log
    log = './log/'
    save_mode = 'best'
    higher = True
    NUM_CLASSES = 40
    th = 120
    #网络超参数
    lr = 1e-3
    epoches = 666
    batch_size = 1
    num_worker = 4
    weight_decay = 1e-5
    previous_loss = 1e6
    previous_accu = 0
    lr_decay = 0.1

    #训练状态控制
    which = "training"
    stanford40 = '/home/share/LabServer/stanford40'
    #数据加载和保存地址
    #原始数据#整理好的数据形式
    train_img_folder = './data/VOCdevkit/train'
    #val_img_folder = './data/VOCdevkit/val'
    #test_img_folder = './data/VOCdevkit/test'
    #saliency训练net1的数据
    #train_img_folder = './data/cam2/saliency'
    val_img_folder = './data/VOCdevkit/val'
    test_img_folder = './data/VOCdevkit/test'
    #第一阶段得到的cam和saliency数据
    cam_folder = './data/cam2'
    saliency_path = './data/saliency/train'

    #模型相关
    save_ckp_path = './checkpoints/1st_train_place365resnet18'
    load_ckp_path = './checkpoints/wideresnet18_places365.pth.tar'

    use_gpu = True
    use_device = [0,1]
    multi_gpu = False

    voc_data_dir = '/home/share/VOCdevkit/VOC2012'
    min_size = 600  # image resize
    max_size = 1000  # image resize
    num_workers = 8
    test_num_workers = 8
    data_path = ''

    combine = 'sum'
    submodel = False

    # sigma for l1_smooth_loss
    rpn_sigma = 3.
    roi_sigma = 1.
    stage2 = False
    # param for optimizer
    # 0.0005 in origin paper but 0.0001 in tf-faster-rcnn


    # visualization
      # visdom env
    port = 8097
    plot_every = 40  # vis every N iter

    # preset
    data = 'voc'
    pretrained_model = 'vgg16'

    # training
    epoch = 14

    multiGPU = True
    use_adam = False  # Use Adam optimizer
    use_chainer = False  # try match everything as chainer
    use_drop = False  # use dropout in RoIHead
    # debug
    debug_file = '/tmp/debugf'

    test_num = 10000
    # model
    exist_pretrain_for_classifier = False
    model_pretrain_for_classifier = ''
    load_path = None

    caffe_pretrain = False  # use caffe pretrained model instead of torchvision
    caffe_pretrain_path = 'checkpoints/vgg16_caffe.pth'

    def _parse(self, kwargs):

        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("WARNING:opt has no attr called %s", k)
            setattr(self, k, v)
        print("user config!")
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('_'):  # 不打印parser
                print(k, getattr(self, k))


opt = Config()  # 实例化config
"""

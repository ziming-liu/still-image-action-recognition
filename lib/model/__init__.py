from .KeyPoint import key50
#from .KeyPoint import twokey
from .resnet import ResNet
from .keypoint_best import onekey as onekey_
from .keypoint_best import twokey as twokey_
from .multiInstance import multi_one,multi_two,multi_three,multi_four
from .keypoint_copy import onekey_copy,twokey_copy
from .vgg import vgg16_bn

from .net import resnet50

from .relation import SelfAttention
from .twostreamhead import Head
from .action_head import ActionHead
__all__ = {'SelfAttention','Head','ActionHead'}


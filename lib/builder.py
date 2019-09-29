from mmcv.runner import obj_from_dict
from torch import nn

from lib.model import (SelfAttention,ActionHead)
__all__ = [
    'SelfAttention','ActionHead',
]


def _build_module(cfg, parrent=None, default_args=None):
    return cfg if isinstance(cfg, nn.Module) else obj_from_dict(
        cfg, parrent, default_args)


def build(cfg, parrent=None, default_args=None):
    if isinstance(cfg, list):
        modules = [_build_module(cfg_, parrent, default_args) for cfg_ in cfg]
        return nn.Sequential(*modules)
    else:
        return _build_module(cfg, parrent, default_args)


def build_relation(cfg):
    return  build(cfg,SelfAttention)

def build_actionhead(cfg):
    return build(cfg,ActionHead)


def build_detector(cfg, train_cfg=None, test_cfg=None):
    from lib import model
    return build(cfg, model, dict(train_cfg=train_cfg, test_cfg=test_cfg))



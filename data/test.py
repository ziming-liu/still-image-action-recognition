from mmdet.datasets import StanfordDataset
from torch.utils.data import DataLoader,Dataset
import collections

import torch
import torch.nn.functional as F
from torch.utils.data.dataloader import default_collate

from mmcv.parallel.data_container import DataContainer


def collate(batch, samples_per_gpu=1):
    """Puts each data field into a tensor/DataContainer with outer dimension
    batch size.

    Extend default_collate to add support for
    :type:`~mmcv.parallel.DataContainer`. There are 3 cases.

    1. cpu_only = True, e.g., meta data
    2. cpu_only = False, stack = True, e.g., images tensors
    3. cpu_only = False, stack = False, e.g., gt bboxes
    """

    if not isinstance(batch, collections.Sequence):
        raise TypeError("{} is not supported.".format(batch.dtype))

    if isinstance(batch[0], DataContainer):
        assert len(batch) % samples_per_gpu == 0
        stacked = []
        if batch[0].cpu_only:
            for i in range(0, len(batch), samples_per_gpu):
                stacked.append(
                    [sample.data for sample in batch[i:i + samples_per_gpu]])
            return DataContainer(
                stacked, batch[0].stack, batch[0].padding_value, cpu_only=True)
        elif batch[0].stack:
            for i in range(0, len(batch), samples_per_gpu):
                assert isinstance(batch[i].data, torch.Tensor)
                if  batch[i].dim() == 3:
                    # TODO: handle tensors other than 3d
                    assert batch[i].dim() == 3
                    #选择一个batch里面最大的 h w
                    c, h, w = batch[0].size()
                    for sample in batch[i:i + samples_per_gpu]:
                        assert c == sample.size(0)
                        h = max(h, sample.size(1))
                        w = max(w, sample.size(2))
                    padded_samples = [
                        F.pad(
                            sample.data,
                            (0, w - sample.size(2), 0, h - sample.size(1)),#在右侧和下册pad
                            value=sample.padding_value)
                        for sample in batch[i:i + samples_per_gpu]
                    ]

                    stacked.append(default_collate(padded_samples))
                elif batch[i].dim() ==1:
                    padded_samples = [sample.data for sample in batch[i:i+samples_per_gpu]]
                    stacked.append(default_collate(padded_samples))

        else:
            for i in range(0, len(batch), samples_per_gpu):
                stacked.append(
                    [sample.data for sample in batch[i:i + samples_per_gpu]])
        return DataContainer(stacked, batch[0].stack, batch[0].padding_value)
    elif isinstance(batch[0], collections.Sequence):
        transposed = zip(*batch)
        return [collate(samples, samples_per_gpu) for samples in transposed]
    elif isinstance(batch[0], collections.Mapping):
        return {
            key: collate([d[key] for d in batch], samples_per_gpu)
            for key in batch[0]
        }
    else:
        return default_collate(batch)
from functools import partial
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

dataset = StanfordDataset(ann_file='/home/share/LabServer/DATASET/stanford40/ImageSplits/test.txt',
                 img_prefix='/home/share/LabServer/DATASET/stanford40/',
                 img_scale=(224,224),
                 img_norm_cfg=img_norm_cfg,
                 size_divisor=32,
                 proposal_file='/home/share/LabServer/GLnet/stanford_test_bbox_new.pkl',
                 test_mode=False,)
print(len(dataset))
dataloader = DataLoader(dataset,shuffle=False,collate_fn=partial(collate,samples_per_gpu=10),batch_size=10,num_workers=4,)
for ii ,data in enumerate(dataloader):
    #print(data)

    pro = data['proposals'].data
    print(len(pro))
    print(len(pro[0]))
    print(type(pro[0].cuda()))

    break


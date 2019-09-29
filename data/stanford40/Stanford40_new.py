from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import os.path as osp
from util.config import cfg
from torchvision import transforms as T
import mmcv
import numpy as np
from data.stanford40.transforms import ImageTransform,BboxTransform
from mmcv.parallel.collate import collate
from mmcv.parallel import DataContainer as DC
from data.datasets.utils import to_tensor
class Stanford40(Dataset):
    def __init__(self, root, phase='train',proposal_file=None ):
        super(Stanford40, self).__init__()
        self.img_norm_cfg= dict(
        mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
        self.size_divisor=32
        self.img_transform = ImageTransform(
            size_divisor=self.size_divisor,
            **self.img_norm_cfg)  # 初始化trainsform  return img, img_shape, pad_shape, scale_factor
        self.bbox_transform = BboxTransform()

        self.proposals = self.load_proposals(proposal_file)
        self.root = root
        split_path = os.path.join(root, 'ImageSplits')
        classes = []
        with open(split_path + '/actions.txt') as f:
            txt = f.readlines()
            for ii, name in enumerate(txt):
                if ii > 0:
                    classes.append(name.strip().split()[0])
        # print(classes)
        # print(len(classes))
        self.classes = classes

        self.sample = []
        self.label = []
        with open(os.path.join(split_path, phase + '.txt')) as f:
            txt = f.readlines()
            for ii, item in enumerate(txt):
                self.sample.append(item.strip())
                cla = item.strip().split('.')[0].split('_')[:-1]
                cla = '_'.join(cla)
                # print(cla)
                self.label.append(self.classes.index(cla))

    def __getitem__(self, item):
        img_path = os.path.join(self.root, 'JPEGImages', self.sample[item])
        #name = [[self.sample[item]]]
        #assert self.name[item][0] == self.sample[item]
        image= mmcv.imread(img_path)
        h,w,c= image.shape
        proposal = self.proposals[item]

        image,ori_shape,now_shape,scale_factor = self.img_transform(image,(224,224),False,keep_ratio=False)
        assert proposal.shape[1] == 5
        proposals_label = proposal[:, 4, None]
        proposals_bboxes = proposal[:, :4]
        proposals_bboxes = self.bbox_transform(proposals_bboxes, now_shape, scale_factor,flip=False)
        proposal = np.hstack([proposals_bboxes, proposals_label])

        label = self.label[item]
        data = dict(image = DC(to_tensor(image),stack=True),
                    label = DC(to_tensor(label)),
                    proposal = DC(to_tensor(proposal)))

        return  data

    def __len__(self):
        return len(self.sample)

    def  load_proposals(self, proposal_file):
        self.proposals = mmcv.load(proposal_file)
        proposal_bboxes = []
        proposal_labels = []
        for i, item in enumerate(self.proposals):
            # 处理一个图片
            # 处理一个图片
            pros = []
            lab = []
            for j, cla in enumerate(item):
                if cla.any():
                    for kk, loc in enumerate(cla):
                        pros.append(loc)
                        lab.append(j)
            if not pros:
                pros = np.zeros((0, 4))
                lab = np.zeros((0,))
            else:
                pros = np.array(pros).astype(np.float32)
                pros = pros[:, :4]
                lab = np.array(lab).astype(np.int64)
            proposal = np.concatenate((pros, lab[:, None]), axis=1)
            assert proposal.shape[1] == 5

            #proposal_labels.append(lab)
            proposal_bboxes.append(proposal)  # bbox and labels 都是有 总共的N个samples的 list
        # dict中两个obj，都是list， 每个list包含N个samples的 nd array对象 ，box或者label
        self.proposals = proposal_bboxes#dict(proposal_bboxes=proposal_bboxes, proposal_labels=proposal_labels)
        return self.proposals

if __name__ == '__main__':
    from torchvision import transforms as T
    import torch as t

    simpletransform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    strongtransform = T.Compose([
        T.RandomCrop((224, 224)),
        T.RandomResizedCrop((224, 224), (0.5, 1.0)),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    dataset = Stanford40(cfg.DATASET.STANFORD40, phase='train',
                      proposal_file='/home/share/LabServer/GLnet/stanford_test_bbox_new.pkl')
    dataloader = DataLoader(dataset, num_workers=1, collate_fn=collate,batch_size=4, shuffle=False)
    for ii, (data) in enumerate(dataloader):
        #imshow(data,"data")
        #imshow(subdata,"subimg")
        print(type(data))
        image,label,proposal = data['image'].data,data['label'].data,data['proposal'].data
        print(type(image))
        print(type(label))
        print(len(label))
        print(type(proposal))
        pros =[]
        for i in range(4):
            p = t.stack(proposal[i]).unsqueeze(0)
            print(p.shape)
            pros.append(p)
        proposal = t.cat(pros,0)
        print(proposal.shape)
        label = t.stack(label[0])
        print(label.shape)
        #print(subdata.shape)
        break

    print('ok')








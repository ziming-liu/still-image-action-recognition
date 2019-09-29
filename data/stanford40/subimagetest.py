import matplotlib.pyplot as plt

from PIL import Image
import os
from util.config import cfg
from torchvision import transforms as T

from lib.module.FindProposal import FindProposal
from lib.model import resnet50

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if __name__ == '__main__':
    phase = 'test'
    root = cfg.DATASET.STANFORD40
    split_path = os.path.join(root, 'ImageSplits')
    classes = []
    with open(split_path + '/actions.txt') as f:
        txt = f.readlines()
        for ii, name in enumerate(txt):
            if ii > 0:
                classes.append(name.strip().split()[0])
    # print(classes)
    # print(len(classes))

    sample = []
    label = []
    with open(os.path.join(split_path, phase + '.txt')) as f:
        txt = f.readlines()
        for ii, item in enumerate(txt):
            sample.append(item.strip())
            cla = item.strip().split('.')[0].split('_')[:-1]
            cla = '_'.join(cla)
            # print(cla)
            label.append(classes.index(cla))
    classifier = resnet50(pretrained=True).cuda()

    simpletransform = T.Compose([
        T.Resize((224, 224)),
        # T.RandomGrayscale(1),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

    ])


    for item in range(len(sample)):
        img_path = os.path.join(root, 'JPEGImages', sample[item])
        # name = [[self.sample[item]]]
        # assert self.name[item][0] == self.sample[item]
        image = Image.open(img_path)
        w, h = image.size
        image = T.Resize((224, 224))(image)

        fullimage = simpletransform(image)
        # run pre network
        box = FindProposal(classifier, fullimage.cuda(), n_grids=5)
        y1, x1, y2, x2 = box
        subimage = image.crop((x1, y1, x2, y2))
        plt.imshow(subimage)
        save_path = os.path.join(root, 'SubImages', sample[item])
        subimage = T.Resize((224, 224))(subimage)
        subimage.save(save_path)
        sh, sw = subimage.size

        label_sub = label[item]
        print("true label {}".format(label_sub))



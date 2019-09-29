from torch.utils.data import DataLoader,Dataset
import torch as t
from data.stanford40.Stanford40Dataset import Stanford40
from torchvision import  transforms as T
from data.BU101 import BU101
from config.config import cfg
def stanford40(cfg):
    # ========= Preparing DataLoader =========#
    data = Stanford40(cfg.DATASET.STANFORD40, 'train')
    trn_dataloader = DataLoader(data, shuffle=True,
                                batch_size=cfg.DATASET.BATCH_SIZE,
                                num_workers=cfg.DATASET.NUM_WORKERS)

    data = Stanford40(cfg.DATASET.STANFORD40, 'test')
    val_dataloader = DataLoader(data, shuffle=False,
                                batch_size=cfg.DATASET.BATCH_SIZE,
                                num_workers=cfg.DATASET.NUM_WORKERS)
    return trn_dataloader, val_dataloader



from data.pascalVOC.basicDataloader import get_basicDataloader
def pascal2012():
    trn_dataloader, val_dataloader = get_basicDataloader()
    return  trn_dataloader, val_dataloader

#load model statedict
def load(model,path):
    ckp_path = path
    ckp = t.load(ckp_path)

    pretrained_dict = ckp
    model_dict = model.state_dict()
    pretrained_dict = {k:v for k,v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model


simpletransform = T.Compose([
        T.Resize((224,224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])

    ])
strongtransform = T.Compose([
        T.Resize((224,224)),
        #T.RandomResizedCrop(224,(0.5,1.0)),
        T.RandomHorizontalFlip(0.7),
        #T.RandomVerticalFlip(0.1),
        #T.RandomGrayscale(1),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])

def BU101Dataset(BU_path = '/home/share/DATASET/BU101'):
    train_dataset = BU101(BU_path,transform=strongtransform)
    print(len(train_dataset))
    train_dataloader = DataLoader(train_dataset, shuffle=True,
                                batch_size=cfg.TRAIN.BATCH_SIZE,
                                num_workers=4)

    return train_dataloader

if __name__ == '__main__':
    trainloader  = BU101Dataset()
    print(len(trainloader))

    for ii,(image,label ) in enumerate(trainloader):
        #print(image.shape)
        print(label)


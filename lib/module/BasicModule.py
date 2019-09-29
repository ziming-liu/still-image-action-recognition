import torch as t
import os
from util.config import cfg
import time
class BasicModule(t.nn.Module):
    """
    封装了nn.Module，主要提供save和load两个方法
    """

    def __init__(self):
        super(BasicModule,self).__init__()
        self.model_name = str(type(self)) # 模型的默认名字

    def load(self, path):
        """
        可加载指定路径的模型
        """
        self.load_state_dict(t.load(path))


    def save(self, name=None):
        """
        保存模型，默认使用“模型名字+时间”作为文件名，
        如AlexNet_0710_23:57:29.pth
        """
        savingpath = os.path.join(cfg.MODEL.SAVE_IN,cfg.SYSTEM.NAME)
        if os.path.exists(savingpath) is False:
            os.makedirs(savingpath)
        if name is None:
            prefix = savingpath + self.model_name + '_'
            name_ = time.strftime(prefix + '%m%d_%H:%M:%S.pth')
        else:
            name_ = os.path.join(savingpath,name)
        t.save(self.state_dict(), name_)
        print('    - [Info] The checkpoint file has been updated.')
        print(name_)
        return name_
if __name__ == '__main__':

    print(help(t.optim.lr_scheduler.CosineAnnealingLR))

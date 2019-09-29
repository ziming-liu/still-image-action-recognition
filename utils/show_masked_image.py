
import cv2
import numpy as np
from lib.core.visImage import tensor_to_np

def inverse_normalize(img):
    # if opt.caffe_pretrain:
    #    img = img + (np.array([122.7717, 115.9465, 102.9801]).reshape(3, 1, 1))
    #    return img[::-1, :, :]
    # approximate un-normalize for visualize
    return (img * 0.225 + 0.45).clip(min=0, max=1) * 255


def show_masked_image(vis,initimg, mask, title=None):
    """
    :param vis: the environment to visulize our result image
    :param initimg: 'tensor' , background image
    :param mask: attention map
    :param title:
    :return: image with attention map on it
    """
    initimg = initimg.numpy()
    # depend on how normallize you use
    initimg = inverse_normalize(initimg)

    mask = mask.repeat(3, 1, 1)  # chw
    mask = tensor_to_np(mask)
    map = cv2.resize(mask, (224, 224))
    map = np.uint8(map)
    heatmap = cv2.applyColorMap(map, cv2.COLORMAP_HSV)
    heatmap = heatmap.transpose(2, 0, 1)

    result = heatmap * 0.4 + initimg * 0.7

    vis.image(result, win=title, opts={'title': title})
    # vis.image(initimg, win='sd12j', opts={'title': 'keypoint'})

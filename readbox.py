import scipy.io as scio
import numpy as np
#datafile = '/home/share/DATASET/VOCdevkit/VOC2012/selective_search_action_data/'
datafile = '/home/share/DATASET/stanford40/selective_search/'

data = scio.loadmat(datafile+'selective_search_train.mat')
print(type(data))
print(data.keys())
images_ids = data['images']
boxes = data['boxes']

print(type(images_ids))
print(images_ids.shape)
print(type(boxes))
print(boxes.shape)
num = boxes.shape[0]
max = -1
for ii in range(num):
    imageid = str(list(images_ids[0,ii])[0])
    print(imageid)
    box = boxes[ii,0]
    N = box.shape[0]
    if N>max:
        max = N
max = 500
    #print(type(box))
ss_box = dict()
for ii in range(num):
    imageid = str(list(images_ids[0,ii])[0])
    print(imageid)
    box = boxes[ii,0]
    boxlen = box.shape[0]
    if boxlen>=max:
        boxlen = max
    else:boxlen = boxlen
    ep = np.zeros((max,4)).astype(np.float32)
    ep[:boxlen,:] = box[:max,:]
    #print(type(box))
    ss_box[imageid] = ep

    #print(ep)
    #assert ep.shape[0] ==max
    print(ii)
#print(ss_box[imageid])
datanew = datafile+'ss_box_train.mat'
scio.savemat(datanew,ss_box)





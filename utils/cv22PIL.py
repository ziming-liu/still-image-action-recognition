import cv2
from PIL import Image
import numpy

def cv22PIL(init,target):
    target = Image.fromarray(cv2.cvtColor(init, cv2.COLOR_BGR2RGB))
    return target

def PIL2cv2(init,target):
    target = cv2.cvtColor(numpy.asarray(init), cv2.COLOR_RGB2BGR)
    return target
#from .voc0712 import VOCDetection, AnnotationTransform, detection_collate, VOC_CLASSES
from .detectionDatasets import ActionDetection, AnnotationTransform, detection_collate, CLASSES
from .config import *
import cv2
import numpy as np



#-----------------------------------------------------------------------------
def base_transform_nimgs(images, size, mean):
    res_imgs = []
    seq_len = images.shape[0]
    for i in range(seq_len):
        res_imgs += [cv2.resize(images[i, :, :, :], (size, size)).astype(np.float32)]

    res_imgs = np.array(res_imgs)

    for i in range(seq_len):
        res_imgs[i, :, :, :] -= mean

    res_imgs = res_imgs.astype(np.float32)

    return res_imgs


class BaseTransform:
    def __init__(self, size, mean):
        self.size = size
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, image, boxes=None, labels=None):
        return base_transform_nimgs(image, self.size, self.mean), boxes, labels


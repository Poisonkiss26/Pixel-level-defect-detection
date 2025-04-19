from ctypes import util
import os
os.sys.path.append(r'D:\Research\WORK\CodeLibrary')

import cv2
import time
import numpy as np

import pylib.utils as utils
import pylib.cv.utils as cvutils
from pylib.gx.GXWS import GXWS as GXWS
import pylib.gx.RESULT as RESULT
from pylib.gx.SAMPLE import SAMPLE as SAMPLE


if __name__ == "__main__":  
    jsonpath = r'C:\Users\zhiqu\Documents\Galileo\Workspace\Workspace1\1\imagedrawn'  
    fileList = utils.genFileList(jsonpath, ['.json'])
    for fname in fileList:
        jsonfile = os.path.join(jsonpath, fname)
        json = utils.loadJson(jsonfile)
        print(fname)
        imgHeight = json['Height']
        imgWidth = json['Width']
        img_label,colorDict = SAMPLE.jsonToLabelImage(json, imgHeight,imgWidth,img_channels=3,ID=None)
        ofile = fname.replace('.json', '.png')
        cv2.imwrite(ofile, img_label)


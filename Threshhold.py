import numpy as np
import cv2 as cv
import os

id = 0

imgpath = r'C:\Users\USER\PycharmProjects\untitled1\SC\Cropeyes'

outpath = r'C:\Users\USER\PycharmProjects\untitled1\SC\Thresholdtmp'

for image_name in (os.listdir(imgpath))[:]:
    if image_name.endswith('.jpg'):
        print(id)
        id += 1

        im_gray = cv.imread(imgpath + '\\' + image_name, cv.IMREAD_GRAYSCALE)

        dst = cv.adaptiveThreshold(im_gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11, 100)

        cv.imwrite(outpath + '\\' + image_name, dst)
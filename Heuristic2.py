import numpy as np
import cv2 as cv
import os

id = 0

imgpath = r'C:\Users\USER\PycharmProjects\untitled1\SC\Threshold35'

#blackpix>whitepix means that closing
precision=0
wrongprecision=0
undetermined=0
for image_name in (os.listdir(imgpath))[:]:
    if image_name.endswith('.jpg'):
        print(id)
        id += 1

        im_gray = cv.imread(imgpath + '\\' + image_name, cv.IMREAD_GRAYSCALE)
        blackpix=0
        whitepix=0
        for i in range(len(im_gray)):
          for j in range(len(im_gray[i])):
            if(im_gray[i][j]<128):
             blackpix+=1
            else:
             whitepix+=1
        if(blackpix>whitepix):#predictingclosure
          if(image_name.endswith('close.jpg')):
            precision+=1
          else:
            wrongprecision+=1
        elif(blackpix<whitepix):#predictingopening
          if(image_name.endswith('open.jpg')):
            precision+=1
          else:
           wrongprecision+=1
        else:
          undetermined+=1
print("Correct prediction:",precision)
print("Wrong prediction:",wrongprecision)
print("Undetermined:",undetermined)
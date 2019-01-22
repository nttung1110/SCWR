import numpy as np
import cv2 as cv
import os
import random

cvpath = r'C:\Users\USER\PycharmProjects\untitled1\SC\venv\Lib\site-packages\cv2\data'
face_cascade = cv.CascadeClassifier(cvpath + r'\haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier(cvpath + r'\haarcascade_eye_tree_eyeglasses.xml')

# imgpath = r'D:\Programming\SC\Files\dataset_B_FacialImages\OpenFace'
outpath = r'C:\Users\USER\PycharmProjects\untitled1\SC\Cropeyes'
# img = cv.imread(imgpath + r'\Alicia_Witt_0001.jpg')
# gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

rand = list(range(3000))
random.shuffle(rand)

id = 0

pathL = [r'C:\Users\USER\PycharmProjects\untitled1\SC\venv\Lib\site-packages\cv2\data\dataset_B_FacialImages\OpenFace',
         r'C:\Users\USER\PycharmProjects\untitled1\SC\venv\Lib\site-packages\cv2\data\dataset_B_FacialImages\ClosedFace']

for imgpath in pathL:
    for image_name in (os.listdir(imgpath))[:]:
        if image_name.endswith('.jpg'):
            print(id)
            id += 1

            img = cv.imread(imgpath + '\\' + image_name)
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                side = w
                face_img = img[y: y + h, x: x + w]
                '''
                cv.imshow('eye_img', face_img)
                cv.waitKey(0)
                cv.destroyAllWindows()
                '''
                eye_img = face_img[7 * side // 32: 17 * side // 32, :].copy()
                eye_gray = cv.cvtColor(eye_img, cv.COLOR_BGR2GRAY)
                '''
                cv.imshow('eye_img', eye_img)
                cv.waitKey(0)
                cv.destroyAllWindows()
                '''
                suffix = ''
                if imgpath.endswith('OpenFace'):
                    suffix = 'open'
                else:
                    suffix = 'close'

                cv.imwrite(outpath + '\\' + str(rand[id]) + suffix + '.jpg', eye_gray)

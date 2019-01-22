import numpy as np
import cv2 as cv
import os
import scipy.optimize as opt
from numpy import loadtxt, where, zeros, e, array, log, ones, append, linspace
from pylab import scatter, show, legend, xlabel, ylabel, contour, title
from scipy.optimize import fmin_cg
def im2double(im):
    min_val = np.min(im.ravel())
    max_val = np.max(im.ravel())
    out = (im.astype('float') - min_val) / (max_val - min_val)
    return out
def sigmoid(values):
    return 1/(1+np.exp(-values))
def Cost_Function_regulariztion(theta,X,y,lambdas,m):
    h = sigmoid(X.dot(theta)/2492)
    thetaR = theta[1:]
    J = (1.0 / m) * ((-y.T.dot(np.log(h))) - ((1 - y).T.dot(np.log(1.0 - h)))) + (1 / (2.0 * m)) * (thetaR.T.dot(thetaR))
    return J
def Error(theta,X,y,m):
    error=()
def Gradient_for_f(theta,X,y,lambdas,m):
    h = sigmoid(X.dot(theta)/2492)
    dif = h - y
    gradient = (1 / m) * (np.transpose(X).dot(dif)) + (lambdas / m) * theta
    gradient[0] = gradient[0] - (lambdas / m) * theta[0]  # not regularized the bias theta0
    return gradient
id = 0
imgpath = r'C:\Users\USER\PycharmProjects\untitled1\SC\Threshold11'
undetermined=0
input=[]
Actualvalues=[]#A list representing whether the index image open or closed
for image_name in (os.listdir(imgpath))[:]:
    if image_name.endswith('.jpg'):
        print(id)
        id += 1
        im_gray = cv.imread(imgpath + '\\' + image_name, cv.IMREAD_GRAYSCALE)
        im_gray=im2double(im_gray)
        EachImage=[1]
        for i in range(len(im_gray)):
          for j in range(len(im_gray[i])):
            EachImage.append(im_gray[i][j])
        EachImage.extend([1 for i in range(2493-len(EachImage))])
        EachImage=np.array(EachImage)
        input.append(EachImage)
        if image_name.endswith('open.jpg'):
            Actualvalues.append(1)
        else:
            Actualvalues.append(0)
input=np.array(input)
Actualvalues=np.array(Actualvalues)
m=Actualvalues.shape[0]
mylambda=3.0
max1=0
for i in range(1000):
 initial_theta=np.random.randn(len(EachImage))#Number Of Feature + biased
 #xopt = fmin_cg(Cost_Function_regulariztion, fprime=Gradient_for_f,x0=initial_theta, args=(input, Actualvalues, mylambda,m), maxiter=50)
 xopt = opt.fmin_tnc(func=Cost_Function_regulariztion, x0=initial_theta, fprime=Gradient_for_f, args=(input, Actualvalues,mylambda,m))
 #xopt=opt.minimize(Cost_Function_regulariztion, x0=initial_theta, args=(input, Actualvalues, mylambda,m),method='TNC',options={'disp': True, 'minfev': 0, 'scale': None, 'rescale': -1, 'offset': None, 'gtol': -1, 'eps': 1e-08, 'eta': -1, 'maxiter': None, 'maxCGit': -1, 'mesg_num': None, 'ftol': -1, 'xtol': -1, 'stepmx': 0, 'accuracy': 0})
 count=0
 for i in range(len(input)):
  if((Actualvalues[i]==1 and sigmoid(input[i].T.dot(xopt[0])/2492)>=0.5)or(Actualvalues[i]==0and sigmoid(input[i].T.dot(xopt[0])/2492)<0.5)):
     count+=1
 if(count >= max1):
     max1=count
print(max1)










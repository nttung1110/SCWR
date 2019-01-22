import numpy as np
import cv2 as cv
import os
import scipy.optimize as opt
from numpy import loadtxt, where, zeros, e, array, log, ones, append, linspace
from pylab import scatter, show, legend, xlabel, ylabel, contour, title
import matplotlib.pyplot as plt
def im2double(im):
    min_val = np.min(im.ravel())
    max_val = np.max(im.ravel())
    out = (im.astype('float') - min_val) / (max_val - min_val)
    return out
def sigmoid(values):
    return 1/(1+np.exp(-values))
def Cost_Function_regulariztion(theta,X,y,lambdas,m):
    h = sigmoid(X.dot(theta)/15001)
    thetaR = theta[1:]
    J = (1.0 / m) * ((-y.T.dot(np.log(h))) - ((1 - y).T.dot(np.log(1.0 - h)))) + (lambdas / (2.0 * m)) * (thetaR.T.dot(thetaR))
    return np.array(J)
def Gradient_for_f(theta,X,y,lambdas,m):
    h = sigmoid(X.dot(theta)/15001)
    dif = h - y
    gradient = (1 / m) * (np.transpose(X).dot(dif)) + (lambdas / m) * theta
    gradient[0] = gradient[0] - (lambdas / m) * theta[0]  # not regularized the bias theta0
    return np.array(gradient)
def Batch_Gradient_Descent(theta,X,y,lambdas,m,learningrate):
    tmp=Gradient_for_f(theta,X,y,lambdas,m)
    new_theta=theta-learningrate*tmp
    return new_theta
def Batch_Gradient_Descent_With_Momentum(theta,X,y,lambdas,m,learningrate,velocity_old):
    updated_theta=theta
    grad=Gradient_for_f(theta,X,y,lambdas,m)
    gamma=0.9
    velocity_new=gamma*velocity_old+learningrate*grad
    updated_theta=updated_theta-velocity_new
    return updated_theta,velocity_new
def error_calculating(theta,X,Y):
    error=0
    for i in range(len(X)):
        if ((Y[i] == 0 and sigmoid(X[i].T.dot(theta) / 15001) >= 0.5) or (
                Y[i] == 1 and sigmoid(X[i].T.dot(theta) / 15001) < 0.5)):
            error+=1
    return error/len(X)
def training_progress_Nomomentum(X,Y,m,initial_theta,max_iteration,learning_rates):
    updated_theta = initial_theta
    for i in range(max_iteration):
        print("Current cost:", Cost_Function_regulariztion(updated_theta, X, Y, 0, m))
        updated_theta = Batch_Gradient_Descent(updated_theta, X, Y, 0, m, learning_rates)
    return updated_theta
def training_progress_Withmomentum(X,Y,m,initial_theta,max_iteration,learning_rates,XVAL,YVAL,XTEST,YTEST):
    updated_theta = initial_theta
    velocity_old=np.zeros_like(initial_theta)
    Accuracy_Train_per_iter=[0]
    Iter=[0]
    Accuracy_Validate_per_iter=[0]
    Accuracy_Test_per_iter = [0]
    for i in range(max_iteration):
        print("Current cost:", Cost_Function_regulariztion(updated_theta, X, Y, 0, m))
        updated_theta,velocity_old = Batch_Gradient_Descent_With_Momentum(updated_theta, X, Y, 0, m, learning_rates,velocity_old)
        Accuracy_Train_per_iter.append(calculating_Accuracy(X,Y,m,updated_theta,len(X)))
        Accuracy_Validate_per_iter.append(calculating_Accuracy(XVAL,YVAL,m,updated_theta,len(XVAL)))
        Accuracy_Test_per_iter.append(calculating_Accuracy(XTEST,YTEST,m,updated_theta,len(XTEST)))
        Iter.append(i)
    return updated_theta,Accuracy_Train_per_iter,Accuracy_Validate_per_iter,Accuracy_Test_per_iter,Iter
def calculating_Accuracy(X,Y,m,optimize_theta,num_images):
    count=0
    for i in range(len(X)):
        if ((Y[i] == 1 and sigmoid(X[i].T.dot(optimize_theta) / 15001) >= 0.5) or (
                Y[i] == 0 and sigmoid(X[i].T.dot(optimize_theta) / 15001) < 0.5)):
            count += 1
    return count/num_images
def calculating_recall(X,Y,m,optimize_theta):
    count_TP=0
    count_FN=0
    for i in range(len(X)):
        if ((Y[i] == 1 and sigmoid(X[i].T.dot(optimize_theta) / 15001) >= 0.5)):
            count_TP += 1
        if((Y[i]==1 and sigmoid(X[i].T.dot(optimize_theta)/15001)<0.5)):
           count_FN+=1
    return count_TP/(count_TP+count_FN)
def calculating_precision(X,Y,m,optimize_theta):
    count_TP=0
    count_FP=0
    for i in range(len(X)):
        if ((Y[i] == 1 and sigmoid(X[i].T.dot(optimize_theta) / 15001) >= 0.5)):
            count_TP += 1
        if((Y[i]==0 and sigmoid(X[i].T.dot(optimize_theta)/15001)>=0.5)):
           count_FP+=1
    return count_TP/(count_TP+count_FP)
def calculating_F1Score(precision,recall):
    return 2*precision*recall/(precision+recall)
id = 0
imgpath = r'C:\Users\USER\PycharmProjects\untitled1\SC\AllData'
undetermined=0
input=[]#1400 images
validate=[]#500 images
test=[]#523
Actualvalues=[]#A list representing whether the index image open or closed(training)
ActualvaluesValidate=[]#validateset
ActualvaluesTest=[]#testset
for image_name in (os.listdir(imgpath))[:]:
    if image_name.endswith('.jpg'):
        print(id)
        id += 1
        img = cv.imread(imgpath + '\\' + image_name)
        img=im2double(img)
        img = np.reshape(img, 30000)#30000 features
        img=img[:15000]
        EachImage=[1]#Bias Feature
        for i in range(len(img)):
         EachImage.append(img[i])
        if(id<1400):#training
            EachImage=np.array(EachImage)
            input.append(EachImage)
            if image_name.endswith('open.jpg'):
                Actualvalues.append(1)
            else:
                Actualvalues.append(0)
        elif (id >= 1400 and id < 1900):#validate
            EachImage = np.array(EachImage)
            validate.append(EachImage)
            if image_name.endswith('open.jpg'):
                ActualvaluesValidate.append(1)
            else:
                ActualvaluesValidate.append(0)
        else:#test
            EachImage = np.array(EachImage)
            test.append(EachImage)
            if image_name.endswith('open.jpg'):
                ActualvaluesTest.append(1)
            else:
                ActualvaluesTest.append(0)
input=np.array(input)
validate=np.array(validate)
test=np.array(test)
Actualvalues=np.array(Actualvalues)
ActualvaluesValidate=np.array(ActualvaluesValidate)
ActualvaluesTest=np.array(ActualvaluesTest)
m=Actualvalues.shape[0]
mylambda=3.0
max1=10
#for i in range(1000):
initial_theta=np.random.randn(len(EachImage))#Number Of Feature + biased
learning_rates=100
maxiteration=200
count=0
error_train=[]
error_validate=[]
Accuracy_Train=[]
Iteration=[]
Accuracy_Validate=[]
Accuracy_Test=[]
optimize_theta,Accuracy_Train,Accuracy_Validate,Accuracy_Test,Iteration=training_progress_Withmomentum(input,Actualvalues,m,initial_theta,maxiteration,learning_rates,validate,ActualvaluesValidate,test,ActualvaluesTest)
plt.xlabel("Iteration")
plt.ylabel("Accuracy")
plt.plot(Iteration,Accuracy_Train,label="Training Accuracy")
plt.plot(Iteration,Accuracy_Validate,label="Validate Accuracy")
plt.plot(Iteration,Accuracy_Test,label="Test Accuracy")
plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=2, mode="expand", borderaxespad=0.)
plt.show()
precision_test=calculating_precision(test,ActualvaluesTest,m,optimize_theta)
recall_test=calculating_recall(test,ActualvaluesTest,m,optimize_theta)
print('Precision test:',precision_test)
print('Recall test:',recall_test)
print('F1 Score test:',calculating_F1Score(precision_test,recall_test))
precision_validate=calculating_precision(validate,ActualvaluesValidate,m,optimize_theta)
recall_validate=calculating_recall(validate,ActualvaluesValidate,m,optimize_theta)
print('Precision test:',precision_validate)
print('Recall test:',recall_validate)
print('F1 Score test:',calculating_F1Score(precision_validate,recall_validate))
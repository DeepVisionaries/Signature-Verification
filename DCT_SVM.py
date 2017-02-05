import numpy as np
import cv2
#from matplotlib import pyplot as plt
from scipy import fftpack
from sklearn.svm import LinearSVC
import glob,os,collections


filenames = list()
test_filenames=list()

dctCoefficients = list()
test_dctCoefficients=list()

Y = list()
test_Y=list()

os.chdir(r'.\Training\1')
for file in glob.glob("*.png"):
    filenames.append(file)
    Y.append(1)

os.chdir(r'..')    
os.chdir(r'.\0')
for file in glob.glob("*.png"):
    filenames.append(file)
    Y.append(0)

os.chdir(r'..\..')

os.chdir(r'.\Testing\1')
for file in glob.glob("*.png"):
    test_filenames.append(file)
    test_Y.append(1) 
    
os.chdir(r'..')    
os.chdir(r'.\0')
for file in glob.glob("*.png"):
    test_filenames.append(file)
    test_Y.append(0)
    
os.chdir(r'..\..')   
os.chdir(r'.\Training\C')
for i in range(len(filenames)):
    name=filenames[i]
    img=cv2.imread(name,0)
    row,col= img.shape[:2]
    ratio=float(row)/float(col)
    if col>row:
        dim=(512,int(ratio*512))
    else:
        ratio=1/ratio
        dim=(int(ratio*512),512)
    im2=cv2.resize(img,dim,interpolation=cv2.INTER_AREA)
    
    ret,binarised = cv2.threshold(im2,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    if col>row:
        resized=cv2.copyMakeBorder(binarised,top=0,bottom=512-binarised.shape[0],left=0,right=0,borderType= cv2.BORDER_CONSTANT, value=255)
    else:
        resized=cv2.copyMakeBorder(binarised,top=0,bottom=0,left=0,right=512-binarised.shape[1],borderType= cv2.BORDER_CONSTANT, value=255)
    
    kernel = np.ones((5,5),np.uint8)
    resized = cv2.erode(resized,kernel,iterations = 1)
    
    resized=np.array(resized,dtype=np.float)
    
    dct=fftpack.dct(fftpack.dct(resized.T, norm='ortho').T, norm='ortho')
    dct_copy = dct.copy()
    dct_copy[64:,:] = 0
    dct_copy[:,64:] = 0
    dctCoefficients.append(dct_copy.ravel())
    
os.chdir(r'..\..')       
os.chdir(r'.\Testing\C')
for i in range(len(test_filenames)):
    name=test_filenames[i]
    img=cv2.imread(name,0)
    row,col= img.shape[:2]
    ratio=float(row)/float(col)
    if col>row:
        dim=(512,int(ratio*512))
    else:
        ratio=1/ratio
        dim=(int(ratio*512),512)
    im2=cv2.resize(img,dim,interpolation=cv2.INTER_AREA)
    
    ret,binarised = cv2.threshold(im2,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    if col>row:
        resized=cv2.copyMakeBorder(binarised,top=0,bottom=512-binarised.shape[0],left=0,right=0,borderType= cv2.BORDER_CONSTANT, value=255)
    else:
        resized=cv2.copyMakeBorder(binarised,top=0,bottom=0,left=0,right=512-binarised.shape[1],borderType= cv2.BORDER_CONSTANT, value=255)
    
    kernel = np.ones((5,5),np.uint8)
    resized = cv2.erode(resized,kernel,iterations = 1)
    
    resized=np.array(resized,dtype=np.float)
    
    dct=fftpack.dct(fftpack.dct(resized.T, norm='ortho').T, norm='ortho')
    dct_copy = dct.copy()
    dct_copy[64:,:] = 0
    dct_copy[:,64:] = 0
    test_dctCoefficients.append(dct_copy.ravel())

    
model = LinearSVC()
model.fit(dctCoefficients,Y)

predict=model.predict(test_dctCoefficients)

print collections.Counter(predict-test_Y)

score= model.score(test_dctCoefficients,test_Y)
print score
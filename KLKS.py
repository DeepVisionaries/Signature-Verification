import numpy as np
import pandas as pd
import cv2
from scipy import fftpack
from scipy.spatial.distance import pdist as dist
import pickle

Ref_df=pd.read_csv('Reference.data',names=['Reference'])
References=list(Ref_df['Reference'])
dct_Reference=[]

for image in References:
    name='./Images/'+image
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
    dct_copy=dct[:20,:20]
    dct_Reference.append(dct_copy.reshape(400))

Reference_dist=dist(dct_Reference,metric='cosine')

model=open('model','wb')
pickle.dump([len(References),Reference_dist,dct_Reference],model)
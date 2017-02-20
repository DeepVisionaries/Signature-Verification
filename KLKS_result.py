import numpy as np
from scipy.stats import ks_2samp as ks_test
from sklearn.metrics import confusion_matrix
from scipy.stats import entropy as kl_test
from math import exp
import pickle

with open('model','rb') as model:
    ref_size,Reference_dist,dct_Reference=pickle.load(model)
with open('result','rb') as result:
    test_size,Test_dist,true=pickle.load(result)

predicted=[]
for index in range(test_size):
    stats,p_ks=ks_test(Reference_dist,Test_dist[index])
    
    hist,bins=np.histogram(Test_dist[index],bins='auto')
    pd_test=hist/(float(hist.sum(0)))
    hist,bins=np.histogram(Reference_dist,bins)
    pd_ref=hist/(float(hist.sum(0)))
    l=kl_test(pd_ref,pd_test)
    p_kl=exp(-l)
    
    prob=(p_kl+p_ks)/2.0

    if prob>0.6:
        predicted.append(0)
    else:
        predicted.append(1)

a=np.array(true)
b=np.array(predicted)

cm=confusion_matrix(a,b)
tp=cm[0][0]
tn=cm[1][1]
fp=cm[1][0]
fn=cm[0][1]
print 'tp:',tp
print 'fp:',fp
print 'fn:',fn
print 'tn:',tn

frr=fn/(float(tp+fn))
far=fp/(float(fp+tn))

print 'error:' ,(far+frr)/2
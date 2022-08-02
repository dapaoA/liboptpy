import numpy as np

from liboptpy.data_preparing import making_gausses

import scipy.sparse as sp

import time
n = 2
a = np.asarray([1.0,3.0])
b = np.asarray([1.0,3.0])
epsilon = 0.01
round = 5000
tau = 100

dim_a = np.shape(a)[0]
dim_b = np.shape(b)[0]
jHr = np.arange(dim_a * dim_b)
iHr = np.repeat(np.arange(dim_a), dim_b)
jHc = np.arange(dim_a * dim_b)
iHc = np.tile(np.arange(dim_b), dim_a)
Hr = sp.csc_matrix((np.ones(dim_a * dim_b), (iHr, jHr)),
                   shape=(dim_a, dim_a * dim_b))
Hc = sp.csc_matrix((np.ones(dim_a * dim_b), (iHc, jHc)),
                   shape=(dim_b, dim_a * dim_b))


x1 = np.asarray([1.0,3.0,1.0,3.0])
x2 = np.asarray([1.0,3.0,1.0,3.0])

def KL(x,y):
    return np.dot(x,np.log(x/y))-x.sum()+y.sum()

KL(Hr.dot(x1),Hr.dot(x2))+KL(Hc.dot(x1),Hc.dot(x2)) - KL(x1,x2)


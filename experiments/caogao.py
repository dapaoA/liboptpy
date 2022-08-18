import numpy as np
from liboptpy.data_preparing import making_gausses
import scipy.sparse as sp
import liboptpy.screening.screenning as sc

n = 10
a,b,M = making_gausses(n)
epsilon = 0.01
round = 10000
tau = 1000

dim_a = np.shape(a)[0]
dim_b = np.shape(b)[0]
m = M.flatten()
jHr = np.arange(dim_a * dim_b)
iHr = np.repeat(np.arange(dim_a), dim_b)
jHc = np.arange(dim_a * dim_b)
iHc = np.tile(np.arange(dim_b), dim_a)
Hr = sp.csc_matrix((np.ones(dim_a * dim_b), (iHr, jHr)),
                   shape=(dim_a, dim_a * dim_b))
Hc = sp.csc_matrix((np.ones(dim_a * dim_b), (iHc, jHc)),
                   shape=(dim_b, dim_a * dim_b))
Hra = Hr.T.dot(a)
Hrb = Hr.T.dot(b)
HrHr = Hr.T.dot(Hr)
HcHc = Hc.T.dot(Hc)
Hca = Hc.T.dot(a)
Hcb = Hc.T.dot(b)

X = sp.vstack((Hc,Hr))
y = np.concatenate((a,b),axis=0)

sc1 = sc.safe_screening(0,0,0,X,y,1/tau,reg="l1")

sc1.update(np.zeros(100))


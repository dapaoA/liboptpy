import numpy as np
import liboptpy.base_optimizer as base
import liboptpy.constr_solvers as cs
import liboptpy.step_size as ss
from liboptpy.data_preparing import making_mnist_with_noise
from liboptpy.data_preparing import making_uot_gausses
from liboptpy.data_preparing import making_gausses
import ot
import matplotlib.pyplot as plt
import scipy.sparse as sp
import time
from Plot_Function import f_speed_log,f_speed_linear,f_time_log
from functions import UOT_kl
from functions import grad_uot_kl as g
from functions import uot_kl_proximal_K_entropy as pk
from functions import uot_kl_proximal_B_entropy as pb
from functions import marginal_kl as makl
from functions import marginal_l2 as mal2
from functions import initial_c
from functions import reconstructe_c as rc
import liboptpy.screening.screenning as sc

#this file is used to disscuss about the convergence of the algorithm according to the tau
# 关于nestrov加速要不要重启
#貌似重启了下降快，但是不准，不重启下降慢，但是准。。。
class Sinkhornalg:
    def __init__(self,log):
        self.log = log
    def get_convergence(self):
        return self.log['primal']
    def get_time(self):
        return self.log['time']

plt.rc("text", usetex=True)
fontsize = 24
figsize = (16, 14)
import seaborn as sns
sns.set_context("talk")
#from tqdm import tqdm

n = 10
a,b,M = making_uot_gausses(n)
epsilon = 0.1
round = 100
tau = 100

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


def func_opt(t, m):


    return np.dot(t,m)

f_opt = lambda x: func_opt(x, m,)



# linear programming
# times = time.time()
# G0 = ot.emd(a, b, M)
# timee = time.time()
# print("lp time: ",timee-times)
# opt = f_opt(G0.flatten())

f = lambda x: UOT_kl(x,a,b,m,tau,Hc,Hr)


grad = lambda x: g(x, a, b, Hc,Hr, dim_a,dim_b)


proximalB = lambda x,h,grad: pb(m/tau,x,h,grad)

proximalK = lambda x,h,grad: pk(m/tau,x,h,grad)



mkl = lambda x: makl(x, a, b,Hc,Hr)
ml2 = lambda x: mal2(x, a, b,Hc,Hr)
def sparsity(t):
    return np.count_nonzero(t==0)/len(t)

spa = lambda x: sparsity(x)



tau = 0.1
trans1 = sc.safe_screening(np.ones_like(m),sp.vstack((Hc,Hr)),np.concatenate((a,b)),m,1/tau)
trans1_m = trans1.update()

plt.imshow(trans1_m.reshape(10,10))
plt.show()


tau = 0.5
trans1 = sc.safe_screening(np.ones_like(m),sp.vstack((Hc,Hr)),np.concatenate((a,b)),m,1/tau)
trans1_m = trans1.update()

plt.imshow(trans1_m.reshape(10,10))
plt.show()
# a_copy,b_copy,M_new,a_exist_id,b_exist_id,M_def = initial_c(a,b,M)
#
# Gs = ot.unbalanced.sinkhorn_unbalanced(a_copy,b_copy,M_new,epsilon,tau)
Gs_org = ot.unbalanced.sinkhorn_unbalanced(a,b,M,epsilon,tau)
#
# Gs_re = rc(a_copy,b_copy,Gs,a_exist_id,b_exist_id,M_def)
#
#
plt.plot(a)
plt.plot(b)
plt.show()
# plt.imshow(Gs)
# plt.show()
# plt.imshow(Gs_re)
# plt.show()
plt.imshow(Gs_org)
plt.show()
#
#
# print("value of f_uot of initial method is: ",f(Gs_re.flatten().squeeze()))
# print("value of f_uot of sinkhorn method is: ",f(Gs_org.flatten()))
# print("value of f_opt of initial method is: ",f_opt(Gs_re.flatten()))
# print("value of f_opt of sinkhorn method is: ",f_opt(Gs_org.flatten()))
import numpy as np
import liboptpy.base_optimizer as base
import liboptpy.constr_solvers as cs
import liboptpy.step_size as ss
from liboptpy.data_preparing import making_mnist_with_noise
from liboptpy.data_preparing import making_gausses
import ot
import matplotlib.pyplot as plt
import scipy.sparse as sp
import time
from Plot_Function import f_speed_log,f_speed_linear,f_time_log
from functions import semi_l2, grad_semi_l2
from functions import projection_simplex as ps
from functions import kl_projection as kp



# 这个是看看让armoji起始搜索速率可以变化会不会加快收敛的，，，
plt.rc("text", usetex=True)
fontsize = 24
figsize = (8, 6)
import seaborn as sns
sns.set_context("talk")
#from tqdm import tqdm

n = 1000
a,b,M = making_gausses(n)
epsilon = 0.01
round = 10000
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
times = time.time()
G0 = ot.emd(a, b, M)
timee = time.time()
print("lp time: ",timee-times)
opt = f_opt(G0.flatten())

f = lambda x: semi_l2(x, a, b, m, tau, Hc)


grad = lambda x: grad_semi_l2(x, a, b, m, tau, dim_a, dim_b, HcHc, Hcb)


projection_simplex = lambda x: ps(x, dim_a, dim_b, a, axis=1)

kl_projection = lambda x: kp(x, a, b, dim_a, dim_b)


def marginal(t,a,b,Hc,Hr):
    return np.linalg.norm(Hc.dot(t),ord=1)+np.linalg.norm(Hr.dot(t),ord=1)-2

mar = lambda x: marginal(x, a, b)

def sparsity(t):
    return np.count_nonzero(t==0)/len(t)

spa = lambda x: sparsity(x)
#"FW": cs.FrankWolfe(f, grad, linsolver, ss.Backtracking(rule_type="Armijo", rho=0.5, beta=0.1, init_alpha=1.)),
#           "PGD": cs.ProjectedGD(f, grad, projection_simplex, ss.Backtracking(rule_type="Armijo", rho=0.5, beta=0.1, init_alpha=1.)),



methods = {
#"FISTA": cs.FISTA(f, grad, projection_simplex, ss.Backtracking_Nestrov(rule_type="Armijo", rho=0.5, beta=0.001, init_alpha=10.)),
#"AMD": cs.AMD(f, grad, kl_projection, ss.Backtracking_Nestrov(rule_type="Armijo", rho=0.5, beta=0.001, init_alpha=10.)),
#"PGD": cs.ProjectedGD(f, grad, projection_simplex, ss.Backtracking(rule_type="Armijo", rho=0.5, beta=0.001, init_alpha=1.)),
#"MD": cs.MirrorD(f, grad, kl_projection, ss.Backtracking(rule_type="Armijo", rho=0.5, beta=0.001, init_alpha=10.)),
"FISTAd": cs.FISTA(f, grad, projection_simplex, ss.D_Backtracking_Nestrov(rule_type="Armijo", rho=0.5, beta=0.001, init_alpha=10.)),
"AMDd": cs.AMD(f, grad, kl_projection, ss.D_Backtracking_Nestrov(rule_type="Armijo", rho=0.5, beta=0.001, init_alpha=10.)),
"PGDd": cs.ProjectedGD(f, grad, projection_simplex, ss.D_Backtracking(rule_type="Armijo", rho=0.5, beta=0.001, init_alpha=1.)),
"MDd": cs.MirrorD(f, grad, kl_projection, ss.D_Backtracking(rule_type="Armijo", rho=0.5, beta=0.001, init_alpha=10.)),
#"AMD-e-c2": cs.AMD_E(f, grad, kl_projection, ss.ConstantInvIterStepSize(1)),
#"AMD-e-c1": cs.AMD_E(f, grad, kl_projection, ss.Backtracking_Bregman_Nestrov(rule_type="Armijo", rho=0.5, beta=0.0001, init_alpha=1.)),
          }

# n =100 tau =100
# Best convergence for FISTA is 0.01
# Best convergence for AMD is 1
# Best convergence for AMD_E is 1
# Best convergence for PGD is 0.001

x0 = np.ones((dim_a,dim_b)).flatten()/(dim_a*dim_b)
max_iter = 5000
tol = 1e-6


for m_name in methods:
    print("\t", m_name)
    time_s = time.time()
    x = methods[m_name].solve(x0=x0, max_iter=max_iter, tol=tol, disp=1)
    time_e = time.time()
    print(m_name,"time costs: ", time_e - time_s, " s")

plt.figure(figsize=figsize)

f_speed_log(methods,f,"f")
f_speed_log(methods,f_opt,"opt")
f_time_log(methods,f,"time")
#f_speed_linear(methods,mar,"marginal error")
#f_speed_linear(methods,spa,"sparsity")
i = 2

for m_name in methods:
    x = methods[m_name].get_convergence()[-1]
    plt.imshow(x.reshape((dim_a,dim_b)), cmap='hot', interpolation='nearest')
    plt.title(m_name)
    plt.show()
    i+=1


# time_s = time.time()
# t2, t_list2, g_list2 = ot.regpath.regularization_path(a, b, M, reg=1/tau,
#                                                       semi_relaxed=True)
# time_e = time.time()
# print("nips", ": ", time_e - time_s)
#
# plt.subplot(1, 2 + len(methods), 1 + len(methods))
# plt.imshow(t2.reshape((dim_a, dim_b)), cmap='hot', interpolation='nearest')
# plt.title("nips")
# plt.show()

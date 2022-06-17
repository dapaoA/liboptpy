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
from Plot_Function import f_speed_log,f_speed_linear,f_conv_comp
from functions import semi_l2, grad_semi_l2
from functions import linsolver,kl_projection,projection_simplex


plt.rc("text", usetex=True)
fontsize = 24
figsize = (8, 6)
import seaborn as sns
sns.set_context("talk")
#from tqdm import tqdm

n = 200
a,b,M = making_gausses(n)
round = 5000
tau = 100
m = M.flatten()
def func_opt(t, m):


    return np.dot(t,m)

f_opt = lambda x: func_opt(x, m,)



# linear programming
times = time.time()
G0 = ot.emd(a, b, M)
timee = time.time()
print("lp time: ",timee-times)
opt = f_opt(G0.flatten())
stopThr_list = [1e-1,1e-2,1e-3,1e-4,1e-5]

# sinkhorn
timelist ={}
error_list ={}
lambd = 1e-3
timelist["sinkhorn"] = []
error_list["sinkhorn"] = []
for tol in stopThr_list:
    times = time.time()
    Gs = ot.sinkhorn(a, b, M, lambd,numItermax=10000,stopThr=tol, verbose=True)
    timee = time.time()
    timelist["sinkhorn"].append(timee-times)
    print("sinkhorn time: ",timee-times)
    error_list["sinkhorn"].append(np.fabs(f_opt(Gs.flatten())-opt))

# UOT
timelist["uot"] = []
error_list["uot"] = []
epsilon = 1e-3 # entropy parameter
alpha = 1000.  # Unbalanced KL relaxation parameter
for tol in stopThr_list:
    times = time.time()
    Gs = ot.unbalanced.sinkhorn_unbalanced(a, b, M, epsilon, alpha,numItermax=10000,stopThr=tol, verbose=True)
    timee = time.time()
    timelist["uot"].append(timee-times)
    print("uot time: ",timee-times)
    error_list["uot"].append(np.fabs(f_opt(Gs.flatten())-opt))




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
Hra = Hr.T.dot(a)
Hrb = Hr.T.dot(b)
HrHr = Hr.T.dot(Hr)
HcHc = Hc.T.dot(Hc)
Hca = Hc.T.dot(a)
Hcb = Hc.T.dot(b)

f = lambda x: semi_l2(x, a, b, m, tau)


grad = lambda x: grad_semi_l2(x, a, b, m, tau)

def marginal(t,a,b):
    return np.linalg.norm(Hc.dot(t),ord=1)+np.linalg.norm(Hr.dot(t),ord=1)-2

mar = lambda x: marginal(x, a, b)

def sparsity(t):
    return np.count_nonzero(t==0)/len(t)

spa = lambda x: sparsity(x)
#"FW": cs.FrankWolfe(f, grad, linsolver, ss.Backtracking(rule_type="Armijo", rho=0.5, beta=0.1, init_alpha=1.)),
#           "PGD": cs.ProjectedGD(f, grad, projection_simplex, ss.Backtracking(rule_type="Armijo", rho=0.5, beta=0.1, init_alpha=1.)),



methods = {
"FISTA": cs.FISTA(f, grad, projection_simplex, ss.Backtracking_Nestrov(rule_type="Armijo", rho=0.5, beta=0.1, init_alpha=1.)),
"AMD": cs.AMD(f, grad, kl_projection, ss.Backtracking_Nestrov(rule_type="Armijo", rho=0.5, beta=0.1, init_alpha=1.)),
"PGD": cs.ProjectedGD(f, grad, projection_simplex, ss.Backtracking(rule_type="Armijo", rho=0.5, beta=0.1, init_alpha=1.)),
"MD": cs.MirrorD(f, grad, kl_projection, ss.Backtracking(rule_type="Armijo", rho=0.5, beta=0.1, init_alpha=1.)),
"AMD-e-c1": cs.AMD_E(f, grad, kl_projection, ss.ConstantInvIterStepSize(1)),
          }

# n =100 tau =100
# Best convergence for FISTA is 0.01
# Best convergence for AMD is 1
# Best convergence for AMD_E is 1
# Best convergence for PGD is 0.001

x0 = np.ones((dim_a,dim_b)).flatten()/(dim_a*dim_b)
max_iter = 5000
tollist = [1e-1, 1e-2, 1e-3, 1e-4]
time_dic = {}
error_dic = {}
for m_name in methods:
    time_dic[m_name] =[]
    error_dic[m_name]=[]
    for tol in tollist:
        print("\t", m_name)
        time_s = time.time()
        x = methods[m_name].solve(x0=x0, max_iter=max_iter, tol=tol, disp=1)
        time_e = time.time()
        time_dic[m_name].append(time_e - time_s)
        error_dic[m_name].append(np.fabs(f_opt(x)-opt))
        print(m_name,"time costs: ", time_e - time_s, " s")

for m_name in methods:
    plt.plot([1 / x ** 0.5 for x in error_dic[m_name]], time_dic[m_name], label=m_name)
plt.plot([1/x**0.5 for x in error_list["sinkhorn"]],timelist["sinkhorn"],label = "sinkhorn")
plt.plot([1/x**0.5 for x in error_list["uot"]],timelist["uot"],label = "uot")
plt.legend()
plt.ylabel("time, $k$")
plt.xlabel("f - f* in 1/x**0.5")
plt.title(" convergence speed")
plt.show()
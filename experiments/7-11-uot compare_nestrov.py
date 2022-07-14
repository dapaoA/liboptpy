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
from functions import UOT_kl
from functions import grad_uot_kl as g
from functions import uot_kl_proximal_K_entropy as pk
from functions import uot_kl_proximal_B_entropy as pb

plt.rc("text", usetex=True)
fontsize = 24
figsize = (8, 6)
import seaborn as sns
sns.set_context("talk")
#from tqdm import tqdm

n = 100
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


def func_opt(t, m):


    return np.dot(t,m)

f_opt = lambda x: func_opt(x, m,)



# linear programming
times = time.time()
G0 = ot.emd(a, b, M)
timee = time.time()
print("lp time: ",timee-times)
opt = f_opt(G0.flatten())

f = lambda x: UOT_kl(x,a,b,m,tau,Hc,Hr)


grad = lambda x: g(x, a, b, Hc,Hr, dim_a,dim_b)


proximalB = lambda x,h,grad: pb(m/tau,x,h,grad)

proximalK = lambda x,h,grad: pk(m/tau,x,h,grad)

def marginal(t,a,b,Hc,Hr):
    return np.linalg.norm(Hc.dot(t),ord=1)+np.linalg.norm(Hr.dot(t),ord=1)-2

mar = lambda x: marginal(x, a, b)

def sparsity(t):
    return np.count_nonzero(t==0)/len(t)

spa = lambda x: sparsity(x)
#"FW": cs.FrankWolfe(f, grad, linsolver, ss.Backtracking(rule_type="Armijo", rho=0.5, beta=0.1, init_alpha=1.)),
#           "PGD": cs.ProjectedGD(f, grad, projection_simplex, ss.Backtracking(rule_type="Armijo", rho=0.5, beta=0.1, init_alpha=1.)),



methods = {
#"FISTA": cs.FISTA(f, grad, projection_simplex, ss.Backtracking_Nestrov(rule_type="Armijo", rho=0.5, beta=0.001, init_alpha=1.)),
#"AMD": cs.AMD(f, grad, kl_projection, ss.Backtracking_Nestrov(rule_type="Armijo", rho=0.5, beta=0.001, init_alpha=1.)),
#"PGD": cs.ProjectedGD(f, grad, projection_simplex, ss.Backtracking(rule_type="Armijo", rho=0.5, beta=0.001, init_alpha=1.)),
"MD-KLb": cs.MirrorD_gen(f, grad, proximalB, ss.ConstantStepSize(0.25)),
"MD-KLk": cs.MirrorD_gen(f, grad, proximalK, ss.ConstantStepSize(0.5)),
#"AMD-KLb": cs.AMD_gen(f, grad, proximalB, ss.D_Backtracking_Nestrov(rule_type="Armijo", rho=0.5, beta=0.001, init_alpha=0.01)),
#"AMD-KLk": cs.AMD_gen(f, grad, proximalK, ss.D_Backtracking_Nestrov(rule_type="Armijo", rho=0.5, beta=0.001, init_alpha=0.01)),
#"AMD-e-c2": cs.AMD_E(f, grad, kl_projection, ss.ConstantInvIterStepSize(1)),
#"AMD-e-c1": cs.AMD_E(f, grad, kl_projection, ss.Backtracking_Bregman_Nestrov(rule_type="Armijo", rho=0.5, beta=0.0001, init_alpha=1.)),
          }

# n =100 tau =100
# Best convergence for FISTA is 0.01
# Best convergence for AMD is 1
# Best convergence for AMD_E is 1
# Best convergence for PGD is 0.001

x0 = np.ones((dim_a,dim_b)).flatten()/(dim_a*dim_b)
max_iter = 10000
tol = 1e-6


# for m_name in methods:
#     print("\t", m_name)
#     time_s = time.time()
#     x = methods[m_name].solve(x0=x0, max_iter=max_iter, tol=tol, disp=1)
#     time_e = time.time()
#     print(m_name,"time costs: ", time_e - time_s, " s")



epsilon = 1e-3 # entropy parameter
alpha = 1000.  # Unbalanced KL relaxation parameter
round = 20000
times = time.time()
Gs = ot.unbalanced.sinkhorn_unbalanced(a, b, M, epsilon, alpha, numItermax=round, stopThr=tol, verbose=True)
timee = time.time()
print("uot time: ", timee - times)

time_s = time.time()
G1kl,log50 = ot.unbalanced.mm_unbalanced(a, b, M, tau, 'kl',numItermax=round,log=True)
time_e = time.time()
print( "time costs: ", time_e - time_s, " s")
time_s = time.time()
G1kla,log50a = ot.unbalanced.mm_unbalanced_revised_nestrov(a, b, M, tau,l_rate=0.5,div= 'kl',numItermax=round,log=True)
time_e = time.time()
print( "time costs: ", time_e - time_s, " s")


# time_s = time.time()
# G1l2 = ot.unbalanced.mm_unbalanced(a, b, M, tau, 'l2',numItermax=10000)
# time_e = time.time()
# print(m_name, "time costs: ", time_e - time_s, " s")

time_s = time.time()
Gmykl25,log25 = ot.unbalanced.mm_unbalanced_revised(a, b, M, tau,l_rate=0.25, div='kl',numItermax=round,log=True)
time_e = time.time()
print( "time costs: ", time_e - time_s, " s")

time_s = time.time()
Gmykl10,log10 = ot.unbalanced.mm_unbalanced_revised(a, b, M, tau,l_rate=0.1, div='kl',numItermax=round,log=True)
time_e = time.time()
print( "time costs: ", time_e - time_s, " s")

time_s = time.time()
Gmykla,log10a = ot.unbalanced.mm_unbalanced_revised_nestrov(a, b, M, tau,l_rate=0.1, div='kl',numItermax=round,log=True)
time_e = time.time()
print( "time costs: ", time_e - time_s, " s")

time_s = time.time()
Gmykl25a,log25a = ot.unbalanced.mm_unbalanced_revised_nestrov(a, b, M, tau,l_rate=0.25, div='kl',numItermax=round,log=True)
time_e = time.time()
print( "time costs: ", time_e - time_s, " s")

pot_names = {'mmkl':log50,
             'mmkla': log50a,
                'myrevise25':log25,
             'myrevise25a': log25a,
             'myrevise10': log10,
             'myrevise10a':log10a
             }


convergence = {'uot-entropy':Gs,
                'mmkl':G1kl,
             'mmkla': G1kla,
                'myrevise25':Gmykl25,
             'myrevise25a': Gmykl25a,
             'myrevise10': Gmykl10,
             'myrevise10a':Gmykla
             }
plt.figure(figsize=figsize)
for con in convergence:
    plt.plot([np.log(f(x.flatten())) for x in convergence[con]['G'][::100]], label=con)
plt.legend()
plt.show()

for con in convergence:
    plt.plot([np.log(f(x.flatten())) for x in convergence[con]['G'][::100]], label=con)
plt.legend()
plt.show()


f_speed_log(methods,f,"f")
f_speed_log(methods,f_opt,"opt")
f_time_log(methods,f,"time")
#f_speed_linear(methods,mar,"marginal error")
#f_speed_linear(methods,spa,"sparsity")
i = 2

# for m_name in methods:
#     x = methods[m_name].get_convergence()[-1]
#     plt.imshow(x.reshape((dim_a,dim_b)), cmap='hot', interpolation='nearest')
#     plt.title(m_name)
#     plt.show()
#     i+=1
for m_name in pot_names:
    x = pot_names[m_name]
    plt.imshow(x, cmap='hot', interpolation='nearest')
    plt.title(m_name)
    plt.show()
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

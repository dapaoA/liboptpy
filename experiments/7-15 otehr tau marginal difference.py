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
from functions import marginal_kl as makl
from functions import marginal_l2 as mal2


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



mkl = lambda x: makl(x, a, b,Hc,Hr)
ml2 = lambda x: mal2(x, a, b,Hc,Hr)
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
  # Unbalanced KL relaxation parameter
round = 10000
times = time.time()
Gs,loguot = ot.unbalanced.sinkhorn_unbalanced(a, b, M, epsilon, tau/10, numItermax=10000, stopThr=tol, verbose=True,log=True)
timee = time.time()
print("uot time: ", timee - times)

nx = ot.backend.get_backend(M, a, b)
K = nx.exp(M / (-epsilon))
loguot['G'] = []
for i in range(len(loguot['u'])):
     loguot['G'].append((loguot['u'][i][:, None] * K * loguot['v'][i][None, :]).flatten())


times = time.time()
Gs2,loguot2 = ot.unbalanced.sinkhorn_unbalanced(a, b, M, epsilon, tau, numItermax=10000, stopThr=tol, verbose=True,log=True)
timee = time.time()
print("uot time: ", timee - times)

loguot2['G'] = []
for i in range(len(loguot2['u'])):
     loguot2['G'].append((loguot2['u'][i][:, None] * K * loguot2['v'][i][None, :]).flatten())


times = time.time()
Gs3,loguot3 = ot.unbalanced.sinkhorn_unbalanced(a, b, M, epsilon, tau*10, numItermax=10000, stopThr=tol, verbose=True,log=True)
timee = time.time()
print("uot time: ", timee - times)

loguot3['G'] = []
for i in range(len(loguot2['u'])):
     loguot3['G'].append((loguot3['u'][i][:, None] * K * loguot3['v'][i][None, :]).flatten())

time_s = time.time()
Gtau,log_tau = ot.unbalanced.mm_unbalanced(a, b, M, tau, div='kl',numItermax=round,log=True)
time_e = time.time()
print( "time costs: ", time_e - time_s, " s")


time_s = time.time()
G1_tau,log1_tau = ot.unbalanced.mm_unbalanced_dynamic(a, b, M,1,tau, div='kl',numItermax=round,log=True)
time_e = time.time()
print( "time costs: ", time_e - time_s, " s")

time_s = time.time()
G5_tau,log5_tau = ot.unbalanced.mm_unbalanced_dynamic(a, b, M,5, tau, div='kl',numItermax=round,log=True)
time_e = time.time()
print( "time costs: ", time_e - time_s, " s")

time_s = time.time()
G10_tau,log10_tau = ot.unbalanced.mm_unbalanced_dynamic(a, b, M,10, tau, div='kl',numItermax=round,log=True)
time_e = time.time()
print( "time costs: ", time_e - time_s, " s")

time_s = time.time()
G50_tau,log50_tau = ot.unbalanced.mm_unbalanced_dynamic(a, b, M,50, tau, div='kl',numItermax=round,log=True)
time_e = time.time()
print( "time costs: ", time_e - time_s, " s")





convergence = {
    'uot-(tau/10)':loguot,
    'uot-tau': loguot2,
    'uot-tau*10': loguot3,
    'mmkl-tau': log_tau,
    "mmkl-dynamic-1-tau": log1_tau,
    "mmkl-dynamic-5-tau": log5_tau,
    "mmkl-dynamic-10-tau": log10_tau,
    "mmkl-dynamic-50-tau": log50_tau,
             }


pot_names = {
    'uot-100': Gs,
    'uot-1000': Gs2,
    'uot-tau*10': Gs3,
    'mmkl-tau-tau': Gtau,
    "mmkl-dynamic-1-tau": G1_tau,
    "mmkl-dynamic-5-tau": G5_tau,
    "mmkl-dynamic-10-tau": G10_tau,
    "mmkl-dynamic-50-tau": G50_tau,
             }
plt.figure(figsize=(13,10))

for con in convergence:
    plt.loglog([f(x.flatten()) for x in convergence[con]['G'][::100]], label=con)
    plt.xlabel('iterations')
    plt.ylabel(r'$(f(x)+\tau(D(Mx,b)+D(Nx,a))$')
    plt.title(r'Convergence spped for $\tau=1000$')
plt.legend()
plt.show()
plt.figure(figsize=(13,10))

for con in convergence:
    plt.loglog([f_opt(x.flatten()) for x in convergence[con]['G'][::100]], label=con)
    plt.xlabel('iterations')
    plt.ylabel(r'$f(x)$')
    plt.title(r'Convergence spped for $\tau=1000$')
plt.legend()
plt.show()
plt.figure(figsize=(13,10))
for con in convergence:
    plt.loglog([mkl(x.flatten()) for x in convergence[con]['G'][::100]], label=con)
    plt.xlabel('iterations')
    plt.ylabel(r'$D_h(Mt,b)+D_h(Nt,a)$')
    plt.title(r'$h=\frac{x^2}{2}$')
plt.legend()
plt.show()
plt.figure(figsize=(13,10))
for con in convergence:
    plt.plot([np.log(ml2(x.flatten())) for x in convergence[con]['G'][::100]], label=con)
    plt.xlabel('iterations')
    plt.ylabel(r'$\ln{(D_h(Mt,b)+D_h(Nt,a))}$')
    plt.title(r'$h=x(\ln{x}-1)$')
plt.legend()
plt.show()
# plt.figure(figsize=(13,10))
# for con in convergence:
#     plt.plot([(ml2(x.flatten())) for x in convergence[con]['G'][::100]], label=con)
#     plt.xlabel('iterations')
#     plt.ylabel(r'$D_h(Mt,b)+D_h(Nt,a)$')
#     plt.title(r'$h=x(\ln{x}-1)$')
# plt.legend()
# plt.show()
#f_speed_linear(methods,mar,"marginal error")
#f_speed_linear(methods,spa,"sparsity")
i = 2

# for m_name in methods:
#     x = methods[m_name].get_convergence()[-1]
#     plt.imshow(x.reshape((dim_a,dim_b)), cmap='hot', interpolation='nearest')
#     plt.title(m_name)
#     plt.show()
#     i+=1
# for m_name in pot_names:
#     x = pot_names[m_name]
#     plt.imshow(x, cmap='hot', interpolation='nearest')
#     plt.title(m_name)
#     plt.show()
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

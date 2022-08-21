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
# n =100 tau =100
# Best convergence for FISTA is 0.01
# Best convergence for AMD is 1
# Best convergence for AMD_E is 1
# Best convergence for PGD is 0.001

x0 = np.ones((dim_a,dim_b)).flatten()/(dim_a*dim_b)
max_iter = 4000
tol = 1e-6


# for m_name in methods:
#     print("\t", m_name)
#     time_s = time.time()
#     x = methods[m_name].solve(x0=x0, max_iter=max_iter, tol=tol, disp=1)
#     time_e = time.time()
#     print(m_name,"time costs: ", time_e - time_s, " s")



epsilon = 1e-3 # entropy parameter
  # Unbalanced KL relaxation parameter
round = 5000
stopThr = 1e-15


time_s = time.time()
Gtau,log_tau = ot.unbalanced.mm_unbalanced_revised_screening(a, b, M, tau, div='l2_2',numItermax=round,log=True)
time_e = time.time()
print( "time costs: ", time_e - time_s, " s")


times = time.time()
Gs,loguot = ot.unbalanced.sinkhorn_unbalanced(a, b, M, epsilon, tau, numItermax=round, stopThr=tol, verbose=True,log=True)
timee = time.time()
print("uot time: ", timee - times)

nx = ot.backend.get_backend(M, a, b)
K = nx.exp(M / (-epsilon))
loguot['G'] = []
for i in range(len(loguot['u'])):
     loguot['G'].append((loguot['u'][i][:, None] * K * loguot['v'][i][None, :]).flatten())


time_s = time.time()
Gtau,log_tau = ot.unbalanced.mm_unbalanced_revised(a, b, M, tau, div='kl',numItermax=round,log=True)
time_e = time.time()
print( "time costs: ", time_e - time_s, " s")

batch = 1
time_s = time.time()
Gtau_c1,log_tau_c1 = ot.unbalanced.mm_unbalanced_revised_coord(a, b, M, tau, batch, div='kl',numItermax=round,log=True)
time_e = time.time()
print( "time costs: ", time_e - time_s, " s")
batch = 5
time_s = time.time()
Gtau_c5,log_tau_c5 = ot.unbalanced.mm_unbalanced_revised_coord(a, b, M, tau, batch, div='kl',numItermax=round,log=True)
time_e = time.time()
print( "time costs: ", time_e - time_s, " s")

batch = 10
time_s = time.time()
Gtau_c10,log_tau_c10 = ot.unbalanced.mm_unbalanced_revised_coord(a, b, M, tau, batch, div='kl',numItermax=round,log=True)
time_e = time.time()
print( "time costs: ", time_e - time_s, " s")

batch = 100
time_s = time.time()
Gtau_c100,log_tau_c100 = ot.unbalanced.mm_unbalanced_revised_coord(a, b, M, tau, batch, div='kl',numItermax=round,log=True)
time_e = time.time()
print( "time costs: ", time_e - time_s, " s")
convergence = {
    'uotsink':loguot,
    'mmkl-tau': log_tau,
    'mmkl-tau_coord1': log_tau_c1,
    'mmkl-tau_coord5': log_tau_c5,
    'mmkl-tau_coord10': log_tau_c10,
    'mmkl-tau_coord100': log_tau_c100,

             }


pot_names = {
    'uotsink':Gs,
    'mmkl-tau': Gtau,
    'mmkl-tau_coord1': Gtau_c1,
    'mmkl-tau_coord5': Gtau_c5,
    'mmkl-tau_coord10': Gtau_c10,
    'mmkl-tau_coord100': Gtau_c100,
             }

plt.figure(figsize=(13,10))

paint_iteration = 10

for con in convergence:
    plt.plot([np.log(f(x.flatten())) for x in convergence[con]['G'][::paint_iteration]], label=con)
    plt.xlabel(r'$iterations \times$ %i' %paint_iteration)
    plt.ylabel(r'$\ln((f(x)+\tau(D(Mx,b)+D(Nx,a)))$')
    plt.title(r'Convergence spped for $\tau=$ %i' %tau)
plt.legend()
plt.show()
plt.figure(figsize=(13,10))

for con in convergence:
    plt.plot([np.log(f(x.flatten())) for x in convergence[con]['G'][::paint_iteration]], label=con)
    plt.xlabel(r'$iterations \times$ %i' %paint_iteration)
    plt.ylabel(r'$\ln((f(x)+\tau(D(Mx,b)+D(Nx,a)))$')
    plt.title(r'Convergence spped for $\tau=$ %i' %tau)
plt.xlim(200,500)
plt.ylim(-2.5,-2.1)
plt.legend()
plt.show()

plt.figure(figsize=(13,10))
for con in convergence:
    plt.loglog([f_opt(x.flatten()) for x in convergence[con]['G'][::paint_iteration]], label=con)
    plt.xlabel(r'$iterations \times$ %i' %paint_iteration)
    plt.ylabel(r'$f(x)$')
    plt.title(r'Convergence spped for $\tau=$ %i' %tau)
plt.legend()
plt.show()
plt.figure(figsize=(13,10))
for con in convergence:
    plt.loglog([ml2(x.flatten()) for x in convergence[con]['G'][::paint_iteration]], label=con)
    plt.xlabel(r'$iterations \times$ %i' %paint_iteration)
    plt.ylabel(r'$D_h(Mt,b)+D_h(Nt,a)$')
    plt.title(r'$h=\frac{x^2}{2}$')
plt.legend()
plt.show()


plt.figure(figsize=(13,10))
for con in convergence:
    plt.plot([spa(x.flatten()) for x in convergence[con]['G'][::paint_iteration]], label=con)
    plt.xlabel(r'$iterations \times 100$')
    plt.ylabel(r'sparsity')
    plt.title(r'Convergence spped for $\tau=1000$')
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
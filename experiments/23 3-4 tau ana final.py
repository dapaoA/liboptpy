import numpy as np
import liboptpy.base_optimizer as base
import liboptpy.constr_solvers as cs
import liboptpy.step_size as ss
from liboptpy.data_preparing import making_mnist_with_noise
from liboptpy.data_preparing import making_gausses, making_mnist_uot
import ot
import matplotlib.pyplot as plt
import scipy.sparse as sp
import time
from Plot_Function import f_speed_log, f_speed_linear, f_time_log
from functions import UOT_kl
from functions import grad_uot_kl as g
from functions import uot_kl_proximal_K_entropy as pk
from functions import uot_kl_proximal_B_entropy as pb
from functions import marginal_kl as makl
from functions import marginal_l2 as mal2

# this one is used for testing about the tau
# we should divide the process into two part and observe the convergence speed to find the stop condition that we long for!!!!!!!!!!!!!!!!!!!


class Sinkhornalg:
    def __init__(self, log):
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
a, b, M = making_mnist_uot('1', '4', 1, 1)
epsilon = 0.01
tau = 100
round = 1000
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
HrHr = Hr.T.dot(Hr)
HcHc = Hc.T.dot(Hc)
Hcb = Hc.T.dot(b)

def func_opt(t, m):
    return np.dot(t, m)

f_opt = lambda x: func_opt(x, m)

# linear programming
times = time.time()
# G0 = ot.emd(a, b, M)
a = a * 1.1
timee = time.time()
print("lp time: ", timee-times)
# opt = f_opt(G0.flatten())

f = lambda x: UOT_kl(x, a, b, m, tau, Hc, Hr)

mkl = lambda x: makl(x, a, b, Hc, Hr)
ml2 = lambda x: mal2(x, a, b, Hc, Hr)
def sparsity(t):
    return np.count_nonzero(t == 0)/len(t)
spa = lambda x: sparsity(x)

# n =100 tau =100
# Best convergence for FISTA is 0.01
# Best convergence for AMD is 1
# Best convergence for AMD_E is 1
# Best convergence for PGD is 0.001
max_iter = 1000
tol = 1e-6

epsilon = 1e-3  # entropy parameter

times = time.time()
Gs, loguot = ot.unbalanced.sinkhorn_unbalanced(a, b, M,
                                               epsilon, tau, numItermax=round, stopThr=tol,
                                               verbose=True, log=True)
timee = time.time()
print("uot time: ", timee - times)

nx = ot.backend.get_backend(M, a, b)
K = nx.exp(M / (-epsilon))
loguot['G'] = []
for i in range(len(loguot['u'])):
     loguot['G'].append((loguot['u'][i][:, None] * K * loguot['v'][i][None, :]).flatten())


time_s = time.time()
Gtau, log_tau = ot.unbalanced.mm_unbalanced(a, b, M, tau, div='kl', numItermax=round, log=True)
time_e = time.time()
print("time costs: ", time_e - time_s, " s")

stopThr = 1e-15

time_s = time.time()
G1_tau_100_2, log1_tau_100_2 = ot.unbalanced.mm_unbalanced_dynamic2(a, b, M, 1, tau, 100, 2, div='kl',numItermax=round,log=True,stopThr=stopThr)
time_e = time.time()
print("time costs: ", time_e - time_s, " s")


time_s = time.time()
Gexptau_1000, logexptau_1000 = ot.unbalanced.mm_unbalanced_dynamic3(a, b, M, 1, tau, 1000, div='kl',numItermax=round,log=True,stopThr=stopThr)
time_e = time.time()
print("time costs: ", time_e - time_s, " s")



time_s = time.time()
Gexp2tau_1000, logexp2tau_1000 = ot.unbalanced.mm_unbalanced_dynamic3(a, b, M, 1, 2*tau, 1000, div='kl',numItermax=round,log=True,stopThr=stopThr)
time_e = time.time()
print("time costs: ", time_e - time_s, " s")

time_s = time.time()
G_d_tau, log_d_tau = ot.unbalanced.mm_unbalanced_inexact(a, b, M, tau, div='kl', numItermax=round, log=True, verbose=True)
time_e = time.time()
print("time costs: ", time_e - time_s, " s")



convergence = {
    'uot-(tau)': loguot,
    'mmkl-tau': log_tau,
    "mmkl-1-tau-100-2": log1_tau_100_2,
    "exp tau 1000":logexptau_1000,
    "exp 2tau 1000": logexp2tau_1000,
    'mmkl-d_tau': log_d_tau,
             }


pot_names = {
    'uot-tau': Gs,
    'mmkl-tau-tau': Gtau,
    "mmkl-1-tau-100": G1_tau_100_2,
    "exp tau 1000": Gexptau_1000,
    "exp 2tau 1000": Gexp2tau_1000,
    'mmkl-d_tau-tau': G_d_tau,
             }
plt.figure(figsize=(13, 10))

for con in convergence:
    a1 = np.asarray([f(x.flatten()) for x in convergence[con]['G'][::10]])
    b1 = np.asarray([mkl(x.flatten()) for x in convergence[con]['G'][::10]])
    plt.plot(a1/(tau * b1), label=con)
    plt.xlabel(r'$iterations \times 100$')
    plt.ylabel(r'$\ln((f(x)+\tau(D(Mx,b)+D(Nx,a)))$')
    plt.title(r'Convergence ratio for $\tau=1000$')
plt.legend()
plt.show()
plt.figure(figsize=(13, 10))

for con in convergence:
    plt.plot([np.log(f(x.flatten())) for x in convergence[con]['G'][::10]], label=con)
    plt.xlabel(r'$iterations \times 100$')
    plt.ylabel(r'$\ln((f(x)+\tau(D(Mx,b)+D(Nx,a)))$')
    plt.title(r'Convergence spped for $\tau=1000$')
plt.legend()
plt.show()
plt.figure(figsize=(13,10))

for con in convergence:
    plt.loglog([f(x.flatten()) for x in convergence[con]['G'][::10]], label=con)
    plt.xlabel(r'$iterations \times 100$')
    plt.ylabel(r'$(f(x)+\tau(D(Mx,b)+D(Nx,a))$')
    plt.title(r'Convergence spped for $\tau=1000$')
plt.legend()
plt.show()
plt.figure(figsize=(13,10))

for con in convergence:
    plt.loglog([f_opt(x.flatten()) for x in convergence[con]['G'][::10]], label=con)
    plt.xlabel(r'$iterations \times 100$')
    plt.ylabel(r'$f(x)$')
    plt.title(r'Convergence spped for $\tau=1000$')
plt.legend()
plt.show()
plt.figure(figsize=(13, 10))
for con in convergence:
    plt.loglog([mkl(x.flatten()) for x in convergence[con]['G'][::10]], label=con)
    plt.xlabel(r'$iterations \times 100$')
    plt.ylabel(r'$D_h(Mt,b)+D_h(Nt,a)$')
    plt.title(r'$h=\frac{x^2}{2}$')
plt.legend()
plt.show()
plt.figure(figsize=(13,10))
for con in convergence:
    plt.plot([np.log(ml2(x.flatten())) for x in convergence[con]['G'][::10]], label=con)
    plt.xlabel(r'$iterations \times 100$')
    plt.ylabel(r'$\ln{(D_h(Mt,b)+D_h(Nt,a))}$')
    plt.title(r'$h=x(\ln{x}-1)$')
plt.legend()
plt.show()

plt.figure(figsize=(13,10))
for con in convergence:
    plt.plot([np.log(f_opt(x.flatten())) for x in convergence[con]['G'][::10]], label=con)
    plt.xlabel(r'$iterations \times 100$')
    plt.ylabel(r'$\ln f(x)$')
    plt.title(r'Convergence spped for $\tau=1000$')
plt.legend()
plt.ylim(-2.8,-2.2)
plt.xlim(0,150)
plt.show()


plt.figure(figsize=(13,10))
for con in convergence:
    plt.plot([spa(x.flatten()) for x in convergence[con]['G'][::10]], label=con)
    plt.xlabel(r'$iterations \times 100$')
    plt.ylabel(r'sparcity')
    plt.title(r'sparcity for $\tau=1000$')
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

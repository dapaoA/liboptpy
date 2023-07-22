import numpy as np
import liboptpy.base_optimizer as base
import liboptpy.constr_solvers as cs
import liboptpy.step_size as ss
from liboptpy.data_preparing import making_mnist_with_noise
from liboptpy.data_preparing import making_gausses_times
import ot
import script_mm_unbalanced2 as sot
import matplotlib.pyplot as plt
import scipy.sparse as sp
import time
from Plot_Function import f_speed_log,f_speed_linear,f_time_log
from functions import UOT_kl_2
import copy
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

Ex_times = 5
n = 100
a_list, b_list, M_list = making_gausses_times(n, Ex_times)
epsilon = 0.01

tau = 1000

# linear programming
times = time.time()

timee = time.time()

max_iter = 4000
tol = 1e-8

epsilon = 1e-3 # entropy parameter
  # Unbalanced KL relaxation parameter
round = 2500

stopThr = 1e-15

G_list = []
log_list = []


f_opt = lambda x, m: np.einsum('ij,ij', x, m)
loguot_list = []
log_mm_list = []
log_mm_d_list = []
log_amm_d_list = []
for i in range(Ex_times):
    a = a_list[i]
    a = a * 1.2
    a_list[i] = a
    b = b_list[i]
    M = M_list[i]

    times = time.time()
    Gs, loguot = sot.sinkhorn_unbalanced(a, b, M, epsilon, tau, numItermax=round, stopThr=tol, verbose=True,log=True)
    timee = time.time()
    print("uot time: ", timee - times)
    nx = ot.backend.get_backend(M, a, b)
    K = nx.exp(M / (-epsilon))
    loguot['G'] = []
    for i in range(len(loguot['u'])):
         loguot['G'].append((loguot['u'][i][:, None] * K * loguot['v'][i][None, :]))
    loguot_list.append(copy.deepcopy(loguot))

    time_s = time.time()
    Gtau, log_tau = sot.mm_unbalanced(a, b, M, tau, div='kl', numItermax=round, log=True)
    time_e = time.time()
    print("time costs: ", time_e - time_s, " s")
    log_mm_list.append(copy.deepcopy(log_tau))

    time_s = time.time()
    G1_q000005, log_q000005 = sot.mm_unbalanced_dynamic2_stop(a, b, M, 0.1, tau,
                                                                        0.0001, 2, div='kl', numItermax=round,
                                                                        log=True, stopThr=stopThr)
    time_e = time.time()
    print("time costs: ", time_e - time_s, " s")
    time_s = time.time()
    log_mm_d_list.append(copy.deepcopy(log_q000005))

    time_s = time.time()
    G1_q000005a2, log_q000005a2 = sot.mm_unbalanced_dynamic2_stop_nestrov2(a, b, M, 0.1, tau,
                                                                                     0.0001, 2, div='kl',
                                                                                     numItermax=round, log=True,
                                                                                     stopThr=stopThr)
    time_e = time.time()
    print("time costs: ", time_e - time_s, " s")
    time_s = time.time()
    log_amm_d_list.append(copy.deepcopy(log_q000005a2))


convergence = {
    'Sinkhorn': loguot_list,
    'MM': log_mm_list,

    "DPMM": log_mm_d_list,
    "DPAMM": log_amm_d_list,

             }

error = {
    'Sinkhorn': loguot,
    'MM': log_tau,

    "DPMM": log_q000005,
    "DPAMM": log_q000005a2,

             }
pot_names = {
    'Sinkhorn': Gs,
    'MM': Gtau,
    "DPMM": G1_q000005,
    "DPAMM": G1_q000005a2,
             }

f = lambda x, a, b, m: UOT_kl_2(x, a, b, m, tau)

paint_iteration = 10
colors = colors = ["#3366FF", "#C00000", "#28D82B", "#990099"]
for con, c in zip(convergence, colors):
    error[con] = np.zeros((Ex_times, len(convergence[con][0]['G'][::paint_iteration])))
    for j in range(Ex_times):
        error[con][j, :] = np.asarray([np.log(f(x, a_list[j], b_list[j], M_list[j]))
                                       for x in convergence[con][j]['G'][::paint_iteration]])

# paint_iteration = 10
# plt.rcParams["font.family"] = "Times New Roman"
# fig, axs = plt.subplots(figsize=(8.0, 6.0), nrows=1, ncols=1)
# for con, c in zip(convergence, colors):
#     avg = np.mean(error[con], axis=0)
#     std = np.std(error[con], axis=0)
#     r1 = list(map(lambda x: (x[0] - x[1]), zip(avg, std)))
#     r2 = list(map(lambda x: (x[0] + x[1]), zip(avg, std)))
#     axs.fill_between(range(len(convergence[con][j]['G'][::paint_iteration])), r1, r2, color=c, alpha=0.2)
#     axs.plot(range(len(convergence[con][j]['G'][::paint_iteration])), avg, c=c, label=con)
# axs.set_xlabel(r'$\text{Iterations} \times$ %i' %paint_iteration)
# axs.set_ylabel(r'$\ln(UOT(T^{k}))$')
# axs.set_title(r'Convergence speed for $\tau=$ %i.' %tau)
# fig.legend(bbox_to_anchor=(0.82, 0.56), loc=4, ncol=1, facecolor='white', edgecolor='black')
# plt.show()
# # - opt_list[j]
# plt.figure(figsize=(13, 10))
# fig, axs = plt.subplots(figsize=(8.0, 8.5), nrows=2, ncols=2)
# for j, m_name in enumerate(pot_names):
#     x = pot_names[m_name]
#     axs[j//2, j%2].imshow(x, cmap='hot', interpolation='nearest')
#     axs[j//2, j%2].set_xticks([0,50,100])
#     axs[j//2, j%2].set_yticks([0,50,100])
#     axs[j//2, j%2].set_title(m_name)
# fig.savefig('ex4.pdf', format='pdf', bbox_inches='tight')
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

plt.rc("text", usetex=True)
plt.rcParams['text.latex.preamble'] = r"\usepackage{amsmath}"

paint_iteration = 10
plt.rcParams["font.family"] = "Times New Roman"
fig, axs = plt.subplots(figsize=(8.0, 6.0), nrows=1, ncols=2)
for con, c in zip(convergence, colors):
    avg = np.mean(error[con], axis=0)
    std = np.std(error[con], axis=0)
    r1 = list(map(lambda x: (x[0] - x[1]), zip(avg, std)))
    r2 = list(map(lambda x: (x[0] + x[1]), zip(avg, std)))
    axs[0].fill_between(range(len(convergence[con][j]['G'][::paint_iteration])), r1, r2, color=c, alpha=0.2)
    axs[0].plot(range(len(convergence[con][j]['G'][::paint_iteration])), avg, c=c)
axs[0].set_xlabel(rf'$\text{{Iterations}} \times$ {paint_iteration}')
axs[0].set_ylim(2.220, 2.24)
axs[0].set_yticks([2.220, 2.225, 2.230, 2.235, 2.240])
axs[0].set_ylabel(r'$\ln(\textbf{P}_{\text{UOT}}(\textbf{T}^{k}))$')
axs[0].set_xlim(0, 250)
for con, c in zip(convergence, colors):
    avg = np.mean(error[con], axis=0)
    std = np.std(error[con], axis=0)
    r1 = list(map(lambda x: (x[0] - x[1]), zip(avg, std)))
    r2 = list(map(lambda x: (x[0] + x[1]), zip(avg, std)))
    axs[1].fill_between(range(len(convergence[con][j]['G'][::paint_iteration])), r1, r2, color=c, alpha=0.2)
    axs[1].plot(range(len(convergence[con][j]['G'][::paint_iteration])), avg, c=c, label=con)
axs[0].set_xlabel(rf'$\text{{Iterations}} \times$ {paint_iteration}')
axs[1].set_xlim(0, 100)
fig.legend(bbox_to_anchor=(0.90, 0.56), loc=4, ncol=1, facecolor='white', edgecolor='black')
fig.savefig('ex3.pdf', format='pdf', bbox_inches='tight')
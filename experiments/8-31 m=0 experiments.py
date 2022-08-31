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
from functions import UOT_l2
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

n = 30
a,b,M = making_uot_gausses(n)
epsilon = 0.001
tau = 20

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



grad = lambda x: g(x, a, b, Hc,Hr, dim_a,dim_b)


proximalB = lambda x,h,grad: pb(m/tau,x,h,grad)

proximalK = lambda x,h,grad: pk(m/tau,x,h,grad)



mkl = lambda x: makl(x, a, b,Hc,Hr)
ml2 = lambda x: mal2(x, a, b,Hc,Hr)
def sparsity(t):
    return np.count_nonzero(t==0)/len(t)

spa = lambda x: sparsity(x)



tau = 0.5
stopThr = 1e-16
xx = sp.vstack((Hr,Hc)).tocsc()
# trans1 = sc.sasvi_screening(np.ones_like(m),xx,np.concatenate((a,b)),m,1/tau)
# time_s = time.time()
# G1_q00001,log_q00001 = ot.unbalanced.mm_unbalanced_revised_screening(a, b, M, tau, l_rate=1/(2*n),screening=trans1,div='l2_2',numItermax=round,log=True)
# time_e = time.time()
# print( "time costs: ", time_e - time_s, " s")
# time_s = time.time()
# plt.imshow(G1_q00001)
# plt.title('uot_mm_solution_05')
# plt.colorbar(aspect=40, pad=0.08, shrink=0.6,
#              orientation='horizontal', extend='both')
# plt.show()

# tau = 1000
# G1= ot.unbalanced.mm_unbalanced_revised(a, b, M, tau, l_rate=1/(2*n),div='l2_2',numItermax=100*round,stopThr=stopThr)
# plt.imshow(G1)
# plt.title('uot_mm_solution_5')
# plt.colorbar(aspect=40, pad=0.08, shrink=0.6,
#              orientation='horizontal', extend='both')
# plt.show()
# trans1 = sc.sasvi_screening_zero_test(np.ones_like(m),xx,np.concatenate((a,b)),m,1/tau,solution=G1.flatten())
# trans1_m = trans1.update(np.zeros((n,n)).flatten())
#
# plt.imshow(trans1_m.reshape(n,n))
# plt.title('sc-savia-5')
# plt.colorbar(aspect=40, pad=0.08, shrink=0.6,
#              orientation='horizontal', extend='both')
# plt.show()
# time_s = time.time()
# G1_q00001= ot.unbalanced.mm_unbalanced_revised_screening(a, b, M, tau, l_rate=1/(2*n),screening=trans1,div='l2_2',numItermax=round,stopThr=stopThr)
# time_e = time.time()
# print( "time costs: ", time_e - time_s, " s")
# time_s = time.time()
#
# plt.imshow(G1_q00001)
# plt.title('uot_mm_solution_5')
# plt.colorbar(aspect=40, pad=0.08, shrink=0.6,
#              orientation='horizontal', extend='both')
# plt.show()




tau =500
round = 1000000
G1= ot.unbalanced.mm_unbalanced_revised(a, b, M, tau, l_rate=1/(2*n),div='l2_2',numItermax=1*round,stopThr=stopThr)
plt.imshow(G1)
plt.title('uot_mm_solution_5')
plt.colorbar(aspect=40, pad=0.08, shrink=0.6,
             orientation='horizontal', extend='both')
plt.show()

trans1 = sc.sasvi_screening_zero_test(np.ones_like(m),xx,np.concatenate((a,b)),m,1/tau,solution=G1.flatten())

time_s = time.time()
G1_q00001,log= ot.unbalanced.mm_unbalanced_revised_screening_for_zero(a, b, M, tau,saveround=10000, l_rate=1/(2*n),screening=trans1,div='l2_2',numItermax=round,stopThr=stopThr,log=True)
time_e = time.time()
print( "time costs: ", time_e - time_s, " s")

plt.figure(figsize=(13,10))
plt.plot(log["opt_alg"],label=r'$\hat{\theta} \sim \theta^{k}$')
plt.plot(log["opt_proj1"],label=r'$\hat{\theta} \sim \tilde{\theta}^{k}_{1}$')

plt.plot(log["alg_proj1"],label=r'$\theta^{k} \sim \tilde{\theta}^{k}_{1}$')

plt.yscale('log')
plt.title("Distances")
plt.xlabel("rounds")
plt.legend()
plt.show()

plt.figure(figsize=(13,10))
plt.plot(log["opt_proj1"],label=r'$\hat{\theta} \sim \tilde{\theta}^{k}_{1}$')

plt.plot(log["alg_proj1"],label=r'$\theta^{k} \sim \tilde{\theta}^{k}_{1}$')

plt.xlabel("rounds")
plt.title("Distances")
plt.legend()
plt.show()

plt.figure(figsize=(13,10))
plt.plot(log["screening_area1"],label=r'$R(\tilde{\theta}^{k}_{1})$')

plt.plot(log["screening_ps"],label=r'$\theta^{k}$')
plt.plot(log["screening_p1"],label=r'$\tilde{\theta}^{k}_{1}$')

plt.title("Sparsity")
plt.xlabel("rounds")
plt.legend()
plt.show()



# import matplotlib.animation as animation
# fig = plt.figure()
# ax = fig.add_subplot(111)
# #Line2D objectを入れるリスト
# ims = []
#
#
# for i in range(len(log['w_screening'])):
#     ax.colorbar(aspect=40, pad=0.08, shrink=0.6,
#                      orientation='horizontal', extend='both')
#     ax.title('Screening Process')
#     im=ax.imshow(log['w_screening'][i])
#
#     ims.append(im) #各フレーム画像をimsに追加
#
# #アニメの生成
# ani = animation.ArtistAnimation(fig, ims, interval=100, blit=True, repeat_delay=1000)
#
# #保存
# ani.save("sample.gif", writer="pillow")
#
#
# plt.imshow(G1_q00001)
# plt.title('uot_mm_solution_5')
# plt.colorbar(aspect=40, pad=0.08, shrink=0.6,
#              orientation='horizontal', extend='both')
# plt.show()


# tau = 50
# trans1 = sc.sasvi_screening(np.ones_like(m),xx,np.concatenate((a,b)),m,1/tau)
# trans1_m = trans1.update(np.zeros((n,n)).flatten())
#
# plt.imshow(trans1_m.reshape(n,n))
# plt.title('sc-savia-50')
# plt.colorbar(aspect=40, pad=0.08, shrink=0.6,
#              orientation='horizontal', extend='both')
# plt.show()
#
# time_s = time.time()
# G1_q00001,log_q00001 = ot.unbalanced.mm_unbalanced_revised_screening(a, b, M, tau, l_rate=1/(2*n),screening=trans1,div='l2_2',numItermax=round,log=True)
# time_e = time.time()
# print( "time costs: ", time_e - time_s, " s")
# time_s = time.time()
#
# plt.imshow(G1_q00001)
# plt.title('uot_mm_solution')
# plt.colorbar(aspect=40, pad=0.08, shrink=0.6,
#              orientation='horizontal', extend='both')
# plt.show()
#
# plt.plot(a)
# plt.plot(b)
# plt.show()
# plt.imshow(Gs)
# plt.show()
# plt.imshow(Gs_re)
# plt.show()
#
#
# print("value of f_uot of initial method is: ",f(Gs_re.flatten().squeeze()))
# print("value of f_uot of sinkhorn method is: ",f(Gs_org.flatten()))
# print("value of f_opt of initial method is: ",f_opt(Gs_re.flatten()))
# print("value of f_opt of sinkhorn method is: ",f_opt(Gs_org.flatten()))
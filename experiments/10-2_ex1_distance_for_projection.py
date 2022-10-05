import numpy as np
from liboptpy.data_preparing import making_uot_gausses
import ot
import matplotlib.pyplot as plt
import time
import liboptpy.screening.screenning as sc


plt.rc("text", usetex=True)
fontsize = 24

n = 30
a, b, M = making_uot_gausses(n)
epsilon = 0.001
stopThr = 1e-16
tau = 100
round = 10000

dim_a = np.shape(a)[0]
dim_b = np.shape(b)[0]
m = M.flatten()

G_opt = ot.unbalanced.mm_unbalanced_revised(a, b, M, tau, l_rate=1/(2*n),div='l2_2',numItermax=100000, stopThr=stopThr)

# plt.imshow(G_opt)
# plt.title('uot_mm_solution_5')
# plt.colorbar(aspect=40, pad=0.08, shrink=0.6,
#              orientation='horizontal', extend='both')
# plt.show()

trans1 = sc.sasvi_screening_matrix(np.ones_like(M), a, b, M, 1/tau, solution=G_opt)

time_s = time.time()
G1_q00001, log = ot.unbalanced.mm_unbalanced_revised_screening_for_divide(a, b, M, tau, saveround=1000, l_rate=1/(2*n),
                                                                          screening=trans1, div='l2_2',
                                                                          numItermax=round, stopThr=stopThr,
                                                                          log=True, distance_log=1)
time_e = time.time()
print( "time costs: ", time_e - time_s, " s")

plt.figure(figsize=(7, 5))
plt.plot(log["opt_alg"],label=r'$\hat{\theta} \sim \theta^{k}$')
plt.plot(log["opt_proj"],label='Dynamic Sasvi', linestyle='dashed')
plt.plot(log["alg_proj"],label='Dynamic Two plane', linestyle='dashed')
plt.yscale('log')
plt.title("Distances")
plt.xlabel("rounds")
plt.legend()
plt.savefig("../ex-photo/ex-1/distance.png")


plt.figure(figsize=(13,10))
plt.plot(log["screening_area1"], label=r'$R(\tilde{\theta}^{k})_{normal}$')
plt.plot(log["screening_area2"], label=r'$R(\tilde{\theta}^{k}_{divide})$', linestyle='dashed')


plt.title("Sparsity")
plt.xlabel("rounds")
plt.legend()
plt.savefig("../ex-photo/ex-1/sparsity.png")

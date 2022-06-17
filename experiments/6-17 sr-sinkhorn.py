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

plt.rc("text", usetex=True)
fontsize = 24
figsize = (8, 6)
import seaborn as sns
sns.set_context("talk")
#from tqdm import tqdm

n = 2000
a,b,M = making_gausses(n)
round = 5000
tau = 100
m = M.flatten()
def func(t, m):


    return np.dot(t,m)

f = lambda x: func(x, m,)



# linear programming
times = time.time()
G0 = ot.emd(a, b, M)
timee = time.time()
print("lp time: ",timee-times)
opt = f(G0.flatten())
stopThr_list = [1e-1,1e-2,1e-3,1e-4,1e-5,1e-6,1e-7,1e-8,1e-9]

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
    error_list["sinkhorn"].append(np.fabs(f(Gs.flatten())-opt))

# UOT
timelist["uot"] = []
error_list["uot"] = []
epsilon = 1e-3 # entropy parameter
alpha = 100.  # Unbalanced KL relaxation parameter
for tol in stopThr_list:
    times = time.time()
    Gs = ot.unbalanced.sinkhorn_unbalanced(a, b, M, epsilon, alpha,numItermax=10000,stopThr=tol, verbose=True)
    timee = time.time()
    timelist["uot"].append(timee-times)
    print("uot time: ",timee-times)
    error_list["uot"].append(np.fabs(f(Gs.flatten())-opt))


plt.plot([1/x**0.5 for x in error_list["sinkhorn"]],timelist["sinkhorn"],label = "sinkhorn")
plt.plot([1/x**0.5 for x in error_list["uot"]],timelist["uot"],label = "uot")
plt.legend()
plt.ylabel("time, $k$")
plt.xlabel("f - f* in 1/x**0.5")
plt.title(" convergence speed")
plt.show()


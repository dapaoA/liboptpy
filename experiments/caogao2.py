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

n = 1000
a,b,M = making_gausses(n)

alpha = 10000000000


Gs = ot.sinkhorn(a, b, M, 1e-3,numItermax=10000,stopThr=1e-5, verbose=True)


plt.imshow(Gs)
plt.show()







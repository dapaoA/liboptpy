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

plt.rc("text", usetex=True)
fontsize = 24
figsize = (8, 6)
import seaborn as sns
sns.set_context("talk")
#from tqdm import tqdm

n = 100
a,b,M = making_gausses(n)
epsilon = 0.01
round = 5000
tau = 100

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

def KL(x,y):
    return np.dot(x,np.log(x/y))-x+y

def l2(x,y):
    a = x-y
    return np.dot(a,a.T)

def func(t, a, b, m, tau):


    return np.dot(t,m) + tau * l2(Hc.dot(t),b)

f = lambda x: func(x, a, b, m, tau)

def grad_f(t, a, b, m, tau):

    return m + 2 *tau* (np.tile(HcHc[:dim_a,:].dot(t),dim_b)-Hcb)

grad = lambda x: grad_f(x, a, b, m, tau)


def linsolver(gradient):
    x = np.zeros(gradient.shape[0])
    pos_grad = gradient > 0
    neg_grad = gradient < 0
    x[pos_grad] = np.zeros(np.sum(pos_grad == True))
    x[neg_grad] = np.ones(np.sum(neg_grad == True))
    return x

def kl_projection(t):
    new_t = np.ones_like(t)
    for i in range(dim_a):
        new_t[i*dim_b:(i+1)*dim_b] = t[i*dim_b:(i+1)*dim_b]*a[i] /np.sum(t[i*dim_b:(i+1)*dim_b] )
    return new_t

def projection(t):
    new_t = np.ones_like(t)
    for i in range(dim_a):
        new_t[i*dim_b:(i+1)*dim_b] = t[i*dim_b:(i+1)*dim_b] /np.sum(t[i*dim_b:(i+1)*dim_b] )
    return new_t


def projection_simplex(x, z=a, axis=1):
    """
    Projection of x onto the simplex, scaled by z:
        P(x; z) = argmin_{y >= 0, sum(y) = z} ||y - x||^2
    z: float or array
        If array, len(z) must be compatible with V
    axis: None or int
        axis=None: project V by P(V.ravel(); z)
        axis=1: project each V[i] by P(V[i]; z[i])
        axis=0: project each V[:, j] by P(V[:, j]; z[j])
    """
    V = np.reshape(x, (dim_a, dim_b))
    if axis == 1:
        n_features = V.shape[1]
        U = np.sort(V, axis=1)[:, ::-1]
        z = np.ones(len(V)) * z
        cssv = np.cumsum(U, axis=1) - z[:, np.newaxis]
        ind = np.arange(n_features) + 1
        cond = U - cssv / ind > 0
        rho = np.count_nonzero(cond, axis=1)
        theta = cssv[np.arange(len(V)), rho - 1] / rho
        return np.maximum(V - theta[:, np.newaxis], 0).flatten()

    elif axis == 0:
        return projection_simplex(V.T, z, axis=1)

    else:
        V = V.ravel().reshape(1, -1)
        return projection_simplex(V, z, axis=1).ravel()

#"FW": cs.FrankWolfe(f, grad, linsolver, ss.Backtracking(rule_type="Armijo", rho=0.5, beta=0.1, init_alpha=1.)),
#           "PGD": cs.ProjectedGD(f, grad, projection_simplex, ss.Backtracking(rule_type="Armijo", rho=0.5, beta=0.1, init_alpha=1.)),



methods = {

"FISTAc": cs.FISTA(f, grad, projection_simplex, ss.ConstantInvIterStepSize(0.01)),
"AMDc": cs.AMD(f, grad, kl_projection, ss.ConstantInvIterStepSize(0.1)),
"PGDc": cs.ProjectedGD(f, grad, projection_simplex, ss.ConstantInvIterStepSize(0.01)),
"MDc": cs.MirrorD(f, grad, kl_projection, ss.ConstantInvIterStepSize(0.1)),

# "FISTA": cs.FISTA(f, grad, projection_simplex, ss.Backtracking_Nestrov(rule_type="Armijo", rho=0.5, beta=0.1, init_alpha=1.)),
# "AMD": cs.AMD(f, grad, kl_projection, ss.Backtracking_Nestrov(rule_type="Armijo", rho=0.5, beta=0.1, init_alpha=1.)),
# "PGD": cs.ProjectedGD(f, grad, projection_simplex, ss.Backtracking(rule_type="Armijo", rho=0.5, beta=0.1, init_alpha=1.)),
# "MD": cs.MirrorD(f, grad, kl_projection, ss.Backtracking(rule_type="Armijo", rho=0.5, beta=0.1, init_alpha=1.)),

          }

x0 = np.ones((dim_a,dim_b)).flatten()/(dim_a*dim_b)
max_iter = 5000
tol = 1e-5


for m_name in methods:
    print("\t", m_name)
    time_s = time.time()
    x = methods[m_name].solve(x0=x0, max_iter=max_iter, tol=tol, disp=1)
    time_e = time.time()
    print(m_name,"time costs: ", time_e - time_s, " s")

plt.figure(figsize=figsize)
for m_name in methods:
    plt.semilogy([f(x) for x in methods[m_name].get_convergence()], label=m_name)

plt.legend(fontsize=fontsize)
plt.xlabel("Number of iteration, $k$", fontsize=fontsize)
plt.ylabel(r"$f(x_k)$", fontsize=fontsize)
plt.xticks(fontsize=fontsize)
_ = plt.yticks(fontsize=fontsize)
plt.title("convergence rate")
plt.show()

i = 2

for m_name in methods:
    x = methods[m_name].get_convergence()[-1]
    plt.imshow(x.reshape((dim_a,dim_b)), cmap='hot', interpolation='nearest')
    plt.title(m_name)
    plt.show()
    i+=1


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

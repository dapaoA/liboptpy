import numpy as np

from liboptpy.data_preparing import making_gausses

import scipy.sparse as sp


def standard_ot(t, m):


    return np.dot(t,m)


def KL(x,y):
    return np.dot(x,np.log(x)-np.log(y))-np.sum(x-y)

def l2(x,y):
    a = x-y
    return np.dot(a,a.T)

def semi_l2(t, a, b, m, tau, Hc):


    return np.dot(t,m) + tau * l2(Hc.dot(t),b)


def grad_semi_l2(t, a, b, m, tau, dim_a, dim_b, HcHc, Hcb):

    return m + 2 *tau* (np.tile(HcHc[:dim_a,:].dot(t),dim_b)-Hcb)

def sto_grad_semi_l2(t, a, b, mdiv, tau, dim_a, dim_b, Hc, i):

    return mdiv + (np.tile(Hc[i,:dim_a].T.dot((Hc[i,:].dot(t)-b[i])),dim_b)/i.shape[0]* 2 *tau)

def sto_grads_semi_l2(t, a, b, mdiv, tau,dim_a,dim_b,Hc, i):

    return np.asarray(np.tile(mdiv,(i.shape[0],1)).T + 2 *tau * Hc[i,:].T.multiply((Hc[i,:].dot(t)-b[i])))


def coord_grads_semi_l2(t,id,a,b,m,tau,Hc,dim_b):
    t[id*(dim_b):(id+1)*(dim_b)] = 0


def semi_kl(t, a, b, m, tau, Hc):


    return np.dot(t,m) + tau * KL(Hc.dot(t),b)



def grad_semi_kl(t, a, b, m, tau, Hc, dim_a,dim_b):

    return m + tau* np.tile(Hc.T[:dim_a,:].dot(np.log(Hc.dot(t))-np.log(b)),dim_b)

def UOT_kl(t,a,b,m,tau,Hc,Hr):

    return np.dot(t,m)+ tau * (KL(Hc.dot(t),b)+KL(Hr.dot(t),a))

def grad_uot_kl(t, a, b,  Hc,Hr, dim_a,dim_b):
# proximal algorithm don't contain c^T t part!!!!!!!!!!!!!
    return   (np.tile(Hc.T[:dim_a,:].dot(np.log(Hc.dot(t))-np.log(b)),dim_b)+np.repeat(Hr.T[::dim_b,:].dot(np.log(Hr.dot(t))-np.log(a)),dim_a,axis=0))

def uot_kl_proximal_B_entropy(ctau,x,h,grad):

    return x/(1+h*np.multiply(ctau-grad,x))


def uot_kl_proximal_K_entropy(ctau, x, h, grad):
    return x / np.exp(h*(ctau-grad))


def UOT_l2(t,a,b,m,tau,Hc,Hr):

    return np.dot(t,m)+ tau/2 * ( l2(Hc.dot(t),b)+(l2(Hr.dot(t),a)))

def grad_uot_l2(t, a, b, HcHc,HrHr,Hcb,Hra, dim_a,dim_b):
    # proximal algorithm don't contain c^T t part!!!!!!!!!!!!!
    return  (np.tile(HcHc[:dim_a,:].dot(t),dim_b)+np.repeat(HrHr[::dim_b,:].dot(t),dim_b,axis=0)-Hcb-Hra)
def uot_l2_proximal_l2(ctau,x,h,grad):
    return np.maximum(x + h *grad -h*ctau,0)

def linsolver(gradient):
    x = np.zeros(gradient.shape[0])
    pos_grad = gradient > 0
    neg_grad = gradient < 0
    x[pos_grad] = np.zeros(np.sum(pos_grad == True))
    x[neg_grad] = np.ones(np.sum(neg_grad == True))
    return x

def kl_projection(t, a, b, dim_a, dim_b):
    new_t = np.ones_like(t)
    for i in range(dim_a):
        new_t[i*dim_b:(i+1)*dim_b] = t[i*dim_b:(i+1)*dim_b]*a[i] /np.sum(t[i*dim_b:(i+1)*dim_b] )
    return new_t

def coor_kl_projection(t,id, a, b, dim_a, dim_b):
    new_t = t
    new_t[id*dim_b:(id+1)*dim_b] = t[id*dim_b:(id+1)*dim_b]*a[id] /np.sum(t[id*dim_b:(id+1)*dim_b] )
    return new_t

def projection_simplex(x, dim_a, dim_b, z, axis=1):
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


def marginal_l2(t,a,b,Hc,Hr):
    return l2(Hc.dot(t),b)+l2(Hr.dot(t),a)

def marginal_kl(t,a,b,Hc,Hr):
    return KL(Hc.dot(t),b)+KL(Hr.dot(t),a)



def sparsity(t):
    return np.count_nonzero(t==0)/len(t)


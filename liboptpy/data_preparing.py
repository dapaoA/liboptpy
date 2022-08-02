import numpy as np
import ot
import ot.plot
from ot.datasets import make_1D_gauss as gauss
from mnist import MNIST

def making_gausses(n):

    # bin positions
    x = np.arange(n, dtype=np.float64)

    # Gaussian distributions
    a = gauss(n, m=n/5, s=n/5)  # m= mean, s= std
    b = gauss(n, m=n/5 *3, s=n/2)

    # make distributions unbalanced

    # loss matrix
    M = ot.dist(x.reshape((n, 1)), x.reshape((n, 1)))
    M /= M.max()
    return a,b,M

def making_uot_gausses(n,vol_of_a=1,vol_of_b=0.5):

    # bin positions
    x = np.arange(n, dtype=np.float64)

    # Gaussian distributions
    a = gauss(n, m=n/5, s=n/5)  # m= mean, s= std
    b = gauss(n, m=n/5 *3, s=n/2)*(vol_of_b/vol_of_a)

    # make distributions unbalanced

    # loss matrix
    M = ot.dist(x.reshape((n, 1)), x.reshape((n, 1)))
    M /= M.max()
    return a,b,M

def err_cal(a,b, M,reg,err):
    K = np.empty_like(M)
    np.divide(M, -reg, out=K)
    np.exp(K, out=K)
    err['errnew']=[]
    for i in range(len(err['u'])):
        G = err['u'][i][:, np.newaxis] * K * err['v'][i][np.newaxis, :]
        viol = G.sum(1) - a
        viol_2 = G.sum(0) - b
        m_viol_1 = np.linalg.norm(viol,1)
        m_viol_2 = np.linalg.norm(viol_2,1)
        err['errnew'].append(m_viol_2+m_viol_1)
    return err

def making_mnist(a_digit,b_digit,a_num,b_num):
    col = 28
    row = 28
    max_rgb = 255
    # import mnist data
    mnist = MNIST(r'D:\github\mud-game\IBP-OT\dataset\MNIST')
    mnist.gz = True
    x_train, y_train = mnist.load_training()  # 60000 samples
    x_test, y_test = mnist.load_testing()  # 10000 samples

    # 1: choose the MNIST
    dic = {}
    for i in range(10):
        dic[str(i)] = [x_train[k] for k in [j for j, x in enumerate(y_train) if x == i]]  # it is useful
    example_list_1 = dic[a_digit][a_num]
    example_list_2 = dic[b_digit][b_num]

    def trans2img(example):
        return np.asarray(example).astype(np.float32).reshape((row, col)) / max_rgb

    def trans2img_normalize(example):
        a = np.asarray(example).astype(np.float32).reshape((row, col))
        a = a / np.sum(a)
        return a

    example_img_1 = trans2img(example_list_1)
    example_img_2 = trans2img(example_list_2)

    del x_test, y_test
    del x_train, y_train

    def cost_fun(img_a, img_b):
        a = img_a.flatten()
        b = img_b.flatten()
        x_a = np.ones((len(example_list_1), 2))
        x_b = np.ones((len(example_list_2), 2))
        j = 0
        for i in range(len(example_list_1)):
            if a[i] != 0:
                x_a[j, 0] = i // col
                x_a[j, 1] = i % col
                j += 1
        x_a = x_a[:j]
        j = 0
        for i in range(len(example_list_2)):
            if b[i] != 0:
                x_b[j, 0] = i // col
                x_b[j, 1] = i % col
                j += 1
        x_b = x_b[:j]
        a = a[a != 0]
        b = b[b != 0]
        M = ot.dist(x_a, x_b)
        M /= M.max()
        return a, b, M

    aa, bb, mm = cost_fun(example_img_1, example_img_2)
    # 除去a中等于0的数
    a = aa / aa.sum()
    b = bb / bb.sum()
    M = mm / mm.max()
    return a,b,M


def making_mnist_uot(a_digit,b_digit,a_num,b_num):
    col = 28
    row = 28
    max_rgb = 255
    # import mnist data
    mnist = MNIST(r'D:\github\mud-game\IBP-OT\dataset\MNIST')
    mnist.gz = True
    x_train, y_train = mnist.load_training()  # 60000 samples
    x_test, y_test = mnist.load_testing()  # 10000 samples

    # 1: choose the MNIST
    dic = {}
    for i in range(10):
        dic[str(i)] = [x_train[k] for k in [j for j, x in enumerate(y_train) if x == i]]  # it is useful
    example_list_1 = dic[a_digit][a_num]
    example_list_2 = dic[b_digit][b_num]

    def trans2img(example):
        return np.asarray(example).astype(np.float32).reshape((row, col)) / max_rgb

    def trans2img_normalize(example):
        a = np.asarray(example).astype(np.float32).reshape((row, col))
        return a

    example_img_1 = trans2img(example_list_1)
    example_img_2 = trans2img(example_list_2)

    del x_test, y_test
    del x_train, y_train

    def cost_fun(img_a, img_b):
        a = img_a.flatten()
        b = img_b.flatten()
        x_a = np.ones((len(example_list_1), 2))
        x_b = np.ones((len(example_list_2), 2))
        j = 0
        for i in range(len(example_list_1)):
            if a[i] != 0:
                x_a[j, 0] = i // col
                x_a[j, 1] = i % col
                j += 1
        x_a = x_a[:j]
        j = 0
        for i in range(len(example_list_2)):
            if b[i] != 0:
                x_b[j, 0] = i // col
                x_b[j, 1] = i % col
                j += 1
        x_b = x_b[:j]
        a = a[a != 0]
        b = b[b != 0]
        M = ot.dist(x_a, x_b)
        M /= M.max()
        return a, b, M

    aa, bb, mm = cost_fun(example_img_1, example_img_2)
    # 除去a中等于0的数
    a = aa
    b = bb
    M = mm / mm.max()
    return a,b,M

def making_mnist_with_noise(a_digit,b_digit,a_num,b_num,noise):
    col = 28
    row = 28
    max_rgb = 255
    # import mnist data
    mnist = MNIST('../dataset/MNIST')
    mnist.gz = True
    x_train, y_train = mnist.load_training()  # 60000 samples
    x_test, y_test = mnist.load_testing()  # 10000 samples

    # 1: choose the MNIST
    dic = {}
    for i in range(10):
        dic[str(i)] = [x_train[k] for k in [j for j, x in enumerate(y_train) if x == i]]  # it is useful
    example_list_1 = dic[a_digit][a_num]
    example_list_2 = dic[b_digit][b_num]
    def trans2img(example):
        return np.asarray(example).astype(np.float32).reshape((row, col)) / max_rgb

    def trans2img_normalize(example):
        a = np.asarray(example).astype(np.float32).reshape((row, col))
        a = a / np.sum(a)
        return a

    example_img_1 = trans2img(example_list_1)
    example_img_2 = trans2img(example_list_2)
    n_img_a = example_img_1 + noise * np.random.rand(row, col)
    n_img_b = example_img_2 + noise * np.random.rand(row, col)

    del x_test, y_test
    del x_train, y_train

    def cost_fun(img_a, img_b):
        a = img_a.flatten()
        b = img_b.flatten()
        x_a = np.ones((len(example_list_1), 2))
        x_b = np.ones((len(example_list_2), 2))
        j = 0
        for i in range(len(example_list_1)):
            if a[i] != 0:
                x_a[j, 0] = i // col
                x_a[j, 1] = i % col
                j += 1
        x_a = x_a[:j]
        j = 0
        for i in range(len(example_list_2)):
            if b[i] != 0:
                x_b[j, 0] = i // col
                x_b[j, 1] = i % col
                j += 1
        x_b = x_b[:j]
        a = a[a != 0]
        b = b[b != 0]
        M = ot.dist(x_a, x_b)
        M /= M.max()
        return a, b, M

    aa, bb, mm = cost_fun(n_img_a, n_img_b)
    # 除去a中等于0的数
    a = aa / aa.sum()
    b = bb / bb.sum()
    M = mm / mm.max()
    return a,b,M

def making_mnist_uot_with_noise(a_digit,b_digit,a_num,b_num,noise):
    col = 28
    row = 28
    max_rgb = 255
    # import mnist data
    mnist = MNIST('./dataset/MNIST')
    mnist.gz = True
    x_train, y_train = mnist.load_training()  # 60000 samples
    x_test, y_test = mnist.load_testing()  # 10000 samples

    # 1: choose the MNIST
    dic = {}
    for i in range(10):
        dic[str(i)] = [x_train[k] for k in [j for j, x in enumerate(y_train) if x == i]]  # it is useful
    example_list_1 = dic[a_digit][a_num]
    example_list_2 = dic[b_digit][b_num]
    def trans2img(example):
        return np.asarray(example).astype(np.float32).reshape((row, col)) / max_rgb

    def trans2img_normalize(example):
        a = np.asarray(example).astype(np.float32).reshape((row, col))
        return a

    example_img_1 = trans2img(example_list_1)
    example_img_2 = trans2img(example_list_2)
    n_img_a = example_img_1 + noise * np.random.rand(row, col)
    n_img_b = example_img_2 + noise * np.random.rand(row, col)

    del x_test, y_test
    del x_train, y_train

    def cost_fun(img_a, img_b):
        a = img_a.flatten()
        b = img_b.flatten()
        x_a = np.ones((len(example_list_1), 2))
        x_b = np.ones((len(example_list_2), 2))
        j = 0
        for i in range(len(example_list_1)):
            if a[i] != 0:
                x_a[j, 0] = i // col
                x_a[j, 1] = i % col
                j += 1
        x_a = x_a[:j]
        j = 0
        for i in range(len(example_list_2)):
            if b[i] != 0:
                x_b[j, 0] = i // col
                x_b[j, 1] = i % col
                j += 1
        x_b = x_b[:j]
        a = a[a != 0]
        b = b[b != 0]
        M = ot.dist(x_a, x_b)
        M /= M.max()
        return a, b, M

    aa, bb, mm = cost_fun(n_img_a, n_img_b)
    # 除去a中等于0的数
    a = aa
    b = bb
    M = mm / mm.max()
    return a,b,M

def c_transform(epsilon, a, b, v, C, n_source, n_target,regularizer='entropic'):
    '''
    The goal is to recover u from the c-transform
    Parameters
    ----------
    epsilon : float
        regularization term > 0
    nu : np.ndarray(nt,)
        target measure
    v : np.ndarray(nt,)
        dual variable
    C : np.ndarray(ns, nt)
        cost matrix
    n_source : np.ndarray(ns,)
        size of the source measure
    n_target : np.ndarray(nt,)
        size of the target measure
    Returns
    -------
    u : np.ndarray(ns,)
    '''

    if regularizer=='entropic':
        u = np.zeros(n_source)
        for i in range(n_source):
            r = C[i,:] - v
            exp_v = np.exp(-r/epsilon) * b
            u[i] = - epsilon * np.log(np.sum(exp_v))
    elif regularizer=='l2':
        u = np.zeros(n_source)
        for i in range(n_source):
            u[i] = (2*epsilon*a[i] - b.sum() + C[i,:].sum())/(n_target)
    return u

def pi_transform(epsilon, a, b, u, v, M, n_source, n_target,regularizer='entropic'):
    if regularizer=='entropic':
        pi = np.exp((u[:, None] + v[None, :] - M[:, :]) / epsilon) * a[:, None] * b[None, :]
    elif regularizer == 'l2':
        pi = ((u[:, None] + v[None, :] - M[:, :]) ) / (2 * epsilon)
    return pi

def err_of_semidual(epsilon,a,b,vlist,M,regularizer = "entropic"):
    n_source = a.shape[0]
    n_target = b.shape[0]
    err = np.zeros((len(vlist)))
    for i in range(len(vlist)):
        v = vlist[i]
        u = c_transform(epsilon, a, b, v, M, n_source, n_target, regularizer=regularizer)
        pi = pi_transform(epsilon, a, b, u, v, M, n_source, n_target, regularizer=regularizer)
        err[i] = np.linalg.norm(a - pi.sum(axis=1), ord=1) + np.linalg.norm(b - pi.sum(axis=0), ord=1)
    return  err


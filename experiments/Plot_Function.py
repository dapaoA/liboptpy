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
figsize = (15, 12)
import seaborn as sns
sns.set_context("talk")
#from tqdm import tqdm


def f_speed_log(methods,f,title,opt = 0,xlabel= "Number of iteration, $k$",
                ylabel=r"$f(x_k)$"):
    for m_name in methods:
        plt.semilogy(np.fabs([f(x)-opt for x in methods[m_name].get_convergence()]), label=m_name)

    plt.legend(fontsize=fontsize)
    plt.xlabel(xlabel, fontsize=fontsize)
    plt.ylabel(ylabel, fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    _ = plt.yticks(fontsize=fontsize)
    plt.title(title+" convergence speed")
    plt.show()


def f_speed_linear(methods,f,title,opt = 0,xlabel= "Number of iteration, $k$",
                ylabel=r"$f(x_k)$"):
    for m_name in methods:
        plt.plot(np.fabs([f(x)-opt for x in methods[m_name].get_convergence()]), label=m_name)

    plt.legend(fontsize=fontsize)
    plt.xlabel(xlabel, fontsize=fontsize)
    plt.ylabel(ylabel, fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    _ = plt.yticks(fontsize=fontsize)
    plt.title(title+" convergence speed")
    plt.show()

def f_conv_comp(methods,time_dic,eplist,title):
    plt.rcParams['text.usetex'] = True
    for m_name in methods:
        plt.plot([1/x**0.5 for x in eplist],time_dic[m_name], label=m_name)
    plt.legend(fontsize=fontsize)
    plt.xlabel("r'\frac{1}{sqrt{epsilon}}', $k$", fontsize=fontsize)
    plt.ylabel("time", fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    _ = plt.yticks(fontsize=fontsize)
    plt.title(title+" convergence speed")
    plt.show()

def f_time_log(methods,f,title,opt=0,xlabel= "Time, $s$",
                ylabel=r"$f(x_k)$"):
    for m_name in methods:
        plt.semilogy(methods[m_name].get_time(),np.fabs([f(x)-opt for x in methods[m_name].get_convergence()]) ,label=m_name)

    plt.legend(fontsize=fontsize)
    plt.xlabel(xlabel, fontsize=fontsize)
    plt.ylabel(ylabel, fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    _ = plt.yticks(fontsize=fontsize)
    plt.title(title+" convergence speed")
    plt.show()

def f_log_time_log(methods,f,title,opt=0,xlabel= "Time, $s$",
                ylabel=r"$f(x_k)$"):
    for m_name in methods:
        plt.loglog(methods[m_name].get_time(),np.fabs([f(x)-opt for x in methods[m_name].get_convergence()]) ,label=m_name)

    plt.legend(fontsize=fontsize)
    plt.xlabel(xlabel, fontsize=fontsize)
    plt.ylabel(ylabel, fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    _ = plt.yticks(fontsize=fontsize)
    plt.title(title+" convergence speed")
    plt.show()
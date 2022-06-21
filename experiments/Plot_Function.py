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
figsize = (14, 10)
import seaborn as sns
sns.set_context("talk")
#from tqdm import tqdm


def f_speed_log(methods,f,title):
    for m_name in methods:
        plt.semilogy([f(x) for x in methods[m_name].get_convergence()], label=m_name)

    plt.legend(fontsize=fontsize)
    plt.xlabel("Number of iteration, $k$", fontsize=fontsize)
    plt.ylabel(r"$f(x_k)$", fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    _ = plt.yticks(fontsize=fontsize)
    plt.title(title+" convergence speed")
    plt.show()


def f_speed_linear(methods,f,title):
    for m_name in methods:
        plt.plot([f(x) for x in methods[m_name].get_convergence()], label=m_name)

    plt.legend(fontsize=fontsize)
    plt.xlabel("Number of iteration, $k$", fontsize=fontsize)
    plt.ylabel(r"$f(x_k)$", fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    _ = plt.yticks(fontsize=fontsize)
    plt.title(title+" convergence speed")
    plt.show()

def f_conv_comp(methods,time_dic,eplist,title):
    for m_name in methods:
        plt.plot([1/x**0.5 for x in eplist],time_dic[m_name], label=m_name)
    plt.legend(fontsize=fontsize)
    plt.xlabel("1/epsilon**2, $k$", fontsize=fontsize)
    plt.ylabel("time", fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    _ = plt.yticks(fontsize=fontsize)
    plt.title(title+" convergence speed")
    plt.show()

def f_time_log(methods,f,title):
    for m_name in methods:
        plt.semilogy(methods[m_name].get_time(),[f(x) for x in methods[m_name].get_convergence()] ,label=m_name)

    plt.legend(fontsize=fontsize)
    plt.xlabel("Number of iteration, $k$", fontsize=fontsize)
    plt.ylabel(r"$f(x_k)$", fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    _ = plt.yticks(fontsize=fontsize)
    plt.title(title+" convergence speed")
    plt.show()
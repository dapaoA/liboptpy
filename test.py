from cProfile import label
from cgitb import text
from tkinter import E
from turtle import color
from matplotlib import rc
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator
from matplotlib.pyplot import MultipleLocator
import math

import matplotlib.colors as cm
colors = [cm.to_hex(plt.cm.tab20(i)) for i in range(20)]



M = 1e9+7
fontsize = 60

print(M)

def lower(length, k):
    y = []
    for i in range(1, length+1):
        Mk = math.pow(M, k)
        res = 1 - math.exp(-0.5 * i * (i - 1) / Mk)
        y.append(res)
    return y

def upper(length, xi, k):
    y = []
    q = 1 - math.pow(2*xi/M, k)
    for i in range(1, length+1):
        res = 1 - math.pow(q, int(i*(i-1)/2))
        y.append(res)
    return y




# rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

rc_fonts = {
    "text.usetex": True,
}
plt.rcParams.update(rc_fonts)

# plt.rcParams["text.latex.preamble"] = r'\usepackage{libertine}'
# plt.rcParams["font.family"] = 'Linux Libertine'

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'
plt.rcParams["xtick.major.size"] = 10
plt.rcParams["ytick.major.size"] = 10

rc('xtick', labelsize=32)
rc('ytick', labelsize=32)
print(plt.rcParams["font.family"])

plt.style.use('seaborn-bright')
plt.rcParams["font.size"] = 35


n_sample = 200000
x = np.array(range(0, n_sample))

y_1  = np.array(lower(n_sample, 1)) * 100
y_2  = np.array(lower(n_sample, 2)) * 100

y_3  = np.array(upper(n_sample, 4, 1)) * 100
y_4  = np.array(upper(n_sample, 4, 2)) * 100
y_5  = np.array(upper(n_sample, 16, 1)) * 100
y_6  = np.array(upper(n_sample, 16, 2)) * 100
y_7  = np.array(upper(n_sample, 64, 1)) * 100
y_8  = np.array(upper(n_sample, 64, 2)) * 100
y_9  = np.array(upper(n_sample, 256, 1)) * 100
y_10 = np.array(upper(n_sample, 256, 2)) * 100
y_11 = np.array(upper(n_sample, 1024, 1)) * 100
y_12 = np.array(upper(n_sample, 1024, 2)) * 100


# fig, axes = plt.subplots(figsize=(6.4*2.2, 4.8*1.66), nrows=1, ncols=2, squeeze=False)
# fig, axes = plt.subplots(figsize=(8.0*4.2, 6.0), nrows=1, ncols=4, squeeze=False)
fig, axes = plt.subplots(figsize=(8.0*4.2, 6.0), nrows=1, ncols=4, squeeze=False)

linewidth = 4


axes[0][0].plot(x, y_1, color=colors[0], label=r"$1-e^{\frac{-N(N-1)}{2M}}$", lw=linewidth, linestyle = "dashed")
axes[0][0].plot(x, y_2, color=colors[1], label=r"$1-e^{\frac{-N(N-1)}{2M^2}}$", lw=linewidth, linestyle = "dashed")
axes[0][0].plot(x, y_3, color=colors[2], label=r"$1-(1-\frac{2 \cdot 4}{M})^{\frac{N(N-1)}{2}}$", lw=linewidth)
axes[0][0].plot(x, y_4, color=colors[3], label=r"$1-(1-(\frac{2 \cdot 4}{M})^2)^{\frac{N(N-1)}{2}}$", lw=linewidth)
axes[0][0].plot(x, y_5, color=colors[4], label=r"$1-(1-\frac{2 \cdot 16}{M})^{\frac{N(N-1)}{2}}$", lw=linewidth)
axes[0][0].plot(x, y_6, color=colors[5], label=r"$1-(1-(\frac{2 \cdot 16}{M})^2)^{\frac{N(N-1)}{2}}$", lw=linewidth)
axes[0][0].plot(x, y_7, color=colors[6], label=r"$1-(1-\frac{2 \cdot 64}{M})^{\frac{N(N-1)}{2}}$", lw=linewidth)
axes[0][0].plot(x, y_8, color=colors[7], label=r"$1-(1-(\frac{2 \cdot 64}{M})^2)^{\frac{N(N-1)}{2}}$", lw=linewidth)
axes[0][0].plot(x, y_9, color=colors[8], label=r"$1-(1-\frac{2 \cdot 256}{M})^{\frac{N(N-1)}{2}}$", lw=linewidth)
axes[0][0].plot(x, y_10, color=colors[9], label=r"$1-(1-(\frac{2 \cdot 256}{M})^2)^{\frac{N(N-1)}{2}}$", lw=linewidth)
axes[0][0].plot(x, y_11, color=colors[10], label=r"$1-(1-\frac{2 \cdot 1024}{M})^{\frac{N(N-1)}{2}}$", lw=linewidth)
axes[0][0].plot(x, y_12, color=colors[11], label=r"$1-(1-(\frac{2 \cdot 1014}{M})^2)^{\frac{N(N-1)}{2}}$", lw=linewidth)
axes[0][0].set_title(r"$k=1,2$", fontsize=fontsize, fontweight="bold")


axes[0][1].plot(x[:int(len(y_1)/8)], y_1[:int(len(y_1)/8)], lw=linewidth, color=colors[0], linestyle="dashed")
axes[0][1].plot(x[:int(len(y_1)/8)], y_3[:int(len(y_1)/8)], lw=linewidth, color=colors[2])
axes[0][1].plot(x[:int(len(y_1)/8)], y_5[:int(len(y_1)/8)], lw=linewidth, color=colors[4])
axes[0][1].plot(x[:int(len(y_1)/8)], y_7[:int(len(y_1)/8)], lw=linewidth, color=colors[6])
axes[0][1].plot(x[:int(len(y_1)/8)], y_9[:int(len(y_1)/8)], lw=linewidth, color=colors[8])
axes[0][1].plot(x[:int(len(y_1)/8)], y_11[:int(len(y_1)/8)], lw=linewidth, color=colors[10])
axes[0][1].set_title(r"$k=1$", fontsize=fontsize, fontweight="bold")

axes[0][2].plot(x, y_2, lw=linewidth, color=colors[1], linestyle="dashed")
axes[0][2].plot(x, y_4, lw=linewidth, color=colors[3])
axes[0][2].plot(x, y_6, lw=linewidth, color=colors[5])
axes[0][2].plot(x, y_8, lw=linewidth, color=colors[7])
axes[0][2].plot(x, y_10, lw=linewidth, color=colors[9])
axes[0][2].plot(x, y_12, lw=linewidth, color=colors[11])
axes[0][2].set_title(r"$k=2$", fontsize=fontsize, fontweight="bold")


axes[0][3].plot(x, y_2, lw=linewidth, color=colors[1], linestyle="dashed")
axes[0][3].plot(x, y_4, lw=linewidth, color=colors[3])
axes[0][3].plot(x, y_6, lw=linewidth, color=colors[5])
# axes[0][3].plot(x, y_8, lw=linewidth, color=colors[7])
# axes[0][3].plot(x, y_10, lw=linewidth, color=colors[9])
# axes[0][2].plot(x, y_12, lw=linewidth, color=colors[11])
axes[0][3].set_title(r"$k=2$", fontsize=fontsize, fontweight="bold")



# axes[0][0].set_xlabel("$N$")
axes[0][0].set_ylabel(r"Collision Probability ($\%$)")

# axes[0][1].set_xlabel("$N$")
# axes[0][1].set_ylabel(r"Collision Probability ($\%$)")

# axes[0][2].set_xlabel("$N$")
# axes[0][2].set_ylabel(r"Collision Probability ($\%$)")


# fig.legend(fontsize=25, bbox_to_anchor=(0.96,0.1), loc=1, ncol=4, facecolor='white', edgecolor='white')
fig.legend(fontsize=35, bbox_to_anchor=(0.85,0.0), loc=1, ncol=4, facecolor='white', edgecolor='white')
fig.subplots_adjust(wspace=0.3, hspace=1)
fig.tight_layout()
plt.savefig("hash.pdf", bbox_inches="tight", pad_inches=0.05)





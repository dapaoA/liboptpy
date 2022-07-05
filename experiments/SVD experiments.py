import numpy as np


n = 4
A = np.eye(n,n)
ATA = np.tile(A,(n,n))
c = np.linalg.eig(ATA)


# 特征值是n个n。。。
# 也就是说UOT的constrain 的奇艺值应该是n个根号n
# SVD 对我们这个没用，我们本来就是稀疏的。。。
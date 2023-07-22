import scipy.sparse as sp
import numpy as np
import time
m = 100000
n = 1000
d = 1
times = 10
x = np.ones((m,1))

A = sp.random(n, m, density=0.01, format='csr')

start_time = time.time()
for i in range(times):
    c = A.dot(x)
end_time = time.time()

print("Ax cost:", end_time - start_time)

row_indices = np.random.choice(n, d, replace=False)
sub_x = x

sub_A = A[row_indices,:]

start_time = time.time()
for i in range(times):
    c = sub_A.dot(sub_x)
end_time = time.time()

print("sub_A dot sub_x cost:", end_time - start_time)
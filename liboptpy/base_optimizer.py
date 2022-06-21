import numpy as np
from collections import deque
import math
import time
import random

class LineSearchOptimizer(object):
    def __init__(self, f, grad, step_size, memory_size=1, **kwargs):
        self.convergence = []
        self.time =[]
        self._f = f
        self._grad = grad
        if step_size is not None:
            step_size.assign_function(f, grad, self._f_update_x_next)
        self._step_size = step_size
        self._par = kwargs
        self._grad_mem = deque(maxlen=memory_size)
        
    def get_convergence(self):
        return self.convergence
    def get_time(self):
        return self.time
    def solve(self, x0, max_iter=100, tol=1e-6, disp=False):
        self.convergence = []
        self.time =[]
        self._x_current = x0.copy()
        self.convergence.append(self._x_current)
        start = time.time()
        self.time.append(time.time()-start)
        iteration = 0
        self._current_grad = None
        while True:
            self._h = self.get_direction(self._x_current)
            if self._current_grad is None:
                raise ValueError("Variable self._current_grad has to be initialized in method get_direction()!")
            self._grad_mem.append(self._current_grad)
            if self.check_convergence(tol):
                if disp > 0:
                    print("Required tolerance achieved!")
                break
            if disp > 1:
                print("Iteration {}/{}".format(iteration, max_iter))
                print("Current function val =", self._f(self._x_current))
                self._print_info()
            self._alpha = self.get_stepsize()
            self._update_x_next()
            self._update_x_current()
            self._append_conv()
            self.time.append(time.time() - start)
            iteration += 1
            if iteration >= max_iter:
                if disp > 0:
                    print("Maximum iteration exceeds!")
                break
        if disp:
            print("Convergence in {} iterations".format(iteration))
            print("Function value = {}".format(self._f(self._x_current)))
            self._print_info()
        return self._get_result_x()
    
    def get_direction(self, x):
        raise NotImplementedError("You have to provide method for finding direction!")
        
    def _update_x_current(self):
        self._x_current = self._x_next
        
    def _update_x_next(self):
        self._x_next = self._f_update_x_next(self._x_current, self._alpha, self._h)
        
    def _f_update_x_next(self, x, alpha, h):
        return x + alpha * h
    # this function has been reloaded in the specific optimizor function。。。
        
    def check_convergence(self, tol):
        return np.linalg.norm(self._current_grad) < tol
        
    def get_stepsize(self):
        raise NotImplementedError("You have to provide method for finding step size!")
    
    def _print_info(self):
        print("Norm of gradient = {}".format(np.linalg.norm(self._current_grad)))
    
    def _append_conv(self):
        self.convergence.append(self._x_next)
        
    def _get_result_x(self):
        return self._x_current


class Acc_LineSearchOptimizer(object):
    def __init__(self, f, grad, step_size, memory_size=1, **kwargs):
        self.convergence = []
        self.time =[]
        self._f = f
        self._grad = grad
        if step_size is not None:
            step_size.assign_function(f, grad, self._f_update_x_next,self._f_update_y_next)
        self._step_size = step_size
        self._par = kwargs
        self._grad_mem = deque(maxlen=memory_size)

    def get_convergence(self):
        return self.convergence
    def get_time(self):
        return self.time
    def solve(self, x0, max_iter=100, tol=1e-6, disp=False):
        self.convergence = []
        self.time =[]
        self._x_current = x0.copy()
        self._y_current = x0.copy()
        self.convergence.append(self._x_current)
        start = time.time()
        self.time.append(time.time()-start)
        iteration = 0
        self.tk = 1
        self._current_grad = None
        while True:
            self._h = self.get_direction(self._y_current)
            if self._current_grad is None:
                raise ValueError("Variable self._current_grad has to be initialized in method get_direction()!")
            self._grad_mem.append(self._current_grad)
            if self.check_convergence(tol):
                if disp > 0:
                    print("Required tolerance achieved!")
                break
            if disp > 1:
                print("Iteration {}/{}".format(iteration, max_iter))
                print("Current function val =", self._f(self._x_current))
                self._print_info()
            self._alpha = self.get_stepsize()
            self._update_x_next()
            self._update_y_next()
            self._update_x_current()
            self._update_y_current()
            self._update_tk()
            self._append_conv()
            self.time.append(time.time() - start)
            iteration += 1
            if iteration >= max_iter:
                if disp > 0:
                    print("Maximum iteration exceeds!")
                break
        if disp:
            print("Convergence in {} iterations".format(iteration))
            print("Function value = {}".format(self._f(self._x_current)))
            self._print_info()
        return self._get_result_x()

    def get_direction(self, x):
        raise NotImplementedError("You have to provide method for finding direction!")

    def _update_x_current(self):
        self._x_current = self._x_next
    def _update_y_current(self):
        self._y_current = self._y_next
    def _update_x_next(self):
        self._x_next = self._f_update_x_next(self._y_current, self._alpha, self._h)
    def _update_y_next(self):
        self._y_next = self._f_update_y_next(self._x_next, self.tk)
    def _update_tk(self):
        self.tk = (1 + math.sqrt(1 + 4 * self.tk ** 2)) / 2
    def _f_update_x_next(self, x, alpha, h):
        return x + alpha * h
    def _f_update_y_next(self, x, alpha, h):
        return x + alpha * h
    # this function has been reloaded in the specific optimizor function。。。

    def check_convergence(self, tol):
        return np.linalg.norm(self._current_grad) < tol

    def get_stepsize(self):
        raise NotImplementedError("You have to provide method for finding step size!")

    def _print_info(self):
        print("Norm of gradient = {}".format(np.linalg.norm(self._current_grad)))

    def _append_conv(self):
        self.convergence.append(self._x_next)

    def _get_result_x(self):
        return self._x_current


class Sto_LineSearchOptimizer(object):
    def __init__(self, f, grad, step_size, dim_a,batch=1, memory_size=1, **kwargs):
        self.convergence = []
        self.time = []
        self._f = f
        self._grad = grad
        if step_size is not None:
            step_size.assign_function(f, grad, self._f_update_x_next)
        self._step_size = step_size
        self._batch = batch
        self._dim_a = dim_a
        self._par = kwargs
        self._grad_mem = deque(maxlen=memory_size)

    def get_convergence(self):
        return self.convergence

    def get_time(self):
        return self.time

    def solve(self, x0, max_iter=100, tol=1e-6, disp=False):
        self.convergence = []
        self.time = []
        self._x_current = x0.copy()
        self.convergence.append(self._x_current)
        start = time.time()
        self.time.append(time.time() - start)
        iteration = 0
        self._current_grad = None
        np.random.seed(42)
        ids = np.arange(self._dim_a)
        while True:
            self._id = np.random.choice(ids,self._batch,replace=False)
            self._h = self.get_direction(self._x_current, self._id)
            if self._current_grad is None:
                raise ValueError("Variable self._current_grad has to be initialized in method get_direction()!")
            self._grad_mem.append(self._current_grad)
            if self.check_convergence(tol):
                if disp > 0:
                    print("Required tolerance achieved!")
                break
            if disp > 1:
                print("Iteration {}/{}".format(iteration, max_iter))
                print("Current function val =", self._f(self._x_current))
                self._print_info()
            self._alpha = self.get_stepsize()
            self._update_x_next()
            self._update_x_current()
            if(iteration%(self._dim_a//self._batch)==0):
                self._append_conv()
                self.time.append(time.time() - start)
            iteration += 1
            if iteration >= max_iter:
                if disp > 0:
                    print("Maximum iteration exceeds!")
                break
        if disp:
            print("Convergence in {} iterations".format(iteration))
            print("Function value = {}".format(self._f(self._x_current)))
            self._print_info()
        return self._get_result_x()

    def get_direction(self, x):
        raise NotImplementedError("You have to provide method for finding direction!")

    def _update_x_current(self):
        self._x_current = self._x_next

    def _update_x_next(self):
        self._x_next = self._f_update_x_next(self._x_current, self._alpha, self._h)

    def _f_update_x_next(self, x, alpha, h):
        return x + alpha * h

    # this function has been reloaded in the specific optimizor function。。。

    def check_convergence(self, tol):
        return np.linalg.norm(self._current_grad) < tol

    def get_stepsize(self):
        raise NotImplementedError("You have to provide method for finding step size!")

    def _print_info(self):
        print("Norm of gradient = {}".format(np.linalg.norm(self._current_grad)))

    def _append_conv(self):
        self.convergence.append(self._x_next)

    def _get_result_x(self):
        return self._x_current

class Sto_Var_LineSearchOptimizer(object):
    def __init__(self, f, grad, step_size, dim_a,batch=1, memory_size=1, **kwargs):
        self.convergence = []
        self.time = []
        self._f = f
        self._grad = grad
        if step_size is not None:
            step_size.assign_function(f, grad, self._f_update_x_next)
        self._step_size = step_size
        self._batch = batch
        self._dim_a = dim_a
        self._par = kwargs
        self._grad_mem = deque(maxlen=memory_size)

    def get_convergence(self):
        return self.convergence

    def get_time(self):
        return self.time

    def solve(self, x0, max_iter=100, tol=1e-6, disp=False):
        self.convergence = []
        self.time = []
        self._x_current = x0.copy()
        self.convergence.append(self._x_current)
        start = time.time()
        self.time.append(time.time() - start)
        iteration = 0
        self._current_grad = None
        np.random.seed(42)
        ids = np.arange(self._dim_a)
        self._saved_gradient = np.zeros((self._dim_a,x0.shape[0]))
        if(iteration == 0): # 先走一步全梯度的，用于保存所有梯度，ppt上看谁是这么说的
            self._h = self.get_direction(self._x_current, ids)
            self._saved_gradient = self._h
            self._sum_grad = self._saved_gradient.sum(axis=1)
        while True:
            self._id = np.random.choice(ids,self._batch,replace=False)
            self._h = self.get_direction(self._x_current, self._id)
            self._sum_grad -= self._saved_gradient[:,self._id].sum(axis=0)
            self._current_grad = self.updating_part()
            self._alpha = self.get_stepsize()
            self._update_x_next()
            if self._current_grad is None:
                raise ValueError("Variable self._current_grad has to be initialized in method get_direction()!")
            self._grad_mem.append(self._current_grad)
            if self.check_convergence(tol):
                if disp > 0:
                    print("Required tolerance achieved!")
                break
            if disp > 1:
                print("Iteration {}/{}".format(iteration, max_iter))
                print("Current function val =", self._f(self._x_current))
                self._print_info()
            self._saved_gradient[:,self._id] = self._h
            self._sum_grad += self._h.sum(axis=0)
            self._update_x_current()
            if(iteration%(self._dim_a//self._batch)==0):
                self._append_conv()
                self.time.append(time.time() - start)
            iteration += 1
            if iteration >= max_iter:
                if disp > 0:
                    print("Maximum iteration exceeds!")
                break
        if disp:
            print("Convergence in {} iterations".format(iteration))
            print("Function value = {}".format(self._f(self._x_current)))
            self._print_info()
        return self._get_result_x()

    def get_direction(self, x):
        raise NotImplementedError("You have to provide method for finding direction!")
    def updating_part(self,x):
        self._current_grad = self.variance_reduction(self.x, self.alpha, self.h,self._sum_grad,self._saved_gradient,self._id,self._dim_a)
        raise NotImplementedError("You have to provide method for finding direction!")

    def _update_x_current(self):
        self._x_current = self._x_next

    def _update_x_next(self):
        self._x_next = self._f_update_x_next(self._x_current, self._alpha, self._current_grad)

    def _f_update_x_next(self, x, alpha, h):
        return x + alpha * h

    def variance_reduction(self,x, alpha, h,sum_grad,saved_grad,id,dim_a):
    # this function has been reloaded in the specific optimizor function。。。

    def check_convergence(self, tol):
        return np.linalg.norm(self._current_grad) < tol

    def get_stepsize(self):
        raise NotImplementedError("You have to provide method for finding step size!")

    def _print_info(self):
        print("Norm of gradient = {}".format(np.linalg.norm(self._current_grad)))

    def _append_conv(self):
        self.convergence.append(self._x_next)

    def _get_result_x(self):
        return self._x_current

class TrustRegionOptimizer(object):
    def __init__(self):
        raise NotImplementedError("Trust region methods are not implemented yet")
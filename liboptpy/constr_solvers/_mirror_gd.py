import numpy as np
from ..base_optimizer import LineSearchOptimizer,Sto_LineSearchOptimizer,Sto_Var_LineSearchOptimizer
import math
class MirrorD(LineSearchOptimizer):
    
    '''
    Class represents projected gradient method
    '''
    
    def __init__(self, f, grad, projector, step_size):
        super().__init__(f, grad, step_size)
        self._projector = projector
        
    def get_direction(self, x):
        self._current_grad = self._grad(x)
        return -self._current_grad
    
    def _f_update_x_next(self, x, alpha, h):
        return self._projector(np.multiply(x,np.exp(alpha * h)))
    
    def check_convergence(self, tol):
        if len(self.convergence) == 1:
            return False
        if math.fabs(self._f(self.convergence[-2]) - self._f(self.convergence[-1])) < tol:
            return True
        else:
            return False
        
    def get_stepsize(self):
        return self._step_size.get_stepsize(-self._grad_mem[-1], self.convergence[-1], len(self.convergence))
    
    def _print_info(self):
        print("Difference in function values = {}".format(self._f(self.convergence[-2]) - self._f(self.convergence[-1])))
        print("Difference in argument = {}".format(np.linalg.norm(self.convergence[-1] - self.convergence[-2])))


class Sto_MirrorD(Sto_LineSearchOptimizer):
    '''
    Class represents projected gradient method
    '''

    def __init__(self, f, grad, projector, step_size, dim_a,batch):
        super().__init__(f, grad, step_size,dim_a,batch=batch)
        self._projector = projector

    def get_direction(self, x, id):
        self._current_grad = self._grad(x, id)
        return -self._current_grad

    def _f_update_x_next(self, x, alpha, h):
        return self._projector(np.multiply(x, np.exp(alpha * h)))

    def check_convergence(self, tol):
        if len(self.convergence) == 1:
            return False
        if math.fabs(self._f(self.convergence[-2]) - self._f(self.convergence[-1])) < tol:
            return True
        else:
            return False

    def get_stepsize(self):
        return self._step_size.get_stepsize(-self._grad_mem[-1], self.convergence[-1], len(self.convergence))

    def _print_info(self):
        print(
            "Difference in function values = {}".format(self._f(self.convergence[-2]) - self._f(self.convergence[-1])))
        print("Difference in argument = {}".format(np.linalg.norm(self.convergence[-1] - self.convergence[-2])))


class Sag_MirrorD(Sto_Var_LineSearchOptimizer):
    '''
    Class represents projected gradient method
    '''

    def __init__(self, f, grad, projector, step_size, dim_a,batch):
        super().__init__(f, grad, step_size,dim_a,batch=batch)
        self._projector = projector

    def get_direction(self, x, id):
        return self._grad(x, id)

    def updating_part(self, h,sum_grad,saved_grad,id,dim_a):
        return (h.sum(axis=1) + saved_grad[:, id].sum(axis=1))/h.shape[1] - sum_grad / dim_a * (dim_a + 1)

    def _f_update_x_next(self, x,alpha,_current_grad):
        return self._projector(np.multiply(x, np.exp(alpha * _current_grad)))


    def check_convergence(self, tol):
        if len(self.convergence) == 1:
            return False
        if math.fabs(self._f(self.convergence[-2]) - self._f(self.convergence[-1])) < tol:
            return True
        else:
            return False

    def get_stepsize(self):
        return self._step_size.get_stepsize(-self._grad_mem[-1], self.convergence[-1], len(self.convergence))

    def _print_info(self):
        print(
            "Difference in function values = {}".format(self._f(self.convergence[-2]) - self._f(self.convergence[-1])))
        print("Difference in argument = {}".format(np.linalg.norm(self.convergence[-1] - self.convergence[-2])))
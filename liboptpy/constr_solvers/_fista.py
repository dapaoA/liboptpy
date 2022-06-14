import math

import numpy as np
from ..base_optimizer import Acc_LineSearchOptimizer

class FISTA(Acc_LineSearchOptimizer):
    
    '''
    Class represents projected gradient method
    '''
    
    def __init__(self, f, grad, projector, step_size):
        super().__init__(f, grad, step_size)
        if step_size is not None:
            step_size.assign_function(f, grad, self._f_update_x_next,self._f_update_y_next)
        self._projector = projector

    def get_direction(self, x):
        self._current_grad = self._grad(x)
        return -self._current_grad
    
    def _f_update_x_next(self, x, alpha, h):
        return self._projector(x + alpha * h)
    def _f_update_y_next(self, x,t_k ):
        t_knew = (1+math.sqrt(1+4*t_k**2))/2
        return x+((t_k-1)/t_knew)*(x - self._x_current)

    def check_convergence(self, tol):
        if len(self.convergence) == 1:
            return False
        if math.fabs(self._f(self.convergence[-2]) - self._f(self.convergence[-1])) < tol:
            return True
        else:
            return False
        
    def get_stepsize(self):
        return self._step_size.get_stepsize(-self._grad_mem[-1], self._y_current, len(self.convergence),self.tk)
    
    def _print_info(self):
        print("Difference in function values = {}".format(self._f(self.convergence[-2]) - self._f(self.convergence[-1])))
        print("Difference in argument = {}".format(np.linalg.norm(self.convergence[-1] - self.convergence[-2])))

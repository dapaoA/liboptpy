import numpy as np
from ..base_optimizer import LineSearchOptimizer

class Dynamic_ProjectedGD(LineSearchOptimizer):
    
    '''
    Class represents projected gradient method
    '''
    
    def __init__(self, f, grad, projector, step_size):
        super().__init__(f, grad, step_size)
        self._projector = projector

    def solve(self, x0, max_iter=100, tol=1e-6, disp=False):
        self.convergence = []
        self._x_current = x0.copy()
        self.convergence.append(self._x_current)
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
            self._alpha, self._x_next = self.get_stepsize()
            self._update_x_current()
            self._append_conv()
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
        self._current_grad = self._grad(x)
        return -self._current_grad
    
    def _f_update_x_next(self, x, alpha, h):
        return self._projector(x + alpha * h)
    
    def check_convergence(self, tol):
        if len(self.convergence) == 1:
            return False
        if self._f(self.convergence[-2]) - self._f(self.convergence[-1]) < tol:
            return True
        else:
            return False
        
    def get_stepsize(self):
        return self._step_size.get_stepsize(-self._grad_mem[-1], self.convergence[-1], len(self.convergence))
    
    def _print_info(self):
        print("Difference in function values = {}".format(self._f(self.convergence[-2]) - self._f(self.convergence[-1])))
        print("Difference in argument = {}".format(np.linalg.norm(self.convergence[-1] - self.convergence[-2])))

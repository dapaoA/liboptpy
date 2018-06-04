import numpy as np

__all__ = ["ConstantStepSize", "Backtracking", "ExactLineSearch4Quad", "InvIterStepSize", "ScaledInvIterStepSize"]

class StepSize(object):
    '''
    Base class for all classes for defining step size
    '''
    def __init__(self):
        pass
    def get_stepsize(self, *args, **kwargs):
        raise NotImplementedError("Method to get current step size has to be implemented!")
        
    def assign_function(self, f, grad):
        pass
    
class ConstantStepSize(StepSize):
    '''
    Class represents interface for constant step size 
    '''
    def __init__(self, stepsize):
        self.stepsize = stepsize
    
    def get_stepsize(self, h, x, num_iter):
        return self.stepsize
    
class ScaledConstantStepSize(StepSize):
    def __init__(self, stepsize):
        self.stepsize = stepsize
    
    def get_stepsize(self, h, x, num_iter):
        return self.stepsize / np.linalg.norm(h)
    
class InvIterStepSize(StepSize):
    def __init__(self):
        pass
    
    def get_stepsize(self, h, x, num_iter):
        return 1. / num_iter
    
class ScaledInvIterStepSize(StepSize):
    def __init__(self):
        pass
    
    def get_stepsize(self, h, x, num_iter):
        s = 1. / num_iter
        return s / np.linalg.norm(h)
    
class InvSqrootIterStepSize(StepSize):
    def __init__(self):
        pass
    
    def get_stepsize(self, h, x, num_iter):
        return 1. / np.sqrt(num_iter)
    
class ProjectedArmijo(StepSize):
    def __init__(self, rho, beta, init_alpha):
        self._rho = rho
        self._beta = beta
        self._init_alpha = init_alpha
        
    def assign_function(self, f, grad):
        self._f = f
        self._grad = grad
    
    def get_stepsize(self, h, x, num_iter):
        alpha = self._init_alpha
        x_next = self._proj(x + self._init_alpha * h)
        current_f = self._f(x)
        while True:
            if np.isnan(self._f(x + alpha * h)):
                    alpha *= self._rho
            else:
                if self._f(x + alpha * h) >= current_f + self._beta * alpha * h.dot(x_next - x):
                    alpha *= self._rho
                else:
                    break
            if alpha < 1e-16:
                break
            x_next = self._proj(x + alpha * h)
        return alpha
        
    
class Backtracking(StepSize):
    '''
    Class represents different rules for backtracking search of step size
    '''
    def __init__(self, rule_type, **kwargs):
        self.rule = rule_type
        self.par = kwargs
    
    def assign_function(self, f, grad):
        self._f = f
        self._grad = grad
    
    def get_stepsize(self, h, x, num_iter):
        alpha = self.par["init_alpha"]
        if self.rule == "Armijo":
            rho = self.par["rho"]
            beta = self.par["beta"]
            assert beta < 0.5, "Armijo rule is applicable for beta less than 0.5"
            assert rho < 1, "Decay factor has to be less than 1"
            current_grad = self._grad(x)
            current_f = self._f(x)
            while True:
                if np.isnan(self._f(x + alpha * h)):
                    alpha *= rho
                else:
                    if self._f(x + alpha * h) >= current_f + beta * alpha * current_grad.dot(h):
                        alpha *= rho
                    else:
                        break
                if alpha < 1e-16:
                    break
            return alpha
        elif self.rule == "Wolfe":
            rho = self.par["rho"]
            assert rho < 1, "Decay factor has to be less than 1"
            beta1 = self.par["beta1"]
            beta2 = self.par["beta2"]
            assert 0 < beta1 < beta2 < 1, "Wolfe rule is applicable for betas such that 0 < beta1 < beta2 < 1"
            current_grad = self._grad(x)
            current_f = self._f(x)
            while True: 
                if np.isnan(self._f(x + alpha * h)):
                    alpha *= rho
                else:
                    if self._f(x + alpha * h) > current_f + beta1 * alpha * current_grad.dot(h):
                        alpha *= rho
                    elif h.dot(self._grad(x + alpha * h)) < beta2 * h.dot(current_grad):
                        alpha *= rho
                    else:
                        break
                if alpha < 1e-10:
                    break
            return alpha
        elif self.rule == "Goldstein":
            pass
        elif self.rule == "Wolfe strong":
            rho = self.par["rho"]
            assert rho < 1, "Decay factor has to be less than 1"
            beta1 = self.par["beta1"]
            beta2 = self.par["beta2"]
            assert 0 < beta1 < beta2 < 1, "Wolfe rule is applicable for betas such that 0 < beta1 < beta2 < 1"
            current_grad = self._grad(x)
            current_f = self._f(x)
            while True: 
                if np.isnan(self._f(x + alpha * h)):
                    alpha *= rho
                else:
                    if self._f(x + alpha * h) > current_f + beta1 * alpha * current_grad.dot(h):
                        alpha *= rho
                    elif np.abs(h.dot(self._grad(x + alpha * h))) > beta2 * np.abs(h.dot(current_grad)):
                        alpha *= rho
                    else:
                        break
                if alpha < 1e-10:
                    break
            return alpha
        else:
            raise NotImplementedError("Available rules for backtracking are 'Armijo', 'Goldstein', 'Wolfe' and 'Wolfe strong'")

class ExactLineSearch4Quad(StepSize):
    def __init__(self, A, b=None):
        self._A = A
        if b is None:
            self._b = np.zeros(A.shape[0])
        else:
            self._b = b
    
    def get_stepsize(self, h, x, num_iter):
        return h.dot(self._b - self._A.dot(x)) / h.dot(self._A.dot(h))
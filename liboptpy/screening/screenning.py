import numpy as np
import math


class screener(object):
    def __init__(self,w, X, y,c,lam, reg="l1" ):
        self.w = w  # primal variable
        self.X = X # constrain matrix
        self.y = y # distributions
        self.c = c
        self.lam = lam
        self.reg = reg

    def update(self):
        return 0


class safe_screening(screener):
    def __init__(self,w, X,y,c, lam,reg="l1" ):
        super().__init__(w,  X,y, c,lam,reg="l1" )
        # self.w = w  # primal variable
        # self.dual = dual  # dual variable
        # self.alpha  #auxiliary
        # self.X = X # constrain matrix
        # self.reg = reg

    def update(self):
        self.w_screening = np.ones_like(self.w)
        self.yTX = self.X.T.dot(self.y)
        self.lam_max = (self.yTX[np.where(self.c>0)]/self.c[np.where(self.c>0)]).max()
        self.countzeros = 0
        for i in range(self.w_screening.shape[0]):
            if(self.yTX[i] < self.lam*self.c[i] - 2 +2* self.lam/self.lam_max):
                self.w_screening[i] = 0
                self.countzeros += 1
        print("screening percent is: ",self.countzeros/self.w_screening.shape[0])
        return self.w_screening


class safe_screening_2(screener):
    def __init__(self,w, X,y,c, lam,reg="l1" ):
        super().__init__(w,  X,y, c,lam,reg="l1" )
        # self.w = w  # primal variable
        # self.dual = dual  # dual variable
        # self.alpha  #auxiliary
        # self.X = X # constrain matrix
        # self.reg = reg

    def update(self):
        self.w_screening = np.ones_like(self.w)
        self.yTX = self.X.T.dot(self.y)
        self.lam_max = self.yTX.max()/self.c[np.where(self.c>0)].min()
        self.countzeros = 0
        for i in range(self.w_screening.shape[0]):
            if(self.yTX[i] < self.lam_max*()):
                self.w_screening[i] = 0
                self.countzeros += 1
        print("screening percent is: ",self.countzeros/self.w_screening.shape[0])
        return self.w_screening

    def P(self,k):
        if (np.linalg.norm(self.g)*np.linalg.norm(self.X[k])**2>=self.d * self.X[k]*self.g):
            P = self.sita_start_0 * self.X[k, :] + self.phi_k * self.d_hat
        else:
            P = -self.y*self.X[k] + np.linalg.norm(self.X[k])*self.d

class dynamic_screening(screener):
    def __init__(self,w, X,y,c, lam,reg="l1" ):
        super().__init__(w,  X,y, c,lam,reg="l1" )

    def update(self,theta):
        self.theta = theta
        self.w_screening = np.ones_like(self.w)
        self.cc = self.y/self.lam
        self.thetaTX = self.X.T.dot(self.theta)
        self.countzeros = 0
        self.mu = max(min(np.dot(self.theta,self.y)/(self.lam*(np.dot(self.theta,self.theta))),
                          1/self.thetaTX.max()),-1/self.thetaTX.max())
        self.r_theta = np.linalg.norm(self.y/self.lam-self.mu*self.theta)
        for i in range(self.w_screening.shape[0]):
            if(self.c[i]-abs(self.X[:,i].T.dot(self.cc)[0])>2*self.r_theta):
                self.w_screening[i] = 0
                self.countzeros += 1
        print("screening percent is: ",self.countzeros/self.w_screening.shape[0])
        return self.w_screening


class sasvi_screening(screener):
    def __init__(self, w, X, y, c, lam, reg="l1"):
        super().__init__(w, X, y, c, lam, reg="l1")

    def update(self,w):
        self.w = w
        self.theta_hat = self.X.dot(self.w) -self.y
        self.Xw = self.X.dot(self.w)
        self.r = 0.5 * np.linalg.norm(self.theta_hat-self.y)
        self.theta_o = 0.5*(self.theta_hat + self.y)
        self.delta = self.lam *np.dot(self.c,self.w) - np.dot(self.theta_o,self.X.dot(self.w))
        self.w_screening = np.ones_like(self.w)

        for i in range(self.w_screening.shape[0]):
            xiXw = np.dot(self.X[:, i], self.Xw)
            if (self.r* xiXw<=self.delta):
                if (self.X[:, i].dot(self.theta_o) +self.r*np.linalg.norm(self.X[:,i])> 1):
                    self.w_screening[i] = 1
                    self.countzeros += 1
            else:
                if (self.X[:, i].dot(self.theta_o)+xiXw/np.dot(self.Xw,self.Xw) + np.linalg.norm(self.X[:,i]
                    - xiXw/np.dot(self.Xw,self.Xw)*self.Xw)*math.sqrt(self.r**2-1/(np.dot(self.Xw,self.Xw)*self.delta**2)) > 1):
                    self.w_screening[i] = 1
                    self.countzeros += 1

        print("screening percent is: ", self.countzeros / self.w_screening.shape[0])
        return self.w_screening


import numpy as np
import math
import matplotlib.pyplot as plt




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
    def __init__(self, w, X, y, c, lam, reg="l1",sratio=0):
        super().__init__(w, X, y, c, lam, reg="l1")
        self.sratio = 0
    def update(self,w):
        self.w = w
        self.Xw = self.X.dot(self.w)
        self.theta_hat = self.lam * min(self.c)*(self.y - self.Xw)/max(1,np.max(self.X.T.dot(self.y-self.Xw)))
        self.Xw_norm2 = np.dot(self.Xw,self.Xw)

        self.r = 0.5 * np.linalg.norm(self.theta_hat-self.y)
        self.theta_o = 0.5*(self.theta_hat + self.y)
        self.delta = self.lam *np.dot(self.c,self.w) - np.dot(self.theta_o,self.Xw )
        self.w_screening = np.ones_like(self.w)
        self.countzeros = 0
        for i in range(self.w_screening.shape[0]):
            xiXw = self.X[:, i].T.dot(self.Xw)[0]
            xi_norm = math.sqrt(self.X[:, i].T.dot(self.X[:, i]).toarray()[0][0])
            if (self.r/xi_norm* xiXw<=self.delta):
                if (self.X[:, i].T.dot(self.theta_o)[0] +self.r*xi_norm< self.lam*self.c[i]):
                    self.w_screening[i] = 0
                    self.countzeros += 1
            else:
                if (self.X[:, i].T.dot(self.theta_o)[0]+xiXw/self.Xw_norm2*self.delta + np.linalg.norm(self.X[:,i].toarray()
                        - xiXw/self.Xw_norm2*self.Xw)*math.sqrt(self.r**2-1/(self.Xw_norm2)*self.delta**2) < self.lam*self.c[i]):
                    self.w_screening[i] = 0
                    self.countzeros += 1
        if(self.countzeros!=0):
            if(self.countzeros / self.w_screening.shape[0]>self.sratio):
                self.sratio = self.countzeros / self.w_screening.shape[0]
                print("screening percent is: ", self.countzeros / self.w_screening.shape[0])
                print("running")
                plt.imshow(self.w_screening.reshape(10, 10))
                plt.title('running!')
                plt.colorbar(aspect=40, pad=0.08, shrink=0.6,
                         orientation='horizontal', extend='both')
                plt.show()
        return self.w_screening


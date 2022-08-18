import numpy as np



class screener(object):
    def __init__(self,w, dual, alpha, X, y,lam, reg="l1" ):
        self.w = w  # primal variable
        self.dual = dual  # dual variable
        self.alpha = alpha #auxiliary
        self.X = X # constrain matrix
        self.y = y
        self.lam = lam
        self.reg = reg

    def update(self):
        return 0


class safe_screening(screener):
    def __init__(self,w, dual, alpha, X,y, lam,reg="l1" ):
        super().__init__(w, dual, alpha, X,y, lam,reg="l1" )
        # self.w = w  # primal variable
        # self.dual = dual  # dual variable
        # self.alpha  #auxiliary
        # self.X = X # constrain matrix
        # self.reg = reg

    def update(self,w_start_0):
        self.lam_0 = self.X.T.dot(self.y).max()
        self.w_start_0 = w_start_0
        self.sita_start_0 = self.X.dot(self.w_start_0)-self.y
        self.g = self.sita_start_0 + self.y
        self.alpha_0 = np.dot(self.sita_start_0,self.sita_start_0)
        self.beta_0 = abs(np.dot(self.y,self.sita_start_0))
        self.r = self.beta_0**2/self.alpha_0*(1-(1-self.alpha_0/self.beta_0*self.lam/self.lam_0)**2)
        self.d = (np.dot(self.y,self.y)-2*self.r)
        self.d_hat = (self.d**2-np.dot(self.g,self.g))**0.5

        self.ids = np.zeros(self.X.shape[1]) # id , if its = 1 , it means the w_i = 0
        if(self.reg =="l1"):
            for k in range(self.X.shape[1]):
                self.phi_k = (self.X[:,k].T.dot(self.X[:,k])-(self.X[:,k].T.dot(self.g)**2)/np.dot(self.g,self.g))**0.5

                if (self.lam > self.P(k)):
                    self.ids[k] = 1


        return 0

    def P(self,k):
        if (np.linalg.norm(self.g)*np.linalg.norm(self.X[k])**2>=self.d * self.X[k]*self.g):
            P = self.sita_start_0 * self.X[k, :] + self.phi_k * self.d_hat
        else:
            P = -self.y*self.X[k] + np.linalg.norm(self.X[k])*self.d




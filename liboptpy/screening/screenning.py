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
        self.Xw_norm2 = np.dot(self.Xw,self.Xw)
        self.theta_hat = self.y - self.Xw
        self.g = self.lam *np.dot(self.c,self.w)
        self.theta_hat = self.lam * min(self.c)*(self.y - self.Xw)/max(1,np.max(self.X.T.dot(self.y-self.Xw)))

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
                        - xiXw/self.Xw_norm2*self.Xw)*math.sqrt(max(self.r**2-1/(self.Xw_norm2)*self.delta**2,0)) < self.lam*self.c[i]):
                    self.w_screening[i] = 0
                    self.countzeros += 1
        if(self.countzeros!=0):
            if(self.countzeros / self.w_screening.shape[0]>self.sratio):
                self.sratio = self.countzeros / self.w_screening.shape[0]
                print("screening percent is: ", self.countzeros / self.w_screening.shape[0])
                print("running")
                plt.imshow(self.w_screening.reshape(math.floor(self.theta_o.shape[0]/2), math.floor(self.theta_o.shape[0]/2)))
                plt.title('running!')
                plt.colorbar(aspect=40, pad=0.08, shrink=0.6,
                         orientation='horizontal', extend='both')
                plt.show()
        return self.w_screening

class sasvi_screening_test(screener):
    def __init__(self, w, X, y, c, lam, reg="l1",sratio=0,solution=None):
        super().__init__(w, X, y, c, lam, reg="l1")
        self.sratio = 0
        self.solution = solution
    def update(self,w):
        self.w = w
        self.Xw = self.X.dot(self.w)
        self.Xw_norm2 = np.dot(self.Xw,self.Xw)
        self.theta_hat = self.y - self.Xw
        self.g = self.lam *np.dot(self.c,self.w)
        # if (self.g - self.Xw.dot(self.theta_hat) < 0):
        #     self.theta_hat += (self.g - np.dot(self.theta_hat, self.Xw)) / self.Xw_norm2 * self.Xw
        #
        # print('primal: ',np.linalg.norm(self.solution-self.w))
        # self.theta_best = self.y - self.X.dot(self.solution)
        # print('dual before projection: ',np.linalg.norm(self.theta_hat-self.theta_best))
        # beilv = self.X.T.dot(self.y-self.Xw)/(self.lam * min(self.c))

        rank = math.floor(self.theta_hat.shape[0] / 2)
        beilv = (self.X.T.dot(self.theta_hat) / (self.lam * self.c)).reshape(rank,rank)
        self.theta_hat[:rank] = (self.y - self.Xw)[:rank]/np.where(beilv.max(axis=1)>1,beilv.max(axis=1),1.0)
        self.theta_hat[rank:] = (self.y - self.Xw)[rank:]/np.where(beilv.max(axis=0)>1,beilv.max(axis=0),1.0)
        print('max ratio',max(1,np.max(beilv)))
        print('increase ratio',((self.y - self.Xw)/self.theta_hat).min())

        plt.imshow(beilv)
        plt.title('ratio!')
        plt.colorbar(aspect=40, pad=0.08, shrink=0.6,
                     orientation='horizontal', extend='both')
        plt.show()

        plt.imshow(w.reshape(rank,rank))
        plt.title('ratio!')
        plt.colorbar(aspect=40, pad=0.08, shrink=0.6,
                     orientation='horizontal', extend='both')
        plt.show()
#         print('dual after projection: ',np.linalg.norm(self.theta_hat-self.theta_best))

        #ggg = self.X.T.dot(self.theta_hat)
        #fff = self.lam * self.c
        #gfr = (ggg - fff).reshape((30, 30))
        #plt.imshow(np.where(gfr < 0, gfr, -1))
        #plt.title('running!')
        #plt.colorbar(aspect=40, pad=0.08, shrink=0.6,
        #             orientation='horizontal', extend='both')
        #plt.show()

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
                        - xiXw/self.Xw_norm2*self.Xw)*math.sqrt(max(self.r**2-1/(self.Xw_norm2)*self.delta**2,0)) < self.lam*self.c[i]):
                    self.w_screening[i] = 0
                    self.countzeros += 1
        if (self.countzeros != 0):
            if (self.countzeros / self.w_screening.shape[0] > self.sratio):
                self.sratio = self.countzeros / self.w_screening.shape[0]
                print("screening percent is: ", self.countzeros / self.w_screening.shape[0])
                print("running")
                plt.imshow(self.w_screening.reshape(math.floor(self.theta_o.shape[0] / 2),
                                                        math.floor(self.theta_o.shape[0] / 2)))
                plt.title('running!')
                plt.colorbar(aspect=40, pad=0.08, shrink=0.6,
                             orientation='horizontal', extend='both')
                plt.show()
        return self.w_screening

class sasvi_screening_zero_test(screener):
    def __init__(self, w, X, y, c, lam, reg="l1",sratio=0,solution=None):
        super().__init__(w, X, y, c, lam, reg="l1")
        self.sratio = 0
        self.solution = solution
    def update(self,w):
        self.w = w
        self.Xw = self.X.dot(self.w)
        self.Xw_norm2 = np.dot(self.Xw,self.Xw)
        self.theta_hat = self.y - self.Xw
        self.g = self.lam *np.dot(self.c,self.w)
        # print('primal: ',np.linalg.norm(self.solution-self.w))
        # self.theta_best = self.y - self.X.dot(self.solution)
        # print('dual before projection: ',np.linalg.norm(self.theta_hat-self.theta_best))
        self.theta_hat = self.lam * min(self.c)*(self.y - self.Xw)/max(1,np.max(self.X.T.dot(self.y-self.Xw)))
        self.theta_hat = -self.theta_hat
#         print('dual after projection: ',np.linalg.norm(self.theta_hat-self.theta_best))

        #ggg = self.X.T.dot(self.theta_hat)
        #fff = self.lam * self.c
        #gfr = (ggg - fff).reshape((30, 30))
        #plt.imshow(np.where(gfr < 0, gfr, -1))
        #plt.title('running!')
        #plt.colorbar(aspect=40, pad=0.08, shrink=0.6,
        #             orientation='horizontal', extend='both')
        #plt.show()

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
                        - xiXw/self.Xw_norm2*self.Xw)*math.sqrt(max(self.r**2-1/(self.Xw_norm2)*self.delta**2,0)) < self.lam*self.c[i]):
                    self.w_screening[i] = 0
                    self.countzeros += 1
        if (self.countzeros != 0):
            if (self.countzeros / self.w_screening.shape[0] > self.sratio):
                self.sratio = self.countzeros / self.w_screening.shape[0]
                print("screening percent is: ", self.countzeros / self.w_screening.shape[0])
                print("running")
                plt.imshow(self.w_screening.reshape(math.floor(self.theta_o.shape[0] / 2),
                                                        math.floor(self.theta_o.shape[0] / 2)))
                plt.title('running!')
                plt.colorbar(aspect=40, pad=0.08, shrink=0.6,
                             orientation='horizontal', extend='both')
                plt.show()
        return self.w_screening

class sasvi_screening_c_trans_test(screener):
    def __init__(self, w, X, y, c, lam, reg="l1",sratio=0,solution=None):
        super().__init__(w, X, y, c, lam, reg="l1")
        self.sratio = 0
        self.solution = solution
    def update(self,w):
        self.w = w
        self.Xw = self.X.dot(self.w)
        self.Xw_norm2 = np.dot(self.Xw,self.Xw)
        self.theta_hat = self.y - self.Xw
        self.g = self.lam *np.dot(self.c,self.w)
        # if (self.g - self.Xw.dot(self.theta_hat) < 0):
        #     self.theta_hat += (self.g - np.dot(self.theta_hat, self.Xw)) / self.Xw_norm2 * self.Xw
        #
        # print('primal: ',np.linalg.norm(self.solution-self.w))
        # self.theta_best = self.y - self.X.dot(self.solution)
        # print('dual before projection: ',np.linalg.norm(self.theta_hat-self.theta_best))
        self.theta_hat = self.y-self.Xw
        rank = int(self.theta_hat.shape[0]/2)
        for i in range(rank):
            self.theta_hat[i] = self.lam * (self.c[i*rank:(i+1)*rank]- self.theta_hat[rank:]).min()
#         print('dual after projection: ',np.linalg.norm(self.theta_hat-self.theta_best))

        #ggg = self.X.T.dot(self.theta_hat)
        #fff = self.lam * self.c
        #gfr = (ggg - fff).reshape((30, 30))
        #plt.imshow(np.where(gfr < 0, gfr, -1))
        #plt.title('running!')
        #plt.colorbar(aspect=40, pad=0.08, shrink=0.6,
        #             orientation='horizontal', extend='both')
        #plt.show()

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
                        - xiXw/self.Xw_norm2*self.Xw)*math.sqrt(max(self.r**2-1/(self.Xw_norm2)*self.delta**2,0)) < self.lam*self.c[i]):
                    self.w_screening[i] = 0
                    self.countzeros += 1
        if (self.countzeros != 0):
            if (self.countzeros / self.w_screening.shape[0] > self.sratio):
                self.sratio = self.countzeros / self.w_screening.shape[0]
                print("screening percent is: ", self.countzeros / self.w_screening.shape[0])
                print("running")
                plt.imshow(self.w_screening.reshape(math.floor(self.theta_o.shape[0] / 2),
                                                        math.floor(self.theta_o.shape[0] / 2)))
                plt.title('running!')
                plt.colorbar(aspect=40, pad=0.08, shrink=0.6,
                             orientation='horizontal', extend='both')
                plt.show()
        return self.w_screening





def dual(theta,y):
    return -0.5* np.dot(theta,theta)+np.dot(y,theta)


class sasvi_screening_ratio_pic_test(screener):
    def __init__(self, w, X, y, c, lam, reg="l1",sratio=0,solution=None):
        super().__init__(w, X, y, c, lam, reg="l1")
        self.sratio = 0
        self.solution = solution
    def update(self,w):
        self.w = w
        self.Xw = self.X.dot(self.w)
        self.Xw_norm2 = np.dot(self.Xw,self.Xw)
        self.theta_hat = self.y - self.Xw
        self.g = self.lam *np.dot(self.c,self.w)
        self.rank = math.floor(self.theta_hat.shape[0] / 2)
        self.theta_p1,beilv1 = self.projection_normal(self.theta_hat)
        self.theta_p2,beilv2 = self.projection_revise(self.theta_hat)
        # if (self.g - self.Xw.dot(self.theta_hat) < 0):
        #     self.theta_hat += (self.g - np.dot(self.theta_hat, self.Xw)) / self.Xw_norm2 * self.Xw
        #
        # print('primal: ',np.linalg.norm(self.solution-self.w))
        self.theta_best = self.y - self.X.dot(self.solution)
        # print('dual before projection: ',np.linalg.norm(self.theta_hat-self.theta_best))
        # beilv = self.X.T.dot(self.y-self.Xw)/(self.lam * min(self.c))


#         print('dual after projection: ',np.linalg.norm(self.theta_hat-self.theta_best))

        #ggg = self.X.T.dot(self.theta_hat)
        #fff = self.lam * self.c
        #gfr = (ggg - fff).reshape((30, 30))
        #plt.imshow(np.where(gfr < 0, gfr, -1))
        #plt.title('running!')
        #plt.colorbar(aspect=40, pad=0.08, shrink=0.6,
        #             orientation='horizontal', extend='both')
        #plt.show()


        d_opt_alg = np.linalg.norm(self.theta_best-self.theta_hat)
        d_opt_proj1 =  np.linalg.norm(self.theta_best-self.theta_p1)
        d_opt_proj2 =  np.linalg.norm(self.theta_best-self.theta_p2)
        d_alg_proj1 =  np.linalg.norm(self.theta_hat-self.theta_p1)
        d_alg_proj2 = np.linalg.norm(self.theta_hat-self.theta_p2)
        dis = {"opt_alg": d_opt_alg,
               "opt_proj1": d_opt_proj1,
               "opt_proj2": d_opt_proj2,
               "alg_proj1": d_alg_proj1,
               "alg_proj2": d_alg_proj2,
               }

        self.w_screening_area1 = self.screening_area(self.theta_p1)
        self.w_screening_area2 = self.screening_area(self.theta_p2)
        self.w_screening_s = self.screening_point(self.theta_hat )
        self.w_screening_s_p1 = self.screening_point(self.theta_p1)
        self.w_screening_s_p2 = self.screening_point(self.theta_p2 )

        w_screening = { "screening_area1":self.w_screening_area1,
                      "screening_area2": self.w_screening_area2,
                      "screening_ps": self.w_screening_s,
                      "screening_p1": self.w_screening_s_p1,
                      "screening_p2": self.w_screening_s_p2,

        }

        return dis,w_screening

    def projection_normal(self, alg_theta):
        beilv = self.X.T.dot(alg_theta)/(self.lam * self.c)
        out = alg_theta / max(1,beilv.max())
        return out, beilv

    def projection_revise(self,alg_theta):
        beilv = (self.X.T.dot(alg_theta) / (self.lam * self.c)).reshape(self.rank,self.rank)
        out = np.ones_like(alg_theta)
        out[:self.rank] = (alg_theta)[:self.rank]/np.where(beilv.max(axis=1)>1,beilv.max(axis=1),1.0)
        out[self.rank:] = (alg_theta)[self.rank:]/np.where(beilv.max(axis=0)>1,beilv.max(axis=0),1.0)
        return out, beilv
    def screening_area(self,theta_projected):
        self.countzeros = 0
        self.w_screening = np.ones_like(self.w)
        self.theta_o = 0.5*(theta_projected + self.y)
        self.r = 0.5 * np.linalg.norm(theta_projected-self.y)
        self.delta = self.lam *np.dot(self.c,self.w) - np.dot(self.theta_o,self.Xw )
        for i in range(self.w_screening.shape[0]):
            xiXw = self.X[:, i].T.dot(self.Xw)[0]
            xi_norm = math.sqrt(self.X[:, i].T.dot(self.X[:, i]).toarray()[0][0])
            if (self.r/xi_norm* xiXw<=self.delta):
                if (self.X[:, i].T.dot(self.theta_o)[0] +self.r*xi_norm< self.lam*self.c[i]):
                    self.w_screening[i] = 0
                    self.countzeros += 1
            else:
                if (self.X[:, i].T.dot(self.theta_o)[0]+xiXw/self.Xw_norm2*self.delta + np.linalg.norm(self.X[:,i].toarray()
                        - xiXw/self.Xw_norm2*self.Xw)*math.sqrt(max(self.r**2-1/(self.Xw_norm2)*self.delta**2,0)) < self.lam*self.c[i]):
                    self.w_screening[i] = 0
                    self.countzeros += 1
        return  self.countzeros / self.w_screening.shape[0]

    def screening_point(self,theta_projected):
        self.countzeros = 0
        self.w_screening = np.ones_like(self.w)
        for i in range(self.w_screening.shape[0]):
            if (self.X[:, i].T.dot(theta_projected)[0]< self.lam*self.c[i]):
                self.w_screening[i] = 0
                self.countzeros += 1
        return self.countzeros / self.w_screening.shape[0]

import numpy as np
import math
import matplotlib.pyplot as plt
import copy




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

class screener_matrix(object):
    def __init__(self,w, a,b,C,lam, reg="l1" ):
        self.w = w  # primal variable
        self.a = a # constrain matrix
        self.b = b # distributions
        self.C = C
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
        self.theta_p2,beilv2 = self.projection_translation(self.theta_hat)

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

    def projection_translation(self,alg_theta):
        trans = ((self.X.T.dot(alg_theta)-(self.lam * self.c)) /2).reshape(self.rank,self.rank)
        # plt.imshow(trans)
        # plt.colorbar(aspect=40, pad=0.08, shrink=0.6,
        #              orientation='horizontal', extend='both')
        # plt.show()
        out = np.ones_like(alg_theta)
        out[:self.rank] = (alg_theta)[:self.rank]-np.where(trans.max(axis=1)>0,trans.max(axis=1),0)
        out[self.rank:] = (alg_theta)[self.rank:]-np.where(trans.max(axis=0)>0,trans.max(axis=0),0)
        return out, trans
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
        self.rank = math.floor(self.theta_hat.shape[0] / 2)
        self.theta_p1,beilv1 = self.projection_translation(self.theta_hat)

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

        d_alg_proj1 =  np.linalg.norm(self.theta_hat-self.theta_p1)

        dis = {"opt_alg": d_opt_alg,
               "opt_proj1": d_opt_proj1,

               "alg_proj1": d_alg_proj1,

               }

        self.w_screening_area1 = self.screening_area(self.theta_p1)
        self.w_screening_s = self.screening_point(self.theta_hat )
        self.w_screening_s_p1 = self.screening_point(self.theta_p1)


        w_screening = { "screening_area1":self.w_screening_area1,

                      "screening_ps": self.w_screening_s,
                      "screening_p1": self.w_screening_s_p1,


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

    def projection_translation(self,alg_theta):
        trans = ((self.X.T.dot(alg_theta)-(self.lam * self.c)) /2).reshape(self.rank,self.rank)
        # plt.imshow(trans)
        # plt.colorbar(aspect=40, pad=0.08, shrink=0.6,
        #              orientation='horizontal', extend='both')
        # plt.show()
        out = np.ones_like(alg_theta)
        out[:self.rank] = (alg_theta)[:self.rank]-np.where(trans.max(axis=1)>0,trans.max(axis=1),0)
        out[self.rank:] = (alg_theta)[self.rank:]-np.where(trans.max(axis=0)>0,trans.max(axis=0),0)
        return out, trans
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



class sasvi_screening_circle_test(screener):
    # This function is used to test about the circle and possiblly activated dual constrain
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
        self.theta_p1,beilv1 = self.projection_translation(self.theta_hat)

        self.theta_best = self.y - self.X.dot(self.solution)

        dis = {

               }

        self.w_screening_area1 = self.screening_area(self.theta_p1)
        self.if_cut_circle(self.theta_p1)
        w_screening = {
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

    def projection_translation(self,alg_theta):
        trans = ((self.X.T.dot(alg_theta)-(self.lam * self.c)) /2).reshape(self.rank,self.rank)
        # plt.imshow(trans)
        # plt.colorbar(aspect=40, pad=0.08, shrink=0.6,
        #              orientation='horizontal', extend='both')
        # plt.show()
        out = np.ones_like(alg_theta)
        out[:self.rank] = (alg_theta)[:self.rank]-np.where(trans.max(axis=1)>0,trans.max(axis=1),0)
        out[self.rank:] = (alg_theta)[self.rank:]-np.where(trans.max(axis=0)>0,trans.max(axis=0),0)
        return out, trans
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

    def if_cut_circle(self,theta_projected):
        self.countzeros = 0
        self.w_distance = np.ones_like(self.w)
        self.theta_o = 0.5*(theta_projected + self.y)
        self.r = 0.5 * np.linalg.norm(theta_projected-self.y)
        self.delta = self.lam *np.dot(self.c,self.w) - np.dot(self.theta_o,self.Xw )
        for i in range(self.w_distance.shape[0]):
            xiXw = self.X[:, i].T.dot(self.Xw)[0]
            xi_norm = math.sqrt(self.X[:, i].T.dot(self.X[:, i]).toarray()[0][0])
            projd = abs((self.lam *self.c[i]-self.X[:,i].T.dot(self.theta_o))/(xi_norm**2))
            self.w_distance[i] = projd[0]

        gg = self.w_distance.reshape(self.rank,self.rank)
        plt.imshow(gg)
        plt.title('running!')
        plt.colorbar(aspect=40, pad=0.08, shrink=0.6,
                     orientation='horizontal', extend='both')
        plt.show()
        return 0




class sasvi_screening_matrix(screener_matrix):
    def __init__(self, w, a,b ,C, lam, reg="l1",sratio=0,solution=None):
        super().__init__(w, a, b, C, lam, reg="l1")
        self.sratio = 0
        self.solution = solution
    def update(self,w):
        self.w = w
        self.u = w.sum(axis=1)
        self.v = w.sum(axis=0)
        # u,v 就是Xw
        self.Xw_norm2 = np.dot(self.u,self.u)+np.dot(self.v,self.v)
        self.theta_hat_u = self.a - self.u
        self.theta_hat_v = self.b - self.v
        self.g = self.lam * np.multiply(self.C,self.w).sum()
        self.dim_a = w.shape[0]
        self.dim_b = w.shape[1]
        self.theta_pu,self.theta_pv = self.projection_translation(self.theta_hat_u,self.theta_hat_v)
        self.theta_best_u = self.a - self.solution.sum(axis=1)
        self.theta_best_v = self.b - self.solution.sum(axis=0)

        d_opt_alg = math.sqrt(np.dot(self.theta_best_u-self.theta_hat_u,self.theta_best_u-self.theta_hat_u)+
                        np.dot(self.theta_best_v-self.theta_hat_v,self.theta_best_v-self.theta_hat_v))
        d_opt_proj =  math.sqrt(np.dot(self.theta_best_u-self.theta_pu,self.theta_best_u-self.theta_pu)+
                        np.dot(self.theta_best_v-self.theta_pv,self.theta_best_v-self.theta_pv))
        d_alg_proj = math.sqrt(np.dot(self.theta_hat_u-self.theta_pu,self.theta_hat_u-self.theta_pu)+
                        np.dot(self.theta_hat_v-self.theta_pv,self.theta_hat_v-self.theta_pv))

        dis = {"opt_alg": d_opt_alg,
               "opt_proj": d_opt_proj,
               "alg_proj": d_alg_proj,
               }

        self.w_screening_area1 = self.screening_area(self.theta_pu,self.theta_pv)
        self.w_screening_area2 = self.screening_divided_area(self.theta_pu,self.theta_pv)



        w_screening = { "screening_area1":self.w_screening_area1,
                      "screening_area2": self.w_screening_area2,

        }

        return dis,w_screening

    # def projection_normal(self, alg_theta):
    #     beilv = self.X.T.dot(alg_theta)/(self.lam * self.c)
    #     out = alg_theta / max(1,beilv.max())
    #     return out, beilv

    def projection_translation(self,at_u,at_v):

        trans = ((at_u[:,None]+at_v[None,:]-(self.lam * self.C))/2 )
        # plt.imshow(trans)
        # plt.colorbar(aspect=40, pad=0.08, shrink=0.6,
        #              orientation='horizontal', extend='both')
        # plt.show
        difu = trans.max(axis=1)
        difv = trans.max(axis=0)
        outu = at_u-np.where(difu>0,difu,0)
        outv = at_v-np.where(difv>0,difv,0)
        return outu,outv
    def screening_area(self,pu,pv):
        self.countzeros = 0
        self.w_screening = np.ones_like(self.w)
        self.theta_ou = 0.5*(pu + self.a)
        self.theta_ov = 0.5*(pv + self.b)
        self.r = 0.5 * math.sqrt(np.dot(pu - self.a,pu - self.a)+np.dot(pv - self.b,pv - self.b))
        self.delta = self.lam *np.multiply(self.C,self.w).sum() - np.dot(self.theta_ou,self.u ) -\
            np.dot(self.theta_ov, self.v)
        xi_norm = math.sqrt(2)
        for i in range(self.w_screening.shape[0]):
            for j in range(self.w_screening.shape[1]):
                xiXw = self.u[i] + self.v[j]
                xitheta_o = self.theta_ou[i] + self.theta_ov[j]
                if (self.r/xi_norm* xiXw<=self.delta):
                    if (xitheta_o +self.r*xi_norm< self.lam*self.C[i,j]):
                        self.w_screening[i] = 0
                        self.countzeros += 1
                else:
                    x_xwu = - xiXw/self.Xw_norm2 * self.u
                    x_xwu[i] = x_xwu[i]+1
                    x_xwv = - xiXw/self.Xw_norm2 * self.v
                    x_xwv[j] = x_xwv[j]+1
                    if (xitheta_o+xiXw/self.Xw_norm2*self.delta +
                            np.linalg.norm(np.concatenate((x_xwu,x_xwv)))*math.sqrt(max(self.r**2-1/(self.Xw_norm2)*self.delta**2,0)) < self.lam*self.C[i,j]):
                        self.w_screening[i] = 0
                        self.countzeros += 1
        return  self.countzeros / (self.w_screening.shape[0]*self.w_screening.shape[1])
    def screening_divided_area(self,pu,pv):
        self.countzeros = 0
        self.w_screening = np.ones_like(self.w)
        self.theta_ou = 0.5*(pu + self.a)
        self.theta_ov = 0.5*(pv + self.b)
        self.r = 0.5 * math.sqrt(np.dot(pu - self.a,pu - self.a)+np.dot(pv - self.b,pv - self.b))
        self.g_matrix = self.lam * np.multiply(self.C,self.w)
        self.g_col_matrix = self.g_matrix.sum(axis=0)
        self.g_row_matrix = self.g_matrix.sum(axis=1)
        self.g_matrix_all = self.g_col_matrix.sum()
        self.theta_x_matrix = self.theta_ou[:,None]+self.theta_ov[None,:]
        self.theta_xqua_matrix = np.multiply(self.theta_x_matrix,self.theta_x_matrix)
        self.delta_matrix = self.g_matrix - (np.multiply(self.theta_x_matrix,self.w))

        self.w_qua = np.multiply(self.w,self.w)
        self.w_col_qua = self.w_qua.sum(axis=0)
        self.w_row_qua = self.w_qua.sum(axis=1)
        
        self.theta_xqua_col =self.theta_xqua_matrix.sum(axis=0)
        self.theta_xqua_row =self.theta_xqua_matrix.sum(axis=1)
        self.theta_xqua_all =self.theta_xqua_row.sum()

        self.sep_col = self.delta_matrix.sum(axis=0)
        self.sep_row = self.delta_matrix.sum(axis=1)
        self.sep_all = self.sep_row.sum()
        xi_norm = math.sqrt(2)
        self.number = 0
        for i in range(self.w.shape[0]):

            for j in range(self.w.shape[1]):
                self.line1 = self.sep_col[j]+self.sep_row[i]-self.delta_matrix[i,j] # constraint plane
                self.line2 = self.sep_all - self.line1 # parallel plane

                line1_include = 0
                line2_include = 0
                # 到L2的距离

                self.Xw_line1_u = copy.copy(self.w[:,j])
                self.Xw_line1_v = copy.copy(self.w[i, :])
                self.Xw_line1_v[j] = self.v[j]
                self.Xw_line1_u[i] = self.u[i]
                self.Xw_line1_norm = np.linalg.norm(np.concatenate((self.Xw_line1_u, self.Xw_line1_v)))
                self.Xw_line2_u = self.u - self.Xw_line1_u
                self.Xw_line2_v = self.v - self.Xw_line1_v
                self.Xw_line2_norm = np.linalg.norm(np.concatenate((self.Xw_line2_u, self.Xw_line2_v)))
                self.dis_line1 = self.line1 / self.Xw_line1_norm
                self.dis_line2 = self.line2 / self.Xw_line2_norm

                if(self.line2>=0):
                    #平面在原点上半
                    if(self.dis_line2>self.r):
                        line2_include = 1
                    else:
                        self.r_new = math.sqrt(self.r**2-self.dis_line2**2)
                else:
                    #平面在原点下半
                    self.r_new = math.sqrt(self.r ** 2 - self.dis_line2 ** 2)

                #L1 部分
                xiXw_line1 = self.Xw_line1_u[i] + self.Xw_line1_v[j]
                xitheta_o = self.theta_ou[i] + self.theta_ov[j]
                if (self.r/xi_norm* xiXw_line1<=self.line1):
                    # l1与球无关系
                    if(line2_include==1):
            #1：L2 全 L1 全
            #3：L2 上交 L1 全
                        if (xitheta_o +self.r*xi_norm< self.lam*self.C[i,j]):
                            self.w_screening[i,j] = 0
                            self.countzeros += 1
                    elif(self.line2<0):
                        if (xitheta_o +self.r_new*xi_norm< self.lam*self.C[i,j]):
                            self.w_screening[i,j] = 0
                            self.countzeros += 1
            #5：L2 下交 L1 全
                else:
                    Xw_line1_norm2 = self.Xw_line1_norm**2
                    x_xw_line1_u = - xiXw_line1 / Xw_line1_norm2 * self.Xw_line1_u
                    x_xw_line1_u[i] = x_xw_line1_u[i] + 1
                    x_xw_line1_v = - xiXw_line1 / Xw_line1_norm2 * self.Xw_line1_v
                    x_xw_line1_v[j] = x_xw_line1_v[j] + 1
                    if (line2_include == 1):
            # 2： L2全 L1 交
                        if (xitheta_o + xiXw_line1 / Xw_line1_norm2 * self.line1 +
                                np.linalg.norm(np.concatenate((x_xw_line1_u, x_xw_line1_v))) * math.sqrt(
                                    max(self.r ** 2 - 1 / Xw_line1_norm2 * self.line1 ** 2, 0)) < self.lam * self.C[
                                    i, j]):
                            self.w_screening[i,j] = 0
                            self.countzeros += 1
                    else:

                        #求投到L2上圆的距离lambda
            # 4: L2上交 L1交
            # 6: L2下交 L1交
                        if(self.line2>=0):
            # 4: L2上半 L1交
                            if (xitheta_o + xiXw_line1 / Xw_line1_norm2 * self.line1 +
                                    np.linalg.norm(np.concatenate((x_xw_line1_u, x_xw_line1_v))) * math.sqrt(
                                        max(self.r ** 2 - 1 / Xw_line1_norm2 * self.line1 ** 2, 0)) < self.lam *
                                    self.C[
                                        i, j]):
                                self.w_screening[i, j] = 0
                                self.countzeros += 1
                        else:
            # 6: L2下交 L1交
                            self.down_theta_u = self.theta_ou + (self.dis_line2 * self.Xw_line2_u/self.Xw_line2_norm)
                            self.down_theta_v = self.theta_ov + (self.dis_line2 * self.Xw_line2_v/self.Xw_line2_norm)
                            #self.g_matrix.sum() - self.g_col_matrix[j] - self.g_row_matrix[i] + self.g_matrix[i, j] - \
                             #                  np.dot(self.down_theta_u, self.u-self.Xw_line1_u) - np.dot(self.down_theta_v,
                             #                                                                  self.v - self.Xw_line1_v)
                            self.opt_down_u = copy.copy(self.down_theta_u)
                            self.opt_down_u[i] = self.down_theta_u[i] + self.r_new/xi_norm
                            self.opt_down_v = copy.copy(self.down_theta_v)
                            self.opt_down_v[j] = self.down_theta_v[j] + self.r_new/xi_norm
            # self.pointandl2 = self.g_matrix.sum() - self.g_col_matrix[j] - self.g_row_matrix[i] + self.g_matrix[i, j] \
            #                  - np.dot(self.opt_down_u, self.u - self.Xw_line1_u) - np.dot(self.opt_down_v,
            #                                                                      self.v - self.Xw_line1_v)
            #               self.line1 = self.sep_col[j]+self.sep_row[i]-self.delta_matrix[i,j]
                            self.opt2forl1 = self.g_col_matrix[j] + self.g_row_matrix[i] - self.g_matrix[i, j]\
                                              -  np.dot(self.opt_down_u, self.Xw_line1_u) - np.dot(self.opt_down_v,
                                                                                              self.Xw_line1_v)

                            if( self.opt2forl1 >=0):
            # 6.1: L2下交 L1交 L2最优在L1内
                                if (xitheta_o + self.r_new * xi_norm < self.lam * self.C[i, j]):
                                    self.w_screening[i, j] = 0
                                    self.countzeros += 1
                            else:
                                self.down_line1_theta_u = self.theta_ou + (
                                            self.dis_line1 * self.Xw_line1_u / self.Xw_line1_norm)
                                self.down_line1_theta_v = self.theta_ov + (
                                            self.dis_line1 * self.Xw_line1_v / self.Xw_line1_norm)
                                self.line2_c1 = self.g_matrix_all - self.g_col_matrix[j] - self.g_row_matrix[i] + self.g_matrix[i, j] \
                                                 - np.dot(self.down_line1_theta_u, self.Xw_line2_u) - np.dot(self.down_line1_theta_v,
                                                                                                     self.Xw_line2_v)
                                if (self.line2_c1>=0):


                                    if (xitheta_o + \
                                            xiXw_line1 / Xw_line1_norm2 * self.line1 + \
                                            np.linalg.norm(np.concatenate((x_xw_line1_u, x_xw_line1_v))) * \
                                            math.sqrt(max(self.r ** 2 - self.dis_line1 ** 2, 0)) < \
                                            self.lam * self.C[i, j]):
                                        self.w_screening[i, j] = 0
                                        self.countzeros += 1

                                else:

            # 6.2: L2下交 L1交 L2最优在L1外
                                    self.dis_c1_line2 = self.line2_c1 / self.Xw_line2_norm

                                    self.down_c1_2_theta_u = self.down_line1_theta_u + (
                                            self.dis_c1_line2 * self.Xw_line2_u / self.Xw_line2_norm)
                                    self.down_c1_2_theta_v = self.down_line1_theta_v + (
                                            self.dis_c1_line2* self.Xw_line2_v / self.Xw_line2_norm)
                                    self.coor_a_i = self.down_theta_u[i]
                                    self.coor_a_j = self.down_theta_v[j]
                                    self.coor_b_i = self.down_c1_2_theta_u[i]
                                    self.coor_b_j = self.down_c1_2_theta_v[j]
                                    self.lab = np.linalg.norm([self.coor_a_i-self.coor_b_i,self.coor_a_j-self.coor_b_j])
                                    self.lac = self.r_new
                                    self.lbc = math.sqrt(self.r**2 - self.dis_line1**2 - self.dis_c1_line2**2)
                                    self.coscab = (self.lab**2 + self.lac**2 - self.lbc**2)/(2*self.lab*self.lac)
                                    # if(abs(self.coscab)>1):
                                    #     aa = 1
                                    self.sincab =math.sqrt(1-min(self.coscab,1)**2)
                                    self.vec = [self.coor_b_i-self.coor_a_i,self.coor_b_j-self.coor_a_j]/self.lab
                                    self.vec1 = [0,0]
                                    self.vec1[0] = -self.vec[1]
                                    self.vec1[1] = self.vec[0]
                                    self.vec2 = [0, 0]
                                    self.vec2[0] = self.vec[1]
                                    self.vec2[1] = -self.vec[0]
                                    self.xifinal = self.coor_a_j +self.coor_a_i + \
                                                    self.lac * self.coscab *(self.vec[0]+self.vec[1])+\
                                                    max(self.lac*self.sincab*(self.vec1[0]+self.vec1[1]),self.lac*self.sincab*(self.vec2[0]+self.vec2[1]))
                                    if (self.xifinal < self.lam * self.C[i, j]):
                                        self.w_screening[i, j] = 0
                                        self.countzeros += 1



                # if (xitheta_o + \
                #         xiXw_line1 / Xw_line1_norm2 * self.line1 + \
                #         math.sqrt(xi_norm - xiXw_line1 ** 2 / (xi_norm * self.Xw_line1_norm ** 2)) * \
                #         math.sqrt(max(self.r ** 2 - self.dis_line1 ** 2, 0)) < \
                                # xiXw = self.u[i] + self.v[j]
                                # x_xwu = - xiXw/self.Xw_norm2 * self.u
                                # x_xwu[i] = x_xwu[i]+1
                                # x_xwv = - xiXw/self.Xw_norm2 * self.v
                                # x_xwv[j] = x_xwv[j]+1
                                # if (xitheta_o+xiXw/self.Xw_norm2*self.delta +
                                #     np.linalg.norm(np.concatenate((x_xwu,x_xwv)))*math.sqrt(max(self.r**2-1/(self.Xw_norm2)*self.delta**2,0)) < self.lam*self.C[i,j]):
                                #     self.w_screening[i,j] = 0
                                #     self.countzeros += 1
                                # self.number+=1

                    #得到L1最值
                    #如果L2与球无关系（包裹了球）
                    #否则：
        plt.imshow(self.w_screening)
        plt.title('running!')
        plt.colorbar(aspect=40, pad=0.08, shrink=0.6,
                     orientation='horizontal', extend='both')
        plt.show()
        print("ratio:",self.number)
        return  self.countzeros / (self.w_screening.shape[0]*self.w_screening.shape[1])
    # def screening_point(self,theta_projected):
    #     self.countzeros = 0
    #     self.w_screening = np.ones_like(self.w)
    #     for i in range(self.w_screening.shape[0]):
    #         if (self.X[:, i].T.dot(theta_projected)[0]< self.lam*self.c[i]):
    #             self.w_screening[i] = 0
    #             self.countzeros += 1
    #     return self.countzeros / self.w_screening.shape[0]

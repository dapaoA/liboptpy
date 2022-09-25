import numpy as np
import math
import matplotlib.pyplot as plt
import copy
from scipy.optimize import fsolve




class screener(object):
    def __init__(self,w, X, y,c,lam, reg="l1" ):
        self.w = w  # primal variable
        self.X = X # constrain matrix
        self.y = y # distributions
        self.c = c
        self.lam = lam
        self.reg = reg

    def update(self,w):
        return 0

class screener_matrix(object):
    def __init__(self,w, a,b,C,lam, reg="l1" ):
        self.w = w  # primal variable
        self.a = a # constrain matrix
        self.b = b # distributions
        self.C = C
        self.lam = lam
        self.reg = reg

    def update(self,w):
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
        w_screening = np.ones_like(self.w)
        self.yTX = self.X.T.dot(self.y)
        self.lam_max = (self.yTX[np.where(self.c>0)]/self.c[np.where(self.c>0)]).max()
        countzeros = 0
        for i in range(w_screening.shape[0]):
            if(self.yTX[i] < self.lam*self.c[i] - 2 +2* self.lam/self.lam_max):
                w_screening[i] = 0
                countzeros += 1
        print("screening percent is: ",countzeros/w_screening.shape[0])
        return w_screening


class safe_screening_2(screener):
    def __init__(self,w, X,y,c, lam,reg="l1" ):
        super().__init__(w,  X,y, c,lam,reg="l1" )
        # self.w = w  # primal variable
        # self.dual = dual  # dual variable
        # self.alpha  #auxiliary
        # self.X = X # constrain matrix
        # self.reg = reg

    def update(self):
        w_screening = np.ones_like(self.w)
        self.yTX = self.X.T.dot(self.y)
        self.lam_max = self.yTX.max()/self.c[np.where(self.c>0)].min()
        countzeros = 0
        for i in range(w_screening.shape[0]):
            if(self.yTX[i] < self.lam_max*()):
                w_screening[i] = 0
                countzeros += 1
        print("screening percent is: ",countzeros/w_screening.shape[0])
        return w_screening

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
        w_screening = np.ones_like(self.w)
        self.cc = self.y/self.lam
        self.thetaTX = self.X.T.dot(self.theta)
        countzeros = 0
        self.mu = max(min(np.dot(self.theta,self.y)/(self.lam*(np.dot(self.theta,self.theta))),
                          1/self.thetaTX.max()),-1/self.thetaTX.max())
        self.r_theta = np.linalg.norm(self.y/self.lam-self.mu*self.theta)
        for i in range(w_screening.shape[0]):
            if(self.c[i]-abs(self.X[:,i].T.dot(self.cc)[0])>2*self.r_theta):
                w_screening[i] = 0
                countzeros += 1
        print("screening percent is: ",countzeros/w_screening.shape[0])
        return w_screening


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
        w_screening = np.ones_like(self.w)
        countzeros = 0
        for i in range(w_screening.shape[0]):
            xiXw = self.X[:, i].T.dot(self.Xw)[0]
            xi_norm = math.sqrt(self.X[:, i].T.dot(self.X[:, i]).toarray()[0][0])
            if (self.r/xi_norm* xiXw<=self.delta):
                if (self.X[:, i].T.dot(self.theta_o)[0] +self.r*xi_norm< self.lam*self.c[i]):
                    w_screening[i] = 0
                    countzeros += 1
            else:
                if (self.X[:, i].T.dot(self.theta_o)[0]+xiXw/self.Xw_norm2*self.delta + np.linalg.norm(self.X[:,i].toarray()
                        - xiXw/self.Xw_norm2*self.Xw)*math.sqrt(max(self.r**2-1/(self.Xw_norm2)*self.delta**2,0)) < self.lam*self.c[i]):
                    w_screening[i] = 0
                    countzeros += 1
        if(countzeros!=0):
            if(countzeros / w_screening.shape[0]>self.sratio):
                self.sratio = countzeros / w_screening.shape[0]
                print("screening percent is: ", countzeros / w_screening.shape[0])
                print("running")
                plt.imshow(w_screening.reshape(math.floor(self.theta_o.shape[0]/2), math.floor(self.theta_o.shape[0]/2)))
                plt.title('running!')
                plt.colorbar(aspect=40, pad=0.08, shrink=0.6,
                         orientation='horizontal', extend='both')
                plt.show()
        return w_screening

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
        w_screening = np.ones_like(self.w)
        countzeros = 0
        for i in range(w_screening.shape[0]):
            xiXw = self.X[:, i].T.dot(self.Xw)[0]
            xi_norm = math.sqrt(self.X[:, i].T.dot(self.X[:, i]).toarray()[0][0])
            if (self.r/xi_norm* xiXw<=self.delta):
                if (self.X[:, i].T.dot(self.theta_o)[0] +self.r*xi_norm< self.lam*self.c[i]):
                    w_screening[i] = 0
                    countzeros += 1
            else:
                if (self.X[:, i].T.dot(self.theta_o)[0]+xiXw/self.Xw_norm2*self.delta + np.linalg.norm(self.X[:,i].toarray()
                        - xiXw/self.Xw_norm2*self.Xw)*math.sqrt(max(self.r**2-1/(self.Xw_norm2)*self.delta**2,0)) < self.lam*self.c[i]):
                    w_screening[i] = 0
                    countzeros += 1
        if (countzeros != 0):
            if (countzeros / w_screening.shape[0] > self.sratio):
                self.sratio = countzeros / w_screening.shape[0]
                print("screening percent is: ", countzeros / w_screening.shape[0])
                print("running")
                plt.imshow(w_screening.reshape(math.floor(self.theta_o.shape[0] / 2),
                                                        math.floor(self.theta_o.shape[0] / 2)))
                plt.title('running!')
                plt.colorbar(aspect=40, pad=0.08, shrink=0.6,
                             orientation='horizontal', extend='both')
                plt.show()
        return w_screening


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
        w_screening = np.ones_like(self.w)
        countzeros = 0
        for i in range(w_screening.shape[0]):
            xiXw = self.X[:, i].T.dot(self.Xw)[0]
            xi_norm = math.sqrt(self.X[:, i].T.dot(self.X[:, i]).toarray()[0][0])
            if (self.r/xi_norm* xiXw<=self.delta):
                if (self.X[:, i].T.dot(self.theta_o)[0] +self.r*xi_norm< self.lam*self.c[i]):
                    w_screening[i] = 0
                    countzeros += 1
            else:
                if (self.X[:, i].T.dot(self.theta_o)[0]+xiXw/self.Xw_norm2*self.delta + np.linalg.norm(self.X[:,i].toarray()
                        - xiXw/self.Xw_norm2*self.Xw)*math.sqrt(max(self.r**2-1/(self.Xw_norm2)*self.delta**2,0)) < self.lam*self.c[i]):
                    w_screening[i] = 0
                    countzeros += 1
        if (countzeros != 0):
            if (countzeros / w_screening.shape[0] > self.sratio):
                self.sratio = countzeros / w_screening.shape[0]
                print("screening percent is: ", countzeros / w_screening.shape[0])
                print("running")
                plt.imshow(w_screening.reshape(math.floor(self.theta_o.shape[0] / 2),
                                                        math.floor(self.theta_o.shape[0] / 2)))
                plt.title('running!')
                plt.colorbar(aspect=40, pad=0.08, shrink=0.6,
                             orientation='horizontal', extend='both')
                plt.show()
        return w_screening





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

        w_screening_area1 = self.screening_area(self.theta_p1)
        w_screening_area2 = self.screening_area(self.theta_p2)

        w_screening_s = self.screening_point(self.theta_hat )
        w_screening_s_p1 = self.screening_point(self.theta_p1)
        w_screening_s_p2 = self.screening_point(self.theta_p2 )


        w_screening = { "screening_area1":w_screening_area1,
                      "screening_area2": w_screening_area2,

                      "screening_ps": w_screening_s,
                      "screening_p1": w_screening_s_p1,
                      "screening_p2": w_screening_s_p2,


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
        countzeros = 0
        w_screening = np.ones_like(self.w)
        self.theta_o = 0.5*(theta_projected + self.y)
        self.r = 0.5 * np.linalg.norm(theta_projected-self.y)
        self.delta = self.lam *np.dot(self.c,self.w) - np.dot(self.theta_o,self.Xw )
        for i in range(w_screening.shape[0]):
            xiXw = self.X[:, i].T.dot(self.Xw)[0]
            xi_norm = math.sqrt(self.X[:, i].T.dot(self.X[:, i]).toarray()[0][0])
            if (self.r/xi_norm* xiXw<=self.delta):
                if (self.X[:, i].T.dot(self.theta_o)[0] +self.r*xi_norm< self.lam*self.c[i]):
                    w_screening[i] = 0
                    countzeros += 1
            else:
                if (self.X[:, i].T.dot(self.theta_o)[0]+xiXw/self.Xw_norm2*self.delta + np.linalg.norm(self.X[:,i].toarray()
                        - xiXw/self.Xw_norm2*self.Xw)*math.sqrt(max(self.r**2-1/(self.Xw_norm2)*self.delta**2,0)) < self.lam*self.c[i]):
                    w_screening[i] = 0
                    countzeros += 1
        return  countzeros / w_screening.shape[0]

    def screening_point(self,theta_projected):
        countzeros = 0
        w_screening = np.ones_like(self.w)
        for i in range(w_screening.shape[0]):
            if (self.X[:, i].T.dot(theta_projected)[0]< self.lam*self.c[i]):
                w_screening[i] = 0
                countzeros += 1
        return countzeros / w_screening.shape[0]


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

        w_screening_area1 = self.screening_area(self.theta_p1)
        w_screening_s = self.screening_point(self.theta_hat )
        w_screening_s_p1 = self.screening_point(self.theta_p1)


        w_screening = { "screening_area1":w_screening_area1,

                      "screening_ps": w_screening_s,
                      "screening_p1": w_screening_s_p1,


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
        countzeros = 0
        w_screening = np.ones_like(self.w)
        self.theta_o = 0.5*(theta_projected + self.y)
        self.r = 0.5 * np.linalg.norm(theta_projected-self.y)
        self.delta = self.lam *np.dot(self.c,self.w) - np.dot(self.theta_o,self.Xw )
        for i in range(w_screening.shape[0]):
            xiXw = self.X[:, i].T.dot(self.Xw)[0]
            xi_norm = math.sqrt(self.X[:, i].T.dot(self.X[:, i]).toarray()[0][0])
            if (self.r/xi_norm* xiXw<=self.delta):
                if (self.X[:, i].T.dot(self.theta_o)[0] +self.r*xi_norm< self.lam*self.c[i]):
                    w_screening[i] = 0
                    countzeros += 1
            else:
                if (self.X[:, i].T.dot(self.theta_o)[0]+xiXw/self.Xw_norm2*self.delta + np.linalg.norm(self.X[:,i].toarray()
                        - xiXw/self.Xw_norm2*self.Xw)*math.sqrt(max(self.r**2-1/(self.Xw_norm2)*self.delta**2,0)) < self.lam*self.c[i]):
                    w_screening[i] = 0
                    countzeros += 1
        return  countzeros / w_screening.shape[0]

    def screening_point(self,theta_projected):
        countzeros = 0
        w_screening = np.ones_like(self.w)
        for i in range(w_screening.shape[0]):
            if (self.X[:, i].T.dot(theta_projected)[0]< self.lam*self.c[i]):
                w_screening[i] = 0
                countzeros += 1
        return countzeros / w_screening.shape[0]



class sasvi_screening_circle_test(screener):
    # This function is used to test about the circle and possiblly activated dual constrain
    def __init__(self, w, X, y, c, lam, reg="l1", sratio=0, solution=None):
        super().__init__(w, X, y, c, lam, reg="l1")
        self.sratio = 0
        self.solution = solution
        self.w = .0
        self.Xw = .0
        self.Xw_norm2 = .0

    def update(self, w):
        self.w = w
        self.Xw = self.X.dot(self.w)
        self.Xw_norm2 = np.dot(self.Xw,self.Xw)
        self.theta_hat = self.y - self.Xw
        self.g = self.lam * np.dot(self.c,self.w)
        self.rank = math.floor(self.theta_hat.shape[0] / 2)
        self.theta_p1,beilv1 = self.projection_translation(self.theta_hat)
        self.theta_best = self.y - self.X.dot(self.solution)

        dis = {

               }

        w_screening_area1 = self.screening_area(self.theta_p1)
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
        countzeros = 0
        w_screening = np.ones_like(self.w)
        self.theta_o = 0.5*(theta_projected + self.y)
        self.r = 0.5 * np.linalg.norm(theta_projected-self.y)
        self.delta = self.lam *np.dot(self.c,self.w) - np.dot(self.theta_o,self.Xw )
        for i in range(w_screening.shape[0]):
            xiXw = self.X[:, i].T.dot(self.Xw)[0]
            xi_norm = math.sqrt(self.X[:, i].T.dot(self.X[:, i]).toarray()[0][0])
            if (self.r/xi_norm* xiXw<=self.delta):
                if (self.X[:, i].T.dot(self.theta_o)[0] +self.r*xi_norm< self.lam*self.c[i]):
                    w_screening[i] = 0
                    countzeros += 1
            else:
                if (self.X[:, i].T.dot(self.theta_o)[0]+xiXw/self.Xw_norm2*self.delta + np.linalg.norm(self.X[:,i].toarray()
                        - xiXw/self.Xw_norm2*self.Xw)*math.sqrt(max(self.r**2-1/(self.Xw_norm2)*self.delta**2,0)) < self.lam*self.c[i]):
                    w_screening[i] = 0
                    countzeros += 1
        return  countzeros / w_screening.shape[0]

    def screening_point(self,theta_projected):
        countzeros = 0
        w_screening = np.ones_like(self.w)
        for i in range(w_screening.shape[0]):
            if (self.X[:, i].T.dot(theta_projected)[0]< self.lam*self.c[i]):
                w_screening[i] = 0
                countzeros += 1
        return countzeros / w_screening.shape[0]

    def if_cut_circle(self,theta_projected):
        countzeros = 0
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
        self.u = .0
        self.v = .0
        self.Xw_norm2 = .0
        self.theta_hat_u = .0
        self.theta_hat_v = .0
        self.g = .0
        self.dim_a = .0
        self.dim_b = .0
        self.theta_pu, self.theta_pv = .0, .0
        self.theta_best_u = .0
        self.theta_best_v = .0
        self.r = 0.0
        self.delta = .0
        g_matrix = .0
        g_col_matrix = .0
        g_row_matrix = .0
        g_matrix_all = .0
        theta_x_matrix = .0
        theta_xqua_matrix = .0
        delta_matrix = .0

        w_qua = .0
        w_col_qua = .0
        w_row_qua = .0

        theta_xqua_col = .0
        theta_xqua_row = .0
        theta_xqua_all = .0

        sep_col = .0
        sep_row = .0
        sep_all = .0
        number = .0

        self.cosline2 = .0
        self.cosline1 = .0
        self.n2_u = .0
        self.n2_v = .0
        self.n1_u = .0
        self.n1_v = .0
        self.n1n2 = .0
        self.sum_nm = .0
        self.nn = .0
        self.nnnn = .0
        self.sum_n = .0
        self.sum_m = .0
        self.sum_mm_all = .0
        self.sum_nn_all = .0

        self.final = .0

    def update(self, w):
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
        self.theta_pu, self.theta_pv = self.projection_translation(self.theta_hat_u,self.theta_hat_v)
        self.theta_best_u = self.a - self.solution.sum(axis=1)
        self.theta_best_v = self.b - self.solution.sum(axis=0)

        d_opt_alg = math.sqrt(np.dot(self.theta_best_u - self.theta_hat_u, self.theta_best_u - self.theta_hat_u) +
                              np.dot(self.theta_best_v - self.theta_hat_v, self.theta_best_v - self.theta_hat_v))
        d_opt_proj = math.sqrt(np.dot(self.theta_best_u - self.theta_pu, self.theta_best_u - self.theta_pu) +
                               np.dot(self.theta_best_v - self.theta_pv, self.theta_best_v - self.theta_pv))
        d_alg_proj = math.sqrt(np.dot(self.theta_hat_u - self.theta_pu, self.theta_hat_u - self.theta_pu) +
                               np.dot(self.theta_hat_v - self.theta_pv, self.theta_hat_v - self.theta_pv))

        dis = {"opt_alg": d_opt_alg,
               "opt_proj": d_opt_proj,
               "alg_proj": d_alg_proj,
               }

        w_screening_area1 = self.screening_area(self.theta_pu, self.theta_pv)
        w_screening_area2 = self.screening_divided_area(self.theta_pu, self.theta_pv)
        w_screening = {"screening_area1": w_screening_area1,
                       "screening_area2": w_screening_area2}
        return dis, w_screening

    # def projection_normal(self, alg_theta):
    #     beilv = self.X.T.dot(alg_theta)/(self.lam * self.c)
    #     out = alg_theta / max(1,beilv.max())
    #     return out, beilv
    def func(self, i):
        l1, l2, l3 = i[0], i[1], i[2]
        return [
            (- self.sum_nm) * l3 - self.sum_nn_all * l2 - 2 * l1 * self.cosline1 * self.r + self.nn,
            (- self.sum_nm) * l2 - self.sum_mm_all * l3 - 2 * l1 * self.cosline2 * self.r,
            (l2 * self.nn - 1 - 0.5 * self.sum_nn_all * l2 ** 2 - 0.5 * self.sum_mm_all * l3 ** 2 -
             self.sum_nm * l3 * l2) + 2 * self.r**2 * l1**2]

    def projection_translation(self, at_u, at_v):
        trans = ((at_u[:, None]+at_v[None, :]-(self.lam * self.C))/2)
        # plt.imshow(trans)
        # plt.colorbar(aspect=40, pad=0.08, shrink=0.6,
        #              orientation='horizontal', extend='both')
        # plt.show
        difu = trans.max(axis=1)
        difv = trans.max(axis=0)
        outu = at_u - np.where(difu > 0, difu, 0)
        outv = at_v - np.where(difv > 0, difv, 0)
        return outu, outv

    def screening_area(self, pu, pv):
        countzeros = 0
        w_screening = np.ones_like(self.w)
        theta_ou = 0.5 * (pu + self.a)
        theta_ov = 0.5 * (pv + self.b)
        self.r = 0.5 * math.sqrt(np.dot(pu - self.a, pu - self.a)+np.dot(pv - self.b, pv - self.b))
        self.delta = self.lam * np.multiply(self.C, self.w).sum() - np.dot(theta_ou, self.u) -\
            np.dot(theta_ov, self.v)
        xi_norm = math.sqrt(2)
        for i in range(w_screening.shape[0]):
            for j in range(w_screening.shape[1]):
                xiXw = self.u[i] + self.v[j]
                xitheta_o = theta_ou[i] + theta_ov[j]
                if self.r/xi_norm * xiXw <= self.delta:
                    if xitheta_o + self.r * xi_norm < self.lam * self.C[i,j]:
                        w_screening[i, j] = 0
                        countzeros += 1
                else:
                    x_xwu = - xiXw/self.Xw_norm2 * self.u
                    x_xwu[i] = x_xwu[i] + 1
                    x_xwv = - xiXw/self.Xw_norm2 * self.v
                    x_xwv[j] = x_xwv[j] + 1
                    if xitheta_o + xiXw/self.Xw_norm2*self.delta + \
                            np.linalg.norm(np.concatenate((x_xwu, x_xwv))) * \
                            math.sqrt(max(self.r**2-1/self.Xw_norm2 * self.delta**2, 0)) < self.lam*self.C[i, j]:
                        w_screening[i, j] = 0
                        countzeros += 1
        plt.imshow(w_screening)
        plt.title('running!')
        plt.colorbar(aspect=40, pad=0.08, shrink=0.6,
                     orientation='horizontal', extend='both')
        plt.show()
        return countzeros / (w_screening.shape[0]*w_screening.shape[1])

    def screening_divided_area(self, pu, pv):
        countzeros = 0
        w_screening = np.ones_like(self.w)
        theta_ou = 0.5 * (pu + self.a)
        theta_ov = 0.5 * (pv + self.b)
        self.r = 0.5 * math.sqrt(np.dot(pu - self.a, pu - self.a)+np.dot(pv - self.b, pv - self.b))
        g_matrix = self.lam * np.multiply(self.C, self.w)
        g_col_matrix = g_matrix.sum(axis=0)
        g_row_matrix = g_matrix.sum(axis=1)
        g_matrix_all = g_col_matrix.sum()
        theta_x_matrix = theta_ou[:, None]+theta_ov[None, :]
        theta_xqua_matrix = np.multiply(theta_x_matrix, theta_x_matrix)
        delta_matrix = g_matrix - (np.multiply(theta_x_matrix, self.w))

        w_qua = np.multiply(self.w, self.w)
        w_col_qua = w_qua.sum(axis=0)
        w_row_qua = w_qua.sum(axis=1)

        theta_xqua_col = theta_xqua_matrix.sum(axis=0)
        theta_xqua_row = theta_xqua_matrix.sum(axis=1)
        theta_xqua_all = theta_xqua_row.sum()

        sep_col = delta_matrix.sum(axis=0)
        sep_row = delta_matrix.sum(axis=1)
        sep_all = sep_row.sum()
        xi_norm = math.sqrt(2)
        number = 0
        for i in range(self.w.shape[0]):

            for j in range(self.w.shape[1]):
                line1 = sep_col[j]+sep_row[i]-delta_matrix[i, j] # constraint plane
                line2 = sep_all - line1 # parallel plane

                line1_include = 0
                line2_include = 0
                # 到L2的距离

                Xw_line1_u = copy.copy(self.w[:,j])
                Xw_line1_v = copy.copy(self.w[i, :])
                Xw_line1_v[j] = self.v[j]
                Xw_line1_u[i] = self.u[i]
                Xw_line1_norm = np.linalg.norm(np.concatenate((Xw_line1_u, Xw_line1_v)))
                Xw_line2_u = self.u - Xw_line1_u
                Xw_line2_v = self.v - Xw_line1_v
                Xw_line2_norm = np.linalg.norm(np.concatenate((Xw_line2_u, Xw_line2_v)))
                dis_line1 = line1 / Xw_line1_norm
                dis_line2 = line2 / Xw_line2_norm

                if line2 >= 0:
                    #平面在原点上半
                    if dis_line2 > self.r:
                        line2_include = 1
                    else:
                        r_new = math.sqrt(self.r**2 - dis_line2**2)
                else:
                    #平面在原点下半
                    r_new = math.sqrt(self.r ** 2 - dis_line2 ** 2)

                #L1 部分
                xiXw_line1 = Xw_line1_u[i] + Xw_line1_v[j]
                xitheta_o = theta_ou[i] + theta_ov[j]
                if self.r/xi_norm * xiXw_line1 <= line1:
                    # l1与球无关系
                    if line2_include == 1:
            #1：L2 全 L1 全
            #3：L2 上交 L1 全
                        if xitheta_o +self.r*xi_norm < self.lam*self.C[i,j]:
                            w_screening[i,j] = 0
                            countzeros += 1
                    elif line2 < 0:
                        if xitheta_o +r_new * xi_norm < self.lam*self.C[i,j]:
                            w_screening[i,j] = 0
                            countzeros += 1
            #5：L2 下交 L1 全
                else:
                    Xw_line1_norm2 = Xw_line1_norm**2
                    x_xw_line1_u = - xiXw_line1 / Xw_line1_norm2 * Xw_line1_u
                    x_xw_line1_u[i] = x_xw_line1_u[i] + 1
                    x_xw_line1_v = - xiXw_line1 / Xw_line1_norm2 * Xw_line1_v
                    x_xw_line1_v[j] = x_xw_line1_v[j] + 1
                    if line2_include == 1:
            # 2： L2全 L1 交
                        if xitheta_o + xiXw_line1 / Xw_line1_norm2 * line1 + \
                                np.linalg.norm(np.concatenate((x_xw_line1_u, x_xw_line1_v))) * math.sqrt(
                                    max(self.r ** 2 - 1 / Xw_line1_norm2 * line1 ** 2, 0)) < self.lam * self.C[
                                    i, j]:
                            w_screening[i, j] = 0
                            countzeros += 1
                    else:

                        #求投到L2上圆的距离lambda
            # 4: L2上交 L1交
            # 6: L2下交 L1交
                        if line2 >= 0:
            # 4: L2上半 L1交
                            if xitheta_o + xiXw_line1 / Xw_line1_norm2 * line1 + \
                                    np.linalg.norm(np.concatenate((x_xw_line1_u, x_xw_line1_v))) * math.sqrt(
                                        max(self.r ** 2 - 1 / Xw_line1_norm2 * line1 ** 2, 0)) < self.lam * self.C[i, j]:
                                w_screening[i, j] = 0
                                countzeros += 1
                        else:
            # 6: L2下交 L1交
                            down_theta_u = theta_ou + (dis_line2 * Xw_line2_u/Xw_line2_norm)
                            down_theta_v = theta_ov + (dis_line2 * Xw_line2_v/Xw_line2_norm)
                            opt_down_u = copy.copy(down_theta_u)
                            opt_down_u[i] = down_theta_u[i] + r_new/xi_norm
                            opt_down_v = copy.copy(down_theta_v)
                            opt_down_v[j] = down_theta_v[j] + r_new/xi_norm
                            opt2forl1 = g_col_matrix[j] + g_row_matrix[i] - g_matrix[i, j] -\
                                        np.dot(opt_down_u, Xw_line1_u) - np.dot(opt_down_v, Xw_line1_v)

                            if opt2forl1 >= 0:
            # 6.1: L2下交 L1交 L2最优在L1内
                                if xitheta_o + r_new * xi_norm < self.lam * self.C[i, j]:
                                    w_screening[i, j] = 0
                                    countzeros += 1
                            else:
                                down_line1_theta_u = theta_ou + (
                                            dis_line1 * Xw_line1_u / Xw_line1_norm)
                                down_line1_theta_v = theta_ov + (
                                            dis_line1 * Xw_line1_v / Xw_line1_norm)
                                line2_c1 = g_matrix_all - g_col_matrix[j] - g_row_matrix[i] + \
                                           g_matrix[i, j] - np.dot(down_line1_theta_u, Xw_line2_u) - \
                                           np.dot(down_line1_theta_v, Xw_line2_v)
                                if line2_c1 >= 0:
                                    if xitheta_o + \
                                            xiXw_line1 / Xw_line1_norm2 * line1 + \
                                            np.linalg.norm(np.concatenate((x_xw_line1_u, x_xw_line1_v))) * \
                                            math.sqrt(max(self.r ** 2 - dis_line1 ** 2, 0)) < \
                                            self.lam * self.C[i, j]:
                                        w_screening[i, j] = 0
                                        countzeros += 1
                                else:
                                    self.cosline2 = (dis_line2/self.r)
                                    self.cosline1 = (dis_line1/self.r)
                                    self.n2_u = (Xw_line2_u/Xw_line2_norm)
                                    self.n2_v= Xw_line2_v/Xw_line2_norm
                                    self.n1_u = Xw_line1_u/Xw_line1_norm
                                    self.n1_v = Xw_line1_v/Xw_line1_norm
                                    self.n1n2 = np.dot(self.n1_u, self.n2_u)+np.dot(self.n1_v,self.n2_v)
                                    self.sum_nm = self.n1n2
                                    self.nn = self.n1_u[i] + self.n1_v[j]
                                    self.nnnn = self.n1_u[i]**2 +self.n1_v[j]**2
                                    self.sum_n = self.n1_u.sum()+self.n1_v.sum() - self.nn
                                    self.sum_m = self.n2_u.sum()+self.n2_v.sum()
                                    self.sum_mm_all = np.dot(self.n2_u,self.n2_u)+np.dot(self.n2_v,self.n2_v)
                                    self.sum_nn_all = np.dot(self.n1_u,self.n1_u)+np.dot(self.n1_v,self.n1_v)
                                    r1 = fsolve(self.func, np.ones(3))
                                    self.final = (1-self.n1_u[i]*r1[1])/(2*r1[0]) + (1-self.n1_v[j]*r1[1])/(2*r1[0])
                                    print(r1[0]/abs(r1[0]), r1[1]/abs(r1[1]), r1[2]/abs(r1[2]))
                                    xiXw = self.u[i] + self.v[j]
                                    x_xwu = - xiXw/self.Xw_norm2 * self.u
                                    x_xwu[i] = x_xwu[i]+1
                                    x_xwv = - xiXw/self.Xw_norm2 * self.v
                                    x_xwv[j] = x_xwv[j]+1
                                    if xitheta_o + self.final < self.theta_best_u[i] + self.theta_best_v[j]:
                                        print("wrong!!")
                                    if xitheta_o + self.final < self.lam * self.C[i, j]:
                                        w_screening[i, j] = 0
                                        countzeros += 1


                                    '''
                                    final_pos_u = (-self.n1_u * r[1] - self.n2_u * r[2]) / (2 * r[0])
                                    final_pos_v = (-self.n1_v * r[1] - self.n2_v * r[2]) / (2 * r[0])
                                    final_pos_u[i] += (1) / (2 * r[0])
                                    final_pos_v[j] += (1) / (2 * r[0])
                                    print(np.dot(final_pos_u,final_pos_u)+np.dot(final_pos_v,final_pos_v) - self.r**2)
                                    print(np.dot(final_pos_u, self.n1_u) + np.dot(final_pos_v, self.n1_v) - self.cosline1*self.r)
                                    print(np.dot(final_pos_u, self.n2_u) + np.dot(final_pos_v, self.n2_v) - self.cosline2*self.r)
'''
                                    # self.c1c2_u = down_line1_theta_u - down_theta_u
                                    # self.c1c2_v = down_line1_theta_v - down_theta_v
                                    # self.dis_c1c2 = np.linalg.norm(np.concatenate((self.c1c2_u,self.c1c2_v)))
                                    # self.dis_c1op = r_new
                                    # self.dis_c2op = math.sqrt(self.r**2 - dis_line1**2)
                                    #
                                    # self.alpha = - (self.c1c2_u[i]+self.c1c2_v[j])**2/(self.dis_c1c2**2) +2
                                    # self.final =  math.sqrt(self.alpha)
                                    # self.coscab = (self.dis_c1c2 ** 2 + self.dis_c2op ** 2 - self.dis_c1op ** 2) / (2 * self.dis_c1c2 * self.dis_c2op)
                                    # self.middle_part = (self.dis_c2op * self.coscab)*(self.c1c2_u[i]+self.c1c2_v[j])
                                    # self.cosfinal = (self.c1c2_u[i]+self.c1c2_v[j])/(xi_norm*self.dis_c1c2)
                                    # self.sinfinal = math.sqrt(1-self.cosfinal**2)
                                    # self.finallength = self.dis_c2op* math.sqrt(1-self.coscab**2)
                                    # if (xitheta_o+self.middle_part+ self.final*self.finallength< self.lam * self.C[i, j]):
                                    #      w_screening[i, j] = 0
                                    #      countzeros += 1
            # 6.2: L2下交 L1交 L2最优在L1外
            #                         self.dis_c1_line2 = line2_c1 / Xw_line2_norm
            #
            #                         self.down_c1_2_theta_u = down_line1_theta_u + (
            #                                 self.dis_c1_line2 * Xw_line2_u / Xw_line2_norm)
            #                         self.down_c1_2_theta_v = down_line1_theta_v + (
            #                                 self.dis_c1_line2* Xw_line2_v / Xw_line2_norm)
                                    # self.coor_a_i = down_theta_u[i]
                                    # self.coor_a_j = down_theta_v[j]
                                    # self.coor_b_i = self.down_c1_2_theta_u[i]
                                    # self.coor_b_j = self.down_c1_2_theta_v[j]
                                    # self.lab = np.linalg.norm([self.coor_a_i-self.coor_b_i,self.coor_a_j-self.coor_b_j])
                                    # self.lac = r_new
                                    # self.lbc = math.sqrt(self.r**2 - dis_line1**2 - self.dis_c1_line2**2)
                                    # self.coscab = (self.lab**2 + self.lac**2 - self.lbc**2)/(2*self.lab*self.lac)
                                    # # if(abs(self.coscab)>1):
                                    # #     aa = 1
                                    # self.sincab =math.sqrt(1-min(self.coscab,1)**2)
                                    # self.vec = [self.coor_b_i-self.coor_a_i,self.coor_b_j-self.coor_a_j]/self.lab
                                    # self.vec1 = [0,0]
                                    # self.vec1[0] = -self.vec[1]
                                    # self.vec1[1] = self.vec[0]
                                    # self.vec2 = [0, 0]
                                    # self.vec2[0] = self.vec[1]
                                    # self.vec2[1] = -self.vec[0]
                                    # self.xifinal = self.coor_a_j +self.coor_a_i + \
                                    #                 self.lac * self.coscab *(self.vec[0]+self.vec[1])+\
                                    #                 max(self.lac*self.sincab*(self.vec1[0]+self.vec1[1]),self.lac*self.sincab*(self.vec2[0]+self.vec2[1]))
                                    # if (self.xifinal < self.lam * self.C[i, j]):
                                    #     w_screening[i, j] = 0
                                    #     countzeros += 1



                # if (xitheta_o + \
                #         xiXw_line1 / Xw_line1_norm2 * line1 + \
                #         math.sqrt(xi_norm - xiXw_line1 ** 2 / (xi_norm * Xw_line1_norm ** 2)) * \
                #         math.sqrt(max(self.r ** 2 - dis_line1 ** 2, 0)) < \
                                # xiXw = self.u[i] + self.v[j]
                                # x_xwu = - xiXw/self.Xw_norm2 * self.u
                                # x_xwu[i] = x_xwu[i]+1
                                # x_xwv = - xiXw/self.Xw_norm2 * self.v
                                # x_xwv[j] = x_xwv[j]+1
                                # if (xitheta_o+xiXw/self.Xw_norm2*self.delta +
                                #     np.linalg.norm(np.concatenate((x_xwu,x_xwv)))*math.sqrt(max(self.r**2-1/(self.Xw_norm2)*self.delta**2,0)) < self.lam*self.C[i,j]):
                                #     w_screening[i,j] = 0
                                #     countzeros += 1
                                # number+=1

                    #得到L1最值
                    #如果L2与球无关系（包裹了球）
                    #否则：
        plt.imshow(w_screening)
        plt.title('running!')
        plt.colorbar(aspect=40, pad=0.08, shrink=0.6,
                     orientation='horizontal', extend='both')
        plt.show()
        print("ratio:", number)
        return countzeros / (w_screening.shape[0] * w_screening.shape[1])
    # def screening_point(self,theta_projected):
    #     countzeros = 0
    #     w_screening = np.ones_like(self.w)
    #     for i in range(w_screening.shape[0]):
    #         if (self.X[:, i].T.dot(theta_projected)[0]< self.lam*self.c[i]):
    #             w_screening[i] = 0
    #             countzeros += 1
    #     return countzeros / w_screening.shape[0]

class sasvi_screening_matrix_debug(screener_matrix):
    def __init__(self, w, a,b ,C, lam, reg="l1",sratio=0,solution=None):
        super().__init__(w, a, b, C, lam, reg="l1")
        self.sratio = 0
        self.solution = solution
        self.u = .0
        self.v = .0
        self.Xw_norm2 = .0
        self.theta_hat_u = .0
        self.theta_hat_v = .0
        self.g = .0
        self.dim_a = .0
        self.dim_b = .0
        self.theta_pu, self.theta_pv = .0, .0
        self.theta_best_u = .0
        self.theta_best_v = .0
        self.r = 0.0
        self.delta = .0
        g_matrix = .0
        g_col_matrix = .0
        g_row_matrix = .0
        g_matrix_all = .0
        theta_x_matrix = .0
        theta_xqua_matrix = .0
        delta_matrix = .0

        w_qua = .0
        w_col_qua = .0
        w_row_qua = .0

        theta_xqua_col = .0
        theta_xqua_row = .0
        theta_xqua_all = .0

        sep_col = .0
        sep_row = .0
        sep_all = .0
        number = .0

        self.cosline2 = .0
        self.cosline1 = .0
        self.n2_u = .0
        self.n2_v = .0
        self.n1_u = .0
        self.n1_v = .0
        self.n1n2 = .0
        self.sum_nm = .0
        self.nn = .0
        self.nnnn = .0
        self.sum_n = .0
        self.sum_m = .0
        self.sum_mm_all = .0
        self.sum_nn_all = .0

        self.final = .0

    def update(self, w):
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
        self.theta_pu, self.theta_pv = self.projection_translation(self.theta_hat_u,self.theta_hat_v)
        self.theta_best_u = self.a - self.solution.sum(axis=1)
        self.theta_best_v = self.b - self.solution.sum(axis=0)

        d_opt_alg = math.sqrt(np.dot(self.theta_best_u - self.theta_hat_u, self.theta_best_u - self.theta_hat_u) +
                              np.dot(self.theta_best_v - self.theta_hat_v, self.theta_best_v - self.theta_hat_v))
        d_opt_proj = math.sqrt(np.dot(self.theta_best_u - self.theta_pu, self.theta_best_u - self.theta_pu) +
                               np.dot(self.theta_best_v - self.theta_pv, self.theta_best_v - self.theta_pv))
        d_alg_proj = math.sqrt(np.dot(self.theta_hat_u - self.theta_pu, self.theta_hat_u - self.theta_pu) +
                               np.dot(self.theta_hat_v - self.theta_pv, self.theta_hat_v - self.theta_pv))

        dis = {"opt_alg": d_opt_alg,
               "opt_proj": d_opt_proj,
               "alg_proj": d_alg_proj,
               }

        w_screening_area1 = self.screening_area(self.theta_pu, self.theta_pv)
        w_screening_area2 = self.screening_divided_area(self.theta_pu, self.theta_pv)
        w_screening = {"screening_area1": w_screening_area1,
                       "screening_area2": w_screening_area2}
        return dis, w_screening

    # def projection_normal(self, alg_theta):
    #     beilv = self.X.T.dot(alg_theta)/(self.lam * self.c)
    #     out = alg_theta / max(1,beilv.max())
    #     return out, beilv
    def func(self, i):
        l1, l2, l3 = i[0], i[1], i[2]
        return [
            (- self.sum_nm) * l3 - self.sum_nn_all * l2 - 2 * l1 * self.cosline1 * self.r + self.nn,
            (- self.sum_nm) * l2 - self.sum_mm_all * l3 - 2 * l1 * self.cosline2 * self.r,
            (l2 * self.nn - 1 - 0.5 * self.sum_nn_all * l2 ** 2 - 0.5 * self.sum_mm_all * l3 ** 2 -
             self.sum_nm * l3 * l2) + 2 * self.r**2 * l1**2]

    def projection_translation(self, at_u, at_v):
        trans = ((at_u[:, None]+at_v[None, :]-(self.lam * self.C))/2)
        # plt.imshow(trans)
        # plt.colorbar(aspect=40, pad=0.08, shrink=0.6,
        #              orientation='horizontal', extend='both')
        # plt.show
        difu = trans.max(axis=1)
        difv = trans.max(axis=0)
        outu = at_u - np.where(difu > 0, difu, 0)
        outv = at_v - np.where(difv > 0, difv, 0)
        return outu, outv

    def screening_area(self, pu, pv):
        countzeros = 0
        w_screening = np.ones_like(self.w)
        theta_ou = 0.5 * (pu + self.a)
        theta_ov = 0.5 * (pv + self.b)
        self.r = 0.5 * math.sqrt(np.dot(pu - self.a, pu - self.a)+np.dot(pv - self.b, pv - self.b))
        self.delta = self.lam * np.multiply(self.C, self.w).sum() - np.dot(theta_ou, self.u) -\
            np.dot(theta_ov, self.v)
        xi_norm = math.sqrt(2)
        for i in range(w_screening.shape[0]):
            for j in range(w_screening.shape[1]):
                xiXw = self.u[i] + self.v[j]
                xitheta_o = theta_ou[i] + theta_ov[j]
                if self.r/xi_norm * xiXw <= self.delta:
                    if xitheta_o + self.r * xi_norm < self.lam * self.C[i,j]:
                        w_screening[i, j] = 0
                        countzeros += 1
                else:
                    x_xwu = - xiXw/self.Xw_norm2 * self.u
                    x_xwu[i] = x_xwu[i] + 1
                    x_xwv = - xiXw/self.Xw_norm2 * self.v
                    x_xwv[j] = x_xwv[j] + 1
                    if xitheta_o + xiXw/self.Xw_norm2*self.delta + \
                            np.linalg.norm(np.concatenate((x_xwu, x_xwv))) * \
                            math.sqrt(max(self.r**2-1/self.Xw_norm2 * self.delta**2, 0)) < self.lam*self.C[i, j]:
                        w_screening[i, j] = 0
                        countzeros += 1
        plt.imshow(w_screening)
        plt.title('running!')
        plt.colorbar(aspect=40, pad=0.08, shrink=0.6,
                     orientation='horizontal', extend='both')
        plt.show()
        aaa = 1
        return countzeros / (w_screening.shape[0]*w_screening.shape[1])

    def screening_selected_area(self, pu, pv):
        countzeros = 0
        usefulcons = 0
        w_screening = np.ones_like(self.w)
        w_cons = np.zeros_like(self.w)
        theta_ou = 0.5 * (pu + self.a)
        theta_ov = 0.5 * (pv + self.b)
        self.r = 0.5 * math.sqrt(np.dot(pu - self.a, pu - self.a)+np.dot(pv - self.b, pv - self.b))
        self.delta = self.lam * np.multiply(self.C, self.w).sum() - np.dot(theta_ou, self.u) -\
            np.dot(theta_ov, self.v)
        xi_norm = math.sqrt(2)
        for i in range(w_screening.shape[0]):
            for j in range(w_screening.shape[1]):
                if theta_ou[i] + 2* self.r /xi_norm +theta_ov[j] > self.lam * self.C[i,j]:
                    usefulcons += 1
                    w_cons[i,j] = 1
        print("useful constraints",usefulcons / (w_screening.shape[0]*w_screening.shape[1]))
        plt.imshow(w_cons)
        plt.title('constraints!')
        plt.colorbar(aspect=40, pad=0.08, shrink=0.6,
                     orientation='horizontal', extend='both')
        plt.show()

        new_w = np.where(w_cons>0,self.w,0)
        new_u = new_w.sum(axis=1)
        new_v = new_w.sum(axis=0)
        # u,v 就是Xw
        Xw_norm2 = np.dot(new_u,new_u)+np.dot(new_v,new_v)
        delta = self.lam * np.multiply(self.C, new_w).sum() - np.dot(theta_ou, new_u) -\
            np.dot(theta_ov, new_v)
        for i in range(w_screening.shape[0]):
            for j in range(w_screening.shape[1]):
                xiXw = new_u[i] + new_v[j]
                xitheta_o = theta_ou[i] + theta_ov[j]
                if self.r/xi_norm * xiXw <= delta:
                    if xitheta_o + self.r * xi_norm < self.lam * self.C[i,j]:
                        w_screening[i, j] = 0
                        countzeros += 1
                else:
                    x_xwu = - xiXw/Xw_norm2 * new_u
                    x_xwu[i] = x_xwu[i] + 1
                    x_xwv = - xiXw/Xw_norm2 * new_v
                    x_xwv[j] = x_xwv[j] + 1
                    if xitheta_o + xiXw/Xw_norm2*delta + \
                            np.linalg.norm(np.concatenate((x_xwu, x_xwv))) * \
                            math.sqrt(max(self.r**2-1/Xw_norm2 * delta**2, 0)) < self.lam*self.C[i, j]:
                        w_screening[i, j] = 0
                        countzeros += 1
        plt.imshow(w_screening)
        plt.title('running!')
        plt.colorbar(aspect=40, pad=0.08, shrink=0.6,
                     orientation='horizontal', extend='both')
        plt.show()
        aaa = 1
        return countzeros / (w_screening.shape[0]*w_screening.shape[1])
    def screening_divided_area(self, pu, pv):
        countzeros = 0
        w_screening = np.ones_like(self.w)
        theta_ou = 0.5 * (pu + self.a)
        theta_ov = 0.5 * (pv + self.b)
        self.r = 0.5 * math.sqrt(np.dot(pu - self.a, pu - self.a)+np.dot(pv - self.b, pv - self.b))
        g_matrix = self.lam * np.multiply(self.C, self.w)
        g_col_matrix = g_matrix.sum(axis=0)
        g_row_matrix = g_matrix.sum(axis=1)
        g_matrix_all = g_col_matrix.sum()
        theta_x_matrix = theta_ou[:, None]+theta_ov[None, :]
        theta_xqua_matrix = np.multiply(theta_x_matrix, theta_x_matrix)
        delta_matrix = g_matrix - (np.multiply(theta_x_matrix, self.w))

        w_qua = np.multiply(self.w, self.w)
        w_col_qua = w_qua.sum(axis=0)
        w_row_qua = w_qua.sum(axis=1)

        theta_xqua_col = theta_xqua_matrix.sum(axis=0)
        theta_xqua_row = theta_xqua_matrix.sum(axis=1)
        theta_xqua_all = theta_xqua_row.sum()

        sep_col = delta_matrix.sum(axis=0)
        sep_row = delta_matrix.sum(axis=1)
        sep_all = sep_row.sum()
        xi_norm = math.sqrt(2)
        number = 0
        for i in range(self.w.shape[0]):

            for j in range(self.w.shape[1]):
                line1 = sep_col[j]+sep_row[i]-delta_matrix[i, j] # constraint plane
                line2 = sep_all - line1 # parallel plane

                line1_include = 0
                line2_include = 0
                # 到L2的距离

                Xw_line1_u = copy.copy(self.w[:,j])
                Xw_line1_v = copy.copy(self.w[i, :])
                Xw_line1_v[j] = self.v[j]
                Xw_line1_u[i] = self.u[i]
                Xw_line1_norm = np.linalg.norm(np.concatenate((Xw_line1_u, Xw_line1_v)))
                Xw_line2_u = self.u - Xw_line1_u
                Xw_line2_v = self.v - Xw_line1_v
                Xw_line2_norm = np.linalg.norm(np.concatenate((Xw_line2_u, Xw_line2_v)))
                dis_line1 = line1 / Xw_line1_norm
                dis_line2 = line2 / Xw_line2_norm

                if line2 >= 0:
                    #平面在原点上半
                    if dis_line2 > self.r:
                        line2_include = 1
                    else:
                        r_new = math.sqrt(self.r**2 - dis_line2**2)
                else:
                    #平面在原点下半
                    r_new = math.sqrt(self.r ** 2 - dis_line2 ** 2)

                #L1 部分
                xiXw_line1 = Xw_line1_u[i] + Xw_line1_v[j]
                xiXw_line2 = Xw_line2_u[i] + Xw_line2_v[j]
                xitheta_o = theta_ou[i] + theta_ov[j]
                if self.r/xi_norm * xiXw_line1 <= line1:
                    # l1与球无关系
                    if line2_include == 1:
            #1：L2 全 L1 全
            #3：L2 上交 L1 全
                        if xitheta_o +self.r*xi_norm < self.lam*self.C[i,j]:
                            w_screening[i,j] = 0
                            countzeros += 1
                    elif line2 < 0:
                        if xitheta_o +r_new * xi_norm < self.lam*self.C[i,j]:
                            w_screening[i,j] = 0
                            countzeros += 1
            #5：L2 下交 L1 全
                else:
                    Xw_line1_norm2 = Xw_line1_norm**2
                    Xw_line2_norm2 = Xw_line2_norm**2
                    x_xw_line1_u = - xiXw_line1 / Xw_line1_norm2 * Xw_line1_u
                    x_xw_line1_u[i] = x_xw_line1_u[i] + 1
                    x_xw_line1_v = - xiXw_line1 / Xw_line1_norm2 * Xw_line1_v
                    x_xw_line1_v[j] = x_xw_line1_v[j] + 1
                    x_xw_line2_u = - xiXw_line2 / Xw_line2_norm2 * Xw_line2_u
                    x_xw_line2_u[i] = x_xw_line2_u[i] + 1
                    x_xw_line2_v = - xiXw_line2 / Xw_line2_norm2 * Xw_line2_v
                    x_xw_line2_v[j] = x_xw_line2_v[j] + 1
                    if line2_include == 1:
            # 2： L2全 L1 交
                        if xitheta_o + xiXw_line1 / Xw_line1_norm2 * line1 + \
                                np.linalg.norm(np.concatenate((x_xw_line1_u, x_xw_line1_v))) * math.sqrt(
                                    max(self.r ** 2 - 1 / Xw_line1_norm2 * line1 ** 2, 0)) < self.lam * self.C[
                                    i, j]:
                            w_screening[i, j] = 0
                            countzeros += 1
                    else:

                        #求投到L2上圆的距离lambda
            # 4: L2上交 L1交
            # 6: L2下交 L1交
                        if line2 >= 0:
            # 4: L2上半 L1交
                            if xitheta_o + xiXw_line1 / Xw_line1_norm2 * line1 + \
                                    np.linalg.norm(np.concatenate((x_xw_line1_u, x_xw_line1_v))) * math.sqrt(
                                        max(self.r ** 2 - 1 / Xw_line1_norm2 * line1 ** 2, 0)) < self.lam * self.C[i, j]:
                                w_screening[i, j] = 0
                                countzeros += 1
                        else:
            # 6: L2下交 L1交
                            down_theta_u = theta_ou + (dis_line2 * Xw_line2_u/Xw_line2_norm)
                            down_theta_v = theta_ov + (dis_line2 * Xw_line2_v/Xw_line2_norm)
                            opt_down_u = copy.copy(down_theta_u)
                            opt_down_u[i] = down_theta_u[i] + r_new/xi_norm
                            opt_down_v = copy.copy(down_theta_v)
                            opt_down_v[j] = down_theta_v[j] + r_new/xi_norm
                            opt2forl1 = g_col_matrix[j] + g_row_matrix[i] - g_matrix[i, j] -\
                                        np.dot(opt_down_u, Xw_line1_u) - np.dot(opt_down_v, Xw_line1_v)

                            if opt2forl1 >= 0:
            # 6.1: L2下交 L1交 L2最优在L1内
                                if xitheta_o + r_new * xi_norm < self.lam * self.C[i, j]:
                                    w_screening[i, j] = 0
                                    countzeros += 1
                            else:
                                down_line1_theta_u = theta_ou + (
                                            dis_line1 * Xw_line1_u / Xw_line1_norm)
                                down_line1_theta_v = theta_ov + (
                                            dis_line1 * Xw_line1_v / Xw_line1_norm)
                                line2_c1 = g_matrix_all - g_col_matrix[j] - g_row_matrix[i] + \
                                           g_matrix[i, j] - np.dot(down_line1_theta_u, Xw_line2_u) - \
                                           np.dot(down_line1_theta_v, Xw_line2_v)
                                if line2_c1 >= 0:
                                    if xitheta_o + \
                                            xiXw_line1 / Xw_line1_norm2 * line1 + \
                                            np.linalg.norm(np.concatenate((x_xw_line1_u, x_xw_line1_v))) * \
                                            math.sqrt(max(self.r ** 2 - dis_line1 ** 2, 0)) < \
                                            self.lam * self.C[i, j]:
                                        w_screening[i, j] = 0
                                        countzeros += 1
                                else:
                                    self.cosline2 = (dis_line2/self.r)
                                    self.cosline1 = (dis_line1/self.r)
                                    self.n2_u = (Xw_line2_u/Xw_line2_norm)
                                    self.n2_v= Xw_line2_v/Xw_line2_norm
                                    self.n1_u = Xw_line1_u/Xw_line1_norm
                                    self.n1_v = Xw_line1_v/Xw_line1_norm
                                    self.n1n2 = np.dot(self.n1_u, self.n2_u)+np.dot(self.n1_v,self.n2_v)
                                    self.sum_nm = self.n1n2
                                    self.nn = self.n1_u[i] + self.n1_v[j]
                                    self.nnnn = self.n1_u[i]**2 +self.n1_v[j]**2
                                    self.sum_n = self.n1_u.sum()+self.n1_v.sum() - self.nn
                                    self.sum_m = self.n2_u.sum()+self.n2_v.sum()
                                    self.sum_mm_all = np.dot(self.n2_u,self.n2_u)+np.dot(self.n2_v,self.n2_v)
                                    self.sum_nn_all = np.dot(self.n1_u,self.n1_u)+np.dot(self.n1_v,self.n1_v)
                                    r1 = fsolve(self.func, np.ones(3),maxfev=1000)
                                    if r1[0] < 0:
                                        r1 = fsolve(self.func, [-r1[0],1,1])
                                    if r1[1] < 0:
                                        self.final = xiXw_line1 / Xw_line1_norm2 * line1 + \
                                            np.linalg.norm(np.concatenate((x_xw_line1_u, x_xw_line1_v))) * \
                                            math.sqrt(max(self.r ** 2 - dis_line1 ** 2, 0))
                                    else:
                                        if r1[2] < 0:
                                            self.final = xiXw_line2 / Xw_line2_norm2 * line2 + \
                                            np.linalg.norm(np.concatenate((x_xw_line2_u, x_xw_line2_v))) * \
                                            math.sqrt(max(self.r ** 2 - dis_line1 ** 2, 0))
                                    self.final = (1-self.n1_u[i]*r1[1])/(2*r1[0]) + (1-self.n1_v[j]*r1[1])/(2*r1[0])
                                    # print(r1[0]/abs(r1[0]), r1[1]/abs(r1[1]), r1[2]/abs(r1[2]))
                                    # xiXw = self.u[i] + self.v[j]
                                    # x_xwu = - xiXw/self.Xw_norm2 * self.u
                                    # x_xwu[i] = x_xwu[i]+1
                                    # x_xwv = - xiXw/self.Xw_norm2 * self.v
                                    # x_xwv[j] = x_xwv[j]+1
                                    # finalln = xiXw/self.Xw_norm2*self.delta + \
                                    #     np.linalg.norm(np.concatenate((x_xwu, x_xwv)))*math.sqrt(max(self.r**2-1/self.Xw_norm2*self.delta**2, 0))
                                    # finall1 = xiXw_line1 / Xw_line1_norm2 * line1 + \
                                    #         np.linalg.norm(np.concatenate((x_xw_line1_u, x_xw_line1_v))) * \
                                    #         math.sqrt(max(self.r ** 2 - dis_line1 ** 2, 0))
                                    # finall2 = xiXw_line2 / Xw_line2_norm2 * line2 + \
                                    #         np.linalg.norm(np.concatenate((x_xw_line2_u, x_xw_line2_v))) * \
                                    #         math.sqrt(max(self.r ** 2 - dis_line1 ** 2, 0))
                                    # print('{:.5g}'.format(self.final ),'{:.5g}'.format(finall1 ),'{:.5g}'.format(finall2 ),'{:.5g}'.format(finalln))
                                    # if xitheta_o + self.final < self.theta_best_u[i] + self.theta_best_v[j]:
                                    #     print("wrong!!")
                                    # if self.final > xiXw/self.Xw_norm2*self.delta + \
                                    #      np.linalg.norm(np.concatenate((x_xwu, x_xwv)))*math.sqrt(max(self.r**2-1/self.Xw_norm2*self.delta**2, 0)):
                                    #     print("wrong!! again")
                                    if i == 29 and j == 29:
                                        print("here")
                                    if xitheta_o + self.final < self.lam * self.C[i, j] - 1e-10:
                                        w_screening[i, j] = 0
                                        countzeros += 1

                                    '''
                                    final_pos_u = (-self.n1_u * r[1] - self.n2_u * r[2]) / (2 * r[0])
                                    final_pos_v = (-self.n1_v * r[1] - self.n2_v * r[2]) / (2 * r[0])
                                    final_pos_u[i] += (1) / (2 * r[0])
                                    final_pos_v[j] += (1) / (2 * r[0])
                                    print(np.dot(final_pos_u,final_pos_u)+np.dot(final_pos_v,final_pos_v) - self.r**2)
                                    print(np.dot(final_pos_u, self.n1_u) + np.dot(final_pos_v, self.n1_v) - self.cosline1*self.r)
                                    print(np.dot(final_pos_u, self.n2_u) + np.dot(final_pos_v, self.n2_v) - self.cosline2*self.r)
'''
                                    # self.c1c2_u = down_line1_theta_u - down_theta_u
                                    # self.c1c2_v = down_line1_theta_v - down_theta_v
                                    # self.dis_c1c2 = np.linalg.norm(np.concatenate((self.c1c2_u,self.c1c2_v)))
                                    # self.dis_c1op = r_new
                                    # self.dis_c2op = math.sqrt(self.r**2 - dis_line1**2)
                                    #
                                    # self.alpha = - (self.c1c2_u[i]+self.c1c2_v[j])**2/(self.dis_c1c2**2) +2
                                    # self.final =  math.sqrt(self.alpha)
                                    # self.coscab = (self.dis_c1c2 ** 2 + self.dis_c2op ** 2 - self.dis_c1op ** 2) / (2 * self.dis_c1c2 * self.dis_c2op)
                                    # self.middle_part = (self.dis_c2op * self.coscab)*(self.c1c2_u[i]+self.c1c2_v[j])
                                    # self.cosfinal = (self.c1c2_u[i]+self.c1c2_v[j])/(xi_norm*self.dis_c1c2)
                                    # self.sinfinal = math.sqrt(1-self.cosfinal**2)
                                    # self.finallength = self.dis_c2op* math.sqrt(1-self.coscab**2)
                                    # if (xitheta_o+self.middle_part+ self.final*self.finallength< self.lam * self.C[i, j]):
                                    #      w_screening[i, j] = 0
                                    #      countzeros += 1
            # 6.2: L2下交 L1交 L2最优在L1外
            #                         self.dis_c1_line2 = line2_c1 / Xw_line2_norm
            #
            #                         self.down_c1_2_theta_u = down_line1_theta_u + (
            #                                 self.dis_c1_line2 * Xw_line2_u / Xw_line2_norm)
            #                         self.down_c1_2_theta_v = down_line1_theta_v + (
            #                                 self.dis_c1_line2* Xw_line2_v / Xw_line2_norm)
                                    # self.coor_a_i = down_theta_u[i]
                                    # self.coor_a_j = down_theta_v[j]
                                    # self.coor_b_i = self.down_c1_2_theta_u[i]
                                    # self.coor_b_j = self.down_c1_2_theta_v[j]
                                    # self.lab = np.linalg.norm([self.coor_a_i-self.coor_b_i,self.coor_a_j-self.coor_b_j])
                                    # self.lac = r_new
                                    # self.lbc = math.sqrt(self.r**2 - dis_line1**2 - self.dis_c1_line2**2)
                                    # self.coscab = (self.lab**2 + self.lac**2 - self.lbc**2)/(2*self.lab*self.lac)
                                    # # if(abs(self.coscab)>1):
                                    # #     aa = 1
                                    # self.sincab =math.sqrt(1-min(self.coscab,1)**2)
                                    # self.vec = [self.coor_b_i-self.coor_a_i,self.coor_b_j-self.coor_a_j]/self.lab
                                    # self.vec1 = [0,0]
                                    # self.vec1[0] = -self.vec[1]
                                    # self.vec1[1] = self.vec[0]
                                    # self.vec2 = [0, 0]
                                    # self.vec2[0] = self.vec[1]
                                    # self.vec2[1] = -self.vec[0]
                                    # self.xifinal = self.coor_a_j +self.coor_a_i + \
                                    #                 self.lac * self.coscab *(self.vec[0]+self.vec[1])+\
                                    #                 max(self.lac*self.sincab*(self.vec1[0]+self.vec1[1]),self.lac*self.sincab*(self.vec2[0]+self.vec2[1]))
                                    # if (self.xifinal < self.lam * self.C[i, j]):
                                    #     w_screening[i, j] = 0
                                    #     countzeros += 1



                # if (xitheta_o + \
                #         xiXw_line1 / Xw_line1_norm2 * line1 + \
                #         math.sqrt(xi_norm - xiXw_line1 ** 2 / (xi_norm * Xw_line1_norm ** 2)) * \
                #         math.sqrt(max(self.r ** 2 - dis_line1 ** 2, 0)) < \
                                # xiXw = self.u[i] + self.v[j]
                                # x_xwu = - xiXw/self.Xw_norm2 * self.u
                                # x_xwu[i] = x_xwu[i]+1
                                # x_xwv = - xiXw/self.Xw_norm2 * self.v
                                # x_xwv[j] = x_xwv[j]+1
                                # if (xitheta_o+xiXw/self.Xw_norm2*self.delta +
                                #     np.linalg.norm(np.concatenate((x_xwu,x_xwv)))*math.sqrt(max(self.r**2-1/(self.Xw_norm2)*self.delta**2,0)) < self.lam*self.C[i,j]):
                                #     w_screening[i,j] = 0
                                #     countzeros += 1
                                # number+=1

                    #得到L1最值
                    #如果L2与球无关系（包裹了球）
                    #否则：
        plt.imshow(w_screening)
        plt.title('running!')
        plt.colorbar(aspect=40, pad=0.08, shrink=0.6,
                     orientation='horizontal', extend='both')
        plt.show()
        print("ratio:", number)
        return countzeros / (w_screening.shape[0] * w_screening.shape[1])
    # def screening_point(self,theta_projected):
    #     countzeros = 0
    #     w_screening = np.ones_like(self.w)
    #     for i in range(w_screening.shape[0]):
    #         if (self.X[:, i].T.dot(theta_projected)[0]< self.lam*self.c[i]):
    #             w_screening[i] = 0
    #             countzeros += 1
    #     return countzeros / w_screening.shape[0]


class Gap_screening_matrix(screener_matrix):
    def __init__(self, w, a,b ,C, lam,epsilon, reg="l1",sratio=0,mint= 1e-6, solution=None):
        super().__init__(w, a, b, C, lam, reg="l1")
        self.sratio = 0
        self.solution = solution
        self.u = .0
        self.v = .0
        self.Xw_norm2 = .0
        self.theta_hat_u = .0
        self.theta_hat_v = .0
        self.g = .0
        self.dim_a = .0
        self.dim_b = .0
        self.theta_pu, self.theta_pv = .0, .0
        self.theta_best_u = .0
        self.theta_best_v = .0
        self.r = 0.0
        self.delta = .0

        self.eps = epsilon
        self.mint = mint
        self.alpha = .0
        self.gap = .0
        self.primal = .0
        self.dual = .0

    def update(self, w):
        self.w = w
        self.w = np.where(w>self.mint,w,self.mint)
        plt.imshow(self.w)
        plt.title('solurion')
        plt.colorbar(aspect=40, pad=0.08, shrink=0.6,
                     orientation='horizontal', extend='both')
        plt.show()
        self.dim_a = w.shape[0]
        self.dim_b = w.shape[1]
        self.u = self.w.sum(axis=1)
        self.v = self.w.sum(axis=0)
        self.theta_hat_u = np.log(self.u/self.a)
        self.theta_hat_v = np.log(self.v/self.b)
        self.primal_func(self.w,self.u,self.v)
        self.dual_func(self.theta_hat_u,self.theta_hat_v)
        self.gap = self.primal - self.dual
        self.g = self.lam * np.multiply(self.C,self.w).sum()

        print("primal: ",self.primal," dual: ", self.dual)
        print("gap: ",self.gap)
        # self.theta_best_u = np.log(self.solution.sum(axis=1)/self.a)
        # self.theta_best_v = np.log(self.solution.sum(axis=0)/self.b)
        # self.primal_func(self.solution,self.solution.sum(axis=1),self.solution.sum(axis=0))
        # self.dual_func(self.theta_best_u,self.theta_best_v)

        w_screening_area1 = self.screening_area(self.theta_hat_u, self.theta_hat_v)
        w_screening = {"screening_area1": w_screening_area1}
        return w_screening

    # def projection_normal(self, alg_theta):
    #     beilv = self.X.T.dot(alg_theta)/(self.lam * self.c)
    #     out = alg_theta / max(1,beilv.max())
    #     return out, beilv
    def dual_func(self,theta_u,theta_v):
        self.dual = -self.a.dot(np.exp(-theta_u))-self.b.dot(np.exp(-theta_v))
        w_screening = np.ones_like(self.w)
        down = self.eps * (np.log(self.mint) + 1)
        for i in range(self.dim_a):
            for j in range(self.dim_b):
                middle = theta_u[i] + theta_v[j] - self.lam * self.C[i,j]
                if middle >= down:
                    self.dual -= self.eps * math.exp(middle/self.eps - 1)
                else:
                    self.dual -= self.mint * (middle - self.eps * math.exp(self.mint))
                    w_screening[i,j] = 0
        plt.imshow(w_screening)
        plt.title('dual condition!')
        plt.colorbar(aspect=40, pad=0.08, shrink=0.6,
                     orientation='horizontal', extend='both')
        plt.show()

    def primal_func(self,w,u,v):
        self.primal = self.lam * np.multiply(self.C,w).sum() + u.dot(np.log(u/self.a))+\
                      v.dot(np.log(v/self.b))+\
                      self.eps * np.multiply(np.log(w),w).sum()

    def projection_translation(self, at_u, at_v):
        trans = ((at_u[:, None]+at_v[None, :]-(self.lam * self.C))/2)
        # plt.imshow(trans)
        # plt.colorbar(aspect=40, pad=0.08, shrink=0.6,
        #              orientation='horizontal', extend='both')
        # plt.show
        difu = trans.max(axis=1)
        difv = trans.max(axis=0)
        outu = at_u - np.where(difu > 0, difu, 0)
        outv = at_v - np.where(difv > 0, difv, 0)
        return outu, outv

    def screening_area(self,theta_u,theta_v):
        countzeros = 0
        w_screening = np.ones_like(self.w)
        self.alpha = self.lam ** 2 ** 1
        self.r = math.sqrt(2*self.gap/self.alpha)
        xi_norm = math.sqrt(2)
        for i in range(w_screening.shape[0]):
            for j in range(w_screening.shape[1]):
                if theta_u[i] + theta_v[j] + self.r *xi_norm < self.lam *self.C[i,j]:
                    w_screening[i,j] = 0
                    countzeros += 1
        plt.imshow(w_screening)
        plt.title('running!')
        plt.colorbar(aspect=40, pad=0.08, shrink=0.6,
                     orientation='horizontal', extend='both')
        plt.show()
        return countzeros / (w_screening.shape[0]*w_screening.shape[1])

    def screening_divided_area(self, pu, pv):
        countzeros = 0
        w_screening = np.ones_like(self.w)
        theta_ou = 0.5 * (pu + self.a)
        theta_ov = 0.5 * (pv + self.b)
        self.r = 0.5 * math.sqrt(np.dot(pu - self.a, pu - self.a)+np.dot(pv - self.b, pv - self.b))
        g_matrix = self.lam * np.multiply(self.C, self.w)
        g_col_matrix = g_matrix.sum(axis=0)
        g_row_matrix = g_matrix.sum(axis=1)
        g_matrix_all = g_col_matrix.sum()
        theta_x_matrix = theta_ou[:, None]+theta_ov[None, :]
        theta_xqua_matrix = np.multiply(theta_x_matrix, theta_x_matrix)
        delta_matrix = g_matrix - (np.multiply(theta_x_matrix, self.w))

        w_qua = np.multiply(self.w, self.w)
        w_col_qua = w_qua.sum(axis=0)
        w_row_qua = w_qua.sum(axis=1)

        theta_xqua_col = theta_xqua_matrix.sum(axis=0)
        theta_xqua_row = theta_xqua_matrix.sum(axis=1)
        theta_xqua_all = theta_xqua_row.sum()

        sep_col = delta_matrix.sum(axis=0)
        sep_row = delta_matrix.sum(axis=1)
        sep_all = sep_row.sum()
        xi_norm = math.sqrt(2)
        number = 0
        for i in range(self.w.shape[0]):

            for j in range(self.w.shape[1]):
                line1 = sep_col[j]+sep_row[i]-delta_matrix[i, j] # constraint plane
                line2 = sep_all - line1 # parallel plane

                line1_include = 0
                line2_include = 0
                # 到L2的距离

                Xw_line1_u = copy.copy(self.w[:,j])
                Xw_line1_v = copy.copy(self.w[i, :])
                Xw_line1_v[j] = self.v[j]
                Xw_line1_u[i] = self.u[i]
                Xw_line1_norm = np.linalg.norm(np.concatenate((Xw_line1_u, Xw_line1_v)))
                Xw_line2_u = self.u - Xw_line1_u
                Xw_line2_v = self.v - Xw_line1_v
                Xw_line2_norm = np.linalg.norm(np.concatenate((Xw_line2_u, Xw_line2_v)))
                dis_line1 = line1 / Xw_line1_norm
                dis_line2 = line2 / Xw_line2_norm

                if line2 >= 0:
                    #平面在原点上半
                    if dis_line2 > self.r:
                        line2_include = 1
                    else:
                        r_new = math.sqrt(self.r**2 - dis_line2**2)
                else:
                    #平面在原点下半
                    r_new = math.sqrt(self.r ** 2 - dis_line2 ** 2)

                #L1 部分
                xiXw_line1 = Xw_line1_u[i] + Xw_line1_v[j]
                xitheta_o = theta_ou[i] + theta_ov[j]
                if self.r/xi_norm * xiXw_line1 <= line1:
                    # l1与球无关系
                    if line2_include == 1:
            #1：L2 全 L1 全
            #3：L2 上交 L1 全
                        if xitheta_o +self.r*xi_norm < self.lam*self.C[i,j]:
                            w_screening[i,j] = 0
                            countzeros += 1
                    elif line2 < 0:
                        if xitheta_o +r_new * xi_norm < self.lam*self.C[i,j]:
                            w_screening[i,j] = 0
                            countzeros += 1
            #5：L2 下交 L1 全
                else:
                    Xw_line1_norm2 = Xw_line1_norm**2
                    x_xw_line1_u = - xiXw_line1 / Xw_line1_norm2 * Xw_line1_u
                    x_xw_line1_u[i] = x_xw_line1_u[i] + 1
                    x_xw_line1_v = - xiXw_line1 / Xw_line1_norm2 * Xw_line1_v
                    x_xw_line1_v[j] = x_xw_line1_v[j] + 1
                    if line2_include == 1:
            # 2： L2全 L1 交
                        if xitheta_o + xiXw_line1 / Xw_line1_norm2 * line1 + \
                                np.linalg.norm(np.concatenate((x_xw_line1_u, x_xw_line1_v))) * math.sqrt(
                                    max(self.r ** 2 - 1 / Xw_line1_norm2 * line1 ** 2, 0)) < self.lam * self.C[
                                    i, j]:
                            w_screening[i, j] = 0
                            countzeros += 1
                    else:

                        #求投到L2上圆的距离lambda
            # 4: L2上交 L1交
            # 6: L2下交 L1交
                        if line2 >= 0:
            # 4: L2上半 L1交
                            if xitheta_o + xiXw_line1 / Xw_line1_norm2 * line1 + \
                                    np.linalg.norm(np.concatenate((x_xw_line1_u, x_xw_line1_v))) * math.sqrt(
                                        max(self.r ** 2 - 1 / Xw_line1_norm2 * line1 ** 2, 0)) < self.lam * self.C[i, j]:
                                w_screening[i, j] = 0
                                countzeros += 1
                        else:
            # 6: L2下交 L1交
                            down_theta_u = theta_ou + (dis_line2 * Xw_line2_u/Xw_line2_norm)
                            down_theta_v = theta_ov + (dis_line2 * Xw_line2_v/Xw_line2_norm)
                            opt_down_u = copy.copy(down_theta_u)
                            opt_down_u[i] = down_theta_u[i] + r_new/xi_norm
                            opt_down_v = copy.copy(down_theta_v)
                            opt_down_v[j] = down_theta_v[j] + r_new/xi_norm
                            opt2forl1 = g_col_matrix[j] + g_row_matrix[i] - g_matrix[i, j] -\
                                        np.dot(opt_down_u, Xw_line1_u) - np.dot(opt_down_v, Xw_line1_v)

                            if opt2forl1 >= 0:
            # 6.1: L2下交 L1交 L2最优在L1内
                                if xitheta_o + r_new * xi_norm < self.lam * self.C[i, j]:
                                    w_screening[i, j] = 0
                                    countzeros += 1
                            else:
                                down_line1_theta_u = theta_ou + (
                                            dis_line1 * Xw_line1_u / Xw_line1_norm)
                                down_line1_theta_v = theta_ov + (
                                            dis_line1 * Xw_line1_v / Xw_line1_norm)
                                line2_c1 = g_matrix_all - g_col_matrix[j] - g_row_matrix[i] + \
                                           g_matrix[i, j] - np.dot(down_line1_theta_u, Xw_line2_u) - \
                                           np.dot(down_line1_theta_v, Xw_line2_v)
                                if line2_c1 >= 0:
                                    if xitheta_o + \
                                            xiXw_line1 / Xw_line1_norm2 * line1 + \
                                            np.linalg.norm(np.concatenate((x_xw_line1_u, x_xw_line1_v))) * \
                                            math.sqrt(max(self.r ** 2 - dis_line1 ** 2, 0)) < \
                                            self.lam * self.C[i, j]:
                                        w_screening[i, j] = 0
                                        countzeros += 1
                                else:
                                    self.cosline2 = (dis_line2/self.r)
                                    self.cosline1 = (dis_line1/self.r)
                                    self.n2_u = (Xw_line2_u/Xw_line2_norm)
                                    self.n2_v= Xw_line2_v/Xw_line2_norm
                                    self.n1_u = Xw_line1_u/Xw_line1_norm
                                    self.n1_v = Xw_line1_v/Xw_line1_norm
                                    self.n1n2 = np.dot(self.n1_u, self.n2_u)+np.dot(self.n1_v,self.n2_v)
                                    self.sum_nm = self.n1n2
                                    self.nn = self.n1_u[i] + self.n1_v[j]
                                    self.nnnn = self.n1_u[i]**2 +self.n1_v[j]**2
                                    self.sum_n = self.n1_u.sum()+self.n1_v.sum() - self.nn
                                    self.sum_m = self.n2_u.sum()+self.n2_v.sum()
                                    self.sum_mm_all = np.dot(self.n2_u,self.n2_u)+np.dot(self.n2_v,self.n2_v)
                                    self.sum_nn_all = np.dot(self.n1_u,self.n1_u)+np.dot(self.n1_v,self.n1_v)
                                    r1 = fsolve(self.func, np.ones(3))
                                    self.final = (1-self.n1_u[i]*r1[1])/(2*r1[0]) + (1-self.n1_v[j]*r1[1])/(2*r1[0])
                                    print(r1[0]/abs(r1[0]), r1[1]/abs(r1[1]), r1[2]/abs(r1[2]))
                                    xiXw = self.u[i] + self.v[j]
                                    x_xwu = - xiXw/self.Xw_norm2 * self.u
                                    x_xwu[i] = x_xwu[i]+1
                                    x_xwv = - xiXw/self.Xw_norm2 * self.v
                                    x_xwv[j] = x_xwv[j]+1
                                    if xitheta_o + self.final < self.theta_best_u[i] + self.theta_best_v[j]:
                                        print("wrong!!")
                                    if xitheta_o + self.final < self.lam * self.C[i, j]:
                                        w_screening[i, j] = 0
                                        countzeros += 1


                                    '''
                                    final_pos_u = (-self.n1_u * r[1] - self.n2_u * r[2]) / (2 * r[0])
                                    final_pos_v = (-self.n1_v * r[1] - self.n2_v * r[2]) / (2 * r[0])
                                    final_pos_u[i] += (1) / (2 * r[0])
                                    final_pos_v[j] += (1) / (2 * r[0])
                                    print(np.dot(final_pos_u,final_pos_u)+np.dot(final_pos_v,final_pos_v) - self.r**2)
                                    print(np.dot(final_pos_u, self.n1_u) + np.dot(final_pos_v, self.n1_v) - self.cosline1*self.r)
                                    print(np.dot(final_pos_u, self.n2_u) + np.dot(final_pos_v, self.n2_v) - self.cosline2*self.r)
'''
                                    # self.c1c2_u = down_line1_theta_u - down_theta_u
                                    # self.c1c2_v = down_line1_theta_v - down_theta_v
                                    # self.dis_c1c2 = np.linalg.norm(np.concatenate((self.c1c2_u,self.c1c2_v)))
                                    # self.dis_c1op = r_new
                                    # self.dis_c2op = math.sqrt(self.r**2 - dis_line1**2)
                                    #
                                    # self.alpha = - (self.c1c2_u[i]+self.c1c2_v[j])**2/(self.dis_c1c2**2) +2
                                    # self.final =  math.sqrt(self.alpha)
                                    # self.coscab = (self.dis_c1c2 ** 2 + self.dis_c2op ** 2 - self.dis_c1op ** 2) / (2 * self.dis_c1c2 * self.dis_c2op)
                                    # self.middle_part = (self.dis_c2op * self.coscab)*(self.c1c2_u[i]+self.c1c2_v[j])
                                    # self.cosfinal = (self.c1c2_u[i]+self.c1c2_v[j])/(xi_norm*self.dis_c1c2)
                                    # self.sinfinal = math.sqrt(1-self.cosfinal**2)
                                    # self.finallength = self.dis_c2op* math.sqrt(1-self.coscab**2)
                                    # if (xitheta_o+self.middle_part+ self.final*self.finallength< self.lam * self.C[i, j]):
                                    #      w_screening[i, j] = 0
                                    #      countzeros += 1
            # 6.2: L2下交 L1交 L2最优在L1外
            #                         self.dis_c1_line2 = line2_c1 / Xw_line2_norm
            #
            #                         self.down_c1_2_theta_u = down_line1_theta_u + (
            #                                 self.dis_c1_line2 * Xw_line2_u / Xw_line2_norm)
            #                         self.down_c1_2_theta_v = down_line1_theta_v + (
            #                                 self.dis_c1_line2* Xw_line2_v / Xw_line2_norm)
                                    # self.coor_a_i = down_theta_u[i]
                                    # self.coor_a_j = down_theta_v[j]
                                    # self.coor_b_i = self.down_c1_2_theta_u[i]
                                    # self.coor_b_j = self.down_c1_2_theta_v[j]
                                    # self.lab = np.linalg.norm([self.coor_a_i-self.coor_b_i,self.coor_a_j-self.coor_b_j])
                                    # self.lac = r_new
                                    # self.lbc = math.sqrt(self.r**2 - dis_line1**2 - self.dis_c1_line2**2)
                                    # self.coscab = (self.lab**2 + self.lac**2 - self.lbc**2)/(2*self.lab*self.lac)
                                    # # if(abs(self.coscab)>1):
                                    # #     aa = 1
                                    # self.sincab =math.sqrt(1-min(self.coscab,1)**2)
                                    # self.vec = [self.coor_b_i-self.coor_a_i,self.coor_b_j-self.coor_a_j]/self.lab
                                    # self.vec1 = [0,0]
                                    # self.vec1[0] = -self.vec[1]
                                    # self.vec1[1] = self.vec[0]
                                    # self.vec2 = [0, 0]
                                    # self.vec2[0] = self.vec[1]
                                    # self.vec2[1] = -self.vec[0]
                                    # self.xifinal = self.coor_a_j +self.coor_a_i + \
                                    #                 self.lac * self.coscab *(self.vec[0]+self.vec[1])+\
                                    #                 max(self.lac*self.sincab*(self.vec1[0]+self.vec1[1]),self.lac*self.sincab*(self.vec2[0]+self.vec2[1]))
                                    # if (self.xifinal < self.lam * self.C[i, j]):
                                    #     w_screening[i, j] = 0
                                    #     countzeros += 1



                # if (xitheta_o + \
                #         xiXw_line1 / Xw_line1_norm2 * line1 + \
                #         math.sqrt(xi_norm - xiXw_line1 ** 2 / (xi_norm * Xw_line1_norm ** 2)) * \
                #         math.sqrt(max(self.r ** 2 - dis_line1 ** 2, 0)) < \
                                # xiXw = self.u[i] + self.v[j]
                                # x_xwu = - xiXw/self.Xw_norm2 * self.u
                                # x_xwu[i] = x_xwu[i]+1
                                # x_xwv = - xiXw/self.Xw_norm2 * self.v
                                # x_xwv[j] = x_xwv[j]+1
                                # if (xitheta_o+xiXw/self.Xw_norm2*self.delta +
                                #     np.linalg.norm(np.concatenate((x_xwu,x_xwv)))*math.sqrt(max(self.r**2-1/(self.Xw_norm2)*self.delta**2,0)) < self.lam*self.C[i,j]):
                                #     w_screening[i,j] = 0
                                #     countzeros += 1
                                # number+=1

                    #得到L1最值
                    #如果L2与球无关系（包裹了球）
                    #否则：
        plt.imshow(w_screening)
        plt.title('running!')
        plt.colorbar(aspect=40, pad=0.08, shrink=0.6,
                     orientation='horizontal', extend='both')
        plt.show()
        print("ratio:", number)
        return countzeros / (w_screening.shape[0] * w_screening.shape[1])
    # def screening_point(self,theta_projected):
    #     countzeros = 0
    #     w_screening = np.ones_like(self.w)
    #     for i in range(w_screening.shape[0]):
    #         if (self.X[:, i].T.dot(theta_projected)[0]< self.lam*self.c[i]):
    #             w_screening[i] = 0
    #             countzeros += 1
    #     return countzeros / w_screening.shape[0]
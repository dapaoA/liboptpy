import numpy as np
import math
import matplotlib.pyplot as plt
import copy
from scipy.optimize import fsolve




class screener(object):
    def __init__(self,w, X, y, c, lam, reg="l1" ):
        self.w = w  # primal variable
        self.X = X  # constrain matrix
        self.y = y  # distributions
        self.c = c
        self.lam = lam
        self.reg = reg

    def update(self, w):
        return 0

class screener_matrix(object):
    def __init__(self, w, a, b, C, lam, reg="l1" ):
        self.w = w  # primal variable
        self.a = a  # constrain matrix
        self.b = b  # distributions
        self.C = C
        self.lam = lam
        self.reg = reg

    def update(self, w):
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
        self.theta_p2,beilv2 = self.projection(self.theta_hat)

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

    def projection(self,alg_theta):
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
        self.theta_p1,beilv1 = self.projection(self.theta_hat)

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

    def projection(self,alg_theta):
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
        self.theta_p1,beilv1 = self.projection(self.theta_hat)
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

    def projection(self,alg_theta):
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
    def __init__(self, w, a, b ,C, lam, reg="l1",sratio=0,solution=None):
        super().__init__(w, a, b, C, lam, reg="l1")

        self.sratio = 0
        self.solution = solution
        self.Nt = .0
        self.Mt = .0
        self.Xw_norm2 = .0
        self.u = .0
        self.v = .0
        self.g = .0
        self.dim_a = .0
        self.dim_b = .0
        self.u_proj, self.v_proj = .0, .0
        self.u_opt = .0
        self.v_opt = .0
        self.r = 0.0
        self.delta = .0
        self.coslineB = .0
        self.coslineA = .0
        self.n2_u = .0
        self.n2_v = .0
        self.n1_u = .0
        self.n1_v = .0
        self.n1n2 = .0
        self.sum_nm = .0
        self.nn = .0
        self.sum_mm_all = .0
        self.sum_nn_all = .0

        self.final = .0

    def update(self, w, w_s_last=None, distance_log=None):
        self.w = w
        # w is t in paper
        self.Nt = w.sum(axis=1)
        self.Mt = w.sum(axis=0)
        # Nt,Mt is the Nt and Mt
        self.Xw_norm2 = np.dot(self.Nt, self.Nt)+np.dot(self.Mt, self.Mt)
        self.u = self.a - self.Nt
        self.v = self.b - self.Mt
        # [u^{T}, v^{T}]^{T} = \theta
        self.g = self.lam * np.multiply(self.C,self.w).sum()
        self.dim_a = self.w.shape[0]
        self.dim_b = self.w.shape[1]
        self.u_proj, self.v_proj = self.projection(self.u,self.v)
        # project the dual variable into the \mathcal{R}^{D}


        self.u_opt = self.a - self.solution.sum(axis=1)
        self.v_opt = self.b - self.solution.sum(axis=0)
        # for test, u,v are the theoretical best dual solution

        d_opt_alg = math.sqrt(np.dot(self.u_opt - self.u, self.u_opt - self.u) +
                              np.dot(self.v_opt - self.v, self.v_opt - self.v))
        d_opt_proj = math.sqrt(np.dot(self.u_opt - self.u_proj, self.u_opt - self.u_proj) +
                               np.dot(self.v_opt - self.v_proj, self.v_opt - self.v_proj))
        d_alg_proj = math.sqrt(np.dot(self.u - self.u_proj, self.u - self.u_proj) +
                               np.dot(self.v - self.v_proj, self.v - self.v_proj))

        if distance_log == 1:
            dis = {"opt_alg": d_opt_alg,
                   "opt_proj": d_opt_proj,
                    "alg_proj": d_alg_proj,
                    }

        # for drawing the distance of different projection in dual sapce


        w_screening_area1 = self.screening_area(self.u_proj, self.v_proj)
        w_screening_area2 = self.screening_divided_area(self.u_proj, self.v_proj)
        w_screening = {"screening_area1": w_screening_area1,
                       "screening_area2": w_screening_area2}
        if distance_log == 1:
            return dis, w_screening
        else:
            return w_screening
    # def projection_normal(self, alg_theta):
    #     beilv = self.X.T.dot(alg_theta)/(self.lam * self.c)
    #     out = alg_theta / max(1,beilv.max())
    #     return out, beilv
    # def func(self, i):
    #     l1, l2, l3 = i[0], i[1], i[2]
    #     return [
    #         - self.sum_nm * l3 - self.sum_nn_all * l2 - 2 * l1 * self.coslineA * self.r + self.nn, #+ or -??
    #         - self.sum_nm * l2 - self.sum_mm_all * l3 - 2 * l1 * self.coslineB * self.r,
    #         - 2 * l2 * self.nn + 2 + self.sum_nn_all * l2 ** 2 + self.sum_mm_all * l3 ** 2 +\
    #         2 * self.sum_nm * l3 * l2 - 4 * self.r ** 2 * l1 ** 2] # 我刚刚吧第一个L2前面的负号系数去掉了

    def func2(self):
        fenmu = self.sum_mm_all * self.sum_nn_all - self.sum_nm**2
        cn1 = 2 * (self.sum_nm * self.coslineB * self.r - self.sum_mm_all * self.coslineA * self.r)/fenmu
        cn2 = (self.nn * self.sum_mm_all)/fenmu
        cm1 = 2 * (self.sum_nm * self.coslineA * self.r - self.sum_nn_all * self.coslineB * self.r)/fenmu
        cm2 = (- self.nn * self.sum_nm)/fenmu
        aa = 4 * self.r**2 - cn1**2 * self.sum_nn_all - cm1**2 * self.sum_mm_all - 2 * cn1 * cm1 * self.sum_nm
        bb = - 2 * cn1 * cn2 * self.sum_nn_all - 2 * cm1 * cm2 * self.sum_mm_all -\
            2 * (cn1 * cm2 + cn2 * cm1) * self.sum_nm + 2 * self.nn * cn1
        cc = -2 - cn2**2 * self.sum_nn_all - cm2**2 * self.sum_mm_all - 2 * cn2 * cm2 * self.sum_nm + 2 * self.nn * cn2
        if bb ** 2 - 4 * aa * cc < 0:
            print("error")
        eta = (-bb + math.sqrt(max(0, bb ** 2 - 4 * aa * cc)))/(2 * aa)
        # when bb ** 2 - 4 * aa * cc < 0, it indicates another situation that the circle is not activated
        mu = cn1 * eta + cn2
        nu = cm1 * eta + cm2
        return [eta, mu, nu]


    def projection_lasso(self, u, v):
        trans = ((u[:, None]+v[None, :])/(self.lam *self.C))
        max_trans = max(1,trans.max().max())
        return u/max_trans, v/max_trans


    def projection(self, u, v):
        trans = ((u[:, None]+v[None, :]-(self.lam * self.C))/2)
        # plt.imshow(trans)
        # plt.colorbar(aspect=40, pad=0.08, shrink=0.6,
        #              orientation='horizontal', extend='both')
        # plt.show
        difu = trans.max(axis=1)
        difv = trans.max(axis=0)
        outu = u - np.where(difu > 0, difu, 0)
        outv = v - np.where(difv > 0, difv, 0)
        return outu, outv


    def projection_activate(self, u, v, w_s):
        # only projected to the activated constraints
        trans = ((u[:, None]+v[None, :]-(self.lam * self.C))/2)
        # plt.imshow(trans)
        # plt.colorbar(aspect=40, pad=0.08, shrink=0.6,
        #              orientation='horizontal', extend='both')
        # plt.show
        trans = np.where(w_s == 1, trans, -100000)
        difu = trans.max(axis=1)
        difv = trans.max(axis=0)
        outu = u - np.where(difu > 0, difu, 0)
        outv = v - np.where(difv > 0, difv, 0)
        return outu, outv


    def screening_area(self, up, vp):
        #Dynamic-Sasvi method for screening, using one sphere and one hyper-plane
        countzeros = 0
        w_screening = np.ones_like(self.w)
        theta_ou = 0.5 * (up + self.a)
        theta_ov = 0.5 * (vp + self.b)
        # center of the cirle theta_o
        self.r = 0.5 * math.sqrt(np.dot(up - self.a, up - self.a)+np.dot(vp - self.b, vp - self.b))
        # radius og the circle
        self.delta = self.lam * np.multiply(self.C, self.w).sum() - np.dot(theta_ou, self.Nt) -\
            np.dot(theta_ov, self.Mt)
        # delta = g(t) - \theta^{T}Xt
        xi_norm = math.sqrt(2)
        for i in range(w_screening.shape[0]):
            for j in range(w_screening.shape[1]):
                xiXw = self.Nt[i] + self.Mt[j]
                xitheta_o = theta_ou[i] + theta_ov[j]
                if self.r/xi_norm * xiXw <= self.delta:
                    if xitheta_o + self.r * xi_norm < self.lam * self.C[i,j]:
                        w_screening[i, j] = 0
                        countzeros += 1
                else:
                    x_xwu = - xiXw/self.Xw_norm2 * self.Nt
                    x_xwu[i] = x_xwu[i] + 1
                    x_xwv = - xiXw/self.Xw_norm2 * self.Mt
                    x_xwv[j] = x_xwv[j] + 1
                    if xitheta_o + xiXw/self.Xw_norm2*self.delta + \
                            np.linalg.norm(np.concatenate((x_xwu, x_xwv))) * \
                            math.sqrt(max(self.r**2-1/self.Xw_norm2 * self.delta**2, 0)) < self.lam*self.C[i, j]:
                        w_screening[i, j] = 0
                        countzeros += 1
        # plt.imshow(w_screening)
        # plt.title('running!')
        # plt.colorbar(aspect=40, pad=0.08, shrink=0.6,
        #              orientation='horizontal', extend='both')
        # plt.show()
        return countzeros / (w_screening.shape[0]*w_screening.shape[1])

    def screening_divided_area(self, up, vp):
        countzeros = 0
        w_screening = np.ones_like(self.w)
        theta_ou = 0.5 * (up + self.a)
        theta_ov = 0.5 * (vp + self.b)
        self.r = 0.5 * math.sqrt(np.dot(up - self.a, up - self.a)+np.dot(vp - self.b, vp - self.b))
        xi_norm = math.sqrt(2)
        g_matrix = self.lam * np.multiply(self.C, self.w)
        g_col_sum = g_matrix.sum(axis=0)
        g_row_sum = g_matrix.sum(axis=1)
        g_sum = g_col_sum.sum()
        theta_x_matrix = theta_ou[:, None]+theta_ov[None, :]
        theta_xqua_matrix = np.multiply(theta_x_matrix, theta_x_matrix)
        delta_matrix = g_matrix - (np.multiply(theta_x_matrix, self.w))

        w_qua = np.multiply(self.w, self.w)
        w_col_qua = w_qua.sum(axis=0)
        w_row_qua = w_qua.sum(axis=1)

        theta_xqua_col = theta_xqua_matrix.sum(axis=0)
        theta_xqua_row = theta_xqua_matrix.sum(axis=1)
        theta_xqua_all = theta_xqua_row.sum()

        delta_col_sum = delta_matrix.sum(axis=0)
        delta_row_sum = delta_matrix.sum(axis=1)
        sep_all = delta_row_sum.sum()
        number = 0
        for i in range(self.w.shape[0]):
            for j in range(self.w.shape[1]):
                deltaA = delta_col_sum[j]+delta_row_sum[i]-delta_matrix[i, j]  # constraint plane
                deltaB = sep_all - deltaA  # parallel plane

                if_A_intersect = 1
                if_B_intersect = 1
                # 到L2的距离

                Xw_u_A = copy.copy(self.w[:,j])
                Xw_v_A = copy.copy(self.w[i, :])
                Xw_v_A[j] = self.Mt[j]
                Xw_u_A[i] = self.Nt[i]
                Xw_A_norm = np.linalg.norm(np.concatenate((Xw_u_A, Xw_v_A)))
                Xw_u_B = self.Nt - Xw_u_A
                Xw_v_B = self.Mt - Xw_v_A
                Xw_B_norm = np.linalg.norm(np.concatenate((Xw_u_B, Xw_v_B)))
                dis_A = deltaA / Xw_A_norm
                dis_B = deltaB / Xw_B_norm
                # distancan from the plane A,B to the center of circle
                if abs(dis_B) >= self.r:
                    if_B_intersect = 0
                    # if the plane B intersect with the sphere
                if abs(dis_A) >= self.r:
                    if_A_intersect = 0
                    # if the plane A intersect with the sphere
                xiXw_A = Xw_u_A[i] + Xw_v_A[j]
                xiXw_B = Xw_u_B[i] + Xw_v_B[j]
                xitheta_o = theta_ou[i] + theta_ov[j]
                if if_A_intersect == 0 and if_B_intersect == 0 :
                        if xitheta_o +self.r*xi_norm < self.lam*self.C[i,j]:
                            w_screening[i, j] = 0
                            countzeros += 1
                            # no A no B, only the sphere
                else:
                    Xw_A_norm2 = Xw_A_norm**2
                    Xw_B_norm2 = Xw_B_norm**2
                    x_Xw_u_A = - xiXw_A / Xw_A_norm2 * Xw_u_A
                    x_Xw_u_A[i] = x_Xw_u_A[i] + 1
                    x_Xw_v_A = - xiXw_A / Xw_A_norm2 * Xw_v_A
                    x_Xw_v_A[j] = x_Xw_v_A[j] + 1
                    x_Xw_u_B = - xiXw_B / Xw_B_norm2 * Xw_u_B
                    x_Xw_u_B[i] = x_Xw_u_B[i] + 1
                    x_Xw_v_B = - xiXw_B / Xw_B_norm2 * Xw_v_B
                    x_Xw_v_B[j] = x_Xw_v_B[j] + 1
                    if if_A_intersect == 1 and if_B_intersect == 0:
                        if xitheta_o + xiXw_A / Xw_A_norm2 * deltaA + \
                                np.linalg.norm(np.concatenate((x_Xw_u_A, x_Xw_v_A))) * math.sqrt(
                                max(self.r ** 2 - 1 / Xw_A_norm2 * deltaA ** 2, 0)) < self.lam * self.C[
                                i, j]:
                            w_screening[i, j] = 0
                            countzeros += 1
                    elif if_A_intersect == 0 and if_B_intersect == 1:
                        if xitheta_o + xiXw_B / Xw_B_norm2 * deltaB + \
                                np.linalg.norm(np.concatenate((x_Xw_u_B, x_Xw_v_B))) * math.sqrt(
                                max(self.r ** 2 - 1 / Xw_B_norm2 * deltaB ** 2, 0)) < self.lam * self.C[
                                i, j]:
                            w_screening[i, j] = 0
                            countzeros += 1
                    else:
                        self.coslineB = (dis_B/self.r)
                        self.coslineA = (dis_A/self.r)
                        self.n2_u = (Xw_u_B/Xw_B_norm)
                        self.n2_v= Xw_v_B/Xw_B_norm
                        self.n1_u = Xw_u_A/Xw_A_norm
                        self.n1_v = Xw_v_A/Xw_A_norm
                        self.n1n2 = np.dot(self.n1_u, self.n2_u)+np.dot(self.n1_v,self.n2_v)
                        self.sum_nm = self.n1n2                                                   # sum a_i b_i
                        self.nn = self.n1_u[i] + self.n1_v[j]                                     # a_I1 + a_I2
                        #  self.sum_n = self.n1_u.sum()+self.n1_v.sum() - self.nn
                        self.sum_mm_all = np.dot(self.n2_u,self.n2_u)+np.dot(self.n2_v,self.n2_v) # sum b_i^2
                        self.sum_nn_all = np.dot(self.n1_u,self.n1_u)+np.dot(self.n1_v,self.n1_v) # sum a_i^2
                        r1 = self.func2()
                        # r1 = fsolve(self.func, np.ones(3),maxfev=1000)

                        if r1[0] < 0:
                            r1 = fsolve(self.func, [-r1[0],1,1])
                            # the circle is not activated
                        elif r1[1] < 0 and r1[2] > 0:
                            self.final = xiXw_B / Xw_B_norm2 * deltaB + \
                                         np.linalg.norm(np.concatenate((x_Xw_u_B, x_Xw_v_B))) * \
                                         math.sqrt(max(self.r ** 2 - dis_B ** 2, 0))
                        elif r1[2] < 0 and r1[1] > 0:
                            self.final = xiXw_A / Xw_A_norm2 * deltaA + \
                                         np.linalg.norm(np.concatenate((x_Xw_u_A, x_Xw_v_A))) * \
                                         math.sqrt(max(self.r ** 2 - dis_A ** 2, 0))
                        elif r1[2] < 0 and r1[1] < 0:
                            self.final = self.r * xi_norm
                        else:
                            self.final = (1 - self.n1_u[i] * r1[1]) / (2 * r1[0]) + (
                                1 - self.n1_v[j] * r1[1]) / (2 * r1[0])

                        # following part is for debugging
                        '''
                        print(r1[0]/abs(r1[0]), r1[1]/abs(r1[1]), r1[2]/abs(r1[2]))
                        xiXw = self.Nt[i] + self.Mt[j]
                        x_xwu = - xiXw/self.Xw_norm2 * self.Nt
                        x_xwu[i] = x_xwu[i]+1
                        x_xwv = - xiXw/self.Xw_norm2 * self.Mt
                        x_xwv[j] = x_xwv[j]+1
                        finalln = xiXw/self.Xw_norm2*self.delta + \
                            np.linalg.norm(np.concatenate((x_xwu, x_xwv)))*math.sqrt(max(self.r**2-1/self.Xw_norm2*self.delta**2, 0))
                        finall1 = xiXw_A / Xw_A_norm2 * deltaA + \
                            np.linalg.norm(np.concatenate((x_Xw_u_A, x_Xw_v_A))) * \
                            math.sqrt(max(self.r ** 2 - dis_A ** 2, 0))
                        finall2 = xiXw_B / Xw_B_norm2 * deltaB + \
                            np.linalg.norm(np.concatenate((x_Xw_u_B, x_Xw_v_B))) * \
                            math.sqrt(max(self.r ** 2 - dis_B ** 2, 0))
                        print('{:.5g}'.format(self.final ),'{:.5g}'.format(finall1 ),'{:.5g}'.format(finall2 ),'{:.5g}'.format(finalln))
                        # if xitheta_o + self.final < self.u_opt[i] + self.v_opt[j]:
                        #     print("wrong!!")
                        if self.final > finalln:
                            print("wrong!! again")
                        if i == 29 and j == 29:
                            print("here")
                        '''

                        # screening test
                        if xitheta_o + self.final < self.lam * self.C[i, j] :
                            w_screening[i, j] = 0
                            countzeros += 1



                        '''
                        final_pos_u = (-self.n1_u * r[1] - self.n2_u * r[2]) / (2 * r[0])
                        final_pos_v = (-self.n1_v * r[1] - self.n2_v * r[2]) / (2 * r[0])
                        final_pos_u[i] += (1) / (2 * r[0])
                        final_pos_v[j] += (1) / (2 * r[0])
                        print(np.dot(final_pos_u,final_pos_u)+np.dot(final_pos_v,final_pos_v) - self.r**2)
                        print(np.dot(final_pos_u, self.n1_u) + np.dot(final_pos_v, self.n1_v) - self.coslineA*self.r)
                        print(np.dot(final_pos_u, self.n2_u) + np.dot(final_pos_v, self.n2_v) - self.coslineB*self.r)
                        '''
                                    # self.c1c2_u = down_lineA_theta_u - down_theta_u
                                    # self.c1c2_v = down_lineA_theta_v - down_theta_v
                                    # self.dis_c1c2 = np.linalg.norm(np.concatenate((self.c1c2_u,self.c1c2_v)))
                                    # self.dis_c1op = r_new
                                    # self.dis_c2op = math.sqrt(self.r**2 - dis_A**2)
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
            #                         self.dis_c1_lineB = lineB_c1 / Xw_B_norm
            #
            #                         self.down_c1_2_theta_u = down_lineA_theta_u + (
            #                                 self.dis_c1_lineB * Xw_u_B / Xw_B_norm)
            #                         self.down_c1_2_theta_v = down_lineA_theta_v + (
            #                                 self.dis_c1_lineB* Xw_v_B / Xw_B_norm)
                                    # self.coor_a_i = down_theta_u[i]
                                    # self.coor_a_j = down_theta_v[j]
                                    # self.coor_b_i = self.down_c1_2_theta_u[i]
                                    # self.coor_b_j = self.down_c1_2_theta_v[j]
                                    # self.lab = np.linalg.norm([self.coor_a_i-self.coor_b_i,self.coor_a_j-self.coor_b_j])
                                    # self.lac = r_new
                                    # self.lbc = math.sqrt(self.r**2 - dis_A**2 - self.dis_c1_lineB**2)
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
                #         xiXw_A / Xw_A_norm2 * deltaA + \
                #         math.sqrt(xi_norm - xiXw_A ** 2 / (xi_norm * Xw_A_norm ** 2)) * \
                #         math.sqrt(max(self.r ** 2 - dis_A ** 2, 0)) < \
                                # xiXw = self.Nt[i] + self.Mt[j]
                                # x_xwu = - xiXw/self.Xw_norm2 * self.Nt
                                # x_xwu[i] = x_xwu[i]+1
                                # x_xwv = - xiXw/self.Xw_norm2 * self.Mt
                                # x_xwv[j] = x_xwv[j]+1
                                # if (xitheta_o+xiXw/self.Xw_norm2*self.delta +
                                #     np.linalg.norm(np.concatenate((x_xwu,x_xwv)))*math.sqrt(max(self.r**2-1/(self.Xw_norm2)*self.delta**2,0)) < self.lam*self.C[i,j]):
                                #     w_screening[i,j] = 0
                                #     countzeros += 1
                                # number+=1

                    #得到L1最值
                    #如果L2与球无关系（包裹了球）
                    #否则：
        # plt.imshow(w_screening)
        # plt.title('running!')
        # plt.colorbar(aspect=40, pad=0.08, shrink=0.6,
        #              orientation='horizontal', extend='both')
        # plt.show()
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
        self.Nt = .0
        self.Mt = .0
        self.Xw_norm2 = .0
        self.u = .0
        self.v = .0
        self.g = .0
        self.dim_a = .0
        self.dim_b = .0
        self.u_proj, self.v_proj = .0, .0
        self.u_opt = .0
        self.v_opt = .0
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
        self.Nt = self.w.sum(axis=1)
        self.Mt = self.w.sum(axis=0)
        self.u = - np.log(self.Nt/self.a)
        self.v = - np.log(self.Mt/self.b)
        # self.primal_func(w, w.sum(axis=1), w.sum(axis=0))
        # self.u_proj,self.v_proj = self.projection(np.log(w.sum(axis=1)/self.a),np.log(self.w.sum(axis=0)/self.b))
        # self.dual_func(self.u_proj,self.v_proj)
        # print("org primal: ",self.primal)
        # print("org p=dual: ", self.dual )
        self.primal_func(self.w,self.Nt,self.Mt)
        self.u_proj,self.v_proj = self.projection(self.u,self.v)
        self.dual_func(self.u_proj,self.v_proj)
        # self.dual_func(self.u,self.v)
        self.gap = self.primal - self.dual
        self.g = self.lam * np.multiply(self.C,self.w).sum()

        print("primal: ",self.primal," dual: ", self.dual)
        print("gap: ",self.gap)
        # self.u_opt = np.log(self.solution.sum(axis=1)/self.a)
        # self.v_opt = np.log(self.solution.sum(axis=0)/self.b)
        # self.primal_func(self.solution,self.solution.sum(axis=1),self.solution.sum(axis=0))
        # self.dual_func(self.u_opt,self.v_opt)
        # self.u_proj,self.v_proj = self.projection(self.u_opt,self.v_opt)
        # print("opt")
        # print("primal: ",self.primal," dual: ", self.dual)
        # self.dual_func(self.u_proj,self.v_proj)
        # print("proj dual: ",self.dual)
        # print("gap: ",self.gap)

        w_screening_area1 = self.screening_area(self.u, self.v)
        w_screening = {"screening_area1": w_screening_area1}
        return w_screening
        return 0

    # def projection_normal(self, alg_theta):
    #     beilv = self.X.T.dot(alg_theta)/(self.lam * self.c)
    #     out = alg_theta / max(1,beilv.max())
    #     return out, beilv
    def dual_func(self,theta_u,theta_v):
        self.dual = -self.a.dot(np.exp(-theta_u))-self.b.dot(np.exp(-theta_v)) + self.a.sum() + self.b.sum()
        w_screening = np.ones_like(self.w)
        down = self.eps * math.log(self.mint)
        for i in range(self.dim_a):
            for j in range(self.dim_b):
                middle = theta_u[i] + theta_v[j] - self.lam * self.C[i,j]
                if middle >= down:
                    self.dual -= self.eps * math.exp(middle/self.eps )
                else:
                    self.dual -= self.mint * (middle - self.eps * (math.log(self.mint) - 1))
                    w_screening[i,j] = 0
        plt.imshow(w_screening)
        plt.title('dual condition!')
        plt.colorbar(aspect=40, pad=0.08, shrink=0.6,
                     orientation='horizontal', extend='both')
        plt.show()

    def primal_func(self,w,u,v):
        self.primal = self.lam * np.multiply(self.C,w).sum() + u.dot(np.log(u/self.a))+\
                      v.dot(np.log(v/self.b))- u.sum()- v.sum() + self.a.sum() + self.b.sum()+\
                      self.eps * np.multiply(np.log(w)-1,w).sum()

    def projection(self, u, v):
        trans = ((u[:, None]+v[None, :]-(self.lam * self.C + self.eps * math.log(self.mint)))/2)
        # plt.imshow(np.where(trans>=0,trans,0))
        # plt.colorbar(aspect=40, pad=0.08, shrink=0.6,
        #              orientation='horizontal', extend='both')
        # plt.show()
        difu = trans.max(axis=1)
        difv = trans.max(axis=0)
        outu = u - np.where(difu > 0, difu, 0)
        outv = v - np.where(difv > 0, difv, 0)
        return outu, outv

    def screening_area(self,theta_u,theta_v):
        countzeros = 0
        w_screening = np.ones_like(self.w)
        self.alpha = ((self.lam) ** 2) * 1000000
        self.r = math.sqrt(2*self.gap/self.alpha)
        xi_norm = math.sqrt(2)
        for i in range(w_screening.shape[0]):
            for j in range(w_screening.shape[1]):
                if theta_u[i] + theta_v[j] + self.r *xi_norm < self.lam *self.C[i,j] + self.eps*math.log(self.mint):
                    w_screening[i,j] = 0
                    countzeros += 1
        plt.imshow(w_screening)
        plt.title('running!')
        plt.colorbar(aspect=40, pad=0.08, shrink=0.6,
                     orientation='horizontal', extend='both')
        plt.show()
        return countzeros / (w_screening.shape[0]*w_screening.shape[1])

    def screening_divided_area(self, up, vp):
        countzeros = 0
        w_screening = np.ones_like(self.w)
        theta_ou = 0.5 * (up + self.a)
        theta_ov = 0.5 * (vp + self.b)
        self.r = 0.5 * math.sqrt(np.dot(up - self.a, up - self.a)+np.dot(vp - self.b, vp - self.b))
        g_matrix = self.lam * np.multiply(self.C, self.w)
        g_col_sum = g_matrix.sum(axis=0)
        g_row_sum = g_matrix.sum(axis=1)
        g_sum = g_col_sum.sum()
        theta_x_matrix = theta_ou[:, None]+theta_ov[None, :]
        theta_xqua_matrix = np.multiply(theta_x_matrix, theta_x_matrix)
        delta_matrix = g_matrix - (np.multiply(theta_x_matrix, self.w))

        w_qua = np.multiply(self.w, self.w)
        w_col_qua = w_qua.sum(axis=0)
        w_row_qua = w_qua.sum(axis=1)

        theta_xqua_col = theta_xqua_matrix.sum(axis=0)
        theta_xqua_row = theta_xqua_matrix.sum(axis=1)
        theta_xqua_all = theta_xqua_row.sum()

        delta_col_sum = delta_matrix.sum(axis=0)
        delta_row_sum = delta_matrix.sum(axis=1)
        sep_all = delta_row_sum.sum()
        xi_norm = math.sqrt(2)
        number = 0
        for i in range(self.w.shape[0]):

            for j in range(self.w.shape[1]):
                deltaA = delta_col_sum[j]+delta_row_sum[i]-delta_matrix[i, j] # constraint plane
                deltaB = sep_all - deltaA # parallel plane

                if_A_intersect = 0
                if_B_intersect = 0
                # 到L2的距离

                Xw_u_A = copy.copy(self.w[:,j])
                Xw_v_A = copy.copy(self.w[i, :])
                Xw_v_A[j] = self.Mt[j]
                Xw_u_A[i] = self.Nt[i]
                Xw_A_norm = np.linalg.norm(np.concatenate((Xw_u_A, Xw_v_A)))
                Xw_u_B = self.Nt - Xw_u_A
                Xw_v_B = self.Mt - Xw_v_A
                Xw_B_norm = np.linalg.norm(np.concatenate((Xw_u_B, Xw_v_B)))
                dis_A = deltaA / Xw_A_norm
                dis_B = deltaB / Xw_B_norm

                if deltaB >= 0:
                    #平面在原点上半
                    if dis_B > self.r:
                        if_B_intersect = 1
                    else:
                        r_new = math.sqrt(self.r**2 - dis_B**2)
                else:
                    #平面在原点下半
                    r_new = math.sqrt(self.r ** 2 - dis_B ** 2)

                #L1 部分
                xiXw_A = Xw_u_A[i] + Xw_v_A[j]
                xitheta_o = theta_ou[i] + theta_ov[j]
                if self.r/xi_norm * xiXw_A <= deltaA:
                    # l1与球无关系
                    if if_B_intersect == 1:
            #1：L2 全 L1 全
            #3：L2 上交 L1 全
                        if xitheta_o +self.r*xi_norm < self.lam*self.C[i,j]:
                            w_screening[i,j] = 0
                            countzeros += 1
                    elif deltaB < 0:
                        if xitheta_o +r_new * xi_norm < self.lam*self.C[i,j]:
                            w_screening[i,j] = 0
                            countzeros += 1
            #5：L2 下交 L1 全
                else:
                    Xw_A_norm2 = Xw_A_norm**2
                    x_Xw_u_A = - xiXw_A / Xw_A_norm2 * Xw_u_A
                    x_Xw_u_A[i] = x_Xw_u_A[i] + 1
                    x_Xw_v_A = - xiXw_A / Xw_A_norm2 * Xw_v_A
                    x_Xw_v_A[j] = x_Xw_v_A[j] + 1
                    if if_B_intersect == 1:
            # 2： L2全 L1 交
                        if xitheta_o + xiXw_A / Xw_A_norm2 * deltaA + \
                                np.linalg.norm(np.concatenate((x_Xw_u_A, x_Xw_v_A))) * math.sqrt(
                                    max(self.r ** 2 - 1 / Xw_A_norm2 * deltaA ** 2, 0)) < self.lam * self.C[
                                    i, j]:
                            w_screening[i, j] = 0
                            countzeros += 1
                    else:

                        #求投到L2上圆的距离lambda
            # 4: L2上交 L1交
            # 6: L2下交 L1交
                        if deltaB >= 0:
            # 4: L2上半 L1交
                            if xitheta_o + xiXw_A / Xw_A_norm2 * deltaA + \
                                    np.linalg.norm(np.concatenate((x_Xw_u_A, x_Xw_v_A))) * math.sqrt(
                                        max(self.r ** 2 - 1 / Xw_A_norm2 * deltaA ** 2, 0)) < self.lam * self.C[i, j]:
                                w_screening[i, j] = 0
                                countzeros += 1
                        else:
            # 6: L2下交 L1交
                            down_theta_u = theta_ou + (dis_B * Xw_u_B/Xw_B_norm)
                            down_theta_v = theta_ov + (dis_B * Xw_v_B/Xw_B_norm)
                            opt_down_u = copy.copy(down_theta_u)
                            opt_down_u[i] = down_theta_u[i] + r_new/xi_norm
                            opt_down_v = copy.copy(down_theta_v)
                            opt_down_v[j] = down_theta_v[j] + r_new/xi_norm
                            opt2forl1 = g_col_sum[j] + g_row_sum[i] - g_matrix[i, j] -\
                                        np.dot(opt_down_u, Xw_u_A) - np.dot(opt_down_v, Xw_v_A)

                            if opt2forl1 >= 0:
            # 6.1: L2下交 L1交 L2最优在L1内
                                if xitheta_o + r_new * xi_norm < self.lam * self.C[i, j]:
                                    w_screening[i, j] = 0
                                    countzeros += 1
                            else:
                                down_lineA_theta_u = theta_ou + (
                                            dis_A * Xw_u_A / Xw_A_norm)
                                down_lineA_theta_v = theta_ov + (
                                            dis_A * Xw_v_A / Xw_A_norm)
                                lineB_c1 = g_sum - g_col_sum[j] - g_row_sum[i] + \
                                           g_matrix[i, j] - np.dot(down_lineA_theta_u, Xw_u_B) - \
                                           np.dot(down_lineA_theta_v, Xw_v_B)
                                if lineB_c1 >= 0:
                                    if xitheta_o + \
                                            xiXw_A / Xw_A_norm2 * deltaA + \
                                            np.linalg.norm(np.concatenate((x_Xw_u_A, x_Xw_v_A))) * \
                                            math.sqrt(max(self.r ** 2 - dis_A ** 2, 0)) < \
                                            self.lam * self.C[i, j]:
                                        w_screening[i, j] = 0
                                        countzeros += 1
                                else:
                                    self.coslineB = (dis_B/self.r)
                                    self.coslineA = (dis_A/self.r)
                                    self.n2_u = (Xw_u_B/Xw_B_norm)
                                    self.n2_v= Xw_v_B/Xw_B_norm
                                    self.n1_u = Xw_u_A/Xw_A_norm
                                    self.n1_v = Xw_v_A/Xw_A_norm
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
                                    xiXw = self.Nt[i] + self.Mt[j]
                                    x_xwu = - xiXw/self.Xw_norm2 * self.Nt
                                    x_xwu[i] = x_xwu[i]+1
                                    x_xwv = - xiXw/self.Xw_norm2 * self.Mt
                                    x_xwv[j] = x_xwv[j]+1
                                    if xitheta_o + self.final < self.u_opt[i] + self.v_opt[j]:
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
                                    print(np.dot(final_pos_u, self.n1_u) + np.dot(final_pos_v, self.n1_v) - self.coslineA*self.r)
                                    print(np.dot(final_pos_u, self.n2_u) + np.dot(final_pos_v, self.n2_v) - self.coslineB*self.r)
'''
                                    # self.c1c2_u = down_lineA_theta_u - down_theta_u
                                    # self.c1c2_v = down_lineA_theta_v - down_theta_v
                                    # self.dis_c1c2 = np.linalg.norm(np.concatenate((self.c1c2_u,self.c1c2_v)))
                                    # self.dis_c1op = r_new
                                    # self.dis_c2op = math.sqrt(self.r**2 - dis_A**2)
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
            #                         self.dis_c1_lineB = lineB_c1 / Xw_B_norm
            #
            #                         self.down_c1_2_theta_u = down_lineA_theta_u + (
            #                                 self.dis_c1_lineB * Xw_u_B / Xw_B_norm)
            #                         self.down_c1_2_theta_v = down_lineA_theta_v + (
            #                                 self.dis_c1_lineB* Xw_v_B / Xw_B_norm)
                                    # self.coor_a_i = down_theta_u[i]
                                    # self.coor_a_j = down_theta_v[j]
                                    # self.coor_b_i = self.down_c1_2_theta_u[i]
                                    # self.coor_b_j = self.down_c1_2_theta_v[j]
                                    # self.lab = np.linalg.norm([self.coor_a_i-self.coor_b_i,self.coor_a_j-self.coor_b_j])
                                    # self.lac = r_new
                                    # self.lbc = math.sqrt(self.r**2 - dis_A**2 - self.dis_c1_lineB**2)
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
                #         xiXw_A / Xw_A_norm2 * deltaA + \
                #         math.sqrt(xi_norm - xiXw_A ** 2 / (xi_norm * Xw_A_norm ** 2)) * \
                #         math.sqrt(max(self.r ** 2 - dis_A ** 2, 0)) < \
                                # xiXw = self.Nt[i] + self.Mt[j]
                                # x_xwu = - xiXw/self.Xw_norm2 * self.Nt
                                # x_xwu[i] = x_xwu[i]+1
                                # x_xwv = - xiXw/self.Xw_norm2 * self.Mt
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
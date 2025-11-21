from credit import Credit, r_to_df
from holee import HoLee, holee_forwards, holee_spots
import numpy as np

class PrepaidOption:

    def __init__(self, cred: Credit, shorts: HoLee):

        self.cred = cred
        self.shorts = shorts
        self.dt = cred.dt
        self.N = self.cred.N

        self.opt_val = np.zeros([self.N + 1, self.N + 1])
        self.opt_r_val = np.zeros([self.N + 1, self.N + 1])
        self.cred_pval = np.zeros([self.N + 1, self.N + 1])

        self.valuate()
    
    def valuate(self):

        for j in range(self.N):
            for i in range(j+1):
                forwards = holee_forwards(i, j, self.shorts)
                spots = holee_spots(i, j, self.shorts, forwards)
                for k in range(j+1, self.N+1):
                    self.cred_pval[i, j] += self.cred.coup[k] * r_to_df(spots[k - j], self.dt)


        for i in range(0,self.cred.N+1):
            self.opt_val[i,self.cred.N] = 0

        for j in reversed(range(self.N)):
            for i in range(j+1):
                early = self.cred_pval[i,j] - (self.cred.nom - self.cred.sum_amort[j])
                cont = 0.5 * (self.opt_val[i, j+1] + self.opt_val[i+1, j+1]) ## ACÁ HAY QUE MULTIPLICAR POR D[i,j]??
                self.opt_val[i,j] = max(early, cont)
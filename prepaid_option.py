from credit import Credit, rate_to_disc_factor
from holee import HoLee
import numpy as np

class PrepaidOption:

    def __init__(self, credit: Credit, holee_rates: np.array):

        self.credit = credit
        self.holee_rates = holee_rates
        self.initial_rate = holee_rates[0][0]
        self.n_periods = self.credit.n_coupons

        self.credit_values = np.full((self.n_periods+1, self.n_periods+1), np.nan)
        self.option_values = np.full((self.n_periods+1, self.n_periods+1), np.nan)
        self.option_rate_values = np.full((self.n_periods+1, self.n_periods+1), np.nan)

        self.valuate()

    def valuate(self):

        for j in reversed(range(self.n_periods)):
            for i in range(j+1):
                self.credit_values[i,j] = rate_to_disc_factor(self.holee_rates[i,j]) * (0.5*self.credit_values[i, j+1] + 0.5*self.credit_values[i+1, j+1] + self.credit.coupon[j])

        for i in range(0,self.n_periods+1):
            self.option_values[i,N] = 0
        for j in reversed(range(N)):
            for i in range(j+1):
                early = self.credit_values[i,j] - (self.credit.nominal - self.credit.sum_amort[j])
                cont =  0.5 * (self.option_values[i, j+1] + self.option_values[i+1, j+1]) ## ACÁ HAY QUE MULTIPLICAR POR D[i,j]??
                self.option_values[i,j] = max(early, cont)
import numpy as np
from credit import rate_to_disc_factor

class HoLee:

    def __init__(self, n_periods: int, initial_rate: int, 
                 dt: float, volat: float, disc_factors: float):


        self.n_periods = n_periods
        self.dt = dt
        self.volat = volat
        self.disc_factors = disc_factors

        self.rates = np.zeros([self.n_periods + 1, self.n_periods + 1])
        self.arr_debr = np.zeros([self.n_periods + 1, self.n_periods + 1])
        self.drifts = np.zeros(self.n_periods + 1)

        self.set_initial_values(initial_rate)


    def set_initial_values(self, initial_rate):
        
        self.rates[0][0] = initial_rate
        self.arr_debr[0][0] = 1
        self.drifts[0] = ((-(2*self.disc_factors[1]-1)+np.sqrt(1 + 4 * self.disc_factors[1]**2 * self.dt**3 * self.volat**2) )/ (2*self.disc_factors[1]*self.dt) - self.rates[0][0])/self.dt

        self.rates[1][1] = self.rates[0][0] + self.drifts[0]*self.dt + self.volat*np.sqrt(self.dt)
        self.rates[0][1] = self.rates[0][0] + self.drifts[0]*self.dt - self.volat*np.sqrt(self.dt)
    

    def arr_debr_ident(self, drift: float, period: int) -> float:

        value = 0
        s = self.volat * np.sqrt(self.dt)
        bottom_rate = self.rates[0][period] + drift*self.dt - s
        top_rate = self.rates[period][period] + drift*self.dt + s

        value = 0.5 * self.arr_debr[0][period] * rate_to_disc_factor(bottom_rate, self.dt)
        value += 0.5 * self.arr_debr[period][period] * rate_to_disc_factor(top_rate, self.dt)
        
        for i in range(1, period + 1):
            rate = self.rates[i][period] + drift*self.dt - s
            value += 0.5*(self.arr_debr[i-1][period] + self.arr_debr[i][period])*rate_to_disc_factor(rate)

        value -= self.disc_factors[period + 1]

        return value
    

    def arr_debr_ident_deriv(self, drift: float, period: int) -> float:

        value = 0
        s = self.volat * np.sqrt(self.dt)
        bottom_rate = self.rates[0][period] + drift*self.dt - s
        top_rate = self.rates[period][period] + drift*self.dt + s

        value -= 0.5 * (self.dt**2) * self.arr_debr[0][period] * rate_to_disc_factor(bottom_rate, self.dt)**2
        value -= 0.5 * (self.dt**2) * self.arr_debr[period][period] * rate_to_disc_factor(top_rate, self.dt)**2

        for i in range(1, period+1):
            rate = self.rates[i][j] + drift*self.dt - s
            value -= 0.5 * (self.dt**2) * ((self.arr_debr[i-1][period] + self.arr_debr[i][period])*rate_to_disc_factor(rate, self.dt)**2)
        return value
    

    def newton_raphson(self, period, initial_guess=0.0, tol=1e-12, max_iter=50):

        drift = initial_guess
        for _ in range(max_iter):
            F  = self.arr_debr_ident(drift, period)
            dF = self.arr_debr_ident_deriv(drift, period)

            if abs(dF) < 1e-18:
                break

            step = F / dF

            # backtracking line search
            t = 1.0
            while t > 1e-6:
                cand = theta - t*step

                # chequear positividad de todos los denominadores
                s = self.volat*np.sqrt(self.dt)
                ok = True
                if 1.0 + self.dt*(self.rates[0][period] + cand*self.dt - s) <= 0: ok = False
                if 1.0 + self.dt*(self.rates[period][period] + cand*self.dt + s) <= 0: ok = False
                for i in range(1, period):
                    if 1.0 + self.dt*(self.rates[i][period] + cand*self.dt) <= 0: ok = False

                if ok and abs(self.arr_debr_ident(cand, period)) <= abs(F):
                    theta = cand
                    break

                t *= 0.5
            else:
                # no mejoró: damos el paso completo igualmente para no quedarnos colgados
                theta = theta - step

            if abs(step) < tol and abs(F) < tol:
                break

        return theta
    
    
    def calibrate(self):

        for j in range(1,self.n_periods):
        # Precios Arrow-Debreu
            for i in range(0, j+1):
                rate = self.rates[i][j]
                if i == 0:
                    arr_debr = self.arr_debr[i][j-1]
                    self.arr_debr[i][j] = rate_to_disc_factor(rate, self.dt) * 0.5 * arr_debr
                elif i == j:
                    arr_debr = self.arr_debr[j-1][j-1]
                    self.arr_debr[i][j] = rate_to_disc_factor(rate, self.dt) * 0.5 * arr_debr
                else:
                    arr_debr = (self.arr_debr[i-1][j-1] + self.arr_debr[i][j-1])
                    self.arr_debr[i][j] = rate_to_disc_factor(rate, self.dt) * 0.5 * arr_debr
            
            # theta
            self.drifts[j] = self.newton_raphson(j)

            # Tasas
            for i in range(0, j+1):
                self.rates[i][j+1] = self.rates[i][j] + self.drifts[j]*self.dt - self.volat*np.sqrt(self.dt)
            self.rates[j+1][j+1] = self.rates[j][j] + self.drifts[j]*self.dt + self.volat*np.sqrt(self.dt)
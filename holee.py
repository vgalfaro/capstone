import numpy as np
from credit import r_to_df

class HoLee:

    def __init__(self, N: int, initial_r: int, 
                 dt: float, volat: float, df: float):


        self.N = N
        self.dt = dt
        self.volat = volat
        self.df = df

        self.r = np.zeros([self.N + 1, self.N + 1])
        self.arr_debr = np.zeros([self.N + 1, self.N + 1])
        self.drifts = np.zeros(self.N + 1)

        self.set_initial_values(initial_r)


    def set_initial_values(self, initial_r):
        
        self.r[0][0] = initial_r
        self.arr_debr[0][0] = 1
        self.drifts[0] = ((-(2*self.df[1]-1)+np.sqrt(1 + 4 * self.df[1]**2 * self.dt**3 * self.volat**2) )/ (2*self.df[1]*self.dt) - self.r[0][0])/self.dt

        self.r[1][1] = self.r[0][0] + self.drifts[0]*self.dt + self.volat*np.sqrt(self.dt)
        self.r[0][1] = self.r[0][0] + self.drifts[0]*self.dt - self.volat*np.sqrt(self.dt)
    

    def arr_debr_id(self, drift: float, j: int) -> float:

        value = 0
        s = self.volat * np.sqrt(self.dt)
        bot_r = self.r[0][j] + drift*self.dt - s
        top_r = self.r[j][j] + drift*self.dt + s

        value = 0.5 * self.arr_debr[0][j] * r_to_df(bot_r, self.dt)
        value += 0.5 * self.arr_debr[j][j] * r_to_df(top_r, self.dt)
        
        for i in range(1, j + 1):
            r = self.r[i][j] + drift*self.dt - s
            value += 0.5*(self.arr_debr[i-1][j] + self.arr_debr[i][j])*r_to_df(r, self.dt)

        value -= self.df[j + 1]

        return value
    

    def arr_debr_id_d(self, drift: float, j: int) -> float:

        value = 0
        s = self.volat * np.sqrt(self.dt)
        bot_r = self.r[0][j] + drift*self.dt - s
        top_r = self.r[j][j] + drift*self.dt + s

        value -= 0.5 * (self.dt**2) * self.arr_debr[0][j] * r_to_df(bot_r, self.dt)**2
        value -= 0.5 * (self.dt**2) * self.arr_debr[j][j] * r_to_df(top_r, self.dt)**2

        for i in range(1, j+1):
            r = self.r[i][j] + drift*self.dt - s
            value -= 0.5 * (self.dt**2) * ((self.arr_debr[i-1][j] + self.arr_debr[i][j])*r_to_df(r, self.dt)**2)
        return value
    

    def newton_raphson(self, j, initial_guess=0.0, tol=1e-12, max_iter=50):

        drift = initial_guess
        for _ in range(max_iter):
            F  = self.arr_debr_id(drift, j)
            dF = self.arr_debr_id_d(drift, j)

            if abs(dF) < 1e-18:
                break

            step = F / dF

            # backtracking line search
            t = 1.0
            while t > 1e-6:
                cand = drift - t*step

                # chequear positividad de todos los denominadores
                s = self.volat*np.sqrt(self.dt)
                ok = True
                if 1.0 + self.dt*(self.r[0][j] + cand*self.dt - s) <= 0: ok = False
                if 1.0 + self.dt*(self.r[j][j] + cand*self.dt + s) <= 0: ok = False
                for i in range(1, j):
                    if 1.0 + self.dt*(self.r[i][j] + cand*self.dt) <= 0: ok = False

                if ok and abs(self.arr_debr_id(cand, j)) <= abs(F):
                    drift = cand
                    break

                t *= 0.5
            else:
                # no mejoró: damos el paso completo igualmente para no quedarnos colgados
                drift = drift - step

            if abs(step) < tol and abs(F) < tol:
                break

        return drift
    
    
    def calibrate(self):

        for j in range(1,self.N):
        # Precios Arrow-Debreu
            for i in range(0, j+1):
                r = self.r[i][j]
                if i == 0:
                    arr_debr = self.arr_debr[i][j-1]
                    self.arr_debr[i][j] = r_to_df(r, self.dt) * 0.5 * arr_debr
                elif i == j:
                    arr_debr = self.arr_debr[j-1][j-1]
                    self.arr_debr[i][j] = r_to_df(r, self.dt) * 0.5 * arr_debr
                else:
                    arr_debr = (self.arr_debr[i-1][j-1] + self.arr_debr[i][j-1])
                    self.arr_debr[i][j] = r_to_df(r, self.dt) * 0.5 * arr_debr
            
            # drift
            self.drifts[j] = self.newton_raphson(j)

            # Tasas
            for i in range(0, j+1):
                self.r[i][j+1] = self.r[i][j] + self.drifts[j]*self.dt - self.volat*np.sqrt(self.dt)
            self.r[j+1][j+1] = self.r[j][j] + self.drifts[j]*self.dt + self.volat*np.sqrt(self.dt)


def holee_forwards(i: int, j: int, holee: HoLee)-> np.array:

    forwards = np.zeros([holee.N - j + 1])
    forwards[0] = holee.r[i,j]

    for k in range(1, len(forwards)):
        forwards[k] = forwards[k-1] + holee.dt*holee.drifts[k-1]
    
    return forwards

def holee_spots(i: int, j: int, holee: HoLee, forwards: np.array)-> np.array:

    spots = np.zeros([holee.N - j + 1])
    spots[0] = 0

    for k in range(1, len(spots)):
        spots[k] = ((1/(r_to_df(spots[k-1], holee.dt)*r_to_df(forwards[k-1], holee.dt))) - 1)*(1/holee.dt)
    return spots
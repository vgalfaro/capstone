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
        self.S = np.zeros([self.N + 1, self.N + 1])


        self.S_t = np.zeros([self.N + 1, self.N + 1])
        self.O_t = np.zeros([self.N + 1, self.N + 1])
        # for i in range(self.N):
        #     print(self.shorts.dfs[i][0][0]*100)

        nom = self.cred.nom
        r_0 = self.cred.ann_r
        dt  = self.dt
        N = self.N
        # print(nom, r_0, dt)
        # print(np.sum(self.shorts.dfs[i][0][0] for i in range(self.N)))
        # print(self.shorts.dfs[self.N - 1][0][0])
        # print(nom*r_0*dt*(np.sum(self.shorts.dfs[i][0][0] for i in range(self.N))))
        # print(nom*self.shorts.dfs[self.N - 1][0][0])
        # print(nom)

        # self.S_t[0,0] = max(nom * r_0 * dt * (np.sum(self.shorts.dfs[i][0][0] for i in range(self.N))) + nom * self.shorts.dfs[self.N - 1][0][0] - nom, 0)
        # print(self.S_t[0,0])

        # Matrices para guardar resultados
        W1 = np.zeros((N + 1, N + 1)) # Valor Continuación (S_t)
        W2 = np.zeros((N + 1, N + 1)) # Valor Prepago (Strike)
        V_banco = np.zeros((N + 1, N + 1)) # Valor del Crédito (Callable)
        V_opcion = np.zeros((N + 1, N + 1)) # Valor de la Opción

        # Inducción hacia atrás
        for j in range(N - 1, -1, -1):
            cupon_j = self.cred.coup[j+1] # Flujo que se recibe al final del periodo j (en t=j+1)
            strike_j = self.cred.saldo[j+1] # Saldo insoluto en t=j+1 (Según tus notas W2)
            
            for i in range(j + 1):
                tasa = self.shorts.r[i, j]
                df = r_to_df(tasa, dt)
                
                # --- PASO 2: Calcular W1 (Valor Continuación) ---
                # W1 = Esperanza descontada del valor futuro + Cupón futuro
                # Nota: En t=j, miramos a t=j+1.
                # Si estamos en el último paso (j=N-1), el valor futuro es 0.
                
                val_futuro_up = V_banco[i+1, j+1] if j < N-1 else 0
                val_futuro_down = V_banco[i, j+1] if j < N-1 else 0
                
                esperanza_futura = 0.5 * val_futuro_up + 0.5 * val_futuro_down
                
                # W1 = (Cupón_{j+1} + Valor_{j+1}) * DF
                # Ojo con la convención: ¿El cupón se paga ahora o se descuenta?
                # Según tu Excel S_0, todo se descuenta.
                W1[i, j] = (cupon_j + esperanza_futura) * df
                
                # --- PASO 3: Calcular W2 (Valor Prepago / Strike) ---
                # Tus notas dicen: W2 = Nocional (Saldo Insoluto)
                # El prepago ocurre "ahora" para evitar los flujos futuros.
                # Pero el strike se compara con el valor presente de los flujos.
                # Usaremos el Saldo Insoluto del periodo actual como costo de prepago inmediato.
                W2[i, j] = self.cred.saldo[j+1] # Saldo remanente si prepago en este instante
                
                # --- PASO 4: Decisión (Min/Max) ---
                # Valor del Crédito (Banco) = min(Continuar, Que me prepaguen)
                V_banco[i, j] = min(W1[i, j], W2[i, j])
                
                # Valor de la Opción (Cliente) = max(Ahorro, 0)
                esperado = 0.5 * (V_opcion[i, j+1] + V_opcion[i+1, j+1]) if j < N-1 else 0
                V_opcion[i, j] = max(W1[i, j] - W2[i, j], esperado)

        print("Valor de la Opción de Prepago (bps):", V_opcion[0,0] / self.cred.nom * 10**4)
        print("Valor del Crédito Callable (bps):", V_banco[0,0] / self.cred.nom * 10**4)
        self.valuate()
        self.valuate_old()

    def valuate(self):
        nom = self.cred.nom
        r_0 = self.cred.ann_r
        dt  = self.dt
        for t in range(self.N):
            for e in range(t + 1):
                # print(f"t: {t}, e: {e}")
                sum_dfs = 0
                for i in range(0, self.N - t):
                    sum_dfs += self.shorts.dfs[i][e][t]
                # print(f"sum_dfs: {sum_dfs}")
                # print(f"shorts_dfs: {self.shorts.dfs[self.N - 1 - t][e][t]}")
                self.S_t[e, t] = max(nom * r_0 * dt * sum_dfs + nom * self.shorts.dfs[self.N - 1 - t][e][t] - nom, 0)
        
        for t in range(self.N):
            for e in range(t + 1):
                pass


        for t in reversed(range(self.N)):
            for e in range(t + 1):
                early = self.S_t[e, t]
                cont  = 0.5 * (self.O_t[e, t + 1] + self.O_t[e + 1, t + 1])
                self.O_t[e, t] = max(early, cont)

        print(f"{self.O_t[0,0] / self.cred.nom * 10**4}bps")


    def valuate_old(self):

        for j in range(self.N):
            for i in range(j+1):
                forwards = holee_forwards(i, j, self.shorts)
                spots = holee_spots(i, j, self.shorts, forwards)
                for k in range(j+1, self.N+1):
                    self.S[i, j] += self.cred.coup[k] * r_to_df(spots[k - j], self.dt)
                
                # expected_free_risk_rate = (1 + 0.5 * (self.shorts.r[i+1, j] + self.shorts.r[i+1, j+1])*self.dt)**-1 
                # print(expected_free_risk_rate)
        for i in range(0,self.cred.N+1):
            self.opt_val[i,self.cred.N] = 0

        for j in reversed(range(self.N)):
            for i in range(j+1):
                early = self.S[i,j] - (self.cred.nom - self.cred.sum_amort[j])
                
                expected_rate = self.shorts.r[i,j] + self.shorts.drifts[j]*self.dt
                cont = 0.5 * (self.opt_val[i, j+1] + self.opt_val[i+1, j+1]) * r_to_df(expected_rate, self.dt)
                self.opt_val[i,j] = max(early, cont)
        print(f"{self.opt_val[0,0] / self.cred.nom * 10**4}bps")
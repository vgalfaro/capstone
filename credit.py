class Credit:

    def __init__(self, dt: float, ann_r: float, N: int, nom: float, bullet: bool):

        self.ann_r = ann_r
        self.eff_r = self.ann_r * dt
        self.N = N
        self.nom = nom
        self.bullet = bullet
        self.dt = dt

        self.coup = [0 for i in range(self.N+1)]
        self.inter = [0 for i in range(self.N+1)]
        self.amort = [0 for i in range(self.N+1)]
        self.sum_amort = [0 for i in range(self.N + 1)]

        self.set_coups()

    def set_coups(self):

        if self.bullet:
            # Calculamos el valor de cada cupón
            coup_val = (self.eff_r)*self.nom
            self.inter = [0] + [coup_val for i in range(1, self.N+1)]
            self.coup = [0] + [coup_val for i in range(1, self.N+1)]
            self.coup[-1] = self.coup[-1] + self.nom
            self.amort[-1] = self.nom
            self.sum_amort[-1] = self.nom
            
        else:
            # Calculamos el valor de cada cupón
            numerator = self.eff_r * (1 + self.eff_r)**self.N
            denominator = ((1 + self.eff_r)**self.N - 1)
            coup_val = self.nom * (numerator/denominator)

            # Descomoponemos cada cupón
            for i in range(1, self.N+1):
                self.coup[i] = coup_val
                self.inter[i] = (self.nom - self.sum_amort[i-1]) * self.eff_r
                self.amort[i] = coup_val - self.inter[i]
                self.sum_amort[i] = self.sum_amort[i-1] + self.amort[i]

def r_to_df(r: float, dt: float)->float:
    return 1 / (1 + dt*r)
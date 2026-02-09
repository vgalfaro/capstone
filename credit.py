import numpy as np

class Credit:

    def __init__(self, dt: float, ann_r: float, N: int, nom: float, credit_type: str):

        self.dt = dt
        self.ann_r = ann_r
        self.eff_r = (1 + self.ann_r) ** self.dt - 1
        self.N = N
        self.nom = nom
        self.credit_type = credit_type
        
        self.saldo     = np.zeros(self.N + 1)
        self.coup      = np.zeros(self.N + 1)
        self.inter     = np.zeros(self.N + 1)
        self.amort     = np.zeros(self.N + 1)
        self.sum_amort = np.zeros(self.N + 1)

        if self.credit_type == "bullet":
            self.set_coups_bullet()

        elif self.credit_type == "french":
            self.set_coups_french()

        elif self.credit_type == "german":
            self.set_coups_german()

        elif self.credit_type == "custom":
            pass

    def set_coups_bullet(self):
        coup_val   = (self.eff_r)*self.nom
        self.inter = [0] + [coup_val for i in range(1, self.N+1)]
        self.coup  = [0] + [coup_val for i in range(1, self.N+1)]
        self.coup[-1] = self.coup[-1] + self.nom
        self.amort[-1] = self.nom
        self.sum_amort[-1] = self.nom

    def set_coups_french(self):
        coup_val = self.nom * (self.eff_r * (1 + self.eff_r) ** self.N) / ((1 + self.eff_r) ** self.N - 1)
        # Descomponemos cada cupón

        self.coup = np.full(self.N + 1, coup_val)
        self.coup[0] = 0.0
        self.saldo[0] = self.nom

        for i in range(1, self.N + 1):
            interes = self.saldo[i - 1] * self.eff_r
            amortizacion = coup_val - interes
            self.saldo[i] = self.saldo[i - 1] - amortizacion
            if self.saldo[i] < 0:
                self.saldo[i] = 0.0
            self.inter[i] = interes
            self.amort[i] = amortizacion
            self.sum_amort[i] = self.sum_amort[i - 1] + amortizacion

        # for i in range(1, self.N+1):
        #     self.inter[i] = (self.nom - self.sum_amort[i-1]) * self.eff_r
        #     self.amort[i] = coup_val - self.inter[i]
        #     self.sum_amort[i] = self.sum_amort[i-1] + self.amort[i]

    def set_coups_german(self):
        # Descomponemos cada cupón
        amort_val = self.nom / self.N
        for i in range(1, self.N+1):
            self.amort[i] = amort_val
            self.inter[i] = (self.nom - self.sum_amort[i-1]) * self.eff_r
            self.coup[i] = self.amort[i] + self.inter[i]
            self.sum_amort[i] = self.sum_amort[i-1] + self.amort[i]

def r_to_df(r: float, dt: float)->float:
    return 1 / (1 + dt*r)

"""
class Credit:
    def __init__(self, dt: float, ann_r: float, N: int, nom: float, bullet: bool,
                 mode: str = "compound"):
        #dt    : fracción de año por período (p.ej., 1/12)
        #ann_r : tasa anual en DECIMAL (p.ej., 0.06)
        #N     : número de pagos
        #nom   : capital inicial
        #bullet: True=bullet, False=francés
        #mode  : "simple" o "compound" para definir la tasa por período i

        self.dt = dt
        self.ann_r = ann_r
        self.N = N
        self.nom = nom
        self.bullet = bullet
        self.mode = mode

        # tasa por período i (¡una sola convención para todo el crédito!)
        if mode == "simple":
            self.i = ann_r * dt
        elif mode == "compound":
            self.i = (1.0 + ann_r)**dt - 1.0
        else:
            raise ValueError("mode debe ser 'simple' o 'compound'")

        # arreglos 0..N (índice 0 = antes del 1er pago)
        self.coup = [0.0]*(N+1)       # Cuota total por período (interés + amort)
        self.inter = [0.0]*(N+1)      # Interés por período
        self.amort = [0.0]*(N+1)      # Amortización por período
        self.sum_amort = [0.0]*(N+1)  # Acumulado de amortización

        self._build_schedule()

    def _build_schedule(self):
        i = self.i
        L = self.nom
        N = self.N

        if self.bullet:
            # intereses constantes: i*L en cada 1..N
            for k in range(1, N+1):
                self.inter[k] = i * L
                self.coup[k] = self.inter[k]
            # al final se paga además el principal
            self.amort[N] = L
            self.coup[N] += L
            self.sum_amort[N] = L

        else:
            # Cuota fija (sistema francés)
            if abs(i) < 1e-14:
                # Caso límite i ~ 0
                cuota = L / N
            else:
                cuota = L * (i * (1 + i)**N) / ((1 + i)**N - 1)

            saldo = L
            for k in range(1, N+1):
                self.inter[k] = saldo * i
                self.amort[k] = cuota - self.inter[k]
                # proteger contra redondeos en el último pago:
                if k == N:
                    self.amort[k] = L - (self.sum_amort[k-1])
                self.coup[k] = self.inter[k] + self.amort[k]
                self.sum_amort[k] = self.sum_amort[k-1] + self.amort[k]
                saldo = L - self.sum_amort[k]
"""

def r_to_df(r: float, dt: float) -> float:
    """Descuento simple por paso (consistente con árbol si usas simple)."""
    return 1.0 / (1.0 + r * dt)


if __name__ == "__main__":
    # Ejemplo de uso
    dt = 1 / 12  # mensual
    ann_r = 0.06  # tasa anual en decimal
    N = 12 * 20  # 20 años
    nom = 100_000_000  # capital inicial
    credit_type = "bullet"  # tipo de crédito
    credit = Credit(dt=dt, ann_r=ann_r, N=N, nom=nom, credit_type=credit_type)
    print("Cuotas:", credit.coup)
    print("Intereses:", credit.inter)
    print("Amortizaciones:", credit.amort[240])
    print("Suma de amortizaciones:", credit.sum_amort[240])
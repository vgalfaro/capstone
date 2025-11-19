class Credit:

    def __init__(self, ann_rate: float, n_coupons: int, nominal: float, bullet: bool):

        self.ann_rate = ann_rate
        self.eff_rate = self.ann_rate/12
        self.n_coupons = n_coupons
        self.nominal = nominal
        self.bullet = bullet

        self.coupon_value = 0
        self.interests = [0 for i in range(self.n_coupons)]
        self.amort = [0 for i in range(self.n_coupons)]
        self.sum_amort = [0 for i in range(self.n_coupons + 1)]

        self.set_coupons()

    def set_coupons(self):

        if self.bullet:
            # Calculamos el valor de cada cupón
            self.coupon_value = (self.eff_rate)*self.nominal
            self.interests = [self.coupon_value for i in range(self.n_coupons)]
            self.amort[-1] = self.nominal
            self.sum_amort[-1] = self.nominal
            
        else:
            # Calculamos el valor de cada cupón
            numerator = self.eff_rate * (1 + self.eff_rate)**self.n_coupons
            denominator = ((1 + self.eff_rate)**self.n_coupons - 1)
            self.coupon_value = self.nominal * (numerator/denominator)

            # Descomoponemos cada cupón
            for i in range(self.n_coupons):
                self.interests[i] = (self.nominal - self.sum_amort[i]) * self.eff_rate
                self.amort[i] = self.coupon_value - self.interests[i]
                self.sum_amort[i+1] = self.sum_amort[i] + self.amort[i]



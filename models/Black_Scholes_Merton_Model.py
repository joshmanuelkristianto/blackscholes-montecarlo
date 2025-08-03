# Import third-party libraries
import numpy as np
from scipy.stats import norm

# Import local packages
from .base import OptionPricingModel

class BlackScholesMertonModel(OptionPricingModel):
    def __init__(self, spot_price, strike_price, days_to_maturity, risk_free_rate, volatility):
        self.S = spot_price
        self.K = strike_price
        self.T = days_to_maturity / 365
        self.r = risk_free_rate
        self.sigma = volatility

    def _calculate_d1_d2(self):
        self.d1 = (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T))
        self.d2 = self.d1 - self.sigma * np.sqrt(self.T)

        # Store normal CDF values
        self.N_d1 = norm.cdf(self.d1)
        self.N_d2 = norm.cdf(self.d2)
        self.N_minus_d1 = norm.cdf(-self.d1)
        self.N_minus_d2 = norm.cdf(-self.d2)


    def _calculate_call_option_price(self):
        self._calculate_d1_d2()
        self.call_price = self.S * self.N_d1 - self.K * np.exp(-self.r * self.T) * self.N_d2
        return self.call_price

    def _calculate_put_option_price(self):
        self._calculate_d1_d2()
        self.put_price = self.K * np.exp(-self.r * self.T) * self.N_minus_d2 - self.S * self.N_minus_d1
        return self.put_price
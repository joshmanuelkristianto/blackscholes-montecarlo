import numpy as np
import matplotlib.pyplot as plt

class MonteCarloOptionPricer:
    """
    Monte Carlo simulation for European call and put option pricing.
    """

    def __init__(self, spot_price, strike_price, days_to_maturity, risk_free_rate, volatility, num_simulations=10000):
        self.S0 = spot_price
        self.K = strike_price
        self.T = days_to_maturity / 365
        self.r = risk_free_rate
        self.sigma = volatility
        self.N = num_simulations
        self.steps = days_to_maturity
        self.dt = self.T / self.steps
        self.price_paths = None

    def simulate_price_paths(self, seed=42):
        np.random.seed(seed)

        # Initialize array to store simulations
        S = np.zeros((self.steps, self.N))
        S[0] = self.S0

        for t in range(1, self.steps):
            Z = np.random.standard_normal(self.N)
            S[t] = S[t - 1] * np.exp(
                (self.r - 0.5 * self.sigma**2) * self.dt + self.sigma * np.sqrt(self.dt) * Z
            )

        self.price_paths = S

    def price_option(self, option_type="call"):
        """
        Price the option using the terminal prices from simulation.
        """
        if self.price_paths is None:
            raise ValueError("You must run simulate_price_paths() first.")

        final_prices = self.price_paths[-1]

        if option_type.lower() == "call":
            payoffs = np.maximum(final_prices - self.K, 0)
        elif option_type.lower() == "put":
            payoffs = np.maximum(self.K - final_prices, 0)
        else:
            raise ValueError("option_type must be 'call' or 'put'")

        discounted_payoff = np.exp(-self.r * self.T) * np.mean(payoffs)
        return discounted_payoff

    def plot_simulations(self, num_paths_to_plot=50):
        """
        Plot sample simulation paths.
        """
        if self.price_paths is None:
            raise ValueError("You must run simulate_price_paths() first.")

        plt.figure(figsize=(12, 6))
        for i in range(min(num_paths_to_plot, self.N)):
            plt.plot(self.price_paths[:, i], alpha=0.5)

        plt.axhline(self.K, color='black', linestyle='--', label='Strike Price')
        plt.title(f'{num_paths_to_plot} Simulated Price Paths')
        plt.xlabel('Days')
        plt.ylabel('Simulated Price')
        plt.legend()
        plt.grid(True)
        return plt
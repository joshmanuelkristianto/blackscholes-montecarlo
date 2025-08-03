"""
This script tests the 
"""
from models.Black_Scholes_Merton_Model import BlackScholesMertonModel
from models import get_data

# Fetching the prices from yahoo finance
data = get_data.get_historical_data('AAPL')
print(get_data.get_columns(data))
print(get_data.get_last_price(data, 'Close'))
get_data.plot_data(data, 'AAPL', 'Close')

# Black-Scholes model testing
BSM = BlackScholesMertonModel(100, 100, 365, 0.1, 0.2)
print(BSM.calculate_option_price('Call Option'))
print(BSM.calculate_option_price('Put Option'))

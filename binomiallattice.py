import pyop3

import numpy as np
import pyop3
import matplotlib.pyplot as plt
import scipy
from scipy import stats

# Example 1: European Options on Non-Dividend Paying Stock
print('1. European Options on Non-Dividend Paying Stock')
print()
#1.a. Calculation based on number of years to expiration
print('1a. Calculation based on number of years to expiration')
print()
# Initialize the binomial tree object
strike = 100
asset_1 = pyop3.binomial_tree(strike, 0.05, 1.0000, sigma = 0.2)

# View underlying asset information
asset_1.underlying_asset_summary()

# Generate lattice of the underlying asset prices
asset_1_tree = asset_1.underlying_asset_tree()
print(asset_1_tree)

# Visualize the tree in a graphic for better illustration
#uncomment below to see graph
pyop3.tree_planter.show_tree(asset_1_tree, "Binomial Tree Price Development of Underlying Asset")

my_american_option = pyop3.american_option(asset_1, strike)

# To calculate call value, we need to first run the .call() method of the option object
my_american_option.call()
print("call value")
print(my_american_option.call_value)

# Calculate call and put option values using fast method
#print(my_american_option.fast_put_call())

#visualize the call tree
#pyop3.tree_planter.show_tree(my_european_option.call_option, \
                             #"Binomial Tree Price Development of Call Option", \
                             #node_color = '#B9FEA1')

# 1.b. Calculation given spot date and expiry date
print('1b. Calculation given spot date and expiry date')
print()
# Initialize the binomial tree object
# can add current date(spot_date), and expiry date (in DD/MM/YYYY format)
asset_2 = pyop3.binomial_tree(300, 0.08, '23/12/2022', sigma = 0.3, spot_date = '01/09/2022')
asset_2.underlying_asset_summary()

# Generate lattice of the underlying asset prices
asset_2_tree = asset_2.underlying_asset_tree()
# Visualize the tree in a graphic for better illustration
#pyop3.tree_planter.show_tree(asset_2_tree, "Binomial Tree Price Development of Underlying Asset")

# Now initialize the European option
asset_2_options = pyop3.european_option(asset_2, 300)
asset_2_options.put()

print(asset_2_options.put_value)
#pyop3.tree_planter.show_tree(asset_2_options.put_option, \
                             #"Binomial Tree Price Development of Put Option", \
                             #node_color = '#420D09')

# The pyop3.binomial_tree object does take into account dividend payments.
# Amongst the optional keyword arguments, one can define known dollar dividend div or dividend yield div_yield.
# For convenience's sake, user can also define, for known dollar dividend,
# either which step does the dividend occurs ex_div_step or the ex-dividend date ex_div_date.

#Example 2: European Option on Dividend Paying Stocks
print('2. European Option on Dividend Paying Stocks')
print()

#2.a. Ex-Dividend coinciding expiry date
print('2a. Ex-Dividend coinciding expiry date')
print()
div_asset_1 = pyop3.binomial_tree(300, 0.08, 0.3333, sigma = 0.30, div = 30, ex_div_step = 4)
div_asset_1.underlying_asset_summary()
#pyop3.tree_planter.show_tree(div_asset_1.underlying_asset_tree(), \
                             #"Binomial Tree Price Development of a Dividend-paying Stock at N = 4")

div_asset_1_options = pyop3.european_option(div_asset_1, 300)
div_asset_1_options.call()
#pyop3.tree_planter.show_tree(div_asset_1_options.call_option, \
                             #"Binomial Tree Price Development of Call Option on Dividend-paying Stock at N = 4")

#print('Call option of asset 1 without dividend: ${:.2f}'.format(my_european_option.call_value))
print('Call option of asset 1 with dividend: ${:.2f}'.format(div_asset_1_options.call_value))

#2.b. Ex-dividend date before expiration date
print('2b. Ex-dividend date before expiration date')
print()
# known dollar dividends may have ex-dividend date before the expiration of the option contracts.
# not a perfect approximation
div_asset_1_b = pyop3.binomial_tree(100, 0.05, 1, N = 8, sigma = 0.20, div = 30, ex_div_step = 4)
unit_asset = pyop3.binomial_tree(1, 0.08, 0.3333, N = 8, sigma = 0.30)
div_asset_1_b.underlying_asset_summary()
#pyop3.tree_planter.show_tree(div_asset_1_b.underlying_asset_tree(), \
                             #"Binomial Tree Price Development of a Dividend-paying Stock at N = 4")
#pyop3.tree_planter.show_tree(div_asset_1_b.__dividend_tree__(), \
                             #"Binomial Tree Price Development of the dividend paid at N = 4")

# Practically, when evaluating options with dividends, we are often provided with the explicit dates.
# When provided with ex_div_date, user needs to define parameter freq_by of the pyop3.binomial_tree object as 'days' instead of N
my_european_option2 = pyop3.european_option(div_asset_1_b, 100)
# To calculate call value, we need to first run the .call() method of the option object
my_european_option2.call()
print("call value")
print(my_european_option2.call_value)

# Calculate call and put option values using fast method
print(my_european_option2.fast_put_call())

#2.c. Ex-dividend date before expiration date, given spot date, expiration date, ex-div date
print('Ex-dividend date before expiration date, given spot date, expiration date, ex-div date')
print()
div_asset_2 = pyop3.binomial_tree(300, 0.08, '23/12/2022', N = None, sigma = 0.3, spot_date = '01/09/2022', \
                                  div = 30, ex_div_date = '01/12/2022', freq_by = 'days')
div_asset_2.underlying_asset_summary()
div_asset_2_options = pyop3.european_option(div_asset_2, 300)
div_asset_2_options.put()
print('Put option of asset 2 without dividend: ${:.2f}'.format(asset_2_options.put_value))
print('Put option of asset 2 with dividend: ${:.2f}'.format(div_asset_2_options.put_value))


#Calibrating European Options, to find volatility
# Some market date - dated 1/12/2020

SPX_index = 3662.45 # spot price
zero_rate = 0.114128/100 # for simplicity, we used the 13-day zero rate instead of applying the interpolation
spot_date = '01/12/2020'
T = '18/12/2020'

SPX_put = 0.75
SPX_put_strike = 2675 # we chose an OTM put

SPX_call = 52.65
SPX_call_strike = 3665 # we chose an OTM call

calibrated_put_tree = pyop3.calibrate_european(SPX_put, SPX_index, SPX_put_strike, zero_rate, T, \
                                               spot_date = spot_date, N = 100, call = False, tree_type = "RB", \
                                               calibrate_range = (1.0001,1.05))

calibrated_put_tree.underlying_asset_summary()
print("Optimal u: ",calibrated_put_tree.u)

calibrated_call_tree = pyop3.calibrate_european(SPX_call, SPX_index, SPX_call_strike, zero_rate, T, \
                                                spot_date = spot_date, tree_type = "RB", N = 100, \
                                                calibrate_range = [1.0001, 1.05])
calibrated_call_tree.underlying_asset_summary()
print("Optimal u: ",calibrated_call_tree.u)


# DeAmericanization,
#deAmericanization is the process of converting available American option data into pseudo-European option prices for further calibration.
# DeAmericanization offers advantage of simpler and fast calibration of American options and is a market standard.

# Some market date - dated 1/12/2020

SPY_index = 366.02 # spot price
zero_rate = 0.114128/100 # for simplicity, we used the 13-day zero rate instead of applying the interpolation
spot_date = '01/12/2020'
T = '18/12/2020'
div  = 1.58
ex_div_date = '18/12/2020'

SPY_put = 0.115
SPY_put_strike = 280 # we chose an OTM put

SPY_call = 3.445
SPY_call_strike = 370 # we chose an OTM call

pyop3.deamericanization(SPY_call, SPY_index, SPY_call_strike, zero_rate, T, \
                        spot_date = spot_date, N = 100, freq_by = "N", \
                        calibrate_range = (1.0001, 1.05), tree_type = "RB", \
                        div = div, ex_div_date = ex_div_date)
pyop3.deamericanization(SPY_put, SPY_index, SPY_put_strike, zero_rate, T, \
                        spot_date = spot_date, N = 100, freq_by = "N", call = False, \
                        calibrate_range = (1.0001, 1.05), tree_type = "RB", \
                        div = div, ex_div_date = ex_div_date)


#If needed
# Create function to price vanilla options analytically using Black-Scholes Model
def black_scholes_option_pricer(S, K, r, T, sigma, call = True):
    '''
    Function calculates option prices based on Black-scholes model.
    Three modes available: Vanilla (default), Cash-or-Nothing (CON), Asset-or-Nothing (AON)

    Inputs:
    S: underlying asset price at t; can be an array of prices
    K: strike price; can be an array of prices
    r: interest rate, annualized
    T: time to expiration (also the T-t in our equations), in number of years
    sigma: implied volatility of the option
    call: default True. True if pricing call options; otherwise False

    Outputs:
    Option Prices.
    '''

    d2 = (np.log(S/K) +(r - 0.5*np.square(sigma))*(T))/(sigma*np.sqrt(T))
    d1 = d2 + sigma*np.sqrt(T)

    d2 = d2 if call == True else -d2
    d1 = d1 if call == True else -d1

    option_values = S*scipy.stats.norm.cdf(d1) - K*np.exp(-r*T)*scipy.stats.norm.cdf(d2)
    option_values = option_values if call == True else -option_values

    return option_values
import pyop3

# Initialize the binomial tree object
asset_1 = pyop3.binomial_tree(300, 0.08, 0.3333, sigma = 0.3)

# View underlying asset information
asset_1.underlying_asset_summary()

# Generate lattice of the underlying asset prices
asset_1_tree = asset_1.underlying_asset_tree()
#print(asset_1_tree)

# Visualize the tree in a graphic for better illustration
#uncomment below to see graph
#pyop3.tree_planter.show_tree(asset_1_tree, "Binomial Tree Price Development of Underlying Asset")

strike = 300
my_european_option = pyop3.european_option(asset_1, strike)

# To calculate call value, we need to first run the .call() method of the option object
my_european_option.call()
print("call value")
print(my_european_option.call_value)

# Calculate call and put option values using fast method
#print(my_european_option.fast_put_call())

#visualize the call tree
#pyop3.tree_planter.show_tree(my_european_option.call_option, \
                             #"Binomial Tree Price Development of Call Option", \
                             #node_color = '#B9FEA1')

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

div_asset_1 = pyop3.binomial_tree(300, 0.08, 0.3333, sigma = 0.30, div = 30, ex_div_step = 4)
div_asset_1.underlying_asset_summary()
#pyop3.tree_planter.show_tree(div_asset_1.underlying_asset_tree(), \
                             #"Binomial Tree Price Development of a Dividend-paying Stock at N = 4")

div_asset_1_options = pyop3.european_option(div_asset_1, 300)
div_asset_1_options.call()
#pyop3.tree_planter.show_tree(div_asset_1_options.call_option, \
                             #"Binomial Tree Price Development of Call Option on Dividend-paying Stock at N = 4")

print('Call option of asset 1 without dividend: ${:.2f}'.format(my_european_option.call_value))
print('Call option of asset 1 with dividend: ${:.2f}'.format(div_asset_1_options.call_value))

# known dollar dividends may have ex-dividend date before the expiration of the option contracts.
# not a perfect approximation
div_asset_1_b = pyop3.binomial_tree(300, 0.08, 0.3333, N = 8, sigma = 0.30, div = 30, ex_div_step = 4)
unit_asset = pyop3.binomial_tree(1, 0.08, 0.3333, N = 8, sigma = 0.30)
div_asset_1_b.underlying_asset_summary()
pyop3.tree_planter.show_tree(div_asset_1_b.underlying_asset_tree(), \
                             "Binomial Tree Price Development of a Dividend-paying Stock at N = 4")
pyop3.tree_planter.show_tree(div_asset_1_b.__dividend_tree__(), \
                             "Binomial Tree Price Development of the dividend paid at N = 4")

# Practically, when evaluating options with dividends, we are often provided with the explicit dates.
# When provided with ex_div_date, user needs to define parameter freq_by of the pyop3.binomial_tree object as 'days' instead of N

div_asset_2 = pyop3.binomial_tree(300, 0.08, '23/12/2022', N = None, sigma = 0.3, spot_date = '01/09/2022', \
                                  div = 30, ex_div_date = '01/12/2022', freq_by = 'days')
div_asset_2.underlying_asset_summary()
div_asset_2_options = pyop3.european_option(div_asset_2, 300)
div_asset_2_options.put()
print('Put option of asset 2 without dividend: ${:.2f}'.format(asset_2_options.put_value))
print('Put option of asset 2 with dividend: ${:.2f}'.format(div_asset_2_options.put_value))
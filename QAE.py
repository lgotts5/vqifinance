from ftplib import print_line

import matplotlib.pyplot as plt

import numpy as np

from qiskit import QuantumCircuit
from qiskit_algorithms import IterativeAmplitudeEstimation, EstimationProblem
from qiskit.circuit.library import LinearAmplitudeFunction
#from qiskit_aer.primitives import Sampler
from qiskit_finance.circuit.library import LogNormalDistribution

# number of qubits to represent the uncertainty
num_uncertainty_qubits = 3

# parameters for considered random distribution
S = 280.4559  # initial spot price
vol = 0.2598  # volatility of 40%
r = 0.2223  # annual interest rate of 4%
T = 1.0000  # 40 days to maturity

# resulting parameters for log-normal distribution
mu = (r - 0.5 * vol**2) * T + np.log(S)
sigma = vol * np.sqrt(T)
mean = np.exp(mu + sigma**2 / 2)
variance = (np.exp(sigma**2) - 1) * np.exp(2 * mu + sigma**2)
stddev = np.sqrt(variance)

# lowest and highest value considered for the spot price; in between, an equidistant discretization is considered.
low = np.maximum(0, mean - 3 * stddev)
high = mean + 3 * stddev

# construct A operator for QAE for the payoff function by
# composing the uncertainty model and the objective
uncertainty_model = LogNormalDistribution(
    num_uncertainty_qubits, mu=mu, sigma=sigma**2, bounds=(low, high)
)

# plot probability distribution
#x = uncertainty_model.values
#y = uncertainty_model.probabilities
#plt.bar(x, y, width=0.2)
#plt.xticks(x, size=15, rotation=90)
#plt.yticks(size=15)
#plt.grid()
#plt.xlabel("Spot Price at Maturity $S_T$ (\$)", size=15)
#plt.ylabel("Probability ($\%$)", size=15)
#plt.show()

# set the strike price (should be within the low and the high value of the uncertainty)
strike_price = S

# set the approximation scaling for the payoff function
c_approx = 0.25

# setup piecewise linear objective fcuntion
breakpoints = sorted([low, strike_price])
slopes = [0, 1]
offsets = [0, 0]
f_min = 0
f_max = high - strike_price
european_call_objective = LinearAmplitudeFunction(
    num_uncertainty_qubits,
    slopes,
    offsets,
    domain=(low, high),
    image=(f_min, f_max),
    breakpoints=breakpoints,
    rescaling_factor=c_approx,
)

# construct A operator for QAE for the payoff function by
# composing the uncertainty model and the objective
num_qubits = european_call_objective.num_qubits
european_call = QuantumCircuit(num_qubits)
european_call.append(uncertainty_model, range(num_uncertainty_qubits))
european_call.append(european_call_objective, range(num_qubits))
print_line("The first plot shows the log-normal probability distribution of the simulated future"
           "asset price discretized into 8 possible values. Each bar represents that the asset ends at that price at maturity."
           "This is the first thing encoded in the circuit.")
print_line("The second plot shows the payoff of a European call option, which is zero below the strike price and increases linearly"
           "above the strike")

# draw the circuit
print("This circuit loads the possible future prices and applies the call-option "
      "payoff so the quantum algorithm can estimate the option’s value.")
print(european_call.draw(output='text'))

# plot exact payoff function (evaluated on the grid of the uncertainty model)
x = uncertainty_model.values
y = np.maximum(0, x - strike_price)
#plt.plot(x, y, "ro-")
#plt.grid()
#plt.title("Payoff Function", size=15)
#plt.xlabel("Spot Price", size=15)
#plt.ylabel("Payoff", size=15)
#plt.xticks(x, size=15, rotation=90)
#plt.yticks(size=15)
#plt.show()

# ---- Better side-by-side plots ----
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Distribution plot
axes[0].bar(uncertainty_model.values, uncertainty_model.probabilities, width=0.2)
axes[0].set_title("Distribution of $S_T$")
axes[0].set_xlabel("Spot Price")
axes[0].set_ylabel("Probability")
axes[0].grid(True)

# Payoff plot
payoff = np.maximum(0, uncertainty_model.values - strike_price)
axes[1].plot(uncertainty_model.values, payoff, "ro-")
axes[1].set_title("Payoff: max($S_T$ - K, 0)")
axes[1].set_xlabel("Spot Price")
axes[1].set_ylabel("Payoff")
axes[1].grid(True)

plt.tight_layout()
plt.show()

# evaluate exact expected value (normalized to the [0, 1] interval)
exact_value = np.dot(uncertainty_model.probabilities, y)
exact_delta = sum(uncertainty_model.probabilities[x >= strike_price])
print("exact expected payoff.:\t%.4f" % exact_value)
print("exact delta value:   \t%.4f" % exact_delta)


# set target precision and confidence level
epsilon = 0.01
alpha = 0.05

problem = EstimationProblem(
    state_preparation=european_call,
    objective_qubits=[3],
    post_processing=european_call_objective.post_processing,
)
# construct amplitude estimation
ae = IterativeAmplitudeEstimation(
    epsilon_target=epsilon,
    alpha=alpha
)

result = ae.estimate(problem) #!!!!
discount_factor = np.exp(-r * T)
# --- Quantum Amplitude Estimation Results ---
estimated_payoff = result.estimation_processed
estimated_price  = discount_factor * estimated_payoff
estimated_delta = estimated_payoff / (f_max * c_approx)   # optional scaling depending on encoding


conf_int_payoff  = np.array(result.confidence_interval_processed)
conf_int_price   = discount_factor * conf_int_payoff

print("\n=== Quantum Amplitude Estimation (QAE) ===")
print(f"Estimated Payoff:     {estimated_payoff: .6f}")
print(f"Estimated Price:      {estimated_price: .6f}")
print(f"Estimated Delta:        {estimated_delta: .6f}")
print(f"Confidence Interval:  [{conf_int_price[0]: .6f}, {conf_int_price[1]: .6f}]")






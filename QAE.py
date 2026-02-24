from ftplib import print_line

import matplotlib.pyplot as plt

import numpy as np

from qiskit import QuantumCircuit
from qiskit_algorithms import IterativeAmplitudeEstimation, EstimationProblem

#from qiskit_aer.primitives import Sampler
from math import comb
from qiskit.circuit.library import StatePreparation


# This program estimates the value of a European call option using a quantum approach.
# First, it models AAPL’s possible future prices using a log-normal distribution,
# based on the stock’s current price, volatility, and expected yearly return.
# This distribution is then encoded into a quantum circuit, where each quantum state
# represents one possible future price and its probability. A second quantum circuit
# applies the payoff function so the amplitude of certain states corresponds to how valuable
# the option is in each scenario. Quantum Amplitude Estimation (QAE) is then used to measure
# these amplitudes and produce an estimate of the option’s expected payoff and price—offering
# a theoretical quadratic speedup compared to classical Monte Carlo simulations.
# The program also plots the distribution and payoff so the user can visualize how the option
# value arises from the modeled stock movements.

# number of qubits to represent the uncertainty
num_uncertainty_qubits = 6

# parameters for considered random distribution, These are what change the option
S = 100  #
vol = 0.2  #
r = 0.05  #
T = 1.0000  #


# ---------------- BINOMIAL (CRR) UNCERTAINTY MODEL ----------------
# Choose number of binomial steps N to match the number of basis states.
# With n uncertainty qubits, you have 2^n states. We'll use N = 2^n - 1 steps so j=0..N fits exactly.
N = (2 ** num_uncertainty_qubits) - 1
dt = T / N

u = np.exp(vol * np.sqrt(dt))
d = 1.0 / u

q = (np.exp(r * dt) - d) / (u - d)  # risk-neutral up probability
if not (0.0 <= q <= 1.0):
    raise ValueError(f"Risk-neutral prob q out of [0,1]: q={q}. Check inputs.")

# j = number of up moves at maturity (0..N)
j = np.arange(N + 1)

# Terminal prices for each j
S_T = S * (u ** j) * (d ** (N - j))

# Binomial probabilities for each j
probs = np.array([comb(N, int(k)) * (q ** k) * ((1 - q) ** (N - k)) for k in j], dtype=float)
probs /= probs.sum()

# Create state |psi> = sum_j sqrt(p_j) |j>
amps = np.sqrt(probs).astype(float)
uncertainty_model = StatePreparation(amps, normalize=True)
uncertainty_model.label = "BinomialDist"

# In this approach, the uncertainty register represents j in [0, N]
domain_low = 0.0
domain_high = float(N)
# -----------------------------------------------------------------
# construct A operator for QAE for the payoff function by
# composing the uncertainty model and the objective
# Creates circuit that encodes all future payoffs at once, essentially "random number generator"


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
strike_price = S
payoff_vals = np.maximum(0.0, S_T - strike_price)

# ---------------- DIRECT AMPLITUDE ENCODING (NO LinearAmplitudeFunction) ----------------
# We want QAE to estimate:
#   a = E[ payoff / max_payoff ]  where payoff = max(S_T - K, 0)
# Then: expected_payoff = a * max_payoff, price = exp(-rT) * expected_payoff

max_payoff = float(payoff_vals.max())
if max_payoff <= 0:
    raise ValueError("max_payoff is 0; option is never ITM on this grid.")

ratios = payoff_vals / max_payoff  # in [0,1]

# Build amplitude vector for |j>|b>, where b is payoff indicator qubit.
# For each j:
#   amp(|j,0>) = sqrt(p_j * (1 - ratio_j))
#   amp(|j,1>) = sqrt(p_j * ratio_j)
combined = []
for pj, fj in zip(probs, ratios):
    combined.append(np.sqrt(pj * (1.0 - float(fj))))  # |j,0>
    combined.append(np.sqrt(pj * float(fj)))          # |j,1>

combined = np.array(combined, dtype=float)
combined /= np.linalg.norm(combined)

# Total qubits = uncertainty qubits + 1 payoff qubit
num_qubits = num_uncertainty_qubits + 1

# State preparation for full system
european_call = QuantumCircuit(num_qubits)
european_call.append(StatePreparation(combined, normalize=True), range(num_qubits))

# Objective qubit is the LAST qubit (the payoff indicator)
objective_qubit = 0
# -------------------------------------------------------------------
print("The first plot shows the log-normal probability distribution of the simulated future"
           "asset price discretized into 8 possible values. Each bar represents that the asset ends at that price at maturity."
           "This is the first thing encoded in the circuit.\n")
print("The second plot shows the payoff of a European call option, which is zero below the strike price and increases linearly"
           "above the strike")

# draw the circuit
print("This circuit loads the possible future prices and applies the call-option "
      "payoff so the quantum algorithm can estimate the option’s value.")
print(european_call.draw(output='text'))

# ---------------- BINOMIAL PLOTS ----------------

# x-axis = terminal stock prices
values = S_T
probabilities = probs
payoff = payoff_vals

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Distribution plot
axes[0].bar(values, probabilities)
axes[0].set_title("Binomial Distribution of $S_T$")
axes[0].set_xlabel("Terminal Spot Price")
axes[0].set_ylabel("Probability")
axes[0].grid(True)

# Payoff plot
axes[1].plot(values, payoff, "ro-")
axes[1].set_title("Payoff: max($S_T$ - K, 0)")
axes[1].set_xlabel("Terminal Spot Price")
axes[1].set_ylabel("Payoff")
axes[1].grid(True)

plt.tight_layout()
plt.show()

# evaluate exact expected value (normalized to the [0, 1] interval)
exact_value = float(np.dot(probs, payoff_vals))  # expected payoff at maturity
exact_price = float(np.exp(-r * T) * exact_value)

print("Exact binomial expected payoff:\t%.6f" % exact_value)
print("Exact binomial price:\t\t%.6f" % exact_price)


# set target precision and confidence level
epsilon = 0.01
alpha = 0.05
problem = EstimationProblem(
    state_preparation=european_call,
    objective_qubits=[objective_qubit],
)
# construct amplitude estimation
ae = IterativeAmplitudeEstimation(
    epsilon_target=epsilon,
    alpha=alpha
)

result = ae.estimate(problem) #!!!!
discount_factor = np.exp(-r * T)
# --- Quantum Amplitude Estimation Results ---
result = ae.estimate(problem)

discount_factor = np.exp(-r * T)

# Raw amplitude estimate a in [0,1]
a_hat = float(result.estimation)
ci = np.array(result.confidence_interval, dtype=float)

estimated_payoff = a_hat * max_payoff
estimated_price = discount_factor * estimated_payoff

conf_int_payoff = ci * max_payoff
conf_int_price = discount_factor * conf_int_payoff

print("\n=== Quantum Amplitude Estimation (QAE) ===")
print(f"Raw amplitude a:      {a_hat: .6f}")
print(f"Estimated Payoff:     {estimated_payoff: .6f}")
print(f"Estimated Price:      {estimated_price: .6f}")
print(f"Confidence Interval:  [{conf_int_price[0]: .6f}, {conf_int_price[1]: .6f}]")








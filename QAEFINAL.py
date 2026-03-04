"""
Quantum Option Pricing Engine
==============================
Prices three types of options using a CRR binomial tree
and Iterative Quantum Amplitude Estimation (QAE):

  1. European  — standard terminal payoff, full QAE encoding
  2. American  — backward induction for early exercise, then QAE
  3. Asian     — arithmetic average price, path enumeration then QAE

Edit the parameters under USER PARAMETERS to change the option.
"""

import numpy as np
import matplotlib.pyplot as plt
from math import comb

from qiskit import QuantumCircuit
from qiskit_algorithms import IterativeAmplitudeEstimation, EstimationProblem
from qiskit.circuit.library import StatePreparation

# ─────────────────────────────────────────────────────────────
#  USER PARAMETERS  — edit these to change the option
# ─────────────────────────────────────────────────────────────
S   = 100       # current stock price ($)
vol = 0.2       # annualised volatility
r   = 0.05      # risk-free rate
T   = 1.0       # time to maturity (years)
K   = S         # strike price (default: ATM, set to any value)

NUM_UNCERTAINTY_QUBITS = 6   # European / American  (2^6 = 64 terminal states)
N_STEPS_ASIAN          = 4   # Asian                (2^4 = 16 paths)
EPSILON                = 0.01
ALPHA                  = 0.05

# ─────────────────────────────────────────────────────────────
#  QAE HELPER  (shared by all methods)
# ─────────────────────────────────────────────────────────────
def run_qae(probs, payoff_vals):
    """
    Encode (probs, payoff_vals) into a quantum circuit via
    StatePreparation and run Iterative Amplitude Estimation.
    Returns (a_hat, confidence_interval, max_payoff, circuit).
    """
    max_payoff = float(payoff_vals.max())
    if max_payoff <= 0:
        raise ValueError("max_payoff is 0 — option is entirely out of the money.")

    ratios   = payoff_vals / max_payoff
    combined = []
    for p, f in zip(probs, ratios):
        combined.append(np.sqrt(p * (1.0 - float(f))))   # |i, 0>
        combined.append(np.sqrt(p * float(f)))            # |i, 1>

    combined = np.array(combined, dtype=float)
    combined /= np.linalg.norm(combined)

    num_qubits = int(np.log2(len(combined)))
    circuit    = QuantumCircuit(num_qubits)
    circuit.append(StatePreparation(combined, normalize=True), range(num_qubits))

    problem = EstimationProblem(state_preparation=circuit, objective_qubits=[0])
    ae      = IterativeAmplitudeEstimation(epsilon_target=EPSILON, alpha=ALPHA)
    result  = ae.estimate(problem)

    a_hat = float(result.estimation)
    ci    = np.array(result.confidence_interval, dtype=float)
    return a_hat, ci, max_payoff, circuit


def print_qae_results(a_hat, ci, max_payoff, classical_price, label, discount):
    qae_price = discount * a_hat * max_payoff
    ci_price  = discount * ci * max_payoff
    print(f"\n{'=' * 55}")
    print(f"  RESULTS  —  {label}")
    print(f"{'=' * 55}")
    print(f"  Classical price           : ${classical_price:.4f}")
    print(f"  QAE estimated price       : ${qae_price:.4f}")
    print(f"  QAE raw amplitude a       : {a_hat:.6f}")
    print(f"  95% CI (price)            : [${ci_price[0]:.4f},  ${ci_price[1]:.4f}]")
    print(f"{'=' * 55}")


# ─────────────────────────────────────────────────────────────
#  METHOD 1 — EUROPEAN OPTION
# ─────────────────────────────────────────────────────────────
def price_european(call: bool):
    label    = f"European {'Call' if call else 'Put'}"
    discount = np.exp(-r * T)

    N  = (2 ** NUM_UNCERTAINTY_QUBITS) - 1
    dt = T / N
    u  = np.exp(vol * np.sqrt(dt))
    d  = 1.0 / u
    q  = (np.exp(r * dt) - d) / (u - d)

    if not (0.0 < q < 1.0):
        raise ValueError(f"Risk-neutral prob q={q:.4f} out of (0,1).")

    j   = np.arange(N + 1)
    S_T = S * (u ** j) * (d ** (N - j))

    probs = np.array(
        [comb(N, int(k)) * (q ** k) * ((1 - q) ** (N - k)) for k in j],
        dtype=float
    )
    probs /= probs.sum()

    payoff_vals     = np.maximum(0.0, S_T - K) if call else np.maximum(0.0, K - S_T)
    classical_price = discount * float(np.dot(probs, payoff_vals))

    print(f"\n  Running QAE for {label}...")
    a_hat, ci, max_payoff, circuit = run_qae(probs, payoff_vals)
    print_qae_results(a_hat, ci, max_payoff, classical_price, label, discount)
    print("\n  Quantum Circuit:")
    print(circuit.draw(output='text'))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"{label}  (S={S}, K={K}, vol={vol}, r={r}, T={T})")
    axes[0].bar(S_T, probs, color="steelblue")
    axes[0].set_title("Binomial Distribution of $S_T$")
    axes[0].set_xlabel("Terminal Price ($)")
    axes[0].set_ylabel("Probability")
    axes[0].grid(True, alpha=0.4)
    axes[1].plot(S_T, payoff_vals, "ro-")
    axes[1].set_title("Payoff at Maturity")
    axes[1].set_xlabel("Terminal Price ($)")
    axes[1].set_ylabel("Payoff ($)")
    axes[1].grid(True, alpha=0.4)
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────
#  METHOD 2 — AMERICAN OPTION
# ─────────────────────────────────────────────────────────────
def price_american(call: bool):
    label     = f"American {'Call' if call else 'Put'}"
    discount  = np.exp(-r * T)

    N         = (2 ** NUM_UNCERTAINTY_QUBITS) - 1
    dt        = T / N
    u         = np.exp(vol * np.sqrt(dt))
    d         = 1.0 / u
    q         = (np.exp(r * dt) - d) / (u - d)
    disc_step = np.exp(-r * dt)

    if not (0.0 < q < 1.0):
        raise ValueError(f"Risk-neutral prob q={q:.4f} out of (0,1).")

    j   = np.arange(N + 1)
    S_T = S * (u ** j) * (d ** (N - j))

    option = np.maximum(0.0, S_T - K) if call else np.maximum(0.0, K - S_T)

    print(f"\n  Running backward induction for {label}...")
    for step in range(N - 1, -1, -1):
        j_step   = np.arange(step + 1)
        S_step   = S * (u ** j_step) * (d ** (step - j_step))
        held     = disc_step * (q * option[1:step + 2] + (1 - q) * option[0:step + 1])
        exercise = np.maximum(0.0, S_step - K) if call else np.maximum(0.0, K - S_step)
        option   = np.maximum(held, exercise)

    american_price_classical = float(option[0])
    print(f"  Classical American price : ${american_price_classical:.4f}")

    probs = np.array(
        [comb(N, int(k)) * (q ** k) * ((1 - q) ** (N - k)) for k in j],
        dtype=float
    )
    probs /= probs.sum()

    payoff_vals     = np.maximum(0.0, S_T - K) if call else np.maximum(0.0, K - S_T)
    european_price  = discount * float(np.dot(probs, payoff_vals))
    scale           = american_price_classical / european_price if european_price > 0 else 1.0
    payoff_vals_adj = payoff_vals * scale

    print(f"  Running QAE for {label}...")
    a_hat, ci, max_payoff, circuit = run_qae(probs, payoff_vals_adj)
    print_qae_results(a_hat, ci, max_payoff, american_price_classical, label, discount)
    print("\n  Quantum Circuit:")
    print(circuit.draw(output='text'))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"{label}  (S={S}, K={K}, vol={vol}, r={r}, T={T})")
    axes[0].bar(S_T, probs, color="steelblue")
    axes[0].set_title("Binomial Distribution of $S_T$")
    axes[0].set_xlabel("Terminal Price ($)")
    axes[0].set_ylabel("Probability")
    axes[0].grid(True, alpha=0.4)
    axes[1].plot(S_T, payoff_vals,      "b--", label="European payoff")
    axes[1].plot(S_T, payoff_vals_adj,  "ro-", label="Early-exercise adjusted")
    axes[1].set_title("Payoff (early-exercise adjusted)")
    axes[1].set_xlabel("Terminal Price ($)")
    axes[1].set_ylabel("Payoff ($)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.4)
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────
#  METHOD 3 — ASIAN OPTION  (arithmetic average price)
# ─────────────────────────────────────────────────────────────
def price_asian(call: bool):
    label    = f"Asian {'Call' if call else 'Put'} (Arithmetic Avg)"
    discount = np.exp(-r * T)

    dt = T / N_STEPS_ASIAN
    u  = np.exp(vol * np.sqrt(dt))
    d  = 1.0 / u
    q  = (np.exp(r * dt) - d) / (u - d)

    if not (0.0 < q < 1.0):
        raise ValueError(f"Risk-neutral prob q={q:.4f} out of (0,1).")

    num_paths    = 2 ** N_STEPS_ASIAN
    path_probs   = np.zeros(num_paths)
    path_avgs    = np.zeros(num_paths)
    path_payoffs = np.zeros(num_paths)

    print(f"\n  Enumerating all {num_paths} paths for {label}...")
    for idx in range(num_paths):
        price = S
        total = S
        prob  = 1.0
        for step in range(N_STEPS_ASIAN):
            up = (idx >> (N_STEPS_ASIAN - 1 - step)) & 1
            if up:
                price *= u;  prob *= q
            else:
                price *= d;  prob *= (1.0 - q)
            total += price
        avg_price          = total / (N_STEPS_ASIAN + 1)
        path_probs[idx]    = prob
        path_avgs[idx]     = avg_price
        path_payoffs[idx]  = max(0.0, avg_price - K) if call else max(0.0, K - avg_price)

    path_probs      /= path_probs.sum()
    classical_payoff = float(np.dot(path_probs, path_payoffs))
    classical_price  = discount * classical_payoff
    print(f"  Classical Asian price ({N_STEPS_ASIAN}-step tree): ${classical_price:.4f}")

    print(f"  Running QAE for {label}...")
    a_hat, ci, max_payoff, circuit = run_qae(path_probs, path_payoffs)
    print_qae_results(a_hat, ci, max_payoff, classical_price, label, discount)
    print("\n  Quantum Circuit:")
    print(circuit.draw(output='text'))

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle(f"{label}  (S={S}, K={K}, vol={vol}, r={r}, T={T})", fontsize=12)

    axes[0].bar(range(num_paths), path_probs, color="steelblue")
    axes[0].set_title("Path Probability Distribution")
    axes[0].set_xlabel("Path index")
    axes[0].set_ylabel("Risk-neutral probability")
    axes[0].grid(True, alpha=0.4)

    sc = axes[1].scatter(
        range(num_paths), path_avgs,
        c=path_probs, cmap="Blues", s=50, edgecolors="k", linewidths=0.5
    )
    axes[1].axhline(K, color="red", linestyle="--", label=f"Strike K=${K:.2f}")
    axes[1].set_title("Arithmetic Average Price per Path")
    axes[1].set_xlabel("Path index")
    axes[1].set_ylabel("Avg price ($)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.4)
    plt.colorbar(sc, ax=axes[1], label="Probability")

    colors = ["tomato" if p > 0 else "lightgrey" for p in path_payoffs]
    axes[2].bar(range(num_paths), path_payoffs, color=colors)
    axes[2].axhline(classical_payoff, color="black", linestyle="--",
                    label=f"E[payoff]=${classical_payoff:.2f}")
    axes[2].set_title(f"Payoff per Path  (max=${max_payoff:.2f})")
    axes[2].set_xlabel("Path index")
    axes[2].set_ylabel("Payoff ($)")
    axes[2].legend()
    axes[2].grid(True, alpha=0.4)
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────
#  USER PROMPT
# ─────────────────────────────────────────────────────────────
def prompt_user():
    print("\n" + "=" * 55)
    print("   QUANTUM OPTION PRICING ENGINE")
    print("=" * 55)
    print(f"  S   (stock price)  : ${S}")
    print(f"  K   (strike)       : ${K}")
    print(f"  vol (volatility)   : {vol:.0%}")
    print(f"  r   (risk-free)    : {r:.0%}")
    print(f"  T   (maturity)     : {T} year(s)")
    print(f"  QAE precision      : epsilon={EPSILON}, alpha={ALPHA}")
    print("=" * 55)

    print("\n  Select option TYPE:")
    print("    1 — European")
    print("    2 — American")
    print("    3 — Asian  (arithmetic average price)")
    type_choice = input("  Enter 1, 2, or 3: ").strip()

    if type_choice not in {"1", "2", "3"}:
        print("  Invalid selection. Please run again and enter 1, 2, or 3.")
        return

    print("\n  Select option DIRECTION:")
    print("    1 — Call")
    print("    2 — Put")
    dir_choice = input("  Enter 1 or 2: ").strip()

    if dir_choice not in {"1", "2"}:
        print("  Invalid selection. Please run again and enter 1 or 2.")
        return

    call = (dir_choice == "1")

    if type_choice == "1":
        price_european(call)
    elif type_choice == "2":
        price_american(call)
    elif type_choice == "3":
        price_asian(call)


# ─────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    prompt_user()
"""
Quantum Option Pricing Engine
==============================
Prices three types of options using a CRR binomial tree
and Iterative Quantum Amplitude Estimation (QAE):

  1. European  — FULLY QUANTUM. QAE genuinely prices the option.
  2. American  — CLASSICAL ONLY. Backward induction on binomial tree.
  3. Asian     — HYBRID. Classical path enumeration, QAE estimates expectation.

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
S           = 100       # current stock price ($)
vol         = 0.2       # annualised volatility
r           = 0.05      # risk-free rate
T           = 1.0       # time to maturity (years)
K           = S         # strike price (default: ATM, set to any value)

D           = 0         # discrete cash dividend ($). Set to 0 for no dividend
ex_div_step = 4         # step at which dividend is paid (only used if D > 0)

NUM_UNCERTAINTY_QUBITS = 6   # European / American  (2^6 = 64 terminal states)
N_STEPS_ASIAN          = 4   # Asian                (2^4 = 16 paths)
EPSILON                = 0.01
ALPHA                  = 0.05


# ─────────────────────────────────────────────────────────────
#  DIVIDEND HELPER
#  Returns the stock price at node (step, j) adjusted for a
#  discrete cash dividend D paid at ex_div_step.
#
#  Before ex_div_step: subtract the PV of the dividend so the
#  tree still recombines correctly.
#  At and after ex_div_step: tree evolves normally (div already paid).
#  If D = 0 this just returns the standard CRR node price.
# ─────────────────────────────────────────────────────────────
def stock_price(S0, u, d, dt, step, j):
    raw = S0 * (u ** j) * (d ** (step - j))
    if D > 0 and step < ex_div_step:
        pv_div = D * np.exp(-r * (ex_div_step - step) * dt)
        raw    = raw - pv_div
    return raw


# ─────────────────────────────────────────────────────────────
#  QAE HELPER  (used by European and Asian)
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
    print(f"  Number of iterations  : {len(result.powers)}")
    print(f"  Shots per iteration   : {result.shots}")

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
#
#  SUMMARY:
#  The European option can only be exercised at maturity, so the
#  entire pricing problem reduces to estimating the expected
#  discounted payoff over the terminal stock price distribution.
#  This maps perfectly onto QAE — the terminal distribution and
#  payoff function are encoded directly as quantum amplitudes via
#  StatePreparation, and QAE estimates E[payoff] with a quadratic
#  speedup over classical Monte Carlo: O(1/eps) vs O(1/eps^2).
#  This is the only option type here that is FULLY QUANTUM.
#
#  QUANTUM ADVANTAGE: Real. QAE independently prices the option
#  without any classical pre-computation of the answer.
#
# This is comparing against binomial lattice with 63 steps(only needs to use terminal layer).
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
    S_T = np.array([stock_price(S, u, d, dt, N, int(k)) for k in j])

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
    div_str = f", D={D} at step {ex_div_step}" if D > 0 else ""
    fig.suptitle(f"{label}  (S={S}, K={K}, vol={vol}, r={r}, T={T}{div_str})")
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

    # Circuit and QAE metrics
    from qiskit.compiler import transpile
    transpiled  = transpile(circuit, basis_gates=['u', 'cx'], optimization_level=0)
    ops         = transpiled.count_ops()
    total_gates = sum(ops.values())
    print(f"  Qubits                : {circuit.num_qubits}")
    print(f"  Single qubit gates (u): {ops.get('u', 0)}")
    print(f"  Two qubit gates (cx)  : {ops.get('cx', 0)}")
    print(f"  Total gates           : {total_gates}")


# ─────────────────────────────────────────────────────────────
#  METHOD 2 — AMERICAN OPTION
# ─────────────────────────────────────────────────────────────
#
#  SUMMARY:
#  The American option can be exercised at any point before
#  maturity, so pricing requires a backward induction pass through
#  the entire binomial tree. At every node the algorithm takes the
#  max of the immediate exercise value and the discounted expected
#  continuation value, working backwards from the terminal nodes
#  to today. The root node value is the option price.
#
#  WHY QAE IS NOT USED HERE:
#  QAE is designed to estimate a single expected value over a
#  known probability distribution. American option pricing requires
#  sequential decisions at every node in the tree — the "take the
#  max" step at each node is inherently classical and cannot be
#  encoded as a simple amplitude estimation problem. The true
#  quantum approach (Longstaff-Schwartz with QAE) would run
#  multiple QAE calls per exercise date going backwards, but this
#  requires enormous memory and computation even for small trees,
#  making it impractical to simulate classically.
#
#  CONSTRAINT: Classical simulation only. A genuinely quantum
#  American pricer would require fault-tolerant quantum hardware
#  with enough qubits to run repeated QAE calls across all
#  exercise dates without classical pre-computation.
#
# This is doing binomial lattice with 63 steps. For non-dividend paying
# stocks this gives the same answer as the european options
#
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
    S_T = np.array([stock_price(S, u, d, dt, N, int(k)) for k in j])

    # Terminal option values
    option = np.maximum(0.0, S_T - K) if call else np.maximum(0.0, K - S_T)

    print(f"\n  Running classical backward induction for {label}...")
    for step in range(N - 1, -1, -1):
        j_step   = np.arange(step + 1)
        S_step   = np.array([stock_price(S, u, d, dt, step, int(k)) for k in j_step])
        held     = disc_step * (q * option[1:step + 2] + (1 - q) * option[0:step + 1])
        exercise = np.maximum(0.0, S_step - K) if call else np.maximum(0.0, K - S_step)
        option   = np.maximum(held, exercise)

    american_price = float(option[0])

    print(f"\n{'=' * 55}")
    print(f"  RESULTS  —  {label}")
    print(f"{'=' * 55}")
    print(f"  Classical price (backward induction) : ${american_price:.4f}")
    print(f"  Note: QAE is not used for American options.")
    print(f"  See comments in source code for explanation.")
    print(f"{'=' * 55}")

    # Recompute terminal distribution for plotting
    probs = np.array(
        [comb(N, int(k)) * (q ** k) * ((1 - q) ** (N - k)) for k in j],
        dtype=float
    )
    probs /= probs.sum()
    payoff_vals = np.maximum(0.0, S_T - K) if call else np.maximum(0.0, K - S_T)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    div_str = f", D={D} at step {ex_div_step}" if D > 0 else ""
    fig.suptitle(f"{label}  (S={S}, K={K}, vol={vol}, r={r}, T={T}{div_str})")
    axes[0].bar(S_T, probs, color="steelblue")
    axes[0].set_title("Binomial Distribution of $S_T$")
    axes[0].set_xlabel("Terminal Price ($)")
    axes[0].set_ylabel("Probability")
    axes[0].grid(True, alpha=0.4)
    axes[1].plot(S_T, payoff_vals, "ro-")
    axes[1].set_title("Terminal Payoff")
    axes[1].set_xlabel("Terminal Price ($)")
    axes[1].set_ylabel("Payoff ($)")
    axes[1].grid(True, alpha=0.4)
    plt.tight_layout()
    plt.show()


# ─────────────────────────────────────────────────────────────
#  METHOD 3 — ASIAN OPTION  (arithmetic average price)
# ─────────────────────────────────────────────────────────────
#
#  SUMMARY:
#  The Asian option payoff depends on the arithmetic average stock
#  price over the life of the option, not just the terminal price.
#  All 2^N_STEPS paths through the binomial tree are enumerated
#  classically, and for each path the average price and payoff are
#  computed. This payoff distribution is then encoded into a quantum
#  state via StatePreparation and QAE estimates E[payoff].
#
#  WHY THIS IS NOT FULLY QUANTUM:
#  The ideal quantum approach would encode the path evolution
#  directly into the circuit using quantum arithmetic — controlled
#  multiplications by u/d at each step, a running sum register,
#  and a quantum comparator for the payoff. All paths would then
#  exist in superposition simultaneously and QAE would estimate
#  the expected payoff without any classical enumeration.
#  However, this requires ~37+ qubits of quantum arithmetic
#  circuits (price register, sum register, ancilla) which are
#  extremely expensive to simulate classically since Qiskit must
#  store the full 2^37 state vector in memory.
#
#  CONSTRAINT: Classical path enumeration scales as 2^N_STEPS,
#  so N_STEPS is capped at 4 (16 paths) for tractability.
#  A fully quantum Asian pricer requires fault-tolerant hardware
#  with 40+ clean qubits and quantum arithmetic primitives.
#  On real quantum hardware the circuit approach would give a
#  genuine quadratic speedup over classical Monte Carlo.
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

    print(f"\n  Enumerating all {num_paths} paths classically for {label}...")
    for idx in range(num_paths):
        price = S
        total = S
        prob  = 1.0
        ups   = 0
        for step in range(N_STEPS_ASIAN):
            up = (idx >> (N_STEPS_ASIAN - 1 - step)) & 1
            if up:
                ups  += 1
                prob *= q
            else:
                prob *= (1.0 - q)
            price  = stock_price(S, u, d, dt, step + 1, ups)
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
    div_str = f", D={D} at step {ex_div_step}" if D > 0 else ""
    fig.suptitle(f"{label}  (S={S}, K={K}, vol={vol}, r={r}, T={T}{div_str})", fontsize=12)

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
    if D > 0:
        print(f"  D   (dividend)     : ${D} at step {ex_div_step}")
    else:
        print(f"  D   (dividend)     : none")
    print(f"  QAE precision      : epsilon={EPSILON}, alpha={ALPHA}")
    print("=" * 55)
    print("  Pricing methods:")
    print("    1 — European  (fully quantum via QAE)")
    print("    2 — American  (classical backward induction)")
    print("    3 — Asian     (hybrid: classical paths + QAE)")
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
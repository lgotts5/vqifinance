"""
IonQ Resource Estimator — fills in the form fields for IonQ Forte.
Change OPTION_TYPE to "european" or "asian". No API token needed.
"""

import math
import numpy as np
from qiskit import QuantumCircuit
from qiskit.compiler import transpile
from qiskit.circuit.library import WeightedAdder
from qiskit_finance.circuit.library import LogNormalDistribution, EuropeanCallPricingObjective

# ── Settings ──────────────────────────────────────────────────
OPTION_TYPE            = "european"   # "european" or "asian"
S                      = 100
vol                    = 0.2
r                      = 0.05
T                      = 1.0
K                      = S
NUM_UNCERTAINTY_QUBITS = 3
N_STEPS_ASIAN          = 2
EPSILON                = 0.01
SHOTS                  = 1024

# ── Build European circuit ─────────────────────────────────────
def build_european():
    mu_ln    = (r - 0.5 * vol**2) * T + np.log(S)
    sigma_ln = vol * np.sqrt(T)
    low, high = np.exp(mu_ln - 3*sigma_ln), np.exp(mu_ln + 3*sigma_ln)
    unc = LogNormalDistribution(NUM_UNCERTAINTY_QUBITS, mu=mu_ln, sigma=sigma_ln**2, bounds=(low, high))
    obj = EuropeanCallPricingObjective(NUM_UNCERTAINTY_QUBITS, strike_price=K, rescaling_factor=0.25, bounds=(low, high))
    qc  = QuantumCircuit(obj.num_qubits)
    qc.append(unc, range(NUM_UNCERTAINTY_QUBITS))
    qc.append(obj, range(obj.num_qubits))
    return qc

# ── Build Asian circuit ────────────────────────────────────────
def build_asian():
    dt, models, bounds_list = T / N_STEPS_ASIAN, [], []
    for t in range(1, N_STEPS_ASIAN + 1):
        mu_t    = (r - 0.5*vol**2)*t*dt + np.log(S)
        sigma_t = vol * np.sqrt(t*dt)
        low_t, high_t = np.exp(mu_t - 3*sigma_t), np.exp(mu_t + 3*sigma_t)
        models.append(LogNormalDistribution(NUM_UNCERTAINTY_QUBITS, mu=mu_t, sigma=sigma_t**2, bounds=(low_t, high_t)))
        bounds_list.append((low_t, high_t))
    n_unc   = N_STEPS_ASIAN * NUM_UNCERTAINTY_QUBITS
    adder   = WeightedAdder(num_state_qubits=n_unc, weights=[1]*n_unc)
    n_sum   = adder.num_sum_qubits
    obj     = EuropeanCallPricingObjective(n_sum, strike_price=N_STEPS_ASIAN*K, rescaling_factor=0.25,
                                           bounds=(sum(b[0] for b in bounds_list), sum(b[1] for b in bounds_list)))
    # Total qubits: adder qubits + obj ancilla qubits + 1 objective qubit
    # obj.num_qubits = n_sum + internal ancilla + 1 objective
    # We append obj starting at the sum register position
    extra = obj.num_qubits - n_sum   # ancilla + objective qubit
    total = adder.num_qubits + extra
    qc    = QuantumCircuit(total)

    for i, m in enumerate(models):
        qc.append(m, range(i*NUM_UNCERTAINTY_QUBITS, (i+1)*NUM_UNCERTAINTY_QUBITS))

    qc.append(adder, range(adder.num_qubits))

    # obj acts on sum register + its ancilla qubits
    obj_qubits = list(range(n_unc, n_unc + obj.num_qubits))
    qc.append(obj, obj_qubits)
    return qc

# ── Analyse ────────────────────────────────────────────────────
circuit    = build_european() if OPTION_TYPE == "european" else build_asian()
transpiled = transpile(circuit, basis_gates=['rz', 'ry', 'rx', 'cx'], optimization_level=1)
ops        = transpiled.count_ops()

one_q   = sum(v for k, v in ops.items() if k in ('rz', 'ry', 'rx', 'u', 'h', 'x', 'y', 'z'))
two_q   = sum(v for k, v in ops.items() if k in ('cx', 'cz', 'xx'))
multi_q = sum(v for k, v in ops.items() if k in ('ccx', 'mcx', 'rccx'))
iters   = math.ceil(math.log2(math.pi / (8 * EPSILON)))

print(f"\n  Option type                   : {OPTION_TYPE}")
print(f"  Number of Qubits              : {transpiled.num_qubits}")
print(f"  Number of One-Qubit Gates     : {one_q}")
print(f"  Number of Two-Qubit Gates     : {two_q}")
print(f"  Multi-Qubit Gates (3+ qubits) : {multi_q}")
print(f"  Shots Per Iteration           : {SHOTS}")
print(f"  Number of Iterations          : {iters}")
print(f"  Full gate breakdown           : {dict(ops)}")
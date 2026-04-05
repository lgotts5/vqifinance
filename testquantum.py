import os
from qiskit_ionq import IonQProvider

# Load your API key from an environment variable named MY_IONQ_API_KEY
provider = IonQProvider(token=os.getenv("IONQ_API_KEY"))

from qiskit_ionq import IonQProvider
provider = IonQProvider()

print(provider.backends())

from qiskit import QuantumCircuit

provider = IonQProvider()
simulator_backend = provider.get_backend("simulator")


# Create a basic Bell State circuit:
qc = QuantumCircuit(2, name="IonQ Qiskit guide - noisy sim example")
qc.h(0)
qc.cx(0, 1)
qc.measure_all()

# Run the circuit on IonQ's platform:
job = simulator_backend.run(qc, shots=10000)

# Print the counts
print(job.get_counts())
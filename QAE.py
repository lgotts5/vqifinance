from qiskit.primitives import BackendSamplerV2
help(BackendSamplerV2.__init__)

# Also check what else is available in primitives
import qiskit.primitives as prim
print(dir(prim))

# And check WeightedAdder signature
from qiskit.circuit.library import WeightedAdder
help(WeightedAdder.__init__)

# And check available IonQ backends
from qiskit_ionq import IonQProvider
provider = IonQProvider(token="YOUR_TOKEN")
print(provider.backends())

from qiskit_ionq import IonQProvider
provider = IonQProvider(token="YOUR_TOKEN")
for b in provider.backends():
    print(b.name)

# /AI_ML_Modules/quantum_integration_module.py

from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import Parameter

def create_quantum_circuit(num_qubits, theta_values):
    # Create a parameterized quantum circuit
    theta = Parameter('θ')
    qc = QuantumCircuit(num_qubits)
    for i in range(num_qubits):
        qc.h(i)
        qc.rz(theta, i)
    qc.cz(0, 1)
    qc.measure_all()
    return qc, theta

def execute_circuit(qc, theta, theta_values, backend='qasm_simulator', shots=1024):
    # Execute the quantum circuit with different theta values
    compiled_circuit = qc.bind_parameters({theta: theta_values})
    backend = Aer.get_backend(backend)
    job = execute(compiled_circuit, backend, shots=shots)
    result = job.result()
    counts = result.get_counts(compiled_circuit)
    return counts

# Example usage
if __name__ == '__main__':
    num_qubits = 2
    theta_values = [0, 1.57, 3.14, 4.71, 6.28]
    qc, theta = create_quantum_circuit(num_qubits, theta_values)
    for theta_val in theta_values:
        counts = execute_circuit(qc, theta, theta_val)
        print(f"Theta = {theta_val}, Counts = {counts}")

from qiskit import QuantumCircuit, Aer, execute
from qiskit.circuit import Parameter

def create_quantum_circuit(num_qubits):
    """
    Create a parameterized quantum circuit.

    Parameters:
    - num_qubits (int): Number of qubits in the quantum circuit.

    Returns:
    - QuantumCircuit: Parameterized quantum circuit.
    - Parameter: Parameter representing theta.
    """
    theta = Parameter('θ')
    qc = QuantumCircuit(num_qubits)
    for i in range(num_qubits):
        qc.h(i)
        qc.rz(theta, i)
    qc.cz(0, 1)
    qc.measure_all()
    return qc, theta

def execute_circuit(qc, theta, theta_values, backend='qasm_simulator', shots=1024):
    """
    Execute the parameterized quantum circuit with different theta values.

    Parameters:
    - qc (QuantumCircuit): Parameterized quantum circuit.
    - theta (Parameter): Parameter representing theta.
    - theta_values (list): List of theta values for parameterizing the circuit.
    - backend (str): Backend to execute the circuit on (default='qasm_simulator').
    - shots (int): Number of shots for each execution (default=1024).

    Returns:
    - list: List of dictionaries containing counts of measurement outcomes for each theta value.
    """
    compiled_circuits = [qc.bind_parameters({theta: theta_val}) for theta_val in theta_values]
    backend = Aer.get_backend(backend)
    job = execute(compiled_circuits, backend, shots=shots)
    results = job.result()
    counts_list = [results.get_counts(circ) for circ in compiled_circuits]
    return counts_list

# Example usage
if __name__ == '__main__':
    num_qubits = 2
    theta_values = [0, 1.57, 3.14, 4.71, 6.28]
    qc, theta = create_quantum_circuit(num_qubits)
    counts_list = execute_circuit(qc, theta, theta_values)
    for theta_val, counts in zip(theta_values, counts_list):
        print(f"Theta = {theta_val}, Counts = {counts}")

from qiskit import QuantumCircuit, execute, Aer
import logging

class QuantumClassicalInterface:
    def __init__(self):
        self.simulator = Aer.get_backend('qasm_simulator')
        logging.basicConfig(level=logging.INFO)
    
    def create_quantum_circuit(self, num_qubits):
        """
        Create a quantum circuit with Hadamard gates applied to all qubits and measurement.
        """
        circuit = QuantumCircuit(num_qubits)
        circuit.h(range(num_qubits))  # Apply Hadamard gate to all qubits
        circuit.measure_all()  # Measure all qubits
        logging.info(f"Created quantum circuit with {num_qubits} qubits.")
        return circuit

    def run_simulation(self, circuit, shots=1000):
        """
        Run a simulation of the quantum circuit.
        """
        logging.info("Running quantum circuit simulation.")
        job = execute(circuit, self.simulator, shots=shots)
        result = job.result()
        counts = result.get_counts(circuit)
        logging.info(f"Simulation results: {counts}")
        return counts

    def batch_process_simulations(self, num_qubits, num_simulations=5):
        """
        Perform multiple simulations of the quantum circuit.
        """
        results = []
        for _ in range(num_simulations):
            qc = self.create_quantum_circuit(num_qubits)
            result = self.run_simulation(qc)
            results.append(result)
        return results

# Example usage
if __name__ == '__main__':
    interface = QuantumClassicalInterface()
    qc = interface.create_quantum_circuit(5)
    results = interface.run_simulation(qc)
    batch_results = interface.batch_process_simulations(5, 3)
    print("Single Simulation Results:", results)
    print("Batch Simulation Results:", batch_results)

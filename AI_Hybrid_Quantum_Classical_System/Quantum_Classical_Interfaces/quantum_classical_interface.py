
# /Quantum_Classical_Interfaces/quantum_classical_interface.py

from qiskit import QuantumCircuit, execute, Aer

class QuantumClassicalInterface:
    def create_quantum_circuit(self, num_qubits):
        circuit = QuantumCircuit(num_qubits)
        for qubit in range(num_qubits):
            circuit.h(qubit)
        circuit.measure_all()
        return circuit

    def run_simulation(self, circuit):
        simulator = Aer.get_backend('qasm_simulator')
        job = execute(circuit, simulator, shots=1000)
        result = job.result()
        counts = result.get_counts(circuit)
        return counts

# Example usage
if __name__ == '__main__':
    interface = QuantumClassicalInterface()
    qc = interface.create_quantum_circuit(5)
    results = interface.run_simulation(qc)
    print("Simulation Results:", results)

from flask import Flask, request, jsonify, render_template
import logging
from quantum_classical_interface import QuantumClassicalInterface

# Configure the logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Flask application
app = Flask(__name__)

# Create an instance of the QuantumClassicalInterface
quantum_interface = QuantumClassicalInterface()

@app.route('/')
def home():
    """Serve the homepage with instructions and possibly a form to submit quantum circuit configurations."""
    return render_template('index.html')

@app.route('/run_simulation', methods=['POST'])
def run_simulation():
    """Endpoint to handle the simulation of quantum circuits based on user input."""
    num_qubits = int(request.form.get('num_qubits', 5))
    qc = quantum_interface.create_quantum_circuit(num_qubits)
    results = quantum_interface.run_simulation(qc)
    return jsonify(results)

@app.route('/batch_simulation', methods=['POST'])
def batch_simulation():
    """Endpoint for performing batch simulations."""
    num_qubits = int(request.form.get('num_qubits', 5))
    num_simulations = int(request.form.get('num_simulations', 3))
    batch_results = quantum_interface.batch_process_simulations(num_qubits, num_simulations)
    return jsonify(batch_results)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

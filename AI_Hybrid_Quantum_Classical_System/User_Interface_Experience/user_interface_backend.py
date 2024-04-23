
# /User_Interface_Experience/user_interface_backend.py

from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/api/submit', methods=['POST'])
def handle_submission():
    data = request.get_json()
    response_message = process_data(data['input'])
    return jsonify({'response': response_message})

def process_data(input_data):
    # Process the input data and return a response
    return f"Processed data: {input_data}"

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
